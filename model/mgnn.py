from typing import Callable, Dict, Optional, Tuple, Union, Any

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Embedding, Linear

from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, SparseTensor

from torch_geometric.nn import MLP
from math import pi

from fairchem.core.models.base import GraphModelMixin
from fairchem.core.common.utils import (
    conditional_grad,
    radius_graph_pbc,
    get_pbc_distances,
    compute_neighbors,
)
from fairchem.core.models.painn.utils import get_edge_id, repeat_blocks
from torch_scatter import scatter, segment_coo
import logging




def orthogonal_init_weight(m):
    if isinstance(m, nn.Linear):
        glorot_orthogonal(m.weight, scale=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def triplets(
    row: Tensor,
    col: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # row, col = edge_index  # col -> row

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=row, col=col, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (j-->i<--k) for triplets.
    # idx_i = row.repeat_interleave(num_triplets)
    idx_j = col.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()

    mask = idx_j != idx_k  # Remove j == k triplets.
    # idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (j->i, k->i) for triplets.
    idx_ji = adj_t_row.storage.row()[mask]
    idx_ki = adj_t_row.storage.value()[mask]

    return row, col, idx_ji, idx_ki


@torch.jit.script
def chebyshev_polynomial(n: int, x: Tensor, R_cut: float, R_min: float = 0.0):
    # x: (E, 1)
    # Map x in interval [R_min, R_cut] to [-1, 1]
    x = 2 * (x - R_min) / (R_cut - R_min) - 1

    if n == 1:
        return x

    else:

        chebyshev_values = [torch.ones_like(x), x]
        for k in range(2, n):
            next_value = 2 * x * chebyshev_values[k - 1] - chebyshev_values[k - 2]
            chebyshev_values.append(next_value)

        return torch.cat(chebyshev_values, dim=1)  # (E, n)


class ChebyshevPolynomialCutoff(nn.Module):
    def __init__(self, beta: int, R_cut: float, R_min: float = 1e-5):
        super().__init__()
        self.beta = beta
        self.R_min = R_min
        self.R_cut = R_cut

    def forward(self, x: Tensor):
        # x: (E, 1) -> (E, beta)
        poly = chebyshev_polynomial(self.beta, x, self.R_cut, self.R_min) * (x - self.R_cut).pow(
            2.0
        )
        poly *= (x < self.R_cut).float()
        return poly


class Gate(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        act: Optional[Union[str, Callable]] = "silu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.out_channels = out_channels

        self.s_mlp = MLP(
            [in_channels + out_channels, hidden_channels, 2 * out_channels],
            act=act,
            act_kwargs=act_kwargs,
            norm=None,
        )
        self.v_lin = nn.Linear(in_channels, 2 * out_channels, bias=False)

    def forward(self, sv: Tuple[torch.Tensor, torch.Tensor]):
        """
        scalar: (N, F)
        vector: (N, 3, F)
        """
        scalar, vector = sv
        vector = self.v_lin(vector)
        vec_s, vec_v = torch.split(vector, self.out_channels, dim=-1)
        vec_s = torch.norm(vec_s, dim=-2)
        scalar = self.s_mlp(torch.cat([scalar, vec_s], dim=-1))
        s_s, s_v = torch.split(scalar, self.out_channels, dim=-1)
        vector = s_v[:, None] * vec_v
        scalar = self.act(s_s)

        return scalar, vector


class MomentInteraction(nn.Module):
    def __init__(
        self,
        F: int,
        act: Optional[Union[str, Callable]] = "silu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.F = F
        self.act = activation_resolver(act, **(act_kwargs or {}))

        # Mixing
        self.split_net = MLP([2 * F, 2 * F, 2 * F], act=act, act_kwargs=act_kwargs, norm=None)

        self.lin_s01 = Linear(F, F)
        self.lin_s02 = Linear(F, F)
        self.lin_s = Linear(2 * F, F)

        self.lin_out = Linear(F, F)

    def forward(
        self,
        f: Tensor,
        M01: Tensor,
        M02: Tensor,
        num_edges: int,
        idx_ji: Tensor,
        idx_ki: Tensor,
    ):

        f = self.act(self.split_net(f))  # (E, 2F)
        f_01, f_02 = torch.split(f, self.F, dim=-1)  # (E, F) * 2

        s01 = f_01[idx_ki] * M01 * f_01[idx_ji]
        s01 = self.act(self.lin_s01(s01))

        s02 = f_02[idx_ki] * M02 * f_02[idx_ji]
        s02 = self.act(self.lin_s02(s02))

        # Scatter: activation -> scatter -> linear
        s = self.lin_s(torch.cat((s01, s02), dim=-1))  # (T, F)
        out = scatter(self.act(s), idx_ji, dim=0, dim_size=num_edges, reduce="sum")  # (E, F)

        return self.lin_out(out)


class Interaction(nn.Module):
    def __init__(
        self,
        F: int,
        beta: int = 8,
        act: Optional[Union[str, Callable]] = "silu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.F = F
        self.beta = beta
        self.act = activation_resolver(act, **(act_kwargs or {}))

        self.lin_z = MLP([F, F, F], act=act, act_kwargs=act_kwargs, norm=None)
        self.lin_c = Linear(2 * F, F * beta)
        self.lin_split = Linear(F, 3 * F)

        self.moment = MomentInteraction(F, act, act_kwargs)
        self.lin_out = Linear(F, F)

    def forward(
        self,
        z: Tensor,
        Q: Tensor,
        M01: Tensor,
        M02: Tensor,
        num_nodes: int,
        num_edges: int,
        i: Tensor,
        j: Tensor,
        idx_ji: Tensor,
        idx_ki: Tensor,
    ):

        z = self.lin_z(z)  # (N, F)
        c = self.lin_c(torch.cat((z[i], z[j]), dim=-1))  # (E, F * beta)
        cQ = torch.sum(c.view(-1, self.F, self.beta) * Q[:, None], dim=-1)  # (E, F)
        z = self.lin_split(cQ)  # (E, 3F)

        dz, f = torch.split(z, [self.F, 2 * self.F], dim=-1)

        # Moment Interaction
        B_alpha = self.moment(f, M01, M02, num_edges, idx_ji, idx_ki)  # (E, F)

        out = dz + B_alpha  # (E, F)
        out = scatter(self.act(out), i, 0, dim_size=num_nodes, reduce="sum")  # (N, F)

        return self.lin_out(out)


class MGNN(nn.Module):
    def __init__(
        self,
        max_z: int = 100,
        cutoff: float = 5,
        max_neighbors: int = 30,
        F: int = 128,
        beta: int = 8,
        num_layers: int = 3,
        n_out: int = 1,
        R_min: float = 0,
        regress_forces: bool = True,
        direct_forces: bool = True,
        regress_stresses: bool = False,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        otf_graph: bool = True,
        enforce_max_neighbors_strictly: Optional[bool]=None,
        post_transforms = None,
        act: Optional[Union[str, Callable]] = "silu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.F = F
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.regress_forces = regress_forces
        self.regress_stresses = regress_stresses
        self.direct_forces = direct_forces
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly
        self.post_transforms = post_transforms

        self.cpc = ChebyshevPolynomialCutoff(beta=beta, R_cut=cutoff, R_min=R_min)

        self.embedding = Embedding(max_z, F)

        self.interactions = nn.ModuleList(
            [Interaction(F, beta, act, act_kwargs) for _ in range(num_layers)]
        )

        self.energy_net = MLP([F, F, n_out], act=act, act_kwargs=act_kwargs, norm=None)

        if self.regress_forces is True and self.direct_forces is True:
            self.weight = nn.Linear(F, F)
            self.act = activation_resolver(act, **(act_kwargs or {}))
            force_net = nn.ModuleList(
                [
                    Gate(in_channels=F, out_channels=F, act=act, act_kwargs=act_kwargs),
                    Gate(in_channels=F, out_channels=1, act=act, act_kwargs=act_kwargs),
                ]
            )
            self.force_net = nn.Sequential(*force_net)

        # self.apply(orthogonal_init_weight)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.energy_net.lins:
            nn.init.xavier_uniform_(lin.weight)
            lin.bias.data.fill_(0.0)

    # Borrowed from GemNet.
    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg) -> torch.Tensor:
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        return tensor_cat[reorder_idx]

    # Borrowed from GemNet.
    def symmetrize_edges(
        self,
        edge_index,
        cell_offsets,
        neighbors,
        reorder_tensors,
        reorder_tensors_invneg,
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """
        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] == 0) & (cell_offsets[:, 2] < 0))
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [edge_index_new, edge_index_new.flip(0)],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        # segment_coo assumes sorted batch_edge
        # Factor 2 since this is only one half of the edges
        ones = batch_edge.new_ones(1).expand_as(batch_edge)
        neighbors_per_image = 2 * segment_coo(ones, batch_edge, dim_size=neighbors.size(0))

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            torch.div(neighbors_per_image, 2, rounding_mode="floor"),
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(cell_offsets, mask, edge_reorder_idx, True)
        reorder_tensors = [
            self.select_symmetric_edges(tensor, mask, edge_reorder_idx, False)
            for tensor in reorder_tensors
        ]
        reorder_tensors_invneg = [
            self.select_symmetric_edges(tensor, mask, edge_reorder_idx, True)
            for tensor in reorder_tensors_invneg
        ]

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_per_image,
            reorder_tensors,
            reorder_tensors_invneg,
        )

    # Copy from FairChem.core.models.base.GraphModelMixin
    def generate_graph(
        self,
        data,
        enforce_max_neighbors_strictly=None,
    ):
        cutoff = self.cutoff
        max_neighbors = self.max_neighbors
        use_pbc = self.use_pbc
        use_pbc_single = self.use_pbc_single
        otf_graph = self.otf_graph

        if enforce_max_neighbors_strictly is not None:
            pass
        elif hasattr(self, "enforce_max_neighbors_strictly"):
            # Not all models will have this attribute
            enforce_max_neighbors_strictly = self.enforce_max_neighbors_strictly
        else:
            # Default to old behavior
            enforce_max_neighbors_strictly = True

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                if use_pbc_single:
                    (
                        edge_index_per_system,
                        cell_offsets_per_system,
                        neighbors_per_system,
                    ) = list(
                        zip(
                            *[
                                radius_graph_pbc(
                                    data[idx],
                                    cutoff,
                                    max_neighbors,
                                    enforce_max_neighbors_strictly,
                                )
                                for idx in range(len(data))
                            ]
                        )
                    )

                    # atom indexs in the edge_index need to be offset
                    atom_index_offset = data.natoms.cumsum(dim=0).roll(1)
                    atom_index_offset[0] = 0
                    edge_index = torch.hstack(
                        [
                            edge_index_per_system[idx] + atom_index_offset[idx]
                            for idx in range(len(data))
                        ]
                    )
                    cell_offsets = torch.vstack(cell_offsets_per_system)
                    neighbors = torch.hstack(neighbors_per_system)
                else:
                    ## TODO this is the original call, but blows up with memory
                    ## using two different samples
                    ## sid='mp-675045-mp-675045-0-7' (MPTRAJ)
                    ## sid='75396' (OC22)
                    edge_index, cell_offsets, neighbors = radius_graph_pbc(
                        data,
                        cutoff,
                        max_neighbors,
                        enforce_max_neighbors_strictly,
                    )

        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
            neighbors = compute_neighbors(data, edge_index)

        data.edge_index = edge_index
        data.cell_offsets = cell_offsets
        data.neighbors = neighbors

        return data

    def generate_graph_values(self, data):

        edge_index = data.edge_index
        pos = data.pos
        neighbors = data.neighbors
        cell = data.cell
        cell_offsets = data.cell_offsets

        row, col = edge_index

        distance_vectors = pos[row] - pos[col]

        # correct for pbc
        cell = torch.repeat_interleave(cell, neighbors, dim=0)
        offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
        distance_vectors += offsets

        # compute distances
        distances = distance_vectors.norm(dim=-1)

        # redundancy: remove zero distances
        nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
        edge_index = edge_index[:, nonzero_idx]
        distances = distances[nonzero_idx]

        distance_vectors = distance_vectors[nonzero_idx]
        offsets = offsets[nonzero_idx]

        # Unit vectors pointing from edge_index[1] to edge_index[0],
        # i.e., edge_index[0] - edge_index[1] divided by the norm.
        # make sure that the distances are not close to zero before dividing
        mask_zero = torch.isclose(distances, torch.tensor(0.0), atol=1e-6)
        distances[mask_zero] = 1.0e-6
        edge_vector = distance_vectors / distances[:, None]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )

        # Symmetrize edges for swapping in symmetric message passing
        (
            edge_index,
            cell_offsets,
            neighbors,
            [edge_dist],
            [edge_vector],
        ) = self.symmetrize_edges(
            edge_index,
            cell_offsets,
            neighbors,
            [distances],
            [edge_vector],
        )

        return edge_index, edge_dist, edge_vector

    @conditional_grad(torch.enable_grad())
    def forward(self, data):

        # Calculate edge_index, cell_offcets, neighbors
        data = self.generate_graph(data)

        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        num_graphs = data.num_graphs
        edge_index = data.edge_index

        if self.regress_forces and not self.direct_forces:
            pos = pos.requires_grad_(True)

        if self.regress_stresses:
            cell = data.cell
            strain = torch.zeros_like(cell)
            strain.requires_grad_()
            strain = strain.transpose(1, 2)

            # strain cell
            data.cell = cell + torch.matmul(cell, strain)

            # strain positions
            batch = data.batch
            strain_i = strain[batch]
            data.pos = pos + torch.matmul(pos[:, None, :], strain_i).squeeze(1)

        edge_index, edge_dist, edge_vector = self.generate_graph_values(data)

        num_nodes = z.size(0)
        num_edges = edge_index.size(1)

        row, col = edge_index
        _, _, idx_ji, idx_ki = triplets(row, col, num_nodes=num_nodes)

        # Calculate distances
        dis = edge_dist.unsqueeze(-1)

        # Edge directions
        dir_ij = edge_vector

        pos_ji = dir_ij[idx_ji]  # j -> i
        pos_ki = dir_ij[idx_ki]  # k -> i

        # M_01: Rji · Rki
        m01 = torch.sum(pos_ji * pos_ki, dim=-1, keepdim=True)
        # M_02: (Rij ⓧ Rij) : (Rij ⓧ Rij)
        m02 = m01.pow(2)

        # Node
        z = self.embedding(z)  # (N, F)

        # Edge
        q = self.cpc(dis)  # Q (E, beta)

        # Interaction
        for interaction in self.interactions:
            v = interaction(z, q, m01, m02, num_nodes, num_edges, row, col, idx_ji, idx_ki)
            z = z + v

        per_atom_energy = self.energy_net(z)  # (N, 1)
        energy = scatter(per_atom_energy, batch, 0, dim_size=num_graphs)  # energy
        outputs = {"energy": energy}

        if self.regress_forces:
            if self.direct_forces:
                weight = self.act(self.weight(z))[col]
                vector_feature = scatter(
                    weight[:, None] * dir_ij[..., None],
                    row,
                    dim=0,
                    dim_size=num_nodes,
                    reduce="sum",
                )
                _, forces = self.force_net((z, vector_feature))
                outputs["forces"] = forces.squeeze(-1)

            else:
                dEdx = torch.autograd.grad(
                    energy,
                    pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=self.training,
                )[0]

                outputs["forces"] = -dEdx

        if self.regress_stresses:
            dEdeps = torch.autograd.grad(
                energy,
                strain,
                grad_outputs=torch.ones_like(energy),
                create_graph=self.training,
            )[
                0
            ]  # [B, 3, 3]
            volume = torch.det(data.cell)
            # Convert Stress unit from eV/A^3 to GPa
            scale = 1 / volume * 160.21766208  # [B,]
            outputs["stresses"] = dEdeps * scale[:, None, None]

        return outputs

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
