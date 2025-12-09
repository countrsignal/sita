import dgl
import torch
from torch import nn, Tensor

from typing import Union, Tuple, Dict

from ..layers.layernorm import AdaLN
from ..layers.embeddings import FourierEmbedding

from ..layers.gvp import EnergyGVPConv, EnergyHead, _rbf
from ..layers.swish import SwishBeta
from ...plans import TrigPlan, expand_t_like


@torch.no_grad()
def target_score_from_velocity_model(
    plan: TrigPlan,
    velocity: Tensor,
    x: Tensor,
    t: Tensor,
) -> Tensor:
    assert isinstance(plan, TrigPlan), f"plan must be an instance of TrigPlan, got {type(plan)}"

    return plan.get_score_from_velocity(t, x, velocity)


def _compute_edge_features(
    node_positions: Tensor,
    src_idx: Tensor,
    dst_idx: Tensor,
    rbf_dmax: float,
    rbf_dim: int,
) -> Tuple[Tensor, Tensor]:
    """Compiled helper to compute normalized edge vectors and RBF distances."""
    x_src = node_positions.index_select(0, src_idx)
    x_dst = node_positions.index_select(0, dst_idx)

    x_diff = x_src - x_dst + 1e-8
    dij = torch.square(x_diff).sum(dim=-1, keepdim=True).sqrt() + 1e-8
    x_diff = x_diff / dij
    d = _rbf(dij.squeeze(-1), D_max=rbf_dmax, D_count=rbf_dim)
    return x_diff, d


class EnergyNet(nn.Module):

    def __init__(
        self,
        n_vec: int = 16,
        n_hidden: int = 64,
        edge_feat_size: int = 32,
        n_message_gvps: int = 1,
        n_output_gvps: int = 1,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        message_norm: str = "sum",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        self.time_to_attr = AdaLN(n_hidden, n_hidden)
        self.temporal_embedding  = FourierEmbedding(n_hidden)

        self.gvp_conv = EnergyGVPConv(
            scalar_size=n_hidden,
            vector_size=n_vec,
            n_message_gvps=n_message_gvps,
            use_dst_feats=False,
            vector_gating=True,
            scalar_activation=SwishBeta,
            vector_activation=nn.Sigmoid,
            rbf_dmax=rbf_dmax,
            rbf_dim=rbf_dim,
            edge_feat_size=edge_feat_size,
            message_norm=message_norm,
            dropout=dropout,
        )
        self.energy_head = EnergyHead(
            n_scalars=n_hidden,
            n_vec_channels=n_vec,
            n_gvps=n_output_gvps,
        )
    
    def forward(
        self,
        x_t: Tensor,
        time: Tensor,
        node_scalars: Tensor,
        node_vectors: Tensor,
        edge_feats: Tensor,
        graph: dgl.DGLGraph,
    ) -> Tensor:
        # x_t: (num_nodes, 3)
        # time: (num_nodes,)
        # node_scalars: (num_nodes, n_hidden)
        # node_vectors: (num_nodes, n_vec_channels, 3)
        # edge_feats: (num_edges, n_hidden)

        # compute the edge features
        src_idx, dst_idx = graph.edges()
        x_diff, d = _compute_edge_features(x_t, src_idx, dst_idx, self.rbf_dmax, self.rbf_dim)
        # x_diff: (num_edges, 3)
        # d: (num_edges, rbf_dim)

        # energy backbone
        hs, vs = self.gvp_conv(graph, node_scalars, node_vectors, edge_feats, x_diff, d)
        # hs: (num_nodes, n_hidden)
        # vs: (num_nodes, n_vec_channels, 3)

        # encode the time
        th = self.temporal_embedding(time)
        # th: (num_nodes, n_hidden)

        # time to scalars
        hs = self.time_to_attr(hs, th)
        # hs: (num_nodes, n_hidden)

        # energy head
        graph.ndata["energy_per_node"] = self.energy_head(hs, vs)
        # graph.ndata["energy_per_node"]: (num_nodes, 1)

        # readout the energy over all atoms in a molecule
        energy = dgl.sum_nodes(graph, "energy_per_node")
        # energy: (num_molecules, )
        return energy

    def fwd_with_grad(
        self,
        x_t: Tensor,
        time: Tensor,
        node_scalars: Tensor,
        node_vectors: Tensor,
        edge_feats: Tensor,
        graph: dgl.DGLGraph,
        require_grad: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        # x_t: (num_nodes, 3)
        # time: (num_nodes,)
        # node_scalars: (num_nodes, n_hidden)
        # node_vectors: (num_nodes, n_vec_channels, 3)
        # edge_feats: (num_edges, n_hidden)

        torch_grad = self.training or require_grad

        if torch_grad:
            x_t = x_t.requires_grad_()

        with torch.set_grad_enabled(torch_grad):
            energy = self(x_t, time, node_scalars, node_vectors, edge_feats, graph)
            # energy: (batch_size, )

            position_grad = torch.autograd.grad(
                energy.sum(), x_t, create_graph=True
            )[0]
            # position_grad: (num_nodes, 3)
            return position_grad, energy
    
    def training_step(
        self,
        z: Tensor,
        x_t: Tensor,
        time: Tensor,
        node_scalars: Tensor,
        node_vectors: Tensor,
        edge_feats: Tensor,
        graph: dgl.DGLGraph,
    ) -> Dict[str, Tensor]:
        # x_t: (num_nodes, 3)
        # time: (num_nodes,)
        # node_scalars: (num_nodes, n_hidden)
        # node_vectors: (num_nodes, n_vec_channels, 3)
        # edge_feats: (num_edges, n_hidden)

        # get the indices of the edges
        src_idx, dst_idx = graph.edges()

        # enable gradient tracking
        x_t = x_t.requires_grad_()

        # score matching loss ================================================================
        with torch.set_grad_enabled(True):
            # compute the edge features
            x_diff, d = _compute_edge_features(x_t, src_idx, dst_idx, self.rbf_dmax, self.rbf_dim)
            # x_diff: (num_edges, 3)
            # d: (num_edges, rbf_dim)

            # energy backbone
            hs, vs = self.gvp_conv(graph, node_scalars, node_vectors, edge_feats, x_diff, d)
            # hs: (num_nodes, n_hidden)
            # vs: (num_nodes, n_vec_channels, 3)

            # encode the time
            th = self.temporal_embedding(time)
            # th: (num_nodes, n_hidden)

            # time to scalars
            hs_t = self.time_to_attr(hs, th)
            # hs: (num_nodes, n_hidden)

            # energy head
            # NOTE: no local scope here because we need to backpropagate through the energy head
            graph.ndata["energy_per_node"] = self.energy_head(hs_t, vs)
            # graph.ndata["energy_per_node"]: (num_nodes, 1)

            # readout the energy over all atoms in a molecule
            energy = dgl.sum_nodes(graph, "energy_per_node")
            # energy: (num_molecules, )

            # compute the gradient of the energy with respect to x_t
            pred_score = torch.autograd.grad(
                energy.sum(), x_t, create_graph=True
            )[0]
            # pred_score: (num_nodes, 3)

        # compute score matching loss
        sigma_t = graph.ndata["sigma_t"]
        with graph.local_scope():
            graph.ndata["sm_loss_per_node"] = torch.square(sigma_t * pred_score - z).mean(dim=-1)
            sm_loss = dgl.mean_nodes(graph, "sm_loss_per_node").mean()
        # ====================================================================================

        # compute nce loss ===================================================================
        # > sample negative times
        batch_size = graph.batch_size
        n_atoms_per_molecule = graph.batch_num_nodes() # NOTE: this is a tensor with the number of atoms in each molecule which can vary across the batch
        perturb = torch.randn(batch_size, device=x_t.device) * 0.025
        negative_t = time + perturb.repeat_interleave(n_atoms_per_molecule)
        negative_t = torch.clamp(negative_t, 0.0, 1.0)

        # > compute the energy of the negative samples
        negative_th = self.temporal_embedding(negative_t)
        negative_hs = self.time_to_attr(hs, negative_th)
        # negative_hs: (num_nodes, n_hidden)

        # > energy head
        with graph.local_scope():
            graph.ndata["negative_energy_per_node"] = self.energy_head(negative_hs, vs)
            # graph.ndata["negative_energy_per_node"]: (num_nodes, 1)

            # > readout the energy over all atoms in a molecule
            negative_energy = dgl.sum_nodes(graph, "negative_energy_per_node")
            # negative_energy: (num_molecules, )

        # > nce loss
        nce_loss = -torch.mean(
            energy - torch.logsumexp(
                torch.cat([energy, negative_energy], dim=-1),
                dim=-1,
                keepdim=True,
            )
        )
        # ====================================================================================

        return {"sm_loss": sm_loss, "nce_loss": nce_loss}