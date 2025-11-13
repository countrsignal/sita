import dgl
import dgl.function as fn

import torch
from torch import nn, Tensor

from typing import Union

from ..layers.swish import SwishBeta
from ..layers.gvp import GVPConv, NodePositionUpdate, EdgeUpdate, _norm_no_nan, _rbf


class GVP_Decoder(nn.Module):

    def __init__(
        self,
        n_vec: int = 16,
        n_layers: int = 5,
        n_hidden_nodes: int = 64,
        n_hidden_edge: int = 32,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        n_coord_gvps: int = 1,
        rbf_dim: int = 16,
        rbf_dmax: float = 20,
        message_norm: Union[float, str] = "sum",
        use_dst_feats: bool = False,
        vector_gating: bool = True,
    ) -> None:
        super().__init__()

        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.n_layers = n_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_edge = n_hidden_edge

        self.convs = nn.ModuleList([])
        self.edge_updater = nn.ModuleList([])
        for _ in range(n_layers):
            self.convs.append(
                GVPConv(
                    scalar_size=n_hidden_nodes,
                    vector_size=n_vec,
                    n_message_gvps=n_message_gvps,
                    n_update_gvps=n_update_gvps,
                    use_dst_feats=use_dst_feats,
                    rbf_dmax=rbf_dmax,
                    rbf_dim=rbf_dim,
                    edge_feat_size=n_hidden_edge,
                    coords_range=10.0,
                    message_norm=message_norm,
                    vector_gating=vector_gating,
                    scalar_activation=SwishBeta,
                    vector_activation=nn.Sigmoid,
                )
            )
            self.edge_updater.append(
                EdgeUpdate(
                    n_node_scalars=n_hidden_nodes,
                    n_edge_feats=n_hidden_edge,
                    update_edge_w_distance=True,
                    rbf_dim=rbf_dim,
                )
            )

        self.position_updater = NodePositionUpdate(
            n_scalars=n_hidden_nodes,
            n_vec_channels=n_vec,
            n_gvps=n_coord_gvps,
        )

    def precompute_distances(self, g: dgl.DGLGraph, node_positions: Tensor):
        """Precompute the pairwise distances between all nodes in the graph."""

        with g.local_scope():

            g.ndata['x_d'] = node_positions

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))
            dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
            x_diff = g.edata['x_diff'] / dij
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        
        return x_diff, d

    def forward(
        self,
        h: Tensor,
        v_init: Tensor,
        x_init: Tensor,
        edge_repr: Tensor,
        edge_mask: Tensor,
        graph: dgl.DGLGraph,
    ):

        hs, vs, xs = h, v_init, x_init
        for conv, edge_nn in zip(self.convs, self.edge_updater):
            x_diff, d = self.precompute_distances(graph, xs)
            edge_repr = edge_nn(graph, hs, edge_repr, d=d) * edge_mask
            hs, vs, xs = conv(graph, hs, xs, vs, edge_feats=edge_repr, x_diff=x_diff, d=d)
            # hs: (num_nodes, n_hidden)
            # vs: (num_nodes, n_vec_channels, 3)
            # xs: (num_nodes, 3)

        vector_field = self.position_updater(hs, vs)
        # vector_field: (num_nodes, 3)
        return vector_field