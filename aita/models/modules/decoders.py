import dgl
import torch
from torch import nn, Tensor

from ..layers.swish import SwishBeta
from ..layers.gvp import GVPConv, NodePositionUpdate


class GVP_Decoder(nn.Module):

    def __init__(
        self,
        n_layers: int = 5,
        n_hidden: int = 64,
        n_vec: int = 16,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        n_coord_gvps: int = 1,
        use_dst_feats: bool = False,
        vector_gating: bool = True,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList(
            [
                GVPConv(
                    scalar_size=n_hidden,
                    vector_size=n_vec,
                    n_message_gvps=n_message_gvps,
                    n_update_gvps=n_update_gvps,
                    use_dst_feats=use_dst_feats,
                    vector_gating=vector_gating,
                    coords_range=10,
                    scalar_activation=SwishBeta,
                )
                for _ in range(n_layers)
            ]
        )

        self.position_updater = NodePositionUpdate(
            n_scalars=n_hidden,
            n_vec_channels=n_vec,
            n_gvps=n_coord_gvps,
        )
    
    def forward(
        self,
        h: Tensor,
        v_init: Tensor,
        x_init: Tensor,
        graph: dgl.DGLGraph,
    ):

        hs, vs, xs = h, v_init, x_init
        for conv in self.convs:
            hs, vs, xs = conv(graph, hs, xs, vs)
            # hs: (num_nodes, n_hidden)
            # vs: (num_nodes, n_vec_channels, 3)
            # xs: (num_nodes, 3)

        vector_field = self.position_updater(hs, vs)
        # vector_field: (num_nodes, 3)
        return vector_field