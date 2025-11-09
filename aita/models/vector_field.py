from typing import Optional

import dgl
import torch
from torch import nn

from .layers.swish import SwishBeta
from .layers.layernorm import AdaLN
from .layers.gvp import GVPConv, NodePositionUpdate
from .layers.embeddings import FourierEmbedding, PositionalEncoding


class GVP_vector_field(nn.Module):
    """Vector field head built on stacks of GVP convolutions."""

    def __init__(
        self,
        n_features: int = 21,
        n_layers: int = 5,
        n_hidden: int = 64,
        n_vec: int = 16,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        n_coord_gvps: int = 1,
        use_dst_feats: bool = False,
        vector_gating: bool = True,
        self_conditioning: bool = False,
    ) -> None:
        super().__init__()

        self.n_vec_channels = n_vec
        self.self_conditioning = self_conditioning

        self.adaln = AdaLN(n_hidden, n_hidden)
        self.temporal_embedding = FourierEmbedding(n_hidden)
        self.positional_embedding = PositionalEncoding(n_hidden)
        self.initial_embedding = nn.Sequential(
            nn.Linear(n_features + n_hidden, n_hidden),
            nn.SiLU(),
        )

        if self_conditioning:
            self.conditional_embedding = nn.Sequential(
                nn.Linear(n_features, n_hidden),
                nn.SiLU(),
            )
            self.conditional_convolution = GVPConv(
                scalar_size=n_hidden,
                vector_size=n_vec,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                use_dst_feats=use_dst_feats,
                vector_gating=vector_gating,
                coords_range=10,
                scalar_activation=SwishBeta,
            )

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
        graph: dgl.DGLGraph,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the vector field forward pass.

        Args:
            t: Diffusion timestep of shape `(batch_size,)`.
            graph: Batched DGL graph with node features `ndata['h']` `(num_nodes, n_features)`
                and coordinates `ndata['x']` `(num_nodes, 3)`.
            condition: Optional conditioning tensor broadcastable per node.

        Returns:
            Vector field predictions of shape `(num_nodes, 3)`.
        """

        x_init = graph.ndata["xt"]  # (num_nodes, 3)
        device = x_init.device
        v_init = torch.zeros(
            graph.num_nodes(),
            self.n_vec_channels,
            3,
            device=device,
        )  # (num_nodes, n_vec_channels, 3)

        th = self.temporal_embedding(graph.ndata["t"].view(-1)) # NOTE: we expect the time to be a 1D tensor
        # th: (num_nodes, n_hidden)

        ph = self.positional_embedding(graph.ndata["atom_index"].view(-1)) # NOTE: we expect the atom index to be a 1D tensor
        # ph: (num_nodes, n_hidden)

        z_init = torch.cat([graph.ndata["attr"], ph], dim=1)
        # z_init: (num_nodes, n_features + 1)

        zs = self.initial_embedding(z_init)
        # zs: (num_nodes, n_hidden)

        # update zs with the time embedding
        zs = self.adaln(zs, th)
        # zs: (num_nodes, n_hidden)

        if condition is not None and self.self_conditioning:
            z_cond = graph.ndata["attr"].clone().float()
            # z_cond: (num_nodes, n_features)

            z_cond = self.conditional_embedding(z_cond)
            # z_cond: (num_nodes, n_hidden)

            v_cond = v_init.clone()
            # v_cond: (num_nodes, n_vec_channels, 3)

            h_cond, v_cond, _ = self.conditional_convolution(
                graph,
                z_cond,
                condition,
                v_cond,
            )
            # h_cond: (num_nodes, n_hidden), v_cond: (num_nodes, n_vec_channels, 3)

            zs = zs + h_cond

        hs, vs, xs = zs, v_init, x_init
        for conv in self.convs:
            hs, vs, xs = conv(graph, hs, xs, vs)
            # hs: (num_nodes, n_hidden)
            # vs: (num_nodes, n_vec_channels, 3)
            # xs: (num_nodes, 3)

        vector_field = self.position_updater(hs, vs)
        # vector_field: (num_nodes, 3)
        return vector_field
    
    def training_step(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # predict velocity
        velocity = self(graph)

        # compute loss
        graph.ndata["loss_per_node"] = torch.square(velocity - graph.ndata["vt"]).mean(dim=-1)
        loss_per_molecule = dgl.mean_nodes(graph, "loss_per_node")
        loss = loss_per_molecule.mean()

        return {"loss": loss}