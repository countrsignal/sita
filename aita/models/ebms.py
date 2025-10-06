from typing import Optional, Tuple

import dgl
import torch
from torch import nn

from .graphormer import Graphormer3D
from .gvp import GVPConv
from .swish import SwishBeta


class GVP_EBM(nn.Module):
    """Energy-based model built on GVP convolutions."""

    def __init__(
        self,
        num_features: int = 21,
        num_layers: int = 8,
        n_hidden: int = 64,
        n_vec: int = 16,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        use_dst_feats: bool = False,
        vector_gating: bool = True,
    ) -> None:
        super().__init__()

        self.n_vec_channels = n_vec

        self.initial_embedding = nn.Sequential(
            nn.Linear(num_features + 1, n_hidden),
            nn.SiLU(),
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
                for _ in range(num_layers)
            ]
        )

        self.output = nn.Sequential(
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(n_hidden, 1, bias=True),
        )

    def forward(
        self,
        t: torch.Tensor,
        data: dgl.DGLGraph,
        return_logprob: bool = False,
        require_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute energies (and optionally gradients) for the input batch.

        Args:
            t: Diffusion timestep tensor `(batch_size,)`.
            data: Batched DGL graph with node features `ndata['h']` `(num_nodes, num_features)`
                and coordinates `ndata['x']` `(num_nodes, 3)`.
            return_logprob: If `True`, return only the energies (log probabilities).
            require_grad: Force gradient tracking of inputs even when not training.

        Returns:
            Either `(position_grad, energy)` with shapes `(num_nodes, 3)` and `(batch_size, 1)`
            or, if `return_logprob` is `True`, the energy tensor `(batch_size, 1)` alone.
        """

        x_init = data.ndata["x"].clone()  # (num_nodes, 3)
        device = x_init.device
        v_init = torch.zeros(
            data.num_nodes(), self.n_vec_channels, 3, device=device
        )  # (num_nodes, n_vec_channels, 3)

        torch_grad = self.training or require_grad

        if torch_grad:
            t = t.requires_grad_()
            x_init = x_init.requires_grad_()
            v_init = v_init.requires_grad_()

        with torch.set_grad_enabled(torch_grad):
            ts = t.repeat_interleave(self.num_particles).view(-1, 1)
            # ts: (num_nodes, 1)

            z_init = torch.cat([data.ndata["h"], ts], dim=1)
            # z_init: (num_nodes, num_features + 1)

            hs, vs, xs = self.initial_embedding(z_init), v_init, x_init

            for conv in self.convs:
                hs, vs, xs = conv(data, hs, xs, vs)
                # hs: (num_nodes, n_hidden)
                # vs: (num_nodes, n_vec_channels, 3)
                # xs: (num_nodes, 3)

            data.ndata["h_out"] = hs
            energy = dgl.mean_nodes(data, "h_out")
            energy = self.output(energy)  # (batch_size, 1)

            if return_logprob:
                return energy

            position_grad = torch.autograd.grad(
                energy.sum(), x_init, create_graph=True
            )[0]
            # position_grad: (num_nodes, 3)
            return position_grad, energy


class graphormer_EBM(nn.Module):
    """Energy-based model powered by Graphormer3D."""

    def __init__(
        self,
        num_features: int = 21,
        num_layers: int = 6,
        embed_dim: int = 512,
        ffn_embed_dim: int = 512,
        attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        num_kernel: int = 50,
        input_dropout: float = 0.1,
        blocks: int = 3,
    ) -> None:
        super().__init__()
        self.graphormer = Graphormer3D(
            num_features=num_features,
            num_layers=num_layers,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            attention_heads=attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            num_kernel=num_kernel,
            input_dropout=input_dropout,
            blocks=blocks,
        )

    def forward(
        self,
        t: torch.Tensor,
        data: dgl.DGLGraph,
        return_logprob: bool = False,
        require_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Graphormer-based energy prediction.

        Args:
            t: Diffusion timestep tensor `(batch_size,)`.
            data: Batched DGL graph.
            return_logprob: If `True`, return energies only.
            require_grad: Force gradient tracking even in eval mode.

        Returns:
            `(position_grad, energy)` or energy only when `return_logprob` is `True`.
        """

        x_init = data.ndata["x"].clone()  # (num_nodes, 3)
        torch_grad = self.training or require_grad

        if torch_grad:
            t = t.requires_grad_()
            x_init = x_init.requires_grad_()

        ts = t.repeat_interleave(data.batch_num_nodes()).view(-1, 1)
        # ts: (num_nodes, 1)

        padded_feats, padded_pos, padded_ts = self.graph_to_padded_sequence(
            data, x_init, ts
        )

        with torch.set_grad_enabled(torch_grad):
            energy, padding_mask = self.graphormer(
                padded_feats, padded_pos, padded_ts
            )
            # energy: (batch_size, 1)
            # padding_mask: (batch_size, max_nodes)

            if return_logprob:
                return energy

            position_grad = torch.autograd.grad(
                energy.sum(), x_init, create_graph=True
            )[0]
            # position_grad: (num_nodes, 3)
            return position_grad, energy

    def graph_to_padded_sequence(
        self,
        data: dgl.DGLGraph,
        x_init: torch.Tensor,
        ts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a batched DGL graph to padded tensors for Graphormer.

        Args:
            data: Batched DGL graph.
            x_init: Node coordinates `(num_nodes, 3)`.
            ts: Node timesteps `(num_nodes, 1)`.

        Returns:
            Tuple `(padded_feats, padded_pos, padded_ts)` each with leading batch
            dimension and padded to `max_nodes`.
        """

        batch_size = data.batch_size
        node_feats = torch.argmax(data.ndata["h"], dim=1) + 1
        # node_feats: (num_nodes,)
        # Index 0 is reserved for padding token.

        num_nodes_per_graph = data.batch_num_nodes()
        max_nodes = num_nodes_per_graph.max().item()

        padded_feats = torch.zeros(
            batch_size, max_nodes, device=node_feats.device, dtype=torch.long
        )
        padded_pos = torch.zeros(
            batch_size, max_nodes, 3, device=node_feats.device, dtype=torch.float
        )
        padded_ts = torch.full(
            (batch_size, max_nodes, 1),
            fill_value=-1.0,
            device=node_feats.device,
            dtype=torch.float,
        )

        node_offsets = torch.zeros_like(num_nodes_per_graph)
        node_offsets[1:] = num_nodes_per_graph[:-1].cumsum(dim=0)

        node_ids = torch.arange(len(node_feats), device=node_feats.device)
        graph_ids = torch.arange(len(num_nodes_per_graph), device=node_feats.device)
        graph_ids = graph_ids.repeat_interleave(num_nodes_per_graph)
        node_pos_in_graph = node_ids - node_offsets[graph_ids]

        padded_feats[graph_ids, node_pos_in_graph] = node_feats
        padded_pos[graph_ids, node_pos_in_graph] = x_init
        padded_ts[graph_ids, node_pos_in_graph] = ts

        return padded_feats, padded_pos, padded_ts
