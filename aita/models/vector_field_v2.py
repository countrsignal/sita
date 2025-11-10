from itertools import chain
from typing import Optional, Tuple, Iterator

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.encoders import AtomEncoder
from .modules.decoders import GVP_Decoder
from ..utils.logging import RankedLogger
from ..utils.graph_utils import dgl_nodes_to_padded_tensor


log = RankedLogger(__name__, on_rank_zero=True)


###################################
# Helper functions
###################################

@torch.no_grad()
def distogram_targets(batch: torch.Tensor, bin_edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute bin indices using torch.bucketize.
    # bin_indices has shape [B, N, N] with values in the range [0, n_bins-1]
    distances = torch.square(batch[:, :, None, :] - batch[:, None, :, :]).sum(dim=-1).sqrt()
    bin_indices = torch.bucketize(distances, bin_edges)
    n_bins = bin_edges.shape[0]  # Total number of bins

    # One-hot encode the bin indices.
    # target has shape [B, N, N, n_bins]
    target = F.one_hot(bin_indices, num_classes=n_bins).float()

    # Create a mask that ignores the diagonal elements.
    # First, create an [N, N] identity matrix, then invert it so that the diagonal is False.
    # Finally, expand it to shape [B, N, N].
    B, N, _ = distances.shape
    mask = ~torch.eye(N, dtype=torch.bool, device=distances.device).unsqueeze(0).expand(B, N, N)

    # Zero out the target values at the diagonal positions.
    # mask.unsqueeze(-1) has shape [B, N, N, 1] and is broadcast over the last dimension.
    target = target * mask.unsqueeze(-1)

    return target, mask


def distogram_loss_fn(preds, targets, mask):
 # Compute the cross-entropy errors:
    # Apply log softmax over the last dimension (bins) and compute the negative log likelihood.
    # The result is summed over the bins to yield errors per pair.
    errors = -1.0 * torch.sum(
        targets * F.log_softmax(preds, dim=-1),  # log softmax over bins
        dim=-1                                # sum over the n_bins dimension
    )  # errors shape: [B, N, N]

    # Compute denominator: total count of valid (off-diagonal) entries per batch sample.
    denom = 1e-5 + torch.sum(mask, dim=(-1, -2))  # shape: [B]

    # Mask the errors so that only off-diagonal contributions remain,
    # then sum over the last dimension (one of the N dimensions).
    mean = errors * mask         # shape: [B, N, N]
    mean = torch.sum(mean, dim=-1)  # shape: [B, N]

    # Normalize by the count of valid entries per row; denom is broadcast to shape [B, N]
    mean = mean / denom[..., None]  # shape: [B, N]

    # Sum the mean loss per token to get the loss per batch sample.
    batch_loss = torch.sum(mean, dim=-1)  # shape: [B]

    # Compute the global loss as the mean over the batch.
    global_loss = torch.mean(batch_loss)  # scalar

    return global_loss


###################################
# Classes
###################################

class VFV2(nn.Module):

    def __init__(
        self,
        n_features: int = 21,
        n_layers: int = 5,
        n_hidden: int = 64,
        n_vec: int = 16,
        pair_dropout: float = 0.0,
        n_transitions: int = 2,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        n_coord_gvps: int = 1,
        use_dst_feats: bool = False,
        vector_gating: bool = True,
        n_distogram_bins: int = 0,
        distance_ranges: Tuple[float, float] = (0.0, 10.0),
        pair_dim_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()

        if pair_dim_hidden is None:
            pair_dim_hidden = n_hidden

        self.n_vec_channels = n_vec
        self.atom_encoder = AtomEncoder(
            n_features=n_features,
            n_hidden=n_hidden,
            n_transitions=n_transitions,
            pair_dropout=pair_dropout,
            pair_dim_hidden=pair_dim_hidden,
        )
        self.gvp_decoder = GVP_Decoder(
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_vec=n_vec,
            edge_feat_size=pair_dim_hidden,
            n_message_gvps=n_message_gvps,
            n_update_gvps=n_update_gvps,
            n_coord_gvps=n_coord_gvps,
            use_dst_feats=use_dst_feats,
            vector_gating=vector_gating,
        )

        # distogram head
        self.bins = n_distogram_bins
        if n_distogram_bins > 0:
            self.distogram_head = torch.nn.Linear(pair_dim_hidden, n_distogram_bins)
            self.register_buffer('bin_edges', torch.linspace(distance_ranges[0], distance_ranges[1], n_distogram_bins + 1)[1:], persistent=False)
        else:
            self.distogram_head = None
            self.register_buffer('bin_edges', None, persistent=False)
    
    def parameters(self) -> Iterator[torch.Tensor]:
        # customized so that we return parameters of both the encoder & decoder, and distogram head if it exists
        if self.distogram_head is not None:
            return chain(self.atom_encoder.parameters(), self.gvp_decoder.parameters(), self.distogram_head.parameters())
        else:
            return chain(self.atom_encoder.parameters(), self.gvp_decoder.parameters())

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # initialize the coordinates and velocities
        x_init = graph.ndata["xt"].clone()  # (num_nodes, 3)
        device = x_init.device
        v_init = torch.zeros(
            graph.num_nodes(),
            self.n_vec_channels,
            3,
            device=device,
        )  # (num_nodes, n_vec_channels, 3)

        # encode the atom features
        node_repr, pair_repr = self.atom_encoder(graph)
        # node_repr: (num_nodes, n_hidden)
        # pair_repr: (num_nodes, num_nodes, pair_dim_hidden)

        # decode the vector field
        velocity = self.gvp_decoder(node_repr, v_init, x_init, graph)
        # velocity: (num_nodes, 3)

        return velocity
    
    def training_step(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # initialize the coordinates and velocities
        x_init = graph.ndata["xt"].clone()  # (num_nodes, 3)
        device = x_init.device
        v_init = torch.zeros(
            graph.num_nodes(),
            self.n_vec_channels,
            3,
            device=device,
        )  # (num_nodes, n_vec_channels, 3)

        # encode the atom features
        node_repr, pair_repr = self.atom_encoder(graph)
        # node_repr: (num_nodes, n_hidden)
        # pair_repr: (num_nodes, num_nodes, pair_dim_hidden)

        # decode the vector field
        velocity = self.gvp_decoder(node_repr, v_init, x_init, graph)
        # vector_field: (num_nodes, 3)

        # compute vector field loss
        graph.ndata["loss_per_node"] = torch.square(velocity - graph.ndata["vt"]).mean(dim=-1)
        loss_per_molecule = dgl.mean_nodes(graph, "loss_per_node")
        loss = loss_per_molecule.mean()

        # compute distogram loss
        if self.bins > 0:
            disto_logits = self.distogram_head(pair_repr)
            # disto_logits: (num_nodes, num_nodes, n_distogram_bins)

            # compute distogram targets
            coords = dgl_nodes_to_padded_tensor(graph, feat_key="x1")[0]
            targets, mask = distogram_targets(coords, self.bin_edges)
            # targets: (num_nodes, num_nodes, n_distogram_bins)
            # mask: (num_nodes, num_nodes)

            # compute distogram loss
            disto_loss = distogram_loss_fn(disto_logits, targets, mask)
        
            return {"loss": loss + disto_loss, "flow_loss": loss, "distogram_loss": disto_loss}
        
        else:
            return {"loss": loss}