import torch
from torch import nn, Tensor
from typing import Any, Union, Optional, Type, Tuple

from .swish import SwishBeta
from .gvp import _norm_no_nan, _rbf, GVPDropout, GVPLayerNorm
from .primitives import LinearNoBias
from .transition import ResidualTransition
from .embeddings import PairEmbedding
from ..modules.encoders import AttributeEncoder


def _make_pair_mask(atom_mask: Tensor) -> Tensor:
    """Build (B, N, N) pair mask from (B, N) atom mask, excluding self-loops."""
    pair_mask = atom_mask[:, :, None] * atom_mask[:, None, :]
    N = atom_mask.shape[1]
    diag = torch.eye(N, dtype=torch.bool, device=atom_mask.device)
    return pair_mask * (~diag).unsqueeze(0)


@torch.compile
def _compute_pairwise_features(
    positions: Tensor,
    pair_mask: Tensor,
    rbf_dmax: float,
    rbf_dim: int,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Compute normalized displacement vectors and RBF distance embeddings
    for all pairs in a dense position tensor.

    Args:
        positions: (B, N, 3)
        pair_mask:  (B, N, N) — 1 for valid pairs, 0 for padding / self-loops
        rbf_dmax:  maximum distance for the RBF basis
        rbf_dim:   number of RBF basis functions

    Returns:
        x_diff: (B, N, N, 3) unit displacement vectors (src − dst convention)
        d_rbf:  (B, N, N, rbf_dim) RBF distance embeddings
    """
    diff = positions[:, :, None, :] - positions[:, None, :, :] + eps
    dij = torch.square(diff).sum(dim=-1).sqrt() + eps
    x_diff = diff / dij.unsqueeze(-1)
    d_rbf = _rbf(dij, D_min=0.0, D_max=rbf_dmax, D_count=rbf_dim)

    mask = pair_mask.unsqueeze(-1)
    return x_diff * mask, d_rbf * mask


@torch.compile
def _edge_update_dense(
    edge_update_fn: nn.Module,
    edge_norm: nn.Module,
    node_scalars: Tensor,
    edge_feats: Tensor,
    pair_mask: Tensor,
    distance_feats: Optional[Tensor] = None,
) -> Tensor:
    """Dense edge-feature update: broadcast node scalars to pairs,
    concatenate with edge features, MLP residual, LayerNorm, re-mask."""
    B, N, _ = node_scalars.shape
    E = edge_feats.shape[-1]

    h_src = node_scalars[:, :, None, :].expand(-1, -1, N, -1)
    h_dst = node_scalars[:, None, :, :].expand(-1, N, -1, -1)

    if distance_feats is not None:
        update_input = torch.cat(
            (h_src, h_dst, edge_feats, distance_feats), dim=-1
        )
    else:
        update_input = torch.cat((h_src, h_dst, edge_feats), dim=-1)

    delta = edge_update_fn(update_input.reshape(B * N * N, -1))
    delta = delta.reshape(B, N, N, E)

    result = edge_norm((edge_feats + delta).reshape(B * N * N, E))
    return result.reshape(B, N, N, E) * pair_mask.unsqueeze(-1)


# ---------------------------------------------------------------------------
# BatchedGVP — GVP with arbitrary leading batch dimensions
# ---------------------------------------------------------------------------

@torch.compile
def _batched_gvp_forward(
    feats: Tensor,
    vectors: Tensor,
    Wh: Tensor,
    Wu: Tensor,
    Wcp: Tensor,
    n_cp_feats: int,
    to_feats_out: nn.Module,
    scalar_to_vector_gates: Optional[nn.Module],
    vectors_activation: nn.Module,
    dim_vectors_out: int,
) -> Tuple[Tensor, Tensor]:
    """GVP forward that works on tensors with arbitrary leading dimensions.

    Unlike ``gvp_forward_optimized`` which requires flat ``(b, d)`` /
    ``(b, v, 3)`` inputs, this version uses ``...`` einsum notation and
    ``dim=-1`` / ``dim=-2`` so that ``(B, N, d)`` / ``(B, N, v, 3)`` tensors
    pass through without reshape.
    """
    Vh = torch.einsum('... v c, v h -> ... h c', vectors, Wh)

    if n_cp_feats > 0:
        Vcp = torch.einsum('... v c, v p -> ... p c', vectors, Wcp)
        cp_src, cp_dst = torch.split(Vcp, n_cp_feats, dim=-2)
        cp = torch.linalg.cross(cp_src, cp_dst, dim=-1)
        Vh = torch.cat((Vh, cp), dim=-2)

    Vu = torch.einsum('... h c, h u -> ... u c', Vh, Wu)

    sh = _norm_no_nan(Vh, axis=-1)
    s = torch.cat((feats, sh), dim=-1)
    feats_out = to_feats_out(s)

    if scalar_to_vector_gates is not None:
        gating = scalar_to_vector_gates(feats_out).unsqueeze(-1)
    else:
        gating = _norm_no_nan(Vu, axis=-1).unsqueeze(-1)

    if dim_vectors_out == 1:
        vector_norms = _norm_no_nan(Vu, axis=-1).unsqueeze(-1)
        Vu = Vu / vector_norms

    vectors_out = vectors_activation(gating) * Vu
    return feats_out, vectors_out


class BatchedGVP(nn.Module):
    """GVP layer that operates on tensors with arbitrary leading batch
    dimensions — ``(..., dim_feats_in)`` and ``(..., dim_vectors_in, 3)``.

    Mirrors ``OptimizedGVP`` but uses ``...`` einsum patterns and ``dim=-1``
    / ``dim=-2`` so no reshape is needed for ``(B, N, ...)`` data.
    """

    def __init__(
        self,
        dim_vectors_in: int,
        dim_vectors_out: int,
        dim_feats_in: int,
        dim_feats_out: int,
        n_cp_feats: int = 0,
        hidden_vectors: Optional[int] = None,
        feats_activation: nn.Module = nn.SiLU(),
        vectors_activation: nn.Module = nn.Sigmoid(),
        vector_gating: bool = True,
    ):
        super().__init__()
        import math

        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in
        self.dim_vectors_out = dim_vectors_out
        self.n_cp_feats = n_cp_feats

        dim_h = max(dim_vectors_in, dim_vectors_out) if hidden_vectors is None else hidden_vectors

        wh_k = 1 / math.sqrt(dim_vectors_in)
        self.Wh = nn.Parameter(
            torch.zeros(dim_vectors_in, dim_h).uniform_(-wh_k, wh_k)
        )

        if n_cp_feats > 0:
            wcp_k = 1 / math.sqrt(dim_vectors_in)
            self.Wcp = nn.Parameter(
                torch.zeros(dim_vectors_in, n_cp_feats * 2).uniform_(-wcp_k, wcp_k)
            )
        else:
            self.register_parameter('Wcp', None)

        wu_in_dim = dim_h + n_cp_feats if n_cp_feats > 0 else dim_h
        wu_k = 1 / math.sqrt(wu_in_dim)
        self.Wu = nn.Parameter(
            torch.zeros(wu_in_dim, dim_vectors_out).uniform_(-wu_k, wu_k)
        )

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + n_cp_feats + dim_feats_in, dim_feats_out),
            feats_activation,
        )

        if vector_gating:
            self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out)
        else:
            self.scalar_to_vector_gates = None

    def forward(self, data: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        feats, vectors = data
        wcp = self.Wcp if self.n_cp_feats > 0 else torch.empty(0, device=vectors.device)
        return _batched_gvp_forward(
            feats, vectors,
            self.Wh, self.Wu, wcp, self.n_cp_feats,
            self.to_feats_out, self.scalar_to_vector_gates,
            self.vectors_activation, self.dim_vectors_out,
        )


class NodePositionUpdateEmu(nn.Module):
    """Stacked ``BatchedGVP`` layers for coordinate refinement that operates
    directly on ``(B, N, ...)`` tensors without any reshape."""

    def __init__(
        self,
        n_scalars: int,
        n_vec_channels: int,
        n_gvps: int = 3,
        n_cp_feats: int = 0,
        vector_gating: bool = True,
    ):
        super().__init__()

        if n_gvps < 1:
            raise ValueError("n_gvps must be >= 1")

        layers = []
        for idx in range(n_gvps):
            last = idx == (n_gvps - 1)
            layers.append(
                BatchedGVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=1 if last else n_vec_channels,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=nn.Identity() if last else nn.Sigmoid(),
                    vector_gating=vector_gating,
                )
            )
        self.gvps = nn.Sequential(*layers)

    def forward(self, scalars: Tensor, vectors: Tensor) -> Tensor:
        """
        Args:
            scalars: (B, N, n_scalars)
            vectors: (B, N, n_vec_channels, 3)

        Returns:
            velocity: (B, N, 3)
        """
        _, vec_updates = self.gvps((scalars, vectors))
        return vec_updates.squeeze(-2)


# ---------------------------------------------------------------------------
# EdgeUpdateEmu
# ---------------------------------------------------------------------------

class EdgeUpdateEmu(nn.Module):
    """Dense pairwise edge update — drop-in replacement for
    ``OptimizedEdgeUpdate`` on fully-connected batched graphs (no DGL)."""

    def __init__(
        self,
        n_node_scalars: int,
        n_edge_feats: int,
        update_edge_w_distance: bool = False,
        rbf_dim: int = 16,
        activation: Type[nn.Module] = nn.SiLU,
    ):
        super().__init__()

        self.update_edge_w_distance = update_edge_w_distance

        input_dim = (n_node_scalars * 2) + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fn = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            activation(),
            nn.Linear(n_edge_feats, n_edge_feats),
            activation(),
        )
        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(
        self,
        node_scalars: Tensor,
        edge_feats: Tensor,
        pair_mask: Tensor,
        d: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            node_scalars: (B, N, D)
            edge_feats:   (B, N, N, E)
            pair_mask:    (B, N, N)
            d:            (B, N, N, rbf_dim) — required when
                          ``update_edge_w_distance=True``

        Returns:
            Updated edge features (B, N, N, E).
        """
        if self.update_edge_w_distance and d is None:
            raise ValueError(
                "Distance features `d` must be provided when "
                "update_edge_w_distance=True."
            )

        return _edge_update_dense(
            self.edge_update_fn,
            self.edge_norm,
            node_scalars,
            edge_feats,
            pair_mask,
            d if self.update_edge_w_distance else None,
        )


# ---------------------------------------------------------------------------
# GVPConvEmu
# ---------------------------------------------------------------------------

class GVPConvEmu(nn.Module):
    """Fully tensorized GVP graph convolution for dense fully-connected
    graphs.

    Replaces ``OptimizedGVPConv`` by removing all DGL dependencies and
    operating on batched ``(B, N, ...)`` tensors throughout — no reshape
    is ever performed.  Uses ``BatchedGVP`` for all internal GVP layers.
    """

    def __init__(
        self,
        scalar_size: int = 128,
        vector_size: int = 16,
        n_cp_feats: int = 0,
        scalar_activation=nn.SiLU,
        vector_activation=nn.Sigmoid,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        edge_feat_size: int = 0,
        coords_range: float = 10.0,
        message_norm: Union[float, str] = 10,
        dropout: float = 0.0,
        vector_gating: bool = True,
    ):
        super().__init__()

        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.n_cp_feats = n_cp_feats
        self.edge_feat_size = edge_feat_size
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.dropout_rate = dropout
        self.message_norm = message_norm
        self.coords_range = coords_range

        # --- message GVP stack ---
        message_gvps = []
        for i in range(n_message_gvps):
            dim_vectors_in = vector_size
            dim_feats_in = scalar_size

            if i == 0:
                dim_vectors_in += 1          # displacement vector channel
                dim_feats_in += rbf_dim + edge_feat_size

            message_gvps.append(
                BatchedGVP(
                    dim_vectors_in=dim_vectors_in,
                    dim_vectors_out=vector_size,
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=dim_feats_in,
                    dim_feats_out=scalar_size,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                    vector_gating=vector_gating,
                )
            )
        self.edge_message = nn.Sequential(*message_gvps)

        # --- node update GVP stack ---
        update_gvps = []
        for _ in range(n_update_gvps):
            update_gvps.append(
                BatchedGVP(
                    dim_vectors_in=vector_size,
                    dim_vectors_out=vector_size,
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=scalar_size,
                    dim_feats_out=scalar_size,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                    vector_gating=vector_gating,
                )
            )
        self.node_update = nn.Sequential(*update_gvps)

        # --- position update GVP ---
        self.node_position_update = BatchedGVP(
            dim_feats_in=scalar_size,
            dim_feats_out=scalar_size,
            dim_vectors_in=vector_size,
            dim_vectors_out=1,
            n_cp_feats=n_cp_feats,
            vectors_activation=nn.Tanh(),
            vector_gating=vector_gating,
        )

        self.dropout = GVPDropout(self.dropout_rate)
        self.message_layer_norm = GVPLayerNorm(self.scalar_size)
        self.update_layer_norm = GVPLayerNorm(self.scalar_size)

        if isinstance(self.message_norm, str):
            if self.message_norm not in ("mean", "sum"):
                raise ValueError(
                    f"message_norm must be 'mean', 'sum', or a number, "
                    f"got {self.message_norm}"
                )

    def forward(
        self,
        scalar_feats: Tensor,
        coord_feats: Tensor,
        vec_feats: Tensor,
        atom_mask: Tensor,
        pair_mask: Optional[Tensor] = None,
        edge_feats: Optional[Tensor] = None,
        x_diff: Optional[Tensor] = None,
        d: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            scalar_feats: (B, N, scalar_size)
            coord_feats:  (B, N, 3)
            vec_feats:    (B, N, vector_size, 3)
            atom_mask:    (B, N)  — 1 for real atoms, 0 for padding
            pair_mask:    (B, N, N) — 1 for valid pairs, 0 for padding / self-loops
            edge_feats:   (B, N, N, edge_feat_size), optional
            x_diff:       (B, N, N, 3)  precomputed unit displacements
            d:            (B, N, N, rbf_dim)  precomputed RBF distances

        Returns:
            scalar_feats_out: (B, N, scalar_size)
            vec_feats_out:    (B, N, vector_size, 3)
            coord_feats_out:  (B, N, 3)
        """
        N = scalar_feats.shape[1]

        if self.edge_feat_size > 0:
            assert edge_feats is not None, "Edge features must be provided."

        if pair_mask is None:
            pair_mask = _make_pair_mask(atom_mask)

        if x_diff is None or d is None:
            x_diff, d = _compute_pairwise_features(
                coord_feats, pair_mask, self.rbf_dmax, self.rbf_dim
            )

        # ==============================================================
        # 1. Build pairwise message inputs — all (B, N, N, ...)
        # ==============================================================
        h_src = scalar_feats[:, :, None, :].expand(-1, -1, N, -1)
        v_src = vec_feats[:, :, None, :, :].expand(-1, -1, N, -1, -1)

        scalar_parts = [h_src, d]
        if self.edge_feat_size > 0:
            scalar_parts.append(edge_feats)
        scalar_input = torch.cat(scalar_parts, dim=-1)

        vec_parts = [x_diff.unsqueeze(-2), v_src]
        vec_input = torch.cat(vec_parts, dim=-2)

        # ==============================================================
        # 2. Message GVP — stays (B, N, N, ...) throughout
        # ==============================================================
        scalar_msg, vec_msg = self.edge_message((scalar_input, vec_input))

        # ==============================================================
        # 3. Mask and aggregate over sources (dim=1)
        # ==============================================================
        scalar_msg = scalar_msg * pair_mask.unsqueeze(-1)
        vec_msg = vec_msg * pair_mask[..., None, None]

        scalar_agg = scalar_msg.sum(dim=1)
        vec_agg = vec_msg.sum(dim=1)

        if self.message_norm == "mean":
            n_nbrs = pair_mask.sum(dim=1).clamp(min=1)
            scalar_agg = scalar_agg / n_nbrs.unsqueeze(-1)
            vec_agg = vec_agg / n_nbrs[..., None, None]
        elif isinstance(self.message_norm, (int, float)):
            scalar_agg = scalar_agg / self.message_norm
            vec_agg = vec_agg / self.message_norm

        # ==============================================================
        # 4. Dropout + residual + message LayerNorm — all (B, N, ...)
        # ==============================================================
        scalar_agg, vec_agg = self.dropout(scalar_agg, vec_agg)

        scalar_new = scalar_feats + scalar_agg
        vec_new = vec_feats + vec_agg
        scalar_new, vec_new = self.message_layer_norm(scalar_new, vec_new)

        # ==============================================================
        # 5. Node update GVP + dropout + residual + LayerNorm
        # ==============================================================
        scalar_res, vec_res = self.node_update((scalar_new, vec_new))
        scalar_res, vec_res = self.dropout(scalar_res, vec_res)
        scalar_out = scalar_new + scalar_res
        vec_out = vec_new + vec_res

        # ==============================================================
        # 6. Position update
        # ==============================================================
        _, pos_update = self.node_position_update((scalar_out, vec_out))
        coord_out = coord_feats + pos_update.squeeze(-2) * self.coords_range

        # ==============================================================
        # 7. Final LayerNorm + mask padding
        # ==============================================================
        scalar_out, vec_out = self.update_layer_norm(scalar_out, vec_out)

        mask_s = atom_mask.unsqueeze(-1)           # (B, N, 1)
        mask_v = atom_mask[..., None, None]        # (B, N, 1, 1)
        scalar_out = scalar_out * mask_s
        vec_out = vec_out * mask_v
        coord_out = coord_out * mask_s

        return scalar_out, vec_out, coord_out


# ---------------------------------------------------------------------------
# GVPDecoderEmu
# ---------------------------------------------------------------------------

class GVPDecoderEmu(nn.Module):
    """Fully tensorized GVP decoder — drop-in replacement for
    ``OptimizedGVPDecoder`` on dense fully-connected batched graphs.

    Composes ``GVPConvEmu`` and ``EdgeUpdateEmu`` layers with a final
    ``NodePositionUpdateEmu`` to produce a velocity vector field.
    """

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
                GVPConvEmu(
                    scalar_size=n_hidden_nodes,
                    vector_size=n_vec,
                    n_message_gvps=n_message_gvps,
                    n_update_gvps=n_update_gvps,
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
                EdgeUpdateEmu(
                    n_node_scalars=n_hidden_nodes,
                    n_edge_feats=n_hidden_edge,
                    update_edge_w_distance=True,
                    rbf_dim=rbf_dim,
                )
            )

        self.position_updater = NodePositionUpdateEmu(
            n_scalars=n_hidden_nodes,
            n_vec_channels=n_vec,
            n_gvps=n_coord_gvps,
            vector_gating=vector_gating,
        )

    def forward(
        self,
        h: Tensor,
        v_init: Tensor,
        x_init: Tensor,
        edge_repr: Tensor,
        edge_mask: Tensor,
        atom_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            h:          (B, N, n_hidden_nodes)  node scalar features
            v_init:     (B, N, n_vec, 3)        initial vector features
            x_init:     (B, N, 3)               initial coordinates
            edge_repr:  (B, N, N, n_hidden_edge) edge features
            edge_mask:  (B, N, N) or (B, N, N, 1) — zeroes out masked edges
            atom_mask:  (B, N)                  1 for real atoms, 0 for padding

        Returns:
            vector_field: (B, N, 3)
            hs:           (B, N, n_hidden_nodes)
            vs:           (B, N, n_vec, 3)
            edge_repr:    (B, N, N, n_hidden_edge)
        """
        if edge_mask.dim() == 4:
            edge_mask = edge_mask.squeeze(-1)

        pair_mask = _make_pair_mask(atom_mask)

        hs, vs, xs = h, v_init, x_init
        for conv, edge_nn in zip(self.convs, self.edge_updater):
            x_diff, d = _compute_pairwise_features(
                xs, pair_mask, self.rbf_dmax, self.rbf_dim
            )
            edge_repr = (
                edge_nn(hs, edge_repr, pair_mask, d=d)
                * edge_mask.unsqueeze(-1)
            )
            hs, vs, xs = conv(
                hs, xs, vs, atom_mask,
                pair_mask=pair_mask,
                edge_feats=edge_repr,
                x_diff=x_diff,
                d=d,
            )

        vector_field = self.position_updater(hs, vs) * atom_mask.unsqueeze(-1)

        return vector_field, hs, vs, edge_repr



class InvariantEncoder(nn.Module):

    def __init__(
        self,
        node_feats_in: int,
        edge_feats_in: int,
        c_atoms: int,
        c_pairs: int,
        dropout_prob: float = 0.0,
    ) -> None:

        super().__init__()

        self.node_feats_in = node_feats_in
        self.edge_feats_in = edge_feats_in
        self.c_atoms = c_atoms
        self.c_pairs = c_pairs
        self.dropout_prob = dropout_prob

        self.attr_encoder = AttributeEncoder(
            n_features=node_feats_in,
            n_hidden=c_atoms,
        )
        self.atom_embedder = nn.Sequential(
            LinearNoBias(2 * c_atoms, c_atoms),
            nn.SiLU(),
        )
        self.pair_embedder = PairEmbedding(
            edge_feats_in=edge_feats_in,
            edge_feats_out=c_pairs,
            dropout_prob=dropout_prob,
        )
        self.message_proj = LinearNoBias(c_pairs, c_atoms)
        self.interaction_residual = ResidualTransition(dim=c_atoms, hidden=c_atoms, dropout_prob=dropout_prob)
    
    def forward(
        self,
        time: Tensor,
        attr: Tensor,
        atom_index: Tensor,
        pair_feats: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        # Latent vectors based on atom attributes
        x_h = self.attr_encoder(
            time=time,
            attr=attr,
            atom_index=atom_index,
        )
        # x_h: (batch_size, n_atoms, c_atoms)

        # Apply atom mask
        x_h = x_h * atom_mask.unsqueeze(-1)

        # Embed the edge features
        edge_repr = self.pair_embedder(pair_features=pair_feats, pair_mask=pair_mask)
        # edge_repr: (batch_size, n_atoms, n_atoms, c_pairs)

        # Project the edge features to the atom features
        msgs = self.message_proj(edge_repr.sum(dim=-2)) # NOTE: sum > mean
        # msgs: (batch_size, n_atoms, c_atoms)

        # Aggregate the edge features to the atom features
        x_h = self.interaction_residual(x_h, msgs)
        # x_h: (batch_size, n_atoms, c_atoms)

        # Apply node mask
        x_h = x_h * atom_mask.unsqueeze(-1)

        return x_h, edge_repr


class EmuModel(nn.Module):

    def __init__(
        self,
        n_vec: int = 16,
        node_feats_in: int = 21,
        edge_feats_in: int = 9,
        n_layers: int = 5,
        n_hidden_nodes: int = 64,
        n_hidden_edge: int = 32,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        n_coord_gvps: int = 1,
        rbf_dim: int = 16,
        rbf_dmax: float = 20,
        message_norm: Union[float, str] = "sum",
        vector_gating: bool = True,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_vec_channels = n_vec

        self.encoder = InvariantEncoder(
            node_feats_in=node_feats_in,
            edge_feats_in=edge_feats_in,
            c_atoms=n_hidden_nodes,
            c_pairs=n_hidden_edge,
            dropout_prob=dropout_prob,
        )

        self.decoder = GVPDecoderEmu(
            n_vec=n_vec,
            n_layers=n_layers,
            n_hidden_nodes=n_hidden_nodes,
            n_hidden_edge=n_hidden_edge,
            n_message_gvps=n_message_gvps,
            n_update_gvps=n_update_gvps,
            n_coord_gvps=n_coord_gvps,
            rbf_dim=rbf_dim,
            rbf_dmax=rbf_dmax,
            message_norm=message_norm,
            vector_gating=vector_gating,
        )

    def load_from_checkpoint(self, checkpoint_path: str, **kwargs: Any) -> "EmuModel":
        checkpoint = torch.load(checkpoint_path, **kwargs)
        self.load_state_dict(checkpoint)
        return self

    def forward(
        self,
        x_t: Tensor,
        time: Tensor,
        attr: Tensor,
        atom_index: Tensor,
        pair_feats: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tensor:
        device = x_t.device
        B, N, _ = x_t.shape

        v_init = torch.zeros(
            B, N, self.n_vec_channels, 3, device=device,
        )

        node_repr, edge_repr = self.encoder(
            time=time,
            attr=attr,
            atom_index=atom_index,
            pair_feats=pair_feats,
            atom_mask=atom_mask,
            pair_mask=pair_mask,
        )

        velocity, *_ = self.decoder(
            h=node_repr,
            v_init=v_init,
            x_init=x_t,
            edge_repr=edge_repr,
            edge_mask=pair_mask,
            atom_mask=atom_mask,
        )

        return velocity

    def inference_fwd(
        self,
        x_t: Tensor,
        time: Tensor,
        attr: Tensor,
        atom_index: Tensor,
        pair_feats: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tensor:
        return self(x_t, time, attr, atom_index, pair_feats, atom_mask, pair_mask)