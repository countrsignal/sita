import torch
from torch import nn, einsum
import math
import dgl
import dgl.function as fn
from typing import Union, Optional, Type

# Import helpers from the original file
from aita.models.layers.gvp import (
    _norm_no_nan,
    _rbf,
    GVPDropout,
    GVPLayerNorm
)
from ...utils.logging import RankedLogger


log = RankedLogger(__name__, on_rank_zero=True)


# Optimized core logic function
# We use torch.compile to optimize this specific computational graph
@torch.compile
def gvp_forward_optimized(
    feats: torch.Tensor,
    vectors: torch.Tensor,
    Wh: torch.Tensor,
    Wu: torch.Tensor,
    Wcp: torch.Tensor, # Can be None/dummy if n_cp_feats is 0, but for compilation handle carefully
    n_cp_feats: int,
    to_feats_out_layer: nn.Module, # Pass the layer/function
    scalar_to_vector_gates_layer: nn.Module, # Can be None
    vectors_activation_fn: nn.Module,
    dim_vectors_in: int,
    dim_feats_in: int,
    dim_vectors_out: int
):
    b, n, _, v, c = *feats.shape, *vectors.shape
    
    # Checks (can be removed for max speed if guaranteed, but good for safety)
    # assert c == 3 and v == dim_vectors_in
    # assert n == dim_feats_in

    Vh = einsum('b v c, v h -> b h c', vectors, Wh)

    if n_cp_feats > 0:
        Vcp = einsum('b v c, v p -> b p c', vectors, Wcp)
        cp_src, cp_dst = torch.split(Vcp, n_cp_feats, dim=1)
        cp = torch.linalg.cross(cp_src, cp_dst, dim=-1)
        Vh = torch.cat((Vh, cp), dim=1)

    Vu = einsum('b h c, h u -> b u c', Vh, Wu)

    sh = _norm_no_nan(Vh)
    s = torch.cat((feats, sh), dim=1)

    feats_out = to_feats_out_layer(s)

    if scalar_to_vector_gates_layer is not None:
        gating = scalar_to_vector_gates_layer(feats_out)
        gating = gating.unsqueeze(dim=-1)
    else:
        gating = _norm_no_nan(Vu)

    if dim_vectors_out == 1:
        vector_norms = _norm_no_nan(Vu)
        Vu = Vu / vector_norms.unsqueeze(-1)

    vectors_out = vectors_activation_fn(gating) * Vu

    return feats_out, vectors_out


@torch.compile
def node_position_update_forward(
    gvp_stack: nn.Module,
    scalars: torch.Tensor,
    vectors: torch.Tensor,
):
    """Compiled helper to run the stacked GVPs used for coordinate updates."""
    _, vector_updates = gvp_stack((scalars, vectors))
    return vector_updates.squeeze(1)


@torch.compile
def edge_update_forward(
    edge_update_fn: nn.Module,
    edge_norm: nn.Module,
    node_scalars: torch.Tensor,
    edge_feats: torch.Tensor,
    src_idxs: torch.Tensor,
    dst_idxs: torch.Tensor,
    distance_feats: Optional[torch.Tensor] = None,
):
    """Compiled helper for the edge feature update residual block."""
    src = torch.index_select(node_scalars, 0, src_idxs)
    dst = torch.index_select(node_scalars, 0, dst_idxs)

    if distance_feats is not None:
        update_inputs = torch.cat((src, dst, edge_feats, distance_feats), dim=-1)
    else:
        update_inputs = torch.cat((src, dst, edge_feats), dim=-1)

    delta = edge_update_fn(update_inputs)
    return edge_norm(edge_feats + delta)


@torch.compile
def energy_head_forward(
    gvp_stack: nn.Module,
    scalars: torch.Tensor,
    vectors: torch.Tensor,
):
    """Compiled helper to run the stacked GVPs used for energy head."""
    energies, _ = gvp_stack((scalars, vectors))
    return energies.squeeze(1)


class OptimizedGVP(nn.Module):
    def __init__(
        self,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        n_cp_feats=0,
        hidden_vectors=None,
        feats_activation=nn.SiLU(),
        vectors_activation=nn.Sigmoid(),
        vector_gating=True,
        xavier_init=False
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in
        self.n_cp_feats = n_cp_feats
        self.dim_vectors_out = dim_vectors_out
        
        dim_h = max(dim_vectors_in, dim_vectors_out) if hidden_vectors is None else hidden_vectors

        # Wh
        wh_k = 1/math.sqrt(dim_vectors_in)
        self.Wh = nn.Parameter(torch.zeros(dim_vectors_in, dim_h, dtype=torch.float32).uniform_(-wh_k, wh_k))

        # Wcp
        if n_cp_feats > 0:
            wcp_k = 1/math.sqrt(dim_vectors_in)
            self.Wcp = nn.Parameter(torch.zeros(dim_vectors_in, n_cp_feats*2, dtype=torch.float32).uniform_(-wcp_k, wcp_k))
        else:
            # Register as buffer or parameter to avoid None issues in some contexts, 
            # but for the function call we can pass a dummy or None if handled.
            # Here we'll just leave it as None on the object, and handle in call.
            self.register_parameter('Wcp', None)

        # Wu
        if n_cp_feats > 0:
            wu_in_dim = dim_h + n_cp_feats
        else:
            wu_in_dim = dim_h
        wu_k = 1/math.sqrt(wu_in_dim)
        self.Wu = nn.Parameter(torch.zeros(wu_in_dim, dim_vectors_out, dtype=torch.float32).uniform_(-wu_k, wu_k))

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + n_cp_feats + dim_feats_in, dim_feats_out),
            feats_activation
        )

        if vector_gating:
            self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out)
            if xavier_init:
                nn.init.xavier_uniform_(self.scalar_to_vector_gates.weight, gain=1)
                nn.init.constant_(self.scalar_to_vector_gates.bias, 0)
        else:
            self.scalar_to_vector_gates = None

    def forward(self, data):
        feats, vectors = data
        
        # Wcp might be None, pass it correctly
        # For torch.compile, it's better if types are stable. 
        # If Wcp is None, we should pass a placeholder or handle it in the compiled function.
        # In the compiled function above, we check n_cp_feats > 0.
        # If n_cp_feats == 0, Wcp usage is skipped.
        # However, passing None to a compiled function might be fine or trigger recompilation if it changes.
        # Since it's constant per instance, it should be fine.
        
        # We default Wcp to a dummy tensor if None to ensure type stability if needed, 
        # but None is usually acceptable in recent torch versions.
        wcp_arg = self.Wcp if self.n_cp_feats > 0 else torch.empty(0, device=vectors.device)

        return gvp_forward_optimized(
            feats,
            vectors,
            self.Wh,
            self.Wu,
            wcp_arg,
            self.n_cp_feats,
            self.to_feats_out,
            self.scalar_to_vector_gates,
            self.vectors_activation,
            self.dim_vectors_in,
            self.dim_feats_in,
            self.dim_vectors_out
        )


class OptimizedGVPConv(nn.Module):
    """Optimized GVP graph convolution on a homogenous graph using OptimizedGVP layers."""

    def __init__(
        self,
        scalar_size: int = 128,
        vector_size: int = 16,
        n_cp_feats: int = 0,
        scalar_activation=nn.SiLU,
        vector_activation=nn.Sigmoid,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        use_dst_feats: bool = False,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        edge_feat_size: int = 0,
        coords_range=10,
        message_norm: Union[float, str] = 10,
        dropout: float = 0.0,
        vector_gating=True,
    ):
        
        super().__init__()

        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.n_cp_feats = n_cp_feats
        self.scalar_activation = scalar_activation
        self.vector_activation = vector_activation
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.edge_feat_size = edge_feat_size
        self.use_dst_feats = use_dst_feats
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.dropout_rate = dropout
        self.message_norm = message_norm
        self.coords_range = coords_range

        # create message passing function using OptimizedGVP
        message_gvps = []
        for i in range(n_message_gvps):

            dim_vectors_in = vector_size
            dim_feats_in = scalar_size

            # on the first layer, there is an extra edge vector for the displacement vector between the two node positions
            if i == 0:
                dim_vectors_in += 1
                dim_feats_in += rbf_dim + edge_feat_size
                
            # if this is the first layer and we are using destination node features to compute messages, add them to the input dimensions
            if use_dst_feats and i == 0:
                dim_vectors_in += vector_size
                dim_feats_in += scalar_size

            message_gvps.append(
                OptimizedGVP(
                    dim_vectors_in=dim_vectors_in, 
                    dim_vectors_out=vector_size,
                    n_cp_feats=n_cp_feats, 
                    dim_feats_in=dim_feats_in, 
                    dim_feats_out=scalar_size, 
                    feats_activation=scalar_activation(), 
                    vectors_activation=vector_activation(), 
                    vector_gating=vector_gating
                )
            )
        self.edge_message = nn.Sequential(*message_gvps)

        # create update function using OptimizedGVP
        update_gvps = []
        for i in range(n_update_gvps):
            update_gvps.append(
                OptimizedGVP(
                    dim_vectors_in=vector_size, 
                    dim_vectors_out=vector_size, 
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=scalar_size, 
                    dim_feats_out=scalar_size, 
                    feats_activation=scalar_activation(), 
                    vectors_activation=vector_activation(), 
                    vector_gating=vector_gating
                )
            )
        self.node_update = nn.Sequential(*update_gvps)
        
        # node position update using OptimizedGVP
        self.node_position_update = OptimizedGVP(
            dim_feats_in=scalar_size,
            dim_feats_out=scalar_size,
            dim_vectors_in=vector_size,
            dim_vectors_out=1,
            n_cp_feats=n_cp_feats,
            vectors_activation=nn.Tanh(),
            vector_gating=vector_gating
        )
        
        self.dropout = GVPDropout(self.dropout_rate)
        self.message_layer_norm = GVPLayerNorm(self.scalar_size)
        self.update_layer_norm = GVPLayerNorm(self.scalar_size)

        if isinstance(self.message_norm, str):
            if self.message_norm not in ['mean', 'sum']:
                raise ValueError(f"message_norm must be either 'mean', 'sum', or a number, got {self.message_norm}")
        else:
            assert isinstance(self.message_norm, (float, int)), "message_norm must be either 'mean', 'sum', or a number"

        if self.message_norm == 'mean':
            self.agg_func = fn.mean
        else:
            self.agg_func = fn.sum

    def forward(self, g: dgl.DGLGraph, 
                scalar_feats: torch.Tensor,
                coord_feats: torch.Tensor,
                vec_feats: torch.Tensor,
                edge_feats: torch.Tensor = None,
                x_diff: torch.Tensor = None,
                d: torch.Tensor = None):
        # vec_feat has shape (n_nodes, n_vectors, 3)

        with g.local_scope():

            g.ndata['h'] = scalar_feats
            g.ndata['x'] = coord_feats
            g.ndata['v'] = vec_feats

            if x_diff is not None and d is not None:
                g.edata['x_diff'] = x_diff
                g.edata['d'] = d

            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feats is not None, "Edge features must be provided."
                g.edata["a"] = edge_feats

            # normalize x_diff and compute rbf embedding of edge distance
            if 'x_diff' not in g.edata:
                # get vectors between node positions
                g.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
                dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
                g.edata['x_diff'] = g.edata['x_diff'] / dij
                g.edata['d'] = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)

            # compute messages on every edge
            g.apply_edges(self.message)

            # aggregate messages from every edge
            g.update_all(fn.copy_e("scalar_msg", "m"), self.agg_func("m", "scalar_msg"))
            g.update_all(fn.copy_e("vec_msg", "m"), self.agg_func("m", "vec_msg"))

            # get aggregated scalar and vector messages
            if isinstance(self.message_norm, str):
                z = 1
            else:
                z = self.message_norm

            scalar_msg = g.ndata["scalar_msg"] / z
            vec_msg = g.ndata["vec_msg"] / z

            # dropout scalar and vector messages
            scalar_msg, vec_msg = self.dropout(scalar_msg, vec_msg)

            # update scalar and vector features, apply layernorm
            scalar_feat_new = g.ndata['h'] + scalar_msg
            vec_feat_new = g.ndata['v'] + vec_msg
            
            scalar_feat_new, vec_feat_new = self.message_layer_norm(scalar_feat_new, vec_feat_new)

            # apply node update function, apply dropout to residuals, apply layernorm
            scalar_residual, vec_residual = self.node_update((scalar_feat_new, vec_feat_new))
            scalar_residual, vec_residual = self.dropout(scalar_residual, vec_residual)
            scalar_feat_return = scalar_feat_new + scalar_residual
            vec_feat_return = vec_feat_new + vec_residual
            
            # position update
            _, position_update = self.node_position_update((scalar_feat_return, vec_feat_return))
            position_update = position_update * self.coords_range
            coord_feats_return = g.ndata['x'] + position_update.squeeze(1)
            
            scalar_feat_return, vec_feat_return = self.update_layer_norm(scalar_feat_return, vec_feat_return)

        return scalar_feat_return, vec_feat_return, coord_feats_return

    def message(self, edges):
        """Compute messages on edges using optimized GVP layers."""

        # concatenate x_diff and v on every edge to produce vector features
        vec_feats = [edges.data["x_diff"].unsqueeze(1), edges.src["v"]]
        if self.use_dst_feats:
            vec_feats.append(edges.dst["v"])
        vec_feats = torch.cat(vec_feats, dim=1)

        # create scalar features
        scalar_feats = [edges.src['h'], edges.data['d']]
        if self.edge_feat_size > 0:
            scalar_feats.append(edges.data['a'])

        if self.use_dst_feats:
            scalar_feats.append(edges.dst['h'])

        scalar_feats = torch.cat(scalar_feats, dim=1)

        scalar_message, vector_message = self.edge_message((scalar_feats, vec_feats))

        return {"scalar_msg": scalar_message, "vec_msg": vector_message}


class OptimizedNodePositionUpdate(nn.Module):
    """Stacked OptimizedGVP blocks for fast coordinate refinement."""

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

        gvp_layers = []
        for idx in range(n_gvps):
            last_layer = idx == (n_gvps - 1)
            gvp_layers.append(
                OptimizedGVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=1 if last_layer else n_vec_channels,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=nn.Identity() if last_layer else nn.Sigmoid(),
                    vector_gating=vector_gating,
                )
            )

        self.gvps = nn.Sequential(*gvp_layers)

    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor):
        return node_position_update_forward(self.gvps, scalars, vectors)


class OptimizedEdgeUpdate(nn.Module):
    """JIT-compiled edge update with optional RBF distance conditioning."""

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
        g: dgl.DGLGraph,
        node_scalars: torch.Tensor,
        edge_feats: torch.Tensor,
        d: Optional[torch.Tensor] = None,
    ):
        src_idxs, dst_idxs = g.edges()
        device = node_scalars.device

        if src_idxs.device != device:
            src_idxs = src_idxs.to(device)
            dst_idxs = dst_idxs.to(device)

        if edge_feats.device != device:
            edge_feats = edge_feats.to(device)

        distance_feats = None
        if self.update_edge_w_distance:
            if d is None:
                raise ValueError(
                    "Distance features `d` must be provided when update_edge_w_distance=True."
                )
            distance_feats = d.to(device) if d.device != device else d

        return edge_update_forward(
            self.edge_update_fn,
            self.edge_norm,
            node_scalars,
            edge_feats,
            src_idxs,
            dst_idxs,
            distance_feats,
        )


class OptimizedEnergyGVPConv(nn.Module):
    """Stripped down version of the Optimized GVPConv for energy head."""

    def __init__(
        self,
        scalar_size: int = 128,
        vector_size: int = 16,
        n_cp_feats: int = 0,
        scalar_activation=nn.SiLU,
        vector_activation=nn.Sigmoid,
        n_message_gvps: int = 1,
        use_dst_feats: bool = False,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        edge_feat_size: int = 0,
        message_norm: str = "sum",
        dropout: float = 0.0,
        vector_gating=True,
    ):
        
        super().__init__()
        assert edge_feat_size > 0, "Edge features must be provided."

        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.n_cp_feats = n_cp_feats
        self.scalar_activation = scalar_activation
        self.vector_activation = vector_activation
        self.n_message_gvps = n_message_gvps
        self.edge_feat_size = edge_feat_size
        self.use_dst_feats = use_dst_feats
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.dropout_rate = dropout
        self.message_norm = message_norm

        # create message passing function using OptimizedGVP
        message_gvps = []
        for i in range(n_message_gvps):

            dim_vectors_in = vector_size
            dim_feats_in = scalar_size

            # on the first layer, there is an extra edge vector for the displacement vector between the two node positions
            if i == 0:
                dim_vectors_in += 1
                dim_feats_in += rbf_dim + edge_feat_size
                
            # if this is the first layer and we are using destination node features to compute messages, add them to the input dimensions
            if use_dst_feats and i == 0:
                dim_vectors_in += vector_size
                dim_feats_in += scalar_size

            message_gvps.append(
                OptimizedGVP(
                    dim_vectors_in=dim_vectors_in, 
                    dim_vectors_out=vector_size,
                    n_cp_feats=n_cp_feats, 
                    dim_feats_in=dim_feats_in, 
                    dim_feats_out=scalar_size, 
                    feats_activation=scalar_activation(), 
                    vectors_activation=vector_activation(), 
                    vector_gating=vector_gating
                )
            )
        self.edge_message = nn.Sequential(*message_gvps)
        self.dropout = GVPDropout(self.dropout_rate)
        self.message_layer_norm = GVPLayerNorm(self.scalar_size)

        if isinstance(self.message_norm, str):
            if self.message_norm not in ['mean', 'sum']:
                raise ValueError(f"message_norm must be either 'mean', 'sum', or a number, got {self.message_norm}")
        else:
            assert isinstance(self.message_norm, (float, int)), "message_norm must be either 'mean', 'sum', or a number"

        if self.message_norm == 'mean':
            self.agg_func = fn.mean
        else:
            self.agg_func = fn.sum

    def forward(self, g: dgl.DGLGraph, 
                scalar_feats: torch.Tensor,
                vec_feats: torch.Tensor,
                edge_feats: torch.Tensor = None,
                x_diff: torch.Tensor = None,
                d: torch.Tensor = None):
        # vec_feat has shape (n_nodes, n_vectors, 3)

        with g.local_scope():

            g.ndata['h'] = scalar_feats
            g.ndata['v'] = vec_feats
            g.edata['x_diff'] = x_diff
            g.edata['d'] = d

            # edge feature
            g.edata["a"] = edge_feats

            # compute messages on every edge
            g.apply_edges(self.message)

            # aggregate messages from every edge
            g.update_all(fn.copy_e("scalar_msg", "m"), self.agg_func("m", "scalar_msg"))
            g.update_all(fn.copy_e("vec_msg", "m"), self.agg_func("m", "vec_msg"))

            # get aggregated scalar and vector messages
            scalar_msg = g.ndata["scalar_msg"]
            vec_msg = g.ndata["vec_msg"]

            # dropout scalar and vector messages
            scalar_msg, vec_msg = self.dropout(scalar_msg, vec_msg)

            # update scalar and vector features, apply layernorm
            scalar_feat_new = g.ndata['h'] + scalar_msg
            vec_feat_new = g.ndata['v'] + vec_msg
            
            scalar_feat_new, vec_feat_new = self.message_layer_norm(scalar_feat_new, vec_feat_new)

        return scalar_feat_new, vec_feat_new

    def message(self, edges):
        """Compute messages on edges using optimized GVP layers."""

        # concatenate x_diff and v on every edge to produce vector features
        vec_feats = [edges.data["x_diff"].unsqueeze(1), edges.src["v"]]
        if self.use_dst_feats:
            vec_feats.append(edges.dst["v"])
        vec_feats = torch.cat(vec_feats, dim=1)

        # create scalar features
        scalar_feats = [edges.src['h'], edges.data['d']]
        if self.edge_feat_size > 0:
            scalar_feats.append(edges.data['a'])

        if self.use_dst_feats:
            scalar_feats.append(edges.dst['h'])

        scalar_feats = torch.cat(scalar_feats, dim=1)

        scalar_message, vector_message = self.edge_message((scalar_feats, vec_feats))

        return {"scalar_msg": scalar_message, "vec_msg": vector_message}


class OptimizedEnergyHead(nn.Module):

    def __init__(
        self,
        n_scalars: int,
        n_vec_channels: int,
        n_gvps: int = 3,
    ) -> None:
        super().__init__()

        if n_gvps < 1:
            raise ValueError("n_gvps must be >= 1")

        gvp_layers = []
        for idx in range(n_gvps):
            last_layer = idx == (n_gvps - 1)
            gvp_layers.append(
                OptimizedGVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=1 if last_layer else n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=1 if last_layer else n_vec_channels,
                    vectors_activation=nn.Identity() if last_layer else nn.Sigmoid(),
                    vector_gating=True,
                )
            )

        self.gvps = nn.Sequential(*gvp_layers)

    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor):
        return energy_head_forward(self.gvps, scalars, vectors)