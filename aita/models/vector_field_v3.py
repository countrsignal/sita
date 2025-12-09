from typing import Union, Any

import dgl
import torch
import torch.nn as nn

from .modules.energy_net import EnergyNet
from .modules.encoders import AtomEncoder
from .modules.decoder_opt import OptimizedGVPDecoder


###################################
# Classes
###################################

class VFV3(nn.Module):

    def __init__(
        self,
        n_vec: int = 16,
        n_features: int = 21,
        edge_feat_size: int = 9,
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

        self.n_vec_channels = n_vec
        self.atom_encoder = AtomEncoder(
            n_features=n_features,
            n_hidden=n_hidden_nodes,
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_feat_size, n_hidden_edge),
            nn.SiLU(),
            nn.Linear(n_hidden_edge, n_hidden_edge),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_edge),
        )

        self.gvp_decoder = OptimizedGVPDecoder(
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
            use_dst_feats=use_dst_feats,
            vector_gating=vector_gating,
        )

        self.ebm = EnergyNet(
            n_vec=n_vec,
            n_hidden=n_hidden_nodes,
            edge_feat_size=n_hidden_edge,
            n_message_gvps=n_message_gvps,
            n_output_gvps=n_update_gvps,
            rbf_dmax=rbf_dmax,
            rbf_dim=rbf_dim,
            message_norm=message_norm,
        )

    def load_from_checkpoint(self, checkpoint_path: str, **kwargs: Any) -> "VFV3":
        checkpoint = torch.load(checkpoint_path, **kwargs)
        self.load_state_dict(checkpoint)
        return self

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
        node_repr = self.atom_encoder(
            time=graph.ndata["t"].view(-1),
            attr=graph.ndata["attr"],
            atom_index=graph.ndata["atom_index"].view(-1),
        )
        # node_repr: (num_nodes, n_hidden)

        # embed the edge features
        edge_mask = ~torch.all(graph.edata["attr"] == 0, dim=-1).unsqueeze(dim=-1)
        edge_repr = self.edge_embedding(graph.edata["attr"]) * edge_mask
        # edge_repr: (num_edges, n_hidden_edge)

        # predict the vector field
        velocity, *_ = self.gvp_decoder(node_repr, v_init, x_init, edge_repr, edge_mask, graph)
        # velocity: (num_nodes, 3)

        return velocity

    def training_step(self, graph: dgl.DGLGraph) -> dict[str, torch.Tensor]:

        # flow training step ================================================================
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
        node_repr = self.atom_encoder(
            time=graph.ndata["t"].view(-1),
            attr=graph.ndata["attr"],
            atom_index=graph.ndata["atom_index"].view(-1),
        )
        # node_repr: (num_nodes, n_hidden)

        # embed the edge features
        edge_mask = ~torch.all(graph.edata["attr"] == 0, dim=-1).unsqueeze(dim=-1)
        edge_repr = self.edge_embedding(graph.edata["attr"]) * edge_mask
        # edge_repr: (num_edges, n_hidden_edge)

        # predict the vector field
        velocity, hs, vs, edge_repr = self.gvp_decoder(node_repr, v_init, x_init, edge_repr, edge_mask, graph)
        # velocity: (num_nodes, 3)
        # hs: (num_nodes, n_hidden)
        # vs: (num_nodes, n_vec_channels, 3)
        # edge_repr: (num_edges, n_hidden_edge)

        # compute vector field loss
        with graph.local_scope():
            graph.ndata["vf_loss_per_node"] = torch.square(velocity - graph.ndata["vt"]).mean(dim=-1)
            vf_loss = dgl.mean_nodes(graph, "vf_loss_per_node").mean()
        # ====================================================================================

        # EBM training step ==================================================================
        # detach from flow computation graph
        z = graph.ndata["z"].detach()
        x_t = graph.ndata["xt"].detach() 
        time = graph.ndata["t"].detach().view(-1)
        hs = hs.detach()
        vs = vs.detach()
        edge_feats = edge_repr.detach()

        loss_dict = self.ebm.training_step(
            z=z,
            x_t=x_t,
            time=time,
            node_scalars=hs,
            node_vectors=vs,
            edge_feats=edge_feats,
            graph=graph,
        )
        # ====================================================================================

        loss_dict = {**loss_dict, "vf_loss": vf_loss}
        loss_dict["loss"] = loss_dict["vf_loss"] + loss_dict["sm_loss"] + loss_dict["nce_loss"]
        return loss_dict