import dgl
import torch
from typing import Optional

from ..utils.graph_utils import GraphAdapter


class InferenceVectorField(torch.nn.Module):

    def __init__(
        self,
        compiled_model: torch.nn.Module,
        remove_com: bool = True,
        dynamo_cache_size_limit: Optional[int] = 16,
    ):
        super(InferenceVectorField, self).__init__()
        self.dynamo_cache_size_limit = dynamo_cache_size_limit
        self.compiled_model = compiled_model
        self.remove_com = remove_com
        self._setup()
    
    def _setup(self) -> None:
        if self.dynamo_cache_size_limit is not None:
            torch._dynamo.config.cache_size_limit = self.dynamo_cache_size_limit

        self.compiled_model.eval();
    
    def inference_fwd(self, graph: dgl.DGLGraph) -> torch.Tensor:

        # adapt and pad the graph
        adapter, data = GraphAdapter.adapt_and_pad(
            graph, target_key=None, use_rbf=True
        )
        # move data to device
        data = tuple(t.to(adapter.device) for t in data)
        times, x_t, node_feats, atom_index, edge_feats, atom_mask, pair_mask = data

        # remove center of mass from current state
        x_t = x_t - x_t.mean(dim=1, keepdim=True)

        # forward pass
        velocity, *_ = self.compiled_model(
            x_t=x_t,
            time=times,
            attr=node_feats,
            atom_index=atom_index,
            pair_feats=edge_feats,
            atom_mask=atom_mask,
            pair_mask=pair_mask,
        )

        # remove center of mass from velocity field
        if self.remove_com:
            velocity = velocity - velocity.mean(dim=1, keepdim=True)

        return velocity