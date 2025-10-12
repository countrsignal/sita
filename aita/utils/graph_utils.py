import torch
from typing import Tuple


def fully_connected_edges(num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes = torch.arange(num_nodes)
    edges = torch.cartesian_prod(nodes, nodes)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return edges[:, 0], edges[:, 1]