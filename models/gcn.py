#!/usr/bin/env python3
"""GCN model for node classification on graph datasets."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network."""

    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 16):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
