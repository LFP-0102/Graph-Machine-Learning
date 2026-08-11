import torch
import torch.nn as nn


class MeanAggregator(nn.Module):
    """论文 AGGREGATE: element-wise mean of neighbor vectors."""
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, neighbor_features):
        """neighbor_features: [neighbor_num, feature_dim] → [feature_dim]"""
        if neighbor_features.numel() == 0:
            return torch.zeros(self.input_dim, device=neighbor_features.device)
        return torch.mean(neighbor_features, dim=0)
