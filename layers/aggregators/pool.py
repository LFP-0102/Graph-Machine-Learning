"""GraphSAGE 的池化聚合器。"""

import torch
import torch.nn as nn


class PoolAggregator(nn.Module):
    """先经 MLP 变换邻居特征，再执行逐维最大池化。"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

    def forward(self, neighbor_features):
        """聚合邻居特征；空邻居集合返回零向量。"""
        if len(neighbor_features) == 0:
            return torch.zeros(self.mlp[0].out_features, device=neighbor_features.device)
        return torch.max(self.mlp(neighbor_features), dim=0)[0]
