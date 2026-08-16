"""GraphSAGE 的均值聚合器。"""

import torch
import torch.nn as nn


class MeanAggregator(nn.Module):
    """按论文的 element-wise mean 聚合邻居向量。"""

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, neighbor_features):
        """将 ``[邻居数, 特征维]`` 聚合为 ``[特征维]``。"""
        if neighbor_features.numel() == 0:
            return torch.zeros(self.input_dim, device=neighbor_features.device)
        return torch.mean(neighbor_features, dim=0)
