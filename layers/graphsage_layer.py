"""GraphSAGE Sample-and-Aggregate 算法中的均值聚合层。"""
import torch
import torch.nn as nn


class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(self, features, edge_index, sampled_neighbors=None):
        """训练时聚合采样邻居，评估时聚合全部邻居。

        ``edge_index[0]`` 表示目标节点，``edge_index[1]`` 表示其一个邻居。
        数据加载器会将引文边处理为双向边。
        """
        if sampled_neighbors is None:
            receivers, neighbors = edge_index
            neighbor_sum = torch.zeros_like(features)
            neighbor_sum.index_add_(0, receivers, features[neighbors])
            counts = torch.zeros(features.size(0), 1, device=features.device)
            counts.index_add_(
                0,
                receivers,
                torch.ones(receivers.numel(), 1, device=features.device),
            )
            neighbor_mean = neighbor_sum / counts.clamp_min(1)
        else:
            neighbor_ids, valid = sampled_neighbors
            gathered = features[neighbor_ids.to(features.device)]
            valid_float = valid.to(features.device).unsqueeze(-1).to(features.dtype)
            neighbor_mean = (gathered * valid_float).sum(dim=1)
            neighbor_mean = neighbor_mean / valid_float.sum(dim=1).clamp_min(1)

        return self.linear(torch.cat([features, neighbor_mean], dim=1))
