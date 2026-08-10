"""
GraphSAGE Layer
对应论文公式:
h_v^k = σ(W · CONCAT(h_v^{k-1}, AGG({h_u, ∀u∈N(v)})))
"""
import torch
import torch.nn as nn
from layers.aggregators.mean import MeanAggregator


class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.aggregator = MeanAggregator(input_dim)
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(self, features, adj_lists):
        outputs = []

        for node in range(features.size(0)):
            self_feat = features[node]
            neighbors = adj_lists.get(node, [])

            if len(neighbors) == 0:
                neighbor_feat = torch.zeros_like(self_feat)
            else:
                neighbor_feat = self.aggregator(
                    features[neighbors]
                )

            h = torch.cat([self_feat, neighbor_feat], dim=0)
            outputs.append(h)

        outputs = torch.stack(outputs)
        return self.linear(outputs)
