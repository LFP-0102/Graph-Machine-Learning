import torch
import torch.nn as nn

class PoolAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

    def forward(self, neighbor_features):
        if len(neighbor_features) == 0:
            return torch.zeros(self.mlp[0].out_features)
        h = self.mlp(neighbor_features)
        return torch.max(h, dim=0)[0]
