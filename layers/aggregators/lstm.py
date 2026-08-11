import torch
import torch.nn as nn

class LSTMAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, neighbor_features):
        if len(neighbor_features) == 0:
            return torch.zeros(self.lstm.hidden_size)
        x = neighbor_features.unsqueeze(0)         # 增加 batch 维
        _, (h, c) = self.lstm(x)
        return h.squeeze(0)
