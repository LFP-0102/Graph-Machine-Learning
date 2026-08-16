"""GraphSAGE 的 LSTM 聚合器。"""

import torch
import torch.nn as nn


class LSTMAggregator(nn.Module):
    """使用 LSTM 编码邻居特征序列。"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, neighbor_features):
        """返回 LSTM 最后一个隐藏状态；空邻居集合返回零向量。"""
        if len(neighbor_features) == 0:
            return torch.zeros(self.lstm.hidden_size, device=neighbor_features.device)
        _, (hidden, _) = self.lstm(neighbor_features.unsqueeze(0))
        return hidden.squeeze(0)
