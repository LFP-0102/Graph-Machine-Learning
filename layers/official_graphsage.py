"""与 williamleif/GraphSAGE TensorFlow 代码对齐的均值聚合层。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficialMeanAggregator(nn.Module):
    """分别投影自身与邻居特征，再将结果拼接。"""

    def __init__(self, input_dim, output_dim, activation, dropout=0.0):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.neighbor_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation
        self.dropout = dropout
        nn.init.xavier_uniform_(self.self_linear.weight)
        nn.init.xavier_uniform_(self.neighbor_linear.weight)

    def forward(self, self_vectors, neighbor_vectors):
        self_vectors = F.dropout(self_vectors, p=self.dropout, training=self.training)
        neighbor_vectors = F.dropout(neighbor_vectors, p=self.dropout, training=self.training)
        neighbor_mean = neighbor_vectors.mean(dim=1)
        output = torch.cat(
            [self.self_linear(self_vectors), self.neighbor_linear(neighbor_mean)], dim=1
        )
        return self.activation(output)
