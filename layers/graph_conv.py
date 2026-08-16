"""GCN 模型使用的图卷积层。"""
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """对稠密或稀疏邻接矩阵计算 ``adj @ (x @ weight)``。"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        return output
