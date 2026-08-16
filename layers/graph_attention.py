"""GAT（图注意力网络）层。

复现论文《Graph Attention Networks》（Veličković 等，ICLR 2018）的
注意力计算公式 (3)--(5)，同时兼容稀疏 ``edge_index`` 与稠密邻接矩阵。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_max(src, index, size):
    """按索引计算最大值，不依赖 torch_scatter。"""
    output = torch.full((size,), float("-inf"), device=src.device, dtype=src.dtype)
    return output.scatter_reduce_(0, index, src, reduce="amax", include_self=False)


def _scatter_sum(src, index, size):
    """按索引求和，不依赖 torch_scatter。"""
    output = torch.zeros(size, device=src.device, dtype=src.dtype)
    return output.scatter_reduce_(0, index, src, reduce="sum", include_self=False)


def _is_edge_index(graph):
    return graph.dim() == 2 and graph.size(0) == 2


class GraphAttention(nn.Module):
    """单头图注意力层。

    参数：
        in_features：输入特征维度。
        out_features：输出特征维度。
        dropout：注意力系数的 dropout 概率。
        alpha：LeakyReLU 的负斜率。
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # 与官方 TensorFlow 实现一致，左右端点使用独立打分向量。
        self.score_left = nn.Parameter(torch.empty(out_features, 1))
        self.score_right = nn.Parameter(torch.empty(out_features, 1))
        self.score_left_bias = nn.Parameter(torch.zeros(()))
        self.score_right_bias = nn.Parameter(torch.zeros(()))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.score_left)
        nn.init.xavier_uniform_(self.score_right)

    def forward(self, x, graph, return_attention=False):
        """计算节点表示；``return_attention=True`` 时额外返回注意力矩阵。"""
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        num_nodes = x.size(0)
        transformed = torch.mm(x, self.W)
        if _is_edge_index(graph):
            return self._forward_sparse(num_nodes, transformed, graph, return_attention)
        return self._forward_dense(transformed, graph, return_attention)

    def _forward_dense(self, transformed, adj, return_attention):
        """稠密邻接矩阵路径，保留以兼容早期实验代码。"""
        left = (transformed @ self.score_left).squeeze(1)
        right = (transformed @ self.score_right).squeeze(1)
        scores = self.leakyrelu(left.unsqueeze(1) + right.unsqueeze(0) + self.score_left_bias + self.score_right_bias)
        attention = F.softmax(scores.masked_fill(adj == 0, float("-inf")), dim=1)
        attention = self.dropout(attention)
        output = torch.mm(attention, F.dropout(transformed, p=self.dropout_rate, training=self.training)) + self.bias
        return (output, attention) if return_attention else output

    def _forward_sparse(self, num_nodes, transformed, edge_index, return_attention):
        """稀疏边列表路径，只在实际边上计算注意力。"""
        # 约定 edge_index[0] 为接收节点，edge_index[1] 为发送节点。
        dst, src = edge_index[0], edge_index[1]
        loops = torch.arange(num_nodes, device=edge_index.device)
        dst_all, src_all = torch.cat([dst, loops]), torch.cat([src, loops])
        scores = self.leakyrelu(
            (transformed[dst_all] @ self.score_left).squeeze(1)
            + (transformed[src_all] @ self.score_right).squeeze(1)
            + self.score_left_bias + self.score_right_bias
        )
        max_scores = _scatter_max(scores, dst_all, num_nodes)
        exp_scores = torch.exp(scores - max_scores[dst_all])
        attention = self.dropout(exp_scores / (_scatter_sum(exp_scores, dst_all, num_nodes)[dst_all] + 1e-37))
        messages = F.dropout(transformed, p=self.dropout_rate, training=self.training)[src_all] * attention.unsqueeze(1)
        output = torch.zeros(num_nodes, self.out_features, device=transformed.device, dtype=transformed.dtype)
        output.index_add_(0, dst_all, messages)
        output = output + self.bias
        if return_attention:
            dense_attention = torch.zeros(num_nodes, num_nodes, device=transformed.device, dtype=transformed.dtype)
            dense_attention[dst_all, src_all] = attention
            return output, dense_attention
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features} -> {self.out_features}, dropout={self.dropout_rate})"


class MultiHeadGraphAttention(nn.Module):
    """多头图注意力层；隐藏层拼接多头输出，输出层可改为均值聚合。"""

    def __init__(self, in_features, out_features, n_heads=8, dropout=0.6, alpha=0.2, concat=True):
        super().__init__()
        self.concat = concat
        self.n_heads = n_heads
        self.heads = nn.ModuleList([GraphAttention(in_features, out_features, dropout, alpha) for _ in range(n_heads)])

    def forward(self, x, graph, return_attention=False):
        """返回多头聚合结果；需要时同时返回每个头的注意力矩阵。"""
        if return_attention:
            results = [head(x, graph, return_attention=True) for head in self.heads]
            outputs, attentions = [item[0] for item in results], [item[1] for item in results]
        else:
            outputs = [head(x, graph) for head in self.heads]
        output = torch.cat(outputs, dim=1) if self.concat else torch.stack(outputs, dim=0).mean(dim=0)
        return (output, attentions) if return_attention else output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.heads[0].in_features} -> {self.heads[0].out_features}, heads={self.n_heads}, concat={self.concat})"
