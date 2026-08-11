"""
Graph Attention Network (GAT).

Paper: Graph Attention Networks (Veličković et al., ICLR 2018)

两层结构：
    Layer 1: K=8 头注意力, 每头 8 维 → ELU → 拼接 → 64 维隐藏
    Layer 2: 单头注意力 (平均) → C 维 → log_softmax

Cora 超参数：hidden_dim=8, n_heads=8, dropout=0.6, lr=0.005, weight_decay=5e-4

支持 edge_index [2, E] 和 dense adj [N, N] 两种图输入。
edge_index 模式走 O(E) 稀疏注意力（推荐，支持大图）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_attention import MultiHeadGraphAttention, _is_edge_index


class GAT(nn.Module):
    graph_type = "edge_index"

    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=8, dropout=0.6):
        super().__init__()
        self.dropout_rate = dropout
        self.gat1 = MultiHeadGraphAttention(
            input_dim, hidden_dim, n_heads=n_heads, concat=True, dropout=dropout)
        self.gat2 = MultiHeadGraphAttention(
            hidden_dim * n_heads, output_dim, n_heads=1, concat=False, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if m is not self and hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, x, graph):
        """graph: edge_index [2, E] (推荐) 或 dense adj [N, N] (兼容)."""
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gat1(x, graph)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gat2(x, graph)
        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def get_embeddings(self, x, graph):
        """返回第一层 GAT 输出（ELU 后）, 维度 [N, n_heads * hidden_dim]."""
        self.eval()
        x = self.gat1(x, graph)
        return F.elu(x)

    @torch.no_grad()
    def get_attention_weights(self, x, graph):
        """提取第一层 GAT 的多头注意力系数.

        返回:
            avg_attn: 8 头平均 attention [N, N]
            all_attn: 8 个头的 attention 列表, 每个 [N, N]

        注意: dense attention [N, N] 在大图上可能 OOM；
              大规模图建议仅抽样部分节点做可视化。
        """
        self.eval()
        _, all_attn = self.gat1(x, graph, return_attention=True)
        avg_attn = torch.stack(all_attn, dim=0).mean(dim=0)
        return avg_attn, all_attn
