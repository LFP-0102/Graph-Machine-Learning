"""GAT（图注意力网络）模型。

复现论文：Graph Attention Networks（Veličković 等，ICLR 2018）。
默认结构与论文 Cora 实验一致：第一层 8 个注意力头，每头 8 维。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_attention import MultiHeadGraphAttention


class GAT(nn.Module):
    """两层多头图注意力网络，支持 ``edge_index`` 与稠密邻接矩阵输入。"""

    graph_type = "edge_index"

    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=8, dropout=0.6):
        super().__init__()
        self.dropout_rate = dropout
        self.gat1 = MultiHeadGraphAttention(
            input_dim, hidden_dim, n_heads=n_heads, concat=True, dropout=dropout
        )
        self.gat2 = MultiHeadGraphAttention(
            hidden_dim * n_heads, output_dim, n_heads=1, concat=False, dropout=dropout
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x, graph):
        """返回节点的多分类对数概率。"""
        x = self.gat1(x, graph)
        x = F.elu(x)
        x = self.gat2(x, graph)
        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def get_embeddings(self, x, graph):
        """提取第一层经 ELU 激活后的节点嵌入。"""
        self.eval()
        return F.elu(self.gat1(x, graph))

    @torch.no_grad()
    def get_attention_weights(self, x, graph):
        """提取第一层多头注意力系数。

        返回平均注意力矩阵和每个头的注意力矩阵。稠密注意力矩阵在大图上可能占用
        较多显存，应仅用于局部可视化。
        """
        self.eval()
        _, all_attn = self.gat1(x, graph, return_attention=True)
        avg_attn = torch.stack(all_attn, dim=0).mean(dim=0)
        return avg_attn, all_attn
