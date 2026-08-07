"""
GAT（Graph Attention Network）模型。

复现论文：Graph Attention Networks (Veličković et al., ICLR 2018)

模型结构（2 层）：
    Layer 1:  K=8 头注意力, 每头 8 维 → ELU → 拼接 → 64 维隐藏表示
    Layer 2:  单头注意力 (对多头取平均) → C 维 → log_softmax

论文中 Cora 超参数：
    - hidden_dim = 8 (每个头)
    - n_heads = 8
    - dropout = 0.6
    - lr = 0.01 (Adam) / 0.005 (SGD)
    - weight_decay = 5e-4
    - 训练 100 epoch (early stopping on val)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_attention import MultiHeadGraphAttention


class GAT(nn.Module):
    """两层 GAT 网络。

    参数：
        input_dim:   输入特征维度
        hidden_dim:  隐藏层每头维度 (论文中 = 8)
        output_dim:  分类类别数
        n_heads:     第一层注意力头数 (论文中 = 8)
        dropout:     dropout 率 (论文中 = 0.6)
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 n_heads=8, dropout=0.6):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)

        # 第一层：K 头注意力，拼接
        # 输入:  [N, F]      → 拼接后: [N, K * hidden_dim]
        self.gat1 = MultiHeadGraphAttention(
            input_dim, hidden_dim, n_heads,
            dropout=dropout, concat=True
        )

        # 第二层：单头注意力（平均），输出类别概率
        # K=1 并 concat=False 等同于单头取平均
        # 输入:  [N, K * hidden_dim]  → 输出: [N, output_dim]
        self.gat2 = MultiHeadGraphAttention(
            hidden_dim * n_heads, output_dim, 1,
            dropout=dropout, concat=False
        )

    def forward(self, x, adj):
        """
        参数：
            x:   节点特征 [N, F]
            adj: 二值邻接矩阵 [N, N]（已添加自环，用于 attention mask）

        返回：
            log_softmax 分类概率 [N, C]
        """
        # 第一层
        x = self.dropout_layer(x)
        x = self.gat1(x, adj)
        x = F.elu(x)

        # 第二层
        x = self.dropout_layer(x)
        x = self.gat2(x, adj)

        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def get_embeddings(self, x, adj):
        """提取第一层 GAT 输出（ELU 后）作为隐藏层嵌入。

        维度: [N, K * hidden_dim]，GAT 论文中 = [N, 64]
        """
        self.eval()
        x = self.dropout_layer(x)
        x = self.gat1(x, adj)
        x = F.elu(x)
        return x
