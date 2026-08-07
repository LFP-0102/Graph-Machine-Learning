"""
GAT（Graph Attention Network）注意力层。

复现论文：Graph Attention Networks (Veličković et al., ICLR 2018)

GAT 核心公式：
    e_ij = LeakyReLU(a^T [W h_i || W h_j])      (3)
    α_ij = softmax_j(e_ij)                        (4)  —— 仅对邻居做 softmax
    h'_i = σ( Σ_j α_ij W h_j )                    (5)

其中 "||" 表示向量拼接。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):
    """单头图注意力层。

    实现论文公式 (3)-(5)。使用稠密邻接矩阵作为 attention mask。

    参数：
        in_features:  输入特征维度
        out_features: 输出特征维度
        dropout:      attention 系数的 dropout 率
        alpha:        LeakyReLU 的负斜率
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout
        self.alpha = alpha

        # W: 共享线性变换矩阵 [in_features, out_features]
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))

        # a: 注意力向量 [2 * out_features, 1]
        #    前半部分用于「源节点」的 attention score
        #    后半部分用于「目标节点」的 attention score
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier 初始化参数。"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, adj):
        """
        参数：
            x:   节点特征 [N, F_in]
            adj: 邻接矩阵 [N, N]  二值矩阵（含自环），用作 attention mask
                 adj[i][j] > 0  → 节点 j 是节点 i 的邻居

        返回：
            h_prime: 更新后的节点表示 [N, F_out]
        """
        N = x.size(0)

        # ── 1. 线性变换 ───────────────────────────────────
        #    公式 (3) 中的 W h_i
        Wh = torch.mm(x, self.W)                                    # [N, F_out]

        # ── 2. 计算 attention scores ──────────────────────
        #    用「分半 a」技巧避免构造 [N, N, 2F_out] 中间张量
        #    e_ij = LeakyReLU(a_left^T Wh_i + a_right^T Wh_j)
        a_left = self.a[:self.out_features, 0]                       # [F_out]
        a_right = self.a[self.out_features:, 0]                     # [F_out]

        attn_left = torch.mm(Wh, a_left.unsqueeze(1))               # [N, 1]
        attn_right = torch.mm(Wh, a_right.unsqueeze(1))             # [N, 1]

        # 广播加法：e[i,j] = attn_left[i] + attn_right[j]
        e = attn_left + attn_right.T                                # [N, N]
        e = self.leakyrelu(e)

        # ── 3. mask 非邻居 ────────────────────────────────
        #    只允许 adj[i][j] > 0 的 j 参与 attention
        attention = e.masked_fill(adj == 0, float("-inf"))

        # ── 4. softmax 归一化 (公式 4) ────────────────────
        attention = F.softmax(attention, dim=1)                     # [N, N]
        attention = self.dropout(attention)

        # ── 5. 加权聚合 (公式 5) ──────────────────────────
        h_prime = torch.mm(attention, Wh)                           # [N, F_out]

        return h_prime

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.in_features} -> {self.out_features}, "
            f"dropout={self.dropout_rate})"
        )


class MultiHeadGraphAttention(nn.Module):
    """多头图注意力层。

    K 个独立的注意力头并行计算。
    - 隐藏层：拼接各头输出 → [N, K * out_features]
    - 输出层（concat=False）：平均各头输出 → [N, out_features]

    参数：
        in_features:  输入特征维度
        out_features: 每个头的输出特征维度
        n_heads:      注意力头数 K
        dropout:      attention dropout
        alpha:        LeakyReLU 负斜率
        concat:       True=拼接（隐藏层） / False=平均（输出层）
    """

    def __init__(self, in_features, out_features, n_heads=8,
                 dropout=0.6, alpha=0.2, concat=True):
        super().__init__()
        self.concat = concat
        self.n_heads = n_heads

        self.heads = nn.ModuleList([
            GraphAttention(in_features, out_features, dropout, alpha)
            for _ in range(n_heads)
        ])

    def forward(self, x, adj):
        """
        返回：
            concat=True:  [N, n_heads * out_features]
            concat=False: [N, out_features]
        """
        if self.concat:
            # 隐藏层：拼接所有头
            return torch.cat([head(x, adj) for head in self.heads], dim=1)
        else:
            # 输出层：平均所有头
            outputs = torch.stack([head(x, adj) for head in self.heads], dim=0)
            return outputs.mean(dim=0)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.heads[0].in_features} -> "
            f"{self.heads[0].out_features}, "
            f"heads={self.n_heads}, "
            f"concat={self.concat})"
        )
