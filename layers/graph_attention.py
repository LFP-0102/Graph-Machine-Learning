"""
GAT（Graph Attention Network）注意力层。

复现论文：Graph Attention Networks (Veličković et al., ICLR 2018)

GAT 核心公式：
    e_ij = LeakyReLU(a^T [W h_i || W h_j])      (3)
    α_ij = softmax_j(e_ij)                        (4)  —— 仅对邻居做 softmax
    h'_i = σ( Σ_j α_ij W h_j )                    (5)

其中 "||" 表示向量拼接。

支持两种图输入格式：
    - edge_index [2, E]: 稀疏边列表（推荐，O(E) 内存）
    - adj [N, N]:        稠密邻接矩阵（向后兼容，O(N²) 内存）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── scatter 工具（不依赖 torch_scatter） ───────────────────────
def _scatter_max(src, index, n):
    out = torch.full((n,), float("-inf"), device=src.device, dtype=src.dtype)
    return out.scatter_reduce_(0, index, src, reduce="amax", include_self=False)


def _scatter_sum(src, index, n):
    out = torch.zeros(n, device=src.device, dtype=src.dtype)
    return out.scatter_reduce_(0, index, src, reduce="sum", include_self=False)


# ── 格式检测 ───────────────────────────────────────────────────
def _is_edge_index(graph):
    return graph.dim() == 2 and graph.size(0) == 2


class GraphAttention(nn.Module):
    """单头图注意力层。

    实现论文公式 (3)-(5)。输入可以是 edge_index 或 dense adj。

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

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, graph, return_attention=False):
        """
        参数：
            x:     节点特征 [N, F_in]
            graph: edge_index [2, E] 或 稠密邻接矩阵 [N, N]
            return_attention: 若为 True，额外返回 attention 系数矩阵 [N, N]

        返回：
            h_prime: 更新后的节点表示 [N, F_out]
            (h_prime, attention): 当 return_attention=True
        """
        N = x.size(0)
        Wh = torch.mm(x, self.W)                                    # [N, F_out]

        a_left = self.a[:self.out_features, 0]                       # [F_out]
        a_right = self.a[self.out_features:, 0]                     # [F_out]

        if _is_edge_index(graph):
            return self._forward_sparse(
                N, Wh, a_left, a_right, graph, return_attention)
        else:
            return self._forward_dense(
                N, Wh, a_left, a_right, graph, return_attention)

    def _forward_dense(self, N, Wh, a_left, a_right, adj, return_attention):
        """稠密路径：O(N²) 内存，向后兼容。"""
        attn_left = torch.mv(Wh, a_left)                             # [N]
        attn_right = torch.mv(Wh, a_right)                           # [N]

        e = attn_left.unsqueeze(1) + attn_right.unsqueeze(0)         # [N, N]
        e = self.leakyrelu(e)
        attention = e.masked_fill(adj == 0, float("-inf"))
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.mm(attention, Wh)

        if return_attention:
            return h_prime, attention
        return h_prime

    def _forward_sparse(self, N, Wh, a_left, a_right, edge_index, return_attention):
        """稀疏路径：仅对边计算 attention，O(E) 内存。"""
        src, dst = edge_index[0], edge_index[1]

        # 添加自环（GAT 标准做法）
        self_loop = torch.arange(N, device=edge_index.device)
        src_all = torch.cat([src, self_loop])
        dst_all = torch.cat([dst, self_loop])

        # 边级 attention scores
        a_src = (Wh[src_all] * a_left).sum(dim=1)                    # [E+N]
        a_dst = (Wh[dst_all] * a_right).sum(dim=1)                   # [E+N]
        e = self.leakyrelu(a_src + a_dst)                            # [E+N]

        # Softmax per destination node (numerical stable)
        e_max = _scatter_max(e, dst_all, N)                          # [N]
        e_exp = torch.exp(e - e_max[dst_all])                        # [E+N]
        e_sum = _scatter_sum(e_exp, dst_all, N)                      # [N]
        alpha = e_exp / (e_sum[dst_all] + 1e-37)                     # [E+N]
        alpha = self.dropout(alpha)

        # 特征聚合
        weighted = Wh[src_all] * alpha.unsqueeze(1)                  # [E+N, F_out]
        device = Wh.device
        h_prime = torch.zeros(N, self.out_features, device=device)
        h_prime.index_add_(0, dst_all, weighted)

        if return_attention:
            attn_dense = torch.zeros(N, N, device=device)
            attn_dense[dst_all, src_all] = alpha
            return h_prime, attn_dense
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

    def forward(self, x, graph, return_attention=False):
        """
        返回：
            concat=True:  [N, n_heads * out_features]
            concat=False: [N, out_features]
            return_attention=True 时额外返回各头 attention 系数列表
        """
        if return_attention:
            results = [head(x, graph, return_attention=True) for head in self.heads]
            outputs = [r[0] for r in results]
            attentions = [r[1] for r in results]
        else:
            outputs = [head(x, graph) for head in self.heads]

        if self.concat:
            out = torch.cat(outputs, dim=1)
        else:
            out = torch.stack(outputs, dim=0).mean(dim=0)

        if return_attention:
            return out, attentions
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.heads[0].in_features} -> "
            f"{self.heads[0].out_features}, "
            f"heads={self.n_heads}, "
            f"concat={self.concat})"
        )
