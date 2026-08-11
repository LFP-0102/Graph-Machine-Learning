import numpy as np

def normalize_adj(adj):
    """计算 GCN 归一化邻接矩阵: Â = D^(-1/2)(A+I)D^(-1/2).

    adj: 原始邻接矩阵 [N, N]
    """
    adj = adj + np.eye(adj.shape[0])           # A + I
    degree = np.array(adj.sum(1))              # d_i = Σ_j A_ij
    degree_inv = np.power(degree, -0.5)        # D^(-1/2)
    degree_inv[np.isinf(degree_inv)] = 0       # 防止孤立节点 1/0 = inf
    D = np.diag(degree_inv)
    return D @ adj @ D                         # D^(-1/2) (A+I) D^(-1/2)


def add_self_loops(adj):
    """添加自环 A+I, 用于 GAT 等需要原始二值邻接矩阵的模型.

    adj: 原始邻接矩阵 [N, N] (numpy)
    """
    adj = adj + np.eye(adj.shape[0])
    return (adj > 0).astype(np.float32)
