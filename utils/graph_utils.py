"""图模型共享的邻接矩阵预处理函数。"""
import numpy as np
import torch


def normalize_adj(adj):
    """为小图返回稠密的 ``D^-1/2 (A + I) D^-1/2``。"""
    adj = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    degree_inv_sqrt = np.power(np.asarray(adj.sum(1)).reshape(-1), -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    return np.diag(degree_inv_sqrt) @ adj @ np.diag(degree_inv_sqrt)


def normalized_sparse_edge_index(edge_index, num_nodes, device=None):
    """根据边索引构建稀疏的 ``D^-1/2 (A + I) D^-1/2``。"""
    device = device or edge_index.device
    edge_index = edge_index.to(device)
    loops = torch.arange(num_nodes, device=device)
    indices = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)
    values = torch.ones(indices.size(1), device=device)
    degree = torch.zeros(num_nodes, device=device)
    degree.index_add_(0, indices[0], values)
    degree_inv_sqrt = degree.clamp_min(1).pow(-0.5)
    values = degree_inv_sqrt[indices[0]] * values * degree_inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), device=device
    ).coalesce()


def add_self_loops(adj):
    """返回包含自环的二值稠密邻接矩阵。"""
    adj = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    return (adj > 0).astype(np.float32)
