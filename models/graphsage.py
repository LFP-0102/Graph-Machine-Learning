"""
GraphSAGE Model
Paper: Inductive Representation Learning on Large Graphs
(Hamilton et al., NeurIPS 2017)

Algorithm 1: GraphSAGE embedding generation
  h⁰_v ← x_v
  for k = 1..K:
      h^k_v ← σ(W^k · CONCAT(h^{k-1}_v, AGG({h^{k-1}_u, ∀u∈N(v)})))
      h^k_v ← h^k_v / ||h^k_v||₂
  z_v ← h^K_v
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.graphsage_layer import GraphSAGELayer


class GraphSAGE(nn.Module):
    """两层 GraphSAGE 网络。

    参数：
        input_dim:   输入特征维度
        hidden_dim:  隐藏层维度 (论文中 = 16)
        output_dim:  分类类别数
        dropout:     Dropout 率 (论文中 = 0.5)
    """

    graph_type = "edge_index"   # runner 据此选择 graph 输入

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.layer1 = GraphSAGELayer(input_dim, hidden_dim)
        self.layer2 = GraphSAGELayer(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.cached_edge_index = None
        self.cached_adj = None

    def forward(self, features, edge_index):
        adj_lists = self._edge_index_to_adj(edge_index)

        # Layer 1: aggregate → linear → relu → L2-norm → dropout
        h = self.layer1(features, adj_lists)
        h = F.relu(h)
        h = F.normalize(h, p=2, dim=1)
        h = self.dropout(h)

        # Layer 2: aggregate → linear → log_softmax
        logits = self.layer2(h, adj_lists)
        return F.log_softmax(logits, dim=1)

    @torch.no_grad()
    def get_embeddings(self, features, edge_index):
        """提取第 1 层 L2 归一化后的隐藏表示（用于 t-SNE / UMAP）。"""
        self.eval()
        adj_lists = self._edge_index_to_adj(edge_index)
        h = self.layer1(features, adj_lists)
        h = F.relu(h)
        return F.normalize(h, p=2, dim=1)

    def _edge_index_to_adj(self, edge_index):
        """将 [2, E] edge_index 转为邻接表 dict，校验缓存避免跨数据集复用。"""
        if (self.cached_adj is not None
                and self.cached_edge_index is not None
                and torch.equal(edge_index, self.cached_edge_index)):
            return self.cached_adj

        adj = {}
        src, dst = edge_index[0], edge_index[1]
        for s, d in zip(src, dst):
            s, d = int(s), int(d)
            if s not in adj:
                adj[s] = []
            adj[s].append(d)

        self.cached_edge_index = edge_index.clone()
        self.cached_adj = adj
        return adj
