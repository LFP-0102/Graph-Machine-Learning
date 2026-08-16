"""带逐层邻居采样的两层 Mean Aggregator GraphSAGE。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graphsage_layer import GraphSAGELayer


class GraphSAGE(nn.Module):
    """使用 Mean Aggregator 实现 GraphSAGE 算法 1。

    训练时每层、每次前向都会重新采样邻居；评估默认聚合全部邻居，以获得确定性
    指标。将 ``sample_in_eval`` 设为真可复现采样推理。``task`` 决定输出多分类
    对数概率（Planetoid 消融）或多标签 logits（原论文 PPI 协议）。
    """

    graph_type = "edge_index"

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.0,
        sample_sizes=(25, 10),
        sample_in_eval=False,
        task="multiclass",
    ):
        super().__init__()
        if len(sample_sizes) != 2 or any(size <= 0 for size in sample_sizes):
            raise ValueError("sample_sizes 必须包含两个正整数采样数")
        if task not in {"multiclass", "multilabel"}:
            raise ValueError("task 只能是 'multiclass' 或 'multilabel'")
        self.layer1 = GraphSAGELayer(input_dim, hidden_dim)
        self.layer2 = GraphSAGELayer(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sample_sizes = tuple(sample_sizes)
        self.sample_in_eval = sample_in_eval
        self.task = task
        self.cached_edge_index = None
        self.cached_neighbors = None

    def forward(self, features, edge_index):
        neighbors = self._edge_index_to_neighbors(edge_index, features.size(0))
        sampled_1 = self._sample_neighbors(neighbors, self.sample_sizes[0]) \
            if self.training or self.sample_in_eval else None
        hidden = self.layer1(features, edge_index, sampled_1)
        hidden = F.relu(hidden)
        hidden = F.normalize(hidden, p=2, dim=1)
        hidden = self.dropout(hidden)

        sampled_2 = self._sample_neighbors(neighbors, self.sample_sizes[1]) \
            if self.training or self.sample_in_eval else None
        logits = self.layer2(hidden, edge_index, sampled_2)
        if self.task == "multilabel":
            return logits
        return F.log_softmax(F.normalize(logits, p=2, dim=1), dim=1)

    @torch.no_grad()
    def get_embeddings(self, features, edge_index):
        self.eval()
        hidden = self.layer1(features, edge_index)
        return F.normalize(F.relu(hidden), p=2, dim=1)

    def _edge_index_to_neighbors(self, edge_index, num_nodes):
        if (
            self.cached_neighbors is not None
            and self.cached_edge_index is not None
            and torch.equal(edge_index, self.cached_edge_index)
        ):
            return self.cached_neighbors

        neighbors = [[] for _ in range(num_nodes)]
        for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            neighbors[source].append(target)
        self.cached_edge_index = edge_index.clone()
        self.cached_neighbors = neighbors
        return neighbors

    @staticmethod
    def _sample_neighbors(neighbors, fanout):
        sampled = []
        valid = []
        for node_neighbors in neighbors:
            degree = len(node_neighbors)
            if degree == 0:
                sampled.append([0] * fanout)
                valid.append([False] * fanout)
            elif degree <= fanout:
                sampled.append(node_neighbors + [node_neighbors[0]] * (fanout - degree))
                valid.append([True] * degree + [False] * (fanout - degree))
            else:
                selection = torch.randperm(degree)[:fanout].tolist()
                sampled.append([node_neighbors[index] for index in selection])
                valid.append([True] * fanout)
        return torch.tensor(sampled, dtype=torch.long), torch.tensor(valid, dtype=torch.bool)
