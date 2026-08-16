"""与 williamleif/GraphSAGE 对齐的小批量监督式 Mean GraphSAGE。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.official_graphsage import OfficialMeanAggregator


class OfficialGraphSAGE(nn.Module):
    """采用 `25 -> 10` 两跳采样的 GraphSAGE 分类器。

    邻接表为固定宽度，并包含全零哨兵行，与官方 NodeMinibatchIterator 的构造
    完全一致。每次前向仅采样目标节点小批量所需的计算树。
    """

    def __init__(self, input_dim, output_dim, dims=(128, 128), samples=(25, 10), dropout=0.0):
        super().__init__()
        if len(dims) != 2 or len(samples) != 2:
            raise ValueError("OfficialGraphSAGE 必须使用恰好两层")
        self.samples = tuple(samples)
        self.dropout = dropout
        self.layer1 = OfficialMeanAggregator(input_dim, dims[0], F.relu, dropout)
        self.layer2 = OfficialMeanAggregator(2 * dims[0], dims[1], lambda value: value, dropout)
        self.classifier = nn.Linear(2 * dims[1], output_dim)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def _sample(adjacency, nodes, count):
        candidates = adjacency[nodes]
        # 官方 UniformNeighborSampler 会转置邻接表，并沿最大度维度对整个
        # 小批量执行一次随机打乱。
        positions = torch.randperm(candidates.shape[1], device=candidates.device)
        return candidates[:, positions[:count]].reshape(-1)

    def forward(self, features, adjacency, target_nodes):
        """返回一个目标节点小批量的 logits。"""
        batch_size = target_nodes.numel()
        samples = [target_nodes]
        support_sizes = [1]
        support_size = 1
        for count in reversed(self.samples):
            support_size *= count
            samples.append(self._sample(adjacency, samples[-1], count))
            support_sizes.append(support_size)

        hidden = [features.index_select(0, nodes) for nodes in samples]
        for layer_index, aggregator in enumerate((self.layer1, self.layer2)):
            next_hidden = []
            for hop in range(len(self.samples) - layer_index):
                node_count = batch_size * support_sizes[hop]
                # ``hidden[hop + 1]`` 来自当前跳的采样器：最外层使用
                # samples_1，内层使用 samples_2。
                count = self.samples[len(self.samples) - hop - 1]
                neighbors = hidden[hop + 1].reshape(node_count, count, -1)
                next_hidden.append(aggregator(hidden[hop], neighbors))
            hidden = next_hidden

        embedding = F.normalize(hidden[0], p=2, dim=1)
        return self.classifier(embedding)
