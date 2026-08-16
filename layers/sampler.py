"""GraphSAGE 邻居采样工具。"""

import random


class NeighborSampler:
    """为给定节点随机采样固定数量的邻居。"""

    def __init__(self, adj_lists, num_samples):
        self.adj_lists = adj_lists
        self.num_samples = num_samples

    def sample(self, nodes):
        """返回 ``{节点: 采样邻居列表}`` 形式的采样结果。"""
        sampled_neighbors = {}
        for node in nodes:
            neighbors = list(self.adj_lists[node])
            if len(neighbors) <= self.num_samples:
                sampled_neighbors[node] = neighbors
            else:
                sampled_neighbors[node] = random.sample(neighbors, self.num_samples)
        return sampled_neighbors
