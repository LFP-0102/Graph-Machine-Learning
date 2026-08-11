"""
GraphSAGE Neighbor Sampler

对应论文: Sample and Aggregate
功能: 给定节点，随机采样邻居
"""
import random

class NeighborSampler:
    def __init__(self, adj_lists, num_samples):
        self.adj_lists = adj_lists
        self.num_samples = num_samples

    def sample(self, nodes):
        """采样一批节点的邻居.

        nodes: [0, 1, 2, ...]
        return: {0: [633, 1862], 1: [...], ...}
        """
        sampled_neighbors = {}
        for node in nodes:
            neighbors = list(self.adj_lists[node])
            if len(neighbors) == 0:
                sampled_neighbors[node] = []
            elif len(neighbors) <= self.num_samples:
                sampled_neighbors[node] = neighbors
            else:
                sampled_neighbors[node] = random.sample(neighbors, self.num_samples)
        return sampled_neighbors
