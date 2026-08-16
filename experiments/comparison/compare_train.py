"""在 Cora 上快速比较 GCN、GraphSAGE 与 GAT 的单次训练结果。"""

import torch

from datasets.base import load_dataset
from models.gat import GAT
from models.gcn import GCN
from models.graphsage import GraphSAGE
from utils.graph_utils import add_self_loops, normalize_adj
from utils.seed import set_seed
from utils.training import evaluate, train_epoch


def run_experiment(model, graph, data, epochs=200, seed=42):
    """训练模型并返回测试集准确率。"""
    set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        train_epoch(model, data.features, graph, data.labels, data.train_mask, optimizer)
    return evaluate(model, data.features, graph, data.labels, data.test_mask)


if __name__ == "__main__":
    data = load_dataset("cora")
    print(f"数据集：Cora（{data.num_features} 维特征，{data.num_classes} 个类别，{data.features.shape[0]} 个节点）\n")
    gcn = GCN(data.num_features, 16, data.num_classes, dropout=0.5)
    gcn_acc = run_experiment(gcn, torch.FloatTensor(normalize_adj(data.adj.numpy())), data)
    sage = GraphSAGE(data.num_features, 16, data.num_classes, dropout=0.5)
    sage_acc = run_experiment(sage, data.edge_index, data)
    gat = GAT(data.num_features, 8, data.num_classes, n_heads=8, dropout=0.6)
    gat_acc = run_experiment(gat, torch.FloatTensor(add_self_loops(data.adj.numpy())), data)
    print(f"GCN 测试准确率：{gcn_acc:.4f}")
    print(f"GraphSAGE 测试准确率：{sage_acc:.4f}")
    print(f"GAT 测试准确率：{gat_acc:.4f}")
