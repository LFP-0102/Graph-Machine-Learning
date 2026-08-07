"""
GCN vs GAT 对比实验。

在 Cora 数据集上比较两种模型的 transductive 节点分类性能。
"""
import torch

from datasets.base import load_dataset
from models.gcn import GCN
from models.gat import GAT
from utils.graph_utils import normalize_adj, add_self_loops
from utils.seed import set_seed
from utils.training import train_epoch, evaluate


def run_experiment(model, adj, data, epochs=200, seed=42):
    """训练并返回 test accuracy。"""
    set_seed(seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4
    )
    for epoch in range(epochs):
        train_epoch(model, data.features, adj, data.labels,
                     data.train_mask, optimizer)

    return evaluate(model, data.features, adj,
                     data.labels, data.test_mask)


if __name__ == "__main__":
    data = load_dataset("cora")
    print(f"Dataset: Cora ({data.num_features} features, "
          f"{data.num_classes} classes, {data.features.shape[0]} nodes)\n")

    # ── GCN ────────────────────────────────────────────────
    adj_gcn = torch.FloatTensor(normalize_adj(data.adj.numpy()))
    gcn = GCN(data.num_features, hidden_dim=16, output_dim=data.num_classes)
    gcn_acc = run_experiment(gcn, adj_gcn, data, seed=42)
    print(f"GCN  Test Acc: {gcn_acc:.4f}")

    # ── GAT ────────────────────────────────────────────────
    adj_gat = torch.FloatTensor(add_self_loops(data.adj.numpy()))
    gat = GAT(data.num_features, hidden_dim=8, output_dim=data.num_classes,
              n_heads=8, dropout=0.6)
    gat_acc = run_experiment(gat, adj_gat, data, seed=42)
    print(f"GAT  Test Acc: {gat_acc:.4f}")

    print(f"\n{'='*40}")
    print(f"GCN: {gcn_acc:.4f}  |  GAT: {gat_acc:.4f}")
