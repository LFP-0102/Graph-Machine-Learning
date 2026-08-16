"""使用 PyTorch 复现 tkipf/gcn 的引文网络训练脚本。"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.base import load_dataset
from models.gcn import GCN
from utils.graph_utils import normalized_sparse_edge_index
from utils.seed import set_seed


@torch.no_grad()
def evaluate(model, features, adjacency, labels, mask):
    model.eval()
    logits = model(features, adjacency)
    loss = F.nll_loss(logits[mask], labels[mask])
    accuracy = (logits[mask].argmax(1) == labels[mask]).float().mean()
    return loss.item(), accuracy.item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer", "pubmed"), default="cora")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    data = load_dataset(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels = data.features.to(device), data.labels.to(device)
    adjacency = normalized_sparse_edge_index(data.edge_index, features.size(0), device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    model = GCN(data.num_features, 16, data.num_classes, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    validation_losses = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adjacency)
        cross_entropy = F.nll_loss(logits[train_mask], labels[train_mask])
        # 原始代码只对第一层图卷积的权重施加 L2 正则化。
        loss = cross_entropy + 5e-4 * 0.5 * model.gc1.weight.square().sum()
        loss.backward()
        optimizer.step()

        val_loss, val_accuracy = evaluate(model, features, adjacency, labels, val_mask)
        validation_losses.append(val_loss)
        train_accuracy = (logits[train_mask].argmax(1) == labels[train_mask]).float().mean().item()
        print(f"轮次 {epoch + 1:04d} | 训练损失={loss.detach().item():.5f} | 训练准确率={train_accuracy:.5f} | 验证损失={val_loss:.5f} | 验证准确率={val_accuracy:.5f}")
        if epoch > 10 and val_loss > sum(validation_losses[-11:-1]) / 10:
            print("触发早停。")
            break

    test_loss, test_accuracy = evaluate(model, features, adjacency, labels, test_mask)
    print(f"测试集结果：损失={test_loss:.5f} | 准确率={test_accuracy:.5f}")
