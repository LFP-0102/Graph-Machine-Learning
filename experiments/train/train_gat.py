"""使用 PyTorch 复现 PetarV-/GAT 的引文网络训练脚本。"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.base import load_dataset
from models.gat import GAT
from utils.seed import set_seed


@torch.no_grad()
def evaluate(model, features, edge_index, labels, mask):
    model.eval()
    logits = model(features, edge_index)
    loss = F.nll_loss(logits[mask], labels[mask])
    accuracy = (logits[mask].argmax(1) == labels[mask]).float().mean()
    return loss.item(), accuracy.item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer", "pubmed"), default="cora")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=100000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    data = load_dataset(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels = data.features.to(device), data.labels.to(device)
    edge_index = data.edge_index.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    model = GAT(data.num_features, 8, data.num_classes, n_heads=8, dropout=0.6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    best_state = None
    min_val_loss, max_val_accuracy, stale_epochs = float("inf"), 0.0, 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_index)
        cross_entropy = F.nll_loss(logits[train_mask], labels[train_mask])
        # 官方 BaseGAttN 代码对全部可训练参数施加 L2 正则化。
        loss = cross_entropy + 5e-4 * 0.5 * sum(parameter.square().sum() for parameter in model.parameters())
        loss.backward()
        optimizer.step()

        val_loss, val_accuracy = evaluate(model, features, edge_index, labels, val_mask)
        train_accuracy = (logits[train_mask].argmax(1) == labels[train_mask]).float().mean().item()
        print(f"轮次 {epoch + 1:05d} | 训练损失={loss.detach().item():.5f} | 训练准确率={train_accuracy:.5f} | 验证损失={val_loss:.5f} | 验证准确率={val_accuracy:.5f}")
        if val_accuracy >= max_val_accuracy or val_loss <= min_val_loss:
            if val_accuracy >= max_val_accuracy and val_loss <= min_val_loss:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            max_val_accuracy = max(max_val_accuracy, val_accuracy)
            min_val_loss = min(min_val_loss, val_loss)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs == 100:
                print("触发早停。")
                break

    model.load_state_dict(best_state)
    test_loss, test_accuracy = evaluate(model, features, edge_index, labels, test_mask)
    print(f"测试集结果：损失={test_loss:.5f} | 准确率={test_accuracy:.5f}")
