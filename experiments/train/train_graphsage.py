"""使用 PyTorch 复现 GraphSAGE 的监督式 PPI 实验。

输入严格遵循 williamleif/GraphSAGE：<prefix>-G.json、<prefix>-id_map.json、
<prefix>-class_map.json 与 <prefix>-feats.npy。
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.graphsage_json import load_graphsage_json
from models.graphsage_official import OfficialGraphSAGE
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-prefix", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-degree", type=int, default=128)
    parser.add_argument("--samples-1", type=int, default=25)
    parser.add_argument("--samples-2", type=int, default=10)
    parser.add_argument("--dim-1", type=int, default=128)
    parser.add_argument("--dim-2", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sigmoid", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, features, labels, nodes, adjacency, sigmoid, batch_size, device):
    model.eval()
    predictions, targets = [], []
    for start in range(0, nodes.numel(), batch_size):
        batch = nodes[start:start + batch_size].to(device)
        predictions.append(model(features, adjacency, batch).cpu())
        targets.append(labels[batch.cpu()])
    scores = torch.cat(predictions)
    truth = torch.cat(targets)
    if sigmoid:
        predicted = (scores.sigmoid() > 0.5).numpy()
        truth = truth.numpy()
    else:
        predicted = scores.argmax(dim=1).numpy()
        truth = truth.argmax(dim=1).numpy()
    return (
        f1_score(truth, predicted, average="micro", zero_division=0),
        f1_score(truth, predicted, average="macro", zero_division=0),
    )


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    data = load_graphsage_json(args.train_prefix, max_degree=args.max_degree, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = data.features.to(device)
    labels = data.labels
    train_adjacency = data.train_adj.to(device)
    test_adjacency = data.test_adj.to(device)
    model = OfficialGraphSAGE(
        data.num_features,
        data.num_classes,
        dims=(args.dim_1, args.dim_2),
        samples=(args.samples_1, args.samples_2),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        train_nodes = data.train_nodes[torch.randperm(data.train_nodes.numel())]
        for start in range(0, train_nodes.numel(), args.batch_size):
            batch = train_nodes[start:start + args.batch_size].to(device)
            logits = model(features, train_adjacency, batch)
            batch_labels = labels[batch.cpu()].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels) if args.sigmoid else F.cross_entropy(logits, batch_labels.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
        val_micro, val_macro = evaluate(model, features, labels, data.val_nodes, test_adjacency, args.sigmoid, args.batch_size, device)
        print(f"轮次 {epoch:03d} | 损失={sum(losses) / len(losses):.4f} | 验证 micro-F1={val_micro:.4f} | 验证 macro-F1={val_macro:.4f}")

    test_micro, test_macro = evaluate(model, features, labels, data.test_nodes, test_adjacency, args.sigmoid, args.batch_size, device)
    elapsed = time.time() - started
    print(f"测试 micro-F1={test_micro:.4f} | macro-F1={test_macro:.4f} | 用时={elapsed:.1f}s")
