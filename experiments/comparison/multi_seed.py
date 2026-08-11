"""
Multi-Seed Experiment

在 Cora 上对 GCN / GraphSAGE / GAT 分别以 5 个随机种子重复训练，
统计 Mean Accuracy ± Std。

输出:
  outputs/results/multi_seed_raw.csv     — 原始记录 (seed, model, accuracy, params, time)
  outputs/results/multi_seed_summary.csv  — 按模型汇总 (mean, std, params, time)
"""
import time
import torch
import pandas as pd
import numpy as np

from datasets.base import load_dataset
from models.gcn import GCN
from models.graphsage import GraphSAGE
from models.gat import GAT
from utils.graph_utils import normalize_adj
from utils.paths import RESULTS_DIR, ensure_dirs
from utils.seed import set_seed
from utils.training import train_epoch, evaluate

# ── 配置 ──────────────────────────────────────────────────────
SEEDS = [10, 20, 30, 40, 50]
EPOCHS = 200

# GAT 论文用更小的 lr
LR_MAP = {"GCN": 0.01, "GraphSAGE": 0.01, "GAT": 0.005}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── 数据 ──────────────────────────────────────────────────────
data = load_dataset("cora")
x, y = data.features, data.labels
adj = torch.FloatTensor(normalize_adj(data.adj.numpy()))
edge_index = data.edge_index


# ── 模型工厂 ───────────────────────────────────────────────────
def build_models():
    return {
        "GCN": GCN(input_dim=data.num_features, hidden_dim=16,
                   output_dim=data.num_classes, dropout=0.5),
        "GraphSAGE": GraphSAGE(input_dim=data.num_features, hidden_dim=16,
                               output_dim=data.num_classes, dropout=0.5),
        "GAT": GAT(input_dim=data.num_features, hidden_dim=8,
                   output_dim=data.num_classes, n_heads=8, dropout=0.6),
    }


# ── 单次训练 ───────────────────────────────────────────────────
def run_one(model_name, model):
    lr = LR_MAP.get(model_name, 0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    graph = adj if model_name == "GCN" else edge_index

    best_val = 0.0
    best_state = None

    t_start = time.time()
    for epoch in range(EPOCHS):
        train_epoch(model, x, graph, y, data.train_mask, optimizer)
        val_acc = evaluate(model, x, graph, y, data.val_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t_start
    model.load_state_dict(best_state)
    test_acc = evaluate(model, x, graph, y, data.test_mask)
    n_params = count_parameters(model)
    return test_acc, train_time, n_params


# ── 主实验 ─────────────────────────────────────────────────────
results = []

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"  Seed: {seed}")
    print(f"{'='*50}")

    set_seed(seed)
    models = build_models()

    for name, model in models.items():
        print(f"  Training {name} ... ", end="", flush=True)
        acc, elapsed, params = run_one(name, model)
        print(f"{acc:.4f}  ({elapsed:.1f}s, {params} params)")
        results.append([seed, name, acc, elapsed, params])

# ── 保存原始数据 ────────────────────────────────────────────────
ensure_dirs(RESULTS_DIR)
df = pd.DataFrame(results, columns=["seed", "model", "accuracy", "train_time", "parameters"])
df.to_csv(RESULTS_DIR / "multi_seed_raw.csv", index=False)
print(f"\nRaw data saved: {RESULTS_DIR / 'multi_seed_raw.csv'}")

# ── 汇总 ──────────────────────────────────────────────────────
summary = df.groupby("model").agg(
    mean_acc=("accuracy", "mean"),
    std_acc=("accuracy", "std"),
    params=("parameters", "first"),
    time=("train_time", "mean"),
).round(4)
summary["params"] = summary["params"].astype(int)

print(f"\n{'='*55}")
print("  Multi-Seed Summary")
print(f"{'='*55}")
print(f"{'Model':12} {'Accuracy':>14} {'Params':>8} {'Time':>8}")
print("-" * 55)
for model_name, row in summary.iterrows():
    print(f"{model_name:12} {row['mean_acc']:.4f} ± {row['std_acc']:.4f}  "
          f"{row['params']:>5}  {row['time']:>6.1f}s")
summary.to_csv(RESULTS_DIR / "multi_seed_summary.csv")
print(f"\nSummary saved: {RESULTS_DIR / 'multi_seed_summary.csv'}")
