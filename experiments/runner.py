"""
通用实验运行器 —— 训练 + 记录 + 可视化一站式。

每个实验自动创建独立输出目录，避免不同模型的图表混在一起。

目录结构：
    outputs/
    ├── visualizations/
    │   ├── gcn/          # GCN 的所有图表
    │   └── gat/          # GAT 的所有图表
    └── runs/
        ├── gcn/
        │   └── history.csv    # 每轮指标
        └── gat/
            └── history.csv

用法：
    from experiments.runner import run_experiment
    from models.gcn import GCN

    run_experiment(
        model_class=GCN,
        model_name="GCN",
        dataset_name="cora",
        ...
    )
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn

# ── 项目模块 ────────────────────────────────────────────────
from datasets.base import load_dataset
from utils.graph_utils import add_self_loops, normalize_adj
from utils.paths import CHECKPOINT_DIR, OUTPUT_DIR, RESULTS_DIR, RUN_DIR, VIS_DIR, ensure_dirs
from utils.seed import set_seed
from utils.training import evaluate, predict, train_epoch

# ── 可视化工具 ──────────────────────────────────────────────
from vis_tool import (
    plot_confusion_matrix,
    plot_embeddings,
    plot_graph,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_train_curves,
)
from vis_tool.config import CATEGORICAL_PALETTE, save_or_show
from vis_tool.statistics.result_table import save_result_table

# ────────────────────────────────────────────────────────────
# Cora 类别名称（全局常量）
# ────────────────────────────────────────────────────────────
CORA_CLASSES = [
    "Case_Based", "Genetic_Algorithms", "Neural_Networks",
    "Probabilistic_Methods", "Reinforcement_Learning",
    "Rule_Learning", "Theory",
]


# ────────────────────────────────────────────────────────────
# 核心运行函数
# ────────────────────────────────────────────────────────────
def run_experiment(
    model_class: Type[nn.Module],
    model_name: str,                # "GCN" | "GAT"
    dataset_name: str = "cora",
    *,
    # 模型参数
    model_kwargs: Optional[Dict[str, Any]] = None,
    # 预处理选择
    use_normalized_adj: bool = True,   # GCN: True,  GAT: False
    # 训练参数
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 100,
    seed: int = 42,
    # 可视化开关
    skip_tsne: bool = False,
    skip_graph: bool = False,
    # 图拓扑参数
    graph_sub_nodes: int = 500,
) -> Dict[str, Any]:
    """训练模型并生成全套可视化图表。

    返回包含 best_epoch, test_acc, history 等信息的字典，
    方便后续做多模型对比。
    """
    # ── 0. 创建输出目录 ──────────────────────────────────────
    vis_out = VIS_DIR / model_name.lower()
    run_out = RUN_DIR / model_name.lower()
    ensure_dirs(vis_out, run_out)

    set_seed(seed)

    # ── 1. 加载数据 ──────────────────────────────────────────
    data = load_dataset(dataset_name)
    print(f"\n{'='*55}")
    print(f"  {model_name} on {dataset_name.capitalize()}")
    print(f"  {data.num_features} features | {data.num_classes} classes "
          f"| {data.features.shape[0]} nodes")
    print(f"{'='*55}\n")

    # ── 2. 预处理 & 模型 ─────────────────────────────────────
    adj_np = normalize_adj(data.adj.numpy()) if use_normalized_adj \
             else add_self_loops(data.adj.numpy())
    adj = torch.FloatTensor(adj_np)

    kwargs = model_kwargs or {}
    model = model_class(
        input_dim=data.num_features,
        output_dim=data.num_classes,
        **kwargs,
    )
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    # ── 3. 训练 + 记录历史 ───────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_acc": [], "test_acc": [],
    }
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    counter = 0

    print(f"{'Epoch':>5}  {'Loss':>8}  {'Train':>7}  "
          f"{'Val':>7}  {'Test':>7}  {'Best':>7}")
    print("-" * 55)

    for epoch in range(epochs):
        loss = train_epoch(model, data.features, adj,
                           data.labels, data.train_mask, optimizer)

        train_acc = evaluate(model, data.features, adj,
                             data.labels, data.train_mask)
        val_acc   = evaluate(model, data.features, adj,
                             data.labels, data.val_mask)
        test_acc  = evaluate(model, data.features, adj,
                             data.labels, data.test_mask)

        history["train_loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        if epoch % 10 == 0 or epoch == epochs - 1:
            marker = " *" if epoch == best_epoch else ""
            print(f"{epoch:5d}  {loss:8.4f}  {train_acc:7.4f}  "
                  f"{val_acc:7.4f}  {test_acc:7.4f}  "
                  f"{best_val_acc:7.4f}{marker}")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── 恢复验证集最佳模型 ──────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    best_test = evaluate(model, data.features, adj,
                         data.labels, data.test_mask)
    print(f"\nBest Epoch: {best_epoch}  |  "
          f"Val Acc: {best_val_acc:.4f}  |  Test Acc: {best_test:.4f}")

    # ── 3.5 保存最佳模型 checkpoint ───────────────────────────
    ensure_dirs(CHECKPOINT_DIR)
    ckpt_name = f"best_{model_name.lower()}.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    torch.save(
        {
            "model_state_dict": best_state,
            "val_acc": best_val_acc,
            "model_kwargs": kwargs,
        },
        ckpt_path,
    )
    print(f"Checkpoint saved: {ckpt_path}")

    # ── 4. 保存训练历史 CSV ──────────────────────────────────
    import pandas as pd
    df_history = pd.DataFrame(history)
    df_history.index.name = "epoch"
    df_history.to_csv(run_out / "history.csv")
    print(f"训练历史已保存: {run_out / 'history.csv'}")

    # ── 5. 预测 & 嵌入 ───────────────────────────────────────
    pred_labels, pred_probs = predict(model, data.features, adj)

    # 类别名称（根据数据集自动选择）
    class_names = CORA_CLASSES if dataset_name == "cora" else None

    # ── 6. 训练曲线（论文 Figure 3b）─────────────────────────
    print("\n>>> 训练曲线 ...")
    plot_train_curves(
        train_values=np.array([history["train_loss"],
                                history["train_acc"]]),
        val_values=np.array([history["val_acc"],
                              history["test_acc"]]),
        metric_names=["Loss", "Accuracy"],
        title=f"{model_name} on {dataset_name.capitalize()} — Training Curves",
        save_path=str(vis_out / "training_curves.png"),
        mark_best=True,
    )

    # ── 7. 隐藏层嵌入可视化（论文 Figure 3a）─────────────────
    if not skip_tsne and hasattr(model, "get_embeddings"):
        hidden_emb = model.get_embeddings(data.features, adj)
        print(f">>> t-SNE 嵌入可视化 (dim={hidden_emb.shape[1]}) ...")

        plot_embeddings(
            embeddings=hidden_emb,
            labels=data.labels,
            method="tsne",
            class_names=class_names,
            title=f"{model_name} Hidden Embeddings — t-SNE ({dataset_name.capitalize()})",
            save_path=str(vis_out / "embeddings_tsne.png"),
            perplexity=30,
            random_state=42,
        )

        # UMAP（可选）
        try:
            print(">>> UMAP 嵌入可视化 ...")
            plot_embeddings(
                embeddings=hidden_emb,
                labels=data.labels,
                method="umap",
                class_names=class_names,
                title=f"{model_name} Hidden Embeddings — UMAP ({dataset_name.capitalize()})",
                save_path=str(vis_out / "embeddings_umap.png"),
                random_state=42,
            )
        except ImportError:
            print("    (跳过 UMAP: pip install umap-learn)")

    # ── 8. 混淆矩阵 ──────────────────────────────────────────
    print(">>> 混淆矩阵 ...")
    test_mask_np = data.test_mask.numpy()
    plot_confusion_matrix(
        y_true=data.labels[test_mask_np],
        y_pred=pred_labels[test_mask_np],
        class_names=class_names,
        normalize=True,
        title=f"{model_name} on {dataset_name.capitalize()} — Confusion Matrix",
        save_path=str(vis_out / "confusion_matrix.png"),
    )

    # ── 9. ROC / PR 曲线 ─────────────────────────────────────
    print(">>> ROC 曲线 ...")
    plot_roc_curves(
        y_true=data.labels[test_mask_np],
        y_score=pred_probs[test_mask_np],
        class_names=class_names,
        title=f"{model_name} on {dataset_name.capitalize()} — ROC",
        save_path=str(vis_out / "roc_curves.png"),
    )

    print(">>> PR 曲线 ...")
    plot_precision_recall_curves(
        y_true=data.labels[test_mask_np],
        y_score=pred_probs[test_mask_np],
        class_names=class_names,
        title=f"{model_name} on {dataset_name.capitalize()} — Precision-Recall",
        save_path=str(vis_out / "pr_curves.png"),
    )

    # ── 10. 图拓扑（只画一次，共享的，不区分模型）────────────
    if not skip_graph:
        print(">>> 图拓扑可视化 ...")
        adj_np_raw = data.adj.numpy()
        sub_adj = adj_np_raw[:graph_sub_nodes, :graph_sub_nodes]
        sub_G = nx.from_numpy_array(sub_adj)
        largest_cc = max(nx.connected_components(sub_G), key=len)
        sub_G = sub_G.subgraph(largest_cc).copy()
        mapping = {old: new for new, old in
                   enumerate(sorted(sub_G.nodes()))}
        sub_G = nx.relabel_nodes(sub_G, mapping)

        edge_index = np.array(list(sub_G.edges())).T
        sub_nodes = sorted(largest_cc)
        sub_labels = data.labels[sub_nodes].numpy()
        sub_names = ([class_names[l] for l in sub_labels]
                     if class_names else None)

        plot_graph(
            edge_index=edge_index,
            node_colors=sub_labels,
            node_labels=sub_names,
            layout="spring",
            title=f"{dataset_name.capitalize()} Graph — Subgraph "
                  f"({len(sub_nodes)} nodes, largest CC)",
            save_path=str(vis_out / "graph_topology.png"),
            node_size=40,
            label_max_nodes=500,
            k=0.5,
        )

    # ── 11. 每类准确率柱状图 ─────────────────────────────────
    print(">>> 每类准确率 ...")
    per_class_acc = {}
    for lbl in range(data.num_classes):
        mask = (data.labels[test_mask_np] == lbl)
        if mask.sum() == 0:
            continue
        correct = (pred_labels[test_mask_np][mask] == lbl).sum().item()
        name = class_names[lbl] if class_names else f"Class {lbl}"
        per_class_acc[name] = correct / mask.sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        per_class_acc.keys(), per_class_acc.values(),
        color=CATEGORICAL_PALETTE[:data.num_classes],
        edgecolor="white", linewidth=1.2,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} on {dataset_name.capitalize()} "
                 f"— Per-Class Accuracy")
    for bar, acc in zip(bars, per_class_acc.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    save_or_show(fig, str(vis_out / "per_class_accuracy.png"))

    # ── 12. 保存结果表 ───────────────────────────────────────
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "test_accuracy": float(best_test),
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": best_epoch,
        "seed": seed,
    }
    save_result_table([result], str(run_out / "results.csv"))

    # ── 12.5 追加到集中 benchmark ─────────────────────────────
    ensure_dirs(RESULTS_DIR)
    benchmark_path = RESULTS_DIR / "benchmark.csv"
    import csv
    write_header = not benchmark_path.exists()
    with open(benchmark_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "dataset", "best_epoch", "val_acc", "test_acc", "seed"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model": model_name,
            "dataset": dataset_name,
            "best_epoch": best_epoch,
            "val_acc": f"{best_val_acc:.4f}",
            "test_acc": f"{best_test:.4f}",
            "seed": seed,
        })
    print(f"Benchmark appended: {benchmark_path}")

    # ── 完成 ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  [{model_name}] 输出目录: {vis_out}")
    print("=" * 55)
    for f in sorted(vis_out.glob("*.png")):
        print(f"    [FIG] {f.name}")
    print(f"  结果表: {run_out / 'results.csv'}")
    print(f"  历史:   {run_out / 'history.csv'}")

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(best_test),
        "history": history,
        "pred_labels": pred_labels,
        "pred_probs": pred_probs,
        "output_dir": vis_out,
    }
