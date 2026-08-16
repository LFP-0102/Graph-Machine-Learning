"""训练一次引文网络模型并生成组会汇报图表。

该运行器与多种子基准脚本分离：它依据验证准确率选择最佳模型，最后只评估一次
测试集，并保存图表和训练历史，但不会写入模型 checkpoint。
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT_DIR))

from datasets.base import load_dataset
from utils.graph_utils import normalized_sparse_edge_index
from utils.seed import set_seed
from vis_tool import (
    plot_confusion_matrix,
    plot_embeddings,
    plot_graph,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_train_curves,
)
from vis_tool.analysis.attention import (
    _dense_attn_to_edge_scores,
    plot_attention_summary,
    plot_attention_weights,
    plot_node_attention_bars,
)
from vis_tool.config import CATEGORICAL_PALETTE, save_or_show


def _accuracy(logits, labels, mask):
    return (logits[mask].argmax(1) == labels[mask]).float().mean().item()


@torch.no_grad()
def _evaluate(model, features, graph, labels, mask):
    model.eval()
    logits = model(features, graph)
    return F.nll_loss(logits[mask], labels[mask]).item(), _accuracy(logits, labels, mask)


def _citation_graph(data, model_name, device):
    if model_name.lower() == "gcn":
        return normalized_sparse_edge_index(data.edge_index, data.features.size(0), device)
    return data.edge_index


def _regularization(model, model_name):
    if model_name.lower() == "gcn":
        return 5e-4 * 0.5 * model.gc1.weight.square().sum()
    if model_name.lower() == "gat":
        return 5e-4 * 0.5 * sum(parameter.square().sum() for parameter in model.parameters())
    return 5e-4 * sum(parameter.square().sum() for parameter in model.parameters())


def _plot_topology(edge_index, labels, destination):
    edges = edge_index.detach().cpu().numpy()
    candidate = edges[:, (edges[0] < 500) & (edges[1] < 500)]
    graph = nx.Graph()
    graph.add_edges_from(zip(candidate[0].tolist(), candidate[1].tolist()))
    component = max(nx.connected_components(graph), key=len)
    component = sorted(component)
    remap = {node: index for index, node in enumerate(component)}
    selected = [(remap[int(src)], remap[int(dst)]) for src, dst in zip(candidate[0], candidate[1]) if int(src) in remap and int(dst) in remap]
    if selected:
        sub_edges = np.asarray(selected, dtype=np.int64).T
        plot_graph(
            sub_edges,
            node_colors=labels[component].detach().cpu().numpy(),
            node_size=35,
            layout="spring",
            k=0.5,
            title=f"Cora Citation Subgraph ({len(component)} nodes)",
            save_path=str(destination),
        )


def _plot_attention(model, features, edge_index, destination):
    average, _ = model.get_attention_weights(features, edge_index)
    average = average.detach().cpu().numpy()
    edges = edge_index.detach().cpu().numpy()
    degrees = np.bincount(edges[0], minlength=average.shape[0])
    hub = int(np.argmax(degrees))
    plot_attention_summary(
        average, edges, focal_nodes=None, num_nodes=6, top_k=8,
        title="GAT Multi-Node Attention Summary",
        save_path=str(destination / "attention_summary.png"),
    )
    plot_node_attention_bars(
        average, edges, focal_node=hub, top_k=10,
        title=f"GAT Hub Node {hub} Neighbor Attention",
        save_path=str(destination / "attention_hub_node.png"),
    )
    neighbors = np.unique(np.concatenate(([hub], edges[1][edges[0] == hub])))
    local = edges[:, np.isin(edges[0], neighbors) & np.isin(edges[1], neighbors)]
    if local.shape[1]:
        scores = _dense_attn_to_edge_scores(average, local)
        plot_attention_weights(
            local, scores, node_size=140, edge_scale=5.0,
            title=f"GAT Local Attention around Node {hub}",
            save_path=str(destination / "attention_graph.png"),
        )


def run_experiment(model_class, model_name, dataset_name="cora", *, model_kwargs=None, lr=0.01,
                   epochs=200, patience=100, seed=42):
    """为一次引文网络训练生成组会汇报图表。"""
    if dataset_name not in {"cora", "citeseer", "pubmed"}:
        raise ValueError("汇报运行器仅支持引文网络数据集")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(dataset_name)
    for name in ("features", "labels", "edge_index", "train_mask", "val_mask", "test_mask"):
        data[name] = data[name].to(device)
    graph = _citation_graph(data, model_name, device)
    model = model_class(data.num_features, output_dim=data.num_classes, **(model_kwargs or {})).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    output = ROOT_DIR / "outputs"
    vis_dir = output / "visualizations" / f"{model_name.lower()}_{dataset_name}"
    run_dir = output / "runs" / f"{model_name.lower()}_{dataset_name}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_accuracy, best_epoch, stale_epochs, best_state = -1.0, 0, 0, None
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.features, graph)
        loss = F.nll_loss(logits[data.train_mask], data.labels[data.train_mask]) + _regularization(model, model_name)
        loss.backward()
        optimizer.step()
        val_loss, val_accuracy = _evaluate(model, data.features, graph, data.labels, data.val_mask)
        history["train_loss"].append(loss.detach().item())
        history["train_accuracy"].append(_accuracy(logits.detach(), data.labels, data.train_mask))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch, stale_epochs = val_accuracy, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    test_loss, test_accuracy = _evaluate(model, data.features, graph, data.labels, data.test_mask)
    elapsed = time.time() - started
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index_label="epoch")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("model", "dataset", "seed", "best_epoch", "val_accuracy", "test_accuracy", "test_loss", "train_time"))
        writer.writeheader()
        writer.writerow({"model": model_name, "dataset": dataset_name, "seed": seed, "best_epoch": best_epoch, "val_accuracy": best_accuracy, "test_accuracy": test_accuracy, "test_loss": test_loss, "train_time": elapsed})

    model.eval()
    with torch.no_grad():
        logits = model(data.features, graph)
        predictions = logits.argmax(1)
        probabilities = logits.exp()
        embeddings = model.get_embeddings(data.features, graph)
    test_mask = data.test_mask.detach().cpu().numpy()
    labels = data.labels.detach().cpu().numpy()
    predicted = predictions.detach().cpu().numpy()
    scores = probabilities.detach().cpu().numpy()
    plot_train_curves(
        np.asarray([history["train_loss"], history["train_accuracy"]]),
        np.asarray([history["val_loss"], history["val_accuracy"]]),
        metric_names=["Loss", "Accuracy"], title=f"{model_name} Training Curves on {dataset_name}",
        save_path=str(vis_dir / "training_curves.png"),
    )
    plot_embeddings(embeddings, labels, method="tsne", title=f"{model_name} Node Embeddings on {dataset_name}", save_path=str(vis_dir / "embeddings_tsne.png"), random_state=seed)
    plot_confusion_matrix(labels[test_mask], predicted[test_mask], title=f"{model_name} Confusion Matrix on {dataset_name}", save_path=str(vis_dir / "confusion_matrix.png"))
    plot_roc_curves(labels[test_mask], scores[test_mask], title=f"{model_name} ROC Curves on {dataset_name}", save_path=str(vis_dir / "roc_curves.png"))
    plot_precision_recall_curves(labels[test_mask], scores[test_mask], title=f"{model_name} PR Curves on {dataset_name}", save_path=str(vis_dir / "pr_curves.png"))
    _plot_topology(data.edge_index, data.labels, vis_dir / "graph_topology.png")
    per_class = []
    for label in range(data.num_classes):
        class_mask = test_mask & (labels == label)
        per_class.append(float((predicted[class_mask] == label).mean()))
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar([f"Class {index}" for index in range(data.num_classes)], per_class, color=CATEGORICAL_PALETTE[:data.num_classes])
    axis.bar_label(bars, labels=[f"{score:.3f}" for score in per_class], padding=3)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Test Accuracy")
    axis.set_title(f"{model_name} Per-Class Accuracy on {dataset_name}")
    figure.tight_layout()
    save_or_show(figure, str(vis_dir / "per_class_accuracy.png"))
    if model_name.lower() == "gat":
        _plot_attention(model, data.features, data.edge_index, vis_dir)

    print(f"{model_name} 在 {dataset_name} 上：测试准确率={test_accuracy:.4f}，最佳轮次={best_epoch}")
    print(f"图表输出目录：{vis_dir}")
    return {"test_accuracy": test_accuracy, "best_epoch": best_epoch, "visualizations": vis_dir}
