"""运行 GCN、GraphSAGE、GAT 在引文网络上的五种子实验。

GCN 与 GAT 使用作者 Cora 风格的超参数。GraphSAGE 在 Planetoid 上的结果
属于引文网络消融；其论文主实验 PPI 结果以 micro/macro-F1 单独记录，不能与
引文网络 accuracy 混合比较。
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT_DIR))

from datasets.base import load_dataset
from models.gat import GAT
from models.gcn import GCN
from models.graphsage import GraphSAGE
from utils.graph_utils import normalized_sparse_edge_index
from utils.paths import OUTPUT_DIR, RESULTS_DIR, ensure_dirs
from utils.seed import set_seed
from vis_tool import (
    plot_confusion_matrix,
    plot_embeddings,
    plot_graph,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_train_curves,
)
from vis_tool.config import CATEGORICAL_PALETTE, save_or_show
from vis_tool.statistics import plot_citation_benchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=("cora", "citeseer", "pubmed"), default=["cora", "citeseer", "pubmed"])
    parser.add_argument("--models", nargs="+", choices=("gcn", "graphsage", "gat"), default=["gcn", "graphsage", "gat"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--gcn-epochs", type=int, default=200)
    parser.add_argument("--graphsage-epochs", type=int, default=200)
    parser.add_argument("--gat-epochs", type=int, default=1000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有 CSV 记录，重新运行本次指定的全部实验。",
    )
    return parser.parse_args()


@torch.no_grad()
def metrics(model, features, graph, labels, mask):
    model.eval()
    logits = model(features, graph)
    loss = F.nll_loss(logits[mask], labels[mask])
    accuracy = (logits[mask].argmax(1) == labels[mask]).float().mean()
    return loss.item(), accuracy.item()


def clone_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}


def save_citation_subgraph(edge_index, labels, destination, title):
    """Draw a small deterministic component rather than the full citation graph."""
    edges = edge_index.detach().cpu().numpy()
    candidate = edges[:, (edges[0] < 500) & (edges[1] < 500)]
    graph = nx.Graph()
    graph.add_edges_from(zip(candidate[0].tolist(), candidate[1].tolist()))
    if not graph.number_of_nodes():
        return
    component = sorted(max(nx.connected_components(graph), key=len))
    remap = {node: index for index, node in enumerate(component)}
    selected = [
        (remap[int(source)], remap[int(target)])
        for source, target in zip(candidate[0], candidate[1])
        if int(source) in remap and int(target) in remap
    ]
    if selected:
        plot_graph(
            np.asarray(selected, dtype=np.int64).T,
            node_colors=labels[component].detach().cpu().numpy(), node_size=35,
            layout="spring", k=0.5, title=title, save_path=str(destination),
        )


def train_gcn(model, features, graph, labels, data, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    validation_losses = []
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, graph)
        loss = F.nll_loss(logits[data.train_mask], labels[data.train_mask])
        loss = loss + 5e-4 * 0.5 * model.gc1.weight.square().sum()
        loss.backward()
        optimizer.step()
        validation_loss, validation_accuracy = metrics(model, features, graph, labels, data.val_mask)
        validation_losses.append(validation_loss)
        history["train_loss"].append(loss.detach().item())
        history["train_accuracy"].append((logits.detach()[data.train_mask].argmax(1) == labels[data.train_mask]).float().mean().item())
        history["val_loss"].append(validation_loss)
        history["val_accuracy"].append(validation_accuracy)
        if epoch > 10 and validation_loss > sum(validation_losses[-11:-1]) / 10:
            return epoch + 1, history
    return epochs, history


def train_graphsage(model, features, graph, labels, data, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_accuracy, best_state, stale_epochs, best_epoch = -1.0, None, 0, 0
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, graph)
        loss = F.nll_loss(logits[data.train_mask], labels[data.train_mask])
        loss.backward()
        optimizer.step()
        validation_loss, validation_accuracy = metrics(model, features, graph, labels, data.val_mask)
        history["train_loss"].append(loss.detach().item())
        history["train_accuracy"].append((logits.detach()[data.train_mask].argmax(1) == labels[data.train_mask]).float().mean().item())
        history["val_loss"].append(validation_loss)
        history["val_accuracy"].append(validation_accuracy)
        if validation_accuracy > best_accuracy:
            best_accuracy, best_state, stale_epochs, best_epoch = validation_accuracy, clone_state(model), 0, epoch + 1
        else:
            stale_epochs += 1
            if stale_epochs >= 100:
                break
    model.load_state_dict(best_state)
    return best_epoch, history


def train_gat(model, features, graph, labels, data, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    min_loss, max_accuracy, stale_epochs, best_state, best_epoch = float("inf"), 0.0, 0, None, 0
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, graph)
        loss = F.nll_loss(logits[data.train_mask], labels[data.train_mask])
        loss = loss + 5e-4 * 0.5 * sum(parameter.square().sum() for parameter in model.parameters())
        loss.backward()
        optimizer.step()
        validation_loss, validation_accuracy = metrics(model, features, graph, labels, data.val_mask)
        history["train_loss"].append(loss.detach().item())
        history["train_accuracy"].append((logits.detach()[data.train_mask].argmax(1) == labels[data.train_mask]).float().mean().item())
        history["val_loss"].append(validation_loss)
        history["val_accuracy"].append(validation_accuracy)
        if validation_accuracy >= max_accuracy or validation_loss <= min_loss:
            if validation_accuracy >= max_accuracy and validation_loss <= min_loss:
                best_state, best_epoch = clone_state(model), epoch + 1
            max_accuracy, min_loss, stale_epochs = max(max_accuracy, validation_accuracy), min(min_loss, validation_loss), 0
        else:
            stale_epochs += 1
            if stale_epochs >= 100:
                break
    model.load_state_dict(best_state)
    return best_epoch, history


def save_run_visualizations(model, model_name, dataset_name, seed, data, graph, history):
    """Save plots for exactly one completed model/dataset/seed training run."""
    vis_dir = OUTPUT_DIR / "visualizations" / f"{model_name}_{dataset_name}" / f"seed_{seed}"
    ensure_dirs(vis_dir)

    model.eval()
    with torch.no_grad():
        logits = model(data.features, graph)
        predictions = logits.argmax(1).detach().cpu().numpy()
        probabilities = logits.exp().detach().cpu().numpy()
        embeddings = model.get_embeddings(data.features, graph)

    labels = data.labels.detach().cpu().numpy()
    test_mask = data.test_mask.detach().cpu().numpy()
    test_labels = labels[test_mask]
    test_predictions = predictions[test_mask]
    test_probabilities = probabilities[test_mask]
    run_title = f"{model_name.upper() if model_name != 'graphsage' else 'GraphSAGE'} on {dataset_name} (seed {seed})"

    pd.DataFrame(history).to_csv(vis_dir / "history.csv", index_label="epoch")
    plot_train_curves(
        np.asarray([history["train_loss"], history["train_accuracy"]]),
        np.asarray([history["val_loss"], history["val_accuracy"]]),
        metric_names=["Loss", "Accuracy"], title=f"{run_title} Training Curves",
        save_path=str(vis_dir / "training_curves.png"),
    )
    plot_embeddings(
        embeddings, labels, method="tsne", title=f"{run_title} Node Embeddings",
        save_path=str(vis_dir / "embeddings_tsne.png"), random_state=seed,
    )
    plot_confusion_matrix(
        test_labels, test_predictions, title=f"{run_title} Confusion Matrix",
        save_path=str(vis_dir / "confusion_matrix.png"),
    )
    plot_roc_curves(
        test_labels, test_probabilities, title=f"{run_title} ROC Curves",
        save_path=str(vis_dir / "roc_curves.png"),
    )
    plot_precision_recall_curves(
        test_labels, test_probabilities, title=f"{run_title} PR Curves",
        save_path=str(vis_dir / "pr_curves.png"),
    )
    save_citation_subgraph(
        data.edge_index, data.labels, vis_dir / "graph_topology.png",
        title=f"{run_title} Citation Subgraph",
    )

    per_class = []
    for label in range(data.num_classes):
        class_mask = test_labels == label
        per_class.append(float((test_predictions[class_mask] == label).mean()) if class_mask.any() else 0.0)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar([f"Class {index}" for index in range(data.num_classes)], per_class,
                    color=CATEGORICAL_PALETTE[:data.num_classes])
    axis.bar_label(bars, labels=[f"{score:.3f}" for score in per_class], padding=3)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Test Accuracy")
    axis.set_title(f"{run_title} Per-Class Accuracy")
    figure.tight_layout()
    save_or_show(figure, str(vis_dir / "per_class_accuracy.png"))
    return vis_dir


def run_one(model_name, dataset_name, seed, args, device):
    set_seed(seed)
    data = load_dataset(dataset_name)
    for key in ("features", "labels", "edge_index", "train_mask", "val_mask", "test_mask"):
        data[key] = data[key].to(device)
    if model_name == "gcn":
        model = GCN(data.num_features, 16, data.num_classes, dropout=0.5).to(device)
        graph = normalized_sparse_edge_index(data.edge_index, data.features.size(0), device)
        trainer, epochs, protocol = train_gcn, args.gcn_epochs, "official_citation"
    elif model_name == "gat":
        model = GAT(data.num_features, 8, data.num_classes, n_heads=8, dropout=0.6).to(device)
        graph = data.edge_index
        trainer, epochs, protocol = train_gat, args.gat_epochs, "official_cora_architecture"
    else:
        model = GraphSAGE(data.num_features, 128, data.num_classes, dropout=0.0, sample_sizes=(25, 10)).to(device)
        graph = data.edge_index
        trainer, epochs, protocol = train_graphsage, args.graphsage_epochs, "planetoid_ablation"
    started = time.time()
    best_epoch, history = trainer(model, data.features, graph, data.labels, data, epochs)
    _, validation_accuracy = metrics(model, data.features, graph, data.labels, data.val_mask)
    test_loss, test_accuracy = metrics(model, data.features, graph, data.labels, data.test_mask)
    train_time = time.time() - started
    visualizations = save_run_visualizations(model, model_name, dataset_name, seed, data, graph, history)
    return {
        "model": model_name.upper() if model_name != "graphsage" else "GraphSAGE",
        "dataset": dataset_name,
        "protocol": protocol,
        "seed": seed,
        "epochs_limit": epochs,
        "best_epoch": best_epoch,
        "val_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "train_time": train_time,
        "visualizations": str(visualizations),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dirs(RESULTS_DIR)
    raw_path = RESULTS_DIR / "citation_multi_seed_raw.csv"
    if raw_path.exists() and not args.force:
        existing = pd.read_csv(raw_path)
        rows = existing.to_dict("records")
    else:
        rows = []
    def result_key(row):
        return (row["model"].lower(), row["dataset"], int(row["seed"]), int(row["epochs_limit"]))

    def visualizations_complete(row):
        path = row.get("visualizations")
        return isinstance(path, str) and (Path(path) / "training_curves.png").is_file()

    completed_keys = {result_key(row) for row in rows if visualizations_complete(row)}
    total = len(args.models) * len(args.datasets) * len(args.seeds)
    completed = 0
    for model_name in args.models:
        for dataset_name in args.datasets:
            for seed in args.seeds:
                completed += 1
                epoch_limit = {
                    "gcn": args.gcn_epochs,
                    "graphsage": args.graphsage_epochs,
                    "gat": args.gat_epochs,
                }[model_name]
                if (model_name, dataset_name, seed, epoch_limit) in completed_keys:
                    print(f"[{completed}/{total}] {model_name} | {dataset_name} | seed={seed}（已有记录，跳过）", flush=True)
                    continue
                print(f"[{completed}/{total}] {model_name} | {dataset_name} | seed={seed}", flush=True)
                result = run_one(model_name, dataset_name, seed, args, device)
                print(f"  测试准确率={result['test_accuracy']:.4f} | 最佳轮次={result['best_epoch']} | 用时={result['train_time']:.1f}s", flush=True)
                key = (model_name, dataset_name, seed, epoch_limit)
                rows = [row for row in rows if result_key(row) != key]
                rows.append(result)
                completed_keys.add(key)
                pd.DataFrame(rows).to_csv(raw_path, index=False)
    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)
    requested_models = {
        "GraphSAGE" if name == "graphsage" else name.upper()
        for name in args.models
    }
    selected = raw[
        raw["model"].isin(requested_models)
        & raw["dataset"].isin(args.datasets)
        & raw["seed"].isin(args.seeds)
    ]
    summary = selected.groupby(["model", "dataset", "protocol"], as_index=False).agg(
        runs=("seed", "count"),
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),
        time_mean=("train_time", "mean"),
    )
    summary.to_csv(RESULTS_DIR / "citation_multi_seed_summary.csv", index=False)
    benchmark_dir = OUTPUT_DIR / "visualizations" / "benchmark_summary"
    plot_citation_benchmark(selected, summary, benchmark_dir)
    print("\n引文网络原始记录：", raw_path)
    print("引文网络汇总结果：", RESULTS_DIR / "citation_multi_seed_summary.csv")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
