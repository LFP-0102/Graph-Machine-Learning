"""Aggregate visualizations for multi-seed benchmark CSV files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vis_tool.config import CATEGORICAL_PALETTE, save_or_show


def _model_order(values: pd.Series) -> list[str]:
    preferred = ["GCN", "GAT", "GraphSAGE"]
    return [name for name in preferred if name in set(values)] + [
        name for name in sorted(set(values)) if name not in preferred
    ]


def plot_citation_benchmark(raw: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Create accuracy, stability, cost, and heatmap views from citation runs."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    datasets = list(summary["dataset"].drop_duplicates())
    models = _model_order(summary["model"])
    paths = []

    figure, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.8), sharey=True)
    for axis, dataset in zip(np.atleast_1d(axes), datasets):
        subset = summary[summary["dataset"] == dataset].set_index("model").reindex(models).dropna()
        bars = axis.bar(subset.index, subset["accuracy_mean"], yerr=subset["accuracy_std"].fillna(0),
                       capsize=5, color=CATEGORICAL_PALETTE[:len(subset)])
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in subset["accuracy_mean"]], padding=3, fontsize=9)
        axis.set_title(dataset)
        axis.set_ylim(0, 1.05)
        axis.set_xlabel("Model")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Test Accuracy")
    figure.suptitle("Citation Benchmarks: Mean Test Accuracy +/- Std", y=1.02)
    figure.tight_layout()
    paths.append(destination / "accuracy_mean_std.png")
    save_or_show(figure, str(paths[-1]))

    figure, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.8), sharey=True)
    for axis, dataset in zip(np.atleast_1d(axes), datasets):
        subset = raw[raw["dataset"] == dataset]
        sns.boxplot(data=subset, x="model", y="test_accuracy", hue="model", order=models,
                    hue_order=models, palette=CATEGORICAL_PALETTE[:len(models)], legend=False,
                    ax=axis, fliersize=0)
        sns.stripplot(data=subset, x="model", y="test_accuracy", order=models, ax=axis,
                      color="#222222", size=4, jitter=0.12)
        axis.set_title(dataset)
        axis.set_xlabel("Model")
        axis.set_ylabel("Test Accuracy")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Citation Benchmarks: Seed Stability", y=1.02)
    figure.tight_layout()
    paths.append(destination / "seed_stability.png")
    save_or_show(figure, str(paths[-1]))

    figure, axis = plt.subplots(figsize=(8, 6))
    for index, row in summary.reset_index(drop=True).iterrows():
        axis.scatter(row["time_mean"], row["accuracy_mean"], s=95,
                     color=CATEGORICAL_PALETTE[index % len(CATEGORICAL_PALETTE)])
        axis.annotate(f"{row['model']}\n{row['dataset']}", (row["time_mean"], row["accuracy_mean"]),
                      xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Mean Training Time (seconds)")
    axis.set_ylabel("Mean Test Accuracy")
    axis.set_title("Citation Benchmarks: Accuracy vs. Training Cost")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    paths.append(destination / "accuracy_vs_time.png")
    save_or_show(figure, str(paths[-1]))

    matrix = summary.pivot(index="model", columns="dataset", values="accuracy_mean").reindex(models)
    figure, axis = plt.subplots(figsize=(1.8 * len(matrix.columns) + 3, 1.0 * len(matrix.index) + 2.5))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1,
                cbar_kws={"label": "Mean Test Accuracy"}, linewidths=0.5, linecolor="white", ax=axis)
    axis.set_title("Citation Benchmarks: Mean Accuracy Heatmap")
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Model")
    figure.tight_layout()
    paths.append(destination / "accuracy_heatmap.png")
    save_or_show(figure, str(paths[-1]))
    return paths


def plot_ppi_benchmark(raw: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Create seed-level and aggregate F1 charts for GraphSAGE PPI runs."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    melted = raw.melt(id_vars="seed", value_vars=["test_micro_f1", "test_macro_f1"],
                      var_name="metric", value_name="f1")
    labels = {"test_micro_f1": "Micro-F1", "test_macro_f1": "Macro-F1"}
    melted["metric"] = melted["metric"].map(labels)

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=melted, x="seed", y="f1", hue="metric", marker="o", linewidth=2, ax=axis)
    axis.set_ylim(0, 1.0)
    axis.set_xlabel("Random Seed")
    axis.set_ylabel("Test F1")
    axis.set_title("GraphSAGE PPI: Per-Seed Test F1")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    paths.append(destination / "ppi_seed_f1.png")
    save_or_show(figure, str(paths[-1]))

    means = melted.groupby("metric", sort=False)["f1"].mean()
    stds = melted.groupby("metric", sort=False)["f1"].std().fillna(0)
    figure, axis = plt.subplots(figsize=(6, 4.5))
    bars = axis.bar(means.index, means.values, yerr=stds.values, capsize=5,
                    color=CATEGORICAL_PALETTE[:len(means)])
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in means.values], padding=3)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Test F1")
    axis.set_title("GraphSAGE PPI: Mean Test F1 +/- Std")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    paths.append(destination / "ppi_f1_mean_std.png")
    save_or_show(figure, str(paths[-1]))
    return paths
