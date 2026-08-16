"""训练曲线可视化。"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from vis_tool.config import CATEGORICAL_PALETTE, save_or_show, to_numpy


def plot_train_curves(train_values: np.ndarray, val_values: np.ndarray,
                      metric_names: Optional[Sequence[str]] = None,
                      title: str = "Training Curves", save_path: Optional[str] = None,
                      figsize: tuple[float, float] = (12, 4), style: Optional[dict] = None,
                      mark_best: bool = True) -> plt.Figure:
    """绘制一个或多个指标的训练集与验证集曲线。"""
    train, val = to_numpy(train_values), to_numpy(val_values)
    if train.ndim == 1:
        train, val = train[np.newaxis, :], val[np.newaxis, :]
    n_metrics, n_epochs = train.shape
    if metric_names is None:
        metric_names = [f"Metric {index}" for index in range(n_metrics)]
    elif len(metric_names) < n_metrics:
        metric_names = list(metric_names) + [f"Metric {index}" for index in range(len(metric_names), n_metrics)]
    line_style = {"train_color": CATEGORICAL_PALETTE[0], "val_color": CATEGORICAL_PALETTE[1], "linewidth": 2, "marker_size": 6}
    if style:
        line_style.update(style)
    width, height = figsize
    figure, axes = plt.subplots(1, n_metrics, figsize=(max(width, 5.5 * n_metrics), height), squeeze=False)
    for index, axis in enumerate(axes.flatten()):
        epochs = np.arange(1, n_epochs + 1)
        axis.plot(epochs, train[index], color=line_style["train_color"], lw=line_style["linewidth"], marker="o", markersize=line_style["marker_size"] // 2, label="Train")
        axis.plot(epochs, val[index], color=line_style["val_color"], lw=line_style["linewidth"], marker="s", markersize=line_style["marker_size"] // 2, label="Validation")
        if mark_best:
            name = metric_names[index].lower()
            minimize = any(keyword in name for keyword in ("loss", "损失", "err", "误差", "mse", "mae", "rmse"))
            best_index = np.nanargmin(val[index]) if minimize else np.nanargmax(val[index])
            best_value = val[index][best_index]
            axis.scatter(best_index + 1, best_value, color="red", s=40, zorder=5, edgecolors="white", linewidth=0.8)
            axis.annotate(f"  {best_value:.4f}", (best_index + 1, best_value), fontsize=8, color="red", va="center")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric_names[index])
        axis.set_title(metric_names[index])
        axis.legend(fontsize=9)
        axis.grid(True, alpha=0.3)
    figure.suptitle(title, fontsize=15, y=1.02)
    figure.tight_layout()
    return save_or_show(figure, save_path)
