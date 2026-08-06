#!/usr/bin/env python
# @File       : curves.py
# @Path       : vis_tool/curves.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:32:16
# @Version    : v1.0.0
# @Description:
#   训练过程曲线可视化：损失、准确率等指标的训练/验证曲线对比。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建

"""Training curve visualizations."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from vis_tool.config import CATEGORICAL_PALETTE, save_or_show, to_numpy


# ---------------------------------------------------------------------------
# 训练曲线
# ---------------------------------------------------------------------------
def plot_train_curves(
    train_values: np.ndarray,
    val_values: np.ndarray,
    metric_names: Optional[Sequence[str]] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (12, 4),
    style: Optional[dict] = None,
    mark_best: bool = True,
) -> plt.Figure:
    """Plot training & validation curves for one or more metrics.

    Parameters
    ----------
    train_values : np.ndarray
        Shape ``(epochs,)`` for a single metric, or ``(metrics, epochs)``
        for multiple metrics.
    val_values : np.ndarray
        Same shape as *train_values*.
    metric_names : sequence of str, optional
        Names for each metric (e.g. ``['Loss', 'Accuracy']``).
        Defaults to ``['Metric 0', 'Metric 1', …]``.
    title : str
        Overall figure title (suptitle).
    save_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size ``(width, height)``.  Width scales with metric count.
    style : dict, optional
        Optional overrides for line style.  Keys:
        ``train_color``, ``val_color``, ``linewidth``, ``marker_size``.
    mark_best : bool
        If True, mark the best-epoch point on each validation curve
        (minimum for loss-like, maximum for accuracy-like metrics).

    Returns
    -------
    plt.Figure
    """
    train = to_numpy(train_values)
    val = to_numpy(val_values)

    # 统一为 (n_metrics, epochs) 形状
    if train.ndim == 1:
        train = train[np.newaxis, :]
        val = val[np.newaxis, :]

    n_metrics, n_epochs = train.shape

    if metric_names is None:
        metric_names = [f"Metric {i}" for i in range(n_metrics)]
    elif len(metric_names) < n_metrics:
        metric_names = list(metric_names) + [
            f"Metric {i}" for i in range(len(metric_names), n_metrics)
        ]

    # 样式默认值
    _style = {
        "train_color": CATEGORICAL_PALETTE[0],
        "val_color": CATEGORICAL_PALETTE[1],
        "linewidth": 2,
        "marker_size": 6,
    }
    if style:
        _style.update(style)

    # 自适应 figsize
    w, h = figsize
    if n_metrics > 1:
        w = max(w, 5.5 * n_metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(w, h), squeeze=False)
    axes = axes.flatten()
    epochs = np.arange(1, n_epochs + 1)

    for i in range(n_metrics):
        ax = axes[i]

        ax.plot(
            epochs, train[i],
            color=_style["train_color"],
            lw=_style["linewidth"],
            marker="o",
            markersize=_style["marker_size"] // 2,
            label="Train",
        )
        ax.plot(
            epochs, val[i],
            color=_style["val_color"],
            lw=_style["linewidth"],
            marker="s",
            markersize=_style["marker_size"] // 2,
            label="Validation",
        )

        # 标注最佳 epoch
        if mark_best:
            # 经验规则：名称中带 'loss'/'Loss'/'err' 的越小越好，否则越大越好
            name_lower = metric_names[i].lower()
            minimize = any(kw in name_lower for kw in ("loss", "err", "mse", "mae", "rmse"))
            if minimize:
                best_idx = np.nanargmin(val[i])
                best_val = val[i][best_idx]
            else:
                best_idx = np.nanargmax(val[i])
                best_val = val[i][best_idx]

            ax.scatter(
                best_idx + 1, best_val,
                color="red", s=40, zorder=5,
                edgecolors="white", linewidth=0.8,
            )
            ax.annotate(
                f"  {best_val:.4f}",
                (best_idx + 1, best_val),
                fontsize=8, color="red", va="center",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_names[i])
        ax.set_title(metric_names[i])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    return save_or_show(fig, save_path)
