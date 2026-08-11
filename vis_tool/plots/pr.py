#!/usr/bin/env python
# @File       : pr.py
# @Path       : vis_tool/plots/pr.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:34:16
# @Version    : v1.0.0
# @Description:
#   Precision-Recall 曲线可视化（多类 one-vs-rest）。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6  | 刘赋平 | v1.0.0 | 初始化创建
#   2026/8/11 | 刘赋平 | v1.1.0 | 从 analysis/classification.py 拆分

"""Precision-Recall curve visualizations (multi-class one-vs-rest)."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from vis_tool.config import get_cmap, save_or_show, to_numpy


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Precision-Recall Curves",
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Plot Precision-Recall curves for multi-class classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels, shape ``(N,)``.
    y_score : np.ndarray
        Predicted probabilities / logits, shape ``(N, C)``.
    class_names : sequence of str, optional
        Class names for the legend.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    plt.Figure
    """
    y_true = to_numpy(y_true).astype(int)
    y_score = to_numpy(y_score)

    n_classes = y_score.shape[1]
    labels = np.unique(y_true)

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    y_bin = label_binarize(y_true, classes=labels)

    colors = get_cmap(n_classes, categorical=True)

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_classes):
        if i >= len(labels):
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_bin[:, i], y_score[:, i])
        ax.plot(recall, precision, color=colors[i], lw=2,
                label=f"{class_names[i]} (AP={ap:.3f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_aspect("equal")

    fig.tight_layout()
    return save_or_show(fig, save_path)
