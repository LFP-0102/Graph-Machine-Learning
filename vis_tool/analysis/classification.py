#!/usr/bin/env python
# @File       : classification.py
# @Path       : vis_tool/classification.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:34:16
# @Version    : v1.0.0
# @Description:
#   分类结果可视化：混淆矩阵、ROC 曲线、Precision-Recall 曲线。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建

"""Classification evaluation visualizations."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from vis_tool.config import save_or_show, to_numpy


# ---------------------------------------------------------------------------
# 混淆矩阵
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Plot a confusion matrix as a seaborn heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels, shape ``(N,)``.
    y_pred : np.ndarray
        Predicted labels, shape ``(N,)``.
    class_names : sequence of str, optional
        Human-readable class names.  If None, integer labels are used.
    normalize : bool
        If True (default), show row-wise proportions; otherwise raw counts.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.  If None, ``plt.show()`` is called.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    plt.Figure
    """
    y_true = to_numpy(y_true).astype(int)
    y_pred = to_numpy(y_pred).astype(int)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        cm_display = cm.astype(float) / row_sums
        fmt = ".2f"
        cbar_label = "Proportion"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Count"

    if class_names is None:
        class_names = [str(lbl) for lbl in labels]
    elif len(class_names) > n_classes:
        class_names = [class_names[i] for i in labels]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1 if normalize else None,
        cbar_kws={"label": cbar_label},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # 标注每行准确率
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    for i, acc in enumerate(per_class_acc):
        ax.text(
            n_classes + 0.35, i + 0.5,
            f"{acc:.1%}",
            va="center", ha="left",
            fontsize=9, color="dimgray",
        )

    fig.tight_layout()
    return save_or_show(fig, save_path)
