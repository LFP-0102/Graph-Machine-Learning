"""分类结果的可视化工具。"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from vis_tool.config import save_or_show, to_numpy


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[Sequence[str]] = None,
    normalize: bool = True, title: str = "Confusion Matrix", save_path: Optional[str] = None,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """以热力图形式绘制分类混淆矩阵。"""
    y_true, y_pred = to_numpy(y_true).astype(int), to_numpy(y_pred).astype(int)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        sums = matrix.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        display, fmt, colorbar_label = matrix.astype(float) / sums, ".2f", "Proportion"
    else:
        display, fmt, colorbar_label = matrix, "d", "Count"
    if class_names is None:
        class_names = [str(label) for label in labels]
    elif len(class_names) > len(labels):
        class_names = [class_names[i] for i in labels]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(display, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, vmin=0, vmax=1 if normalize else None,
                cbar_kws={"label": colorbar_label}, linewidths=0.5, linecolor="white", ax=ax)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(title)
    for index, value in enumerate(matrix.diagonal() / matrix.sum(axis=1).clip(min=1)):
        ax.text(len(labels) + 0.35, index + 0.5, f"{value:.1%}", va="center", ha="left", fontsize=9, color="dimgray")
    fig.tight_layout()
    return save_or_show(fig, save_path)
