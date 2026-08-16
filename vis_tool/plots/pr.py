"""多分类 one-vs-rest Precision-Recall 曲线。"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from vis_tool.config import get_cmap, save_or_show, to_numpy


def plot_precision_recall_curves(
    y_true: np.ndarray, y_score: np.ndarray, class_names: Optional[Sequence[str]] = None,
    title: str = "Precision-Recall Curves", save_path: Optional[str] = None,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """绘制多分类 one-vs-rest Precision-Recall 曲线。"""
    y_true, y_score = to_numpy(y_true).astype(int), to_numpy(y_score)
    num_classes = y_score.shape[1]
    class_names = class_names or [str(i) for i in range(num_classes)]
    fig, ax = plt.subplots(figsize=figsize)
    colors = get_cmap(num_classes, categorical=True)
    for index in range(num_classes):
        targets = (y_true == index).astype(int)
        if targets.min() == targets.max():
            continue
        precision, recall, _ = precision_recall_curve(targets, y_score[:, index])
        ap = average_precision_score(targets, y_score[:, index])
        ax.plot(recall, precision, color=colors[index], lw=2, label=f"{class_names[index]} (AP={ap:.3f})")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Recall", ylabel="Precision", title=title)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    return save_or_show(fig, save_path)
