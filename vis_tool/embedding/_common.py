"""嵌入表示可视化的内部绘图函数。"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from vis_tool.config import save_or_show


def _plot_embedding_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Sequence[str]],
    title: str,
    save_path: Optional[str],
    figsize: tuple[float, float],
    method_label: str = "",
) -> plt.Figure:
    """将二维降维后的节点表示按类别绘制为散点图。"""
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    if class_names is None:
        class_names = [str(i) for i in unique_labels]

    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=figsize)
    for index, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1], c=[cmap(index % cmap.N)],
            label=class_names[index] if index < len(class_names) else str(label),
            alpha=0.7, s=25, edgecolors="none",
        )

    prefix = f"{method_label} " if method_label else ""
    ax.set_xlabel(f"{prefix}Dimension 1")
    ax.set_ylabel(f"{prefix}Dimension 2")
    ax.set_title(f"{title} ({method_label})" if method_label else title)
    ax.legend(loc="best", fontsize=9, markerscale=2, ncol=max(1, n_classes // 10))
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return save_or_show(fig, save_path)
