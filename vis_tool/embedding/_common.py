#!/usr/bin/env python
# @File       : _common.py
# @Path       : vis_tool/embedding/_common.py
# @Author     : 刘赋平
# @Date       : 2026-08-11
# @Version    : v1.0.0
# @Description:
#   嵌入可视化的共享绘图逻辑（内部模块）。
# -----------------------------------------------------------------------------

"""Shared plotting helpers for embedding visualizations (internal)."""

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
    """Plot 2D-reduced coordinates as a scatter plot coloured by class."""

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    if class_names is None:
        class_names = [str(i) for i in unique_labels]

    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)

    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[idx]],
            label=class_names[idx] if idx < len(class_names) else str(lbl),
            alpha=0.7, s=25, edgecolors="none",
        )

    ax.set_xlabel(f"{method_label} Dimension 1")
    ax.set_ylabel(f"{method_label} Dimension 2")
    ax.set_title(f"{title} ({method_label})" if method_label else title)
    ax.legend(
        loc="best",
        fontsize=9,
        markerscale=2,
        ncol=max(1, n_classes // 10),
    )
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    return save_or_show(fig, save_path)
