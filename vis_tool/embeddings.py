#!/usr/bin/env python
# @File       : embeddings.py
# @Path       : vis_tool/embeddings.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:32:49
# @Version    : v1.0.0
# @Description:
#   节点嵌入可视化：通过 t-SNE 或 UMAP 将高维嵌入降维到 2D 并绘图。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建

"""Node embedding visualizations via dimensionality reduction."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from vis_tool.config import save_or_show, to_numpy


# ---------------------------------------------------------------------------
# 嵌入可视化
# ---------------------------------------------------------------------------
def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    class_names: Optional[Sequence[str]] = None,
    title: str = "Node Embeddings",
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (9, 7),
    **kwargs,
) -> plt.Figure:
    """Reduce high-dimensional embeddings to 2D and plot as a scatter plot.

    Parameters
    ----------
    embeddings : np.ndarray
        Node embeddings, shape ``(N, d)``.
    labels : np.ndarray
        Node labels, shape ``(N,)``.
    method : str
        ``'tsne'`` or ``'umap'``.
    class_names : sequence of str, optional
        Human-readable class names for the legend.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size in inches.
    **kwargs
        Forwarded to the reducer constructor (e.g. ``perplexity=30``,
        ``random_state=42`` for t-SNE; ``n_neighbors=15`` for UMAP).

    Returns
    -------
    plt.Figure
    """
    embeddings = to_numpy(embeddings)
    labels = to_numpy(labels).astype(int)

    method_lower = method.lower()

    # 降维
    if method_lower == "umap":
        try:
            import umap  # noqa: F811
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP.  Install with: "
                "pip install umap-learn"
            )
        reducer = umap.UMAP(
            n_components=2,
            random_state=kwargs.pop("random_state", 42),
            **kwargs,
        )
        coords = reducer.fit_transform(embeddings)
        method_label = "UMAP"
    elif method_lower == "tsne":
        reducer = TSNE(
            n_components=2,
            random_state=kwargs.pop("random_state", 42),
            **kwargs,
        )
        coords = reducer.fit_transform(embeddings)
        method_label = "t-SNE"
    else:
        raise ValueError(f"Unknown method '{method}'.  Choose 'tsne' or 'umap'.")

    # 绘图
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    if class_names is None:
        class_names = [str(i) for i in unique_labels]

    # 为每个类别选颜色
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
    ax.set_title(f"{title} ({method_label})")
    ax.legend(
        loc="best",
        fontsize=9,
        markerscale=2,
        ncol=max(1, n_classes // 10),
    )
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    return save_or_show(fig, save_path)
