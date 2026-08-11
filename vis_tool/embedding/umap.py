#!/usr/bin/env python
# @File       : umap.py
# @Path       : vis_tool/embedding/umap.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:32:49
# @Version    : v1.0.0
# @Description:
#   UMAP 降维可视化：将高维节点嵌入通过 UMAP 降到 2D 并绘图。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建
#   2026/8/11 | 刘赋平 | v1.1.0 | 从 embeddings.py 拆分独立

"""UMAP node embedding visualization."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from vis_tool.config import save_or_show, to_numpy
from vis_tool.embedding._common import _plot_embedding_scatter


def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Node Embeddings — UMAP",
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (9, 7),
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    **kwargs,
) -> plt.Figure:
    """Reduce embeddings with UMAP and plot 2D scatter.

    Parameters
    ----------
    embeddings : np.ndarray
        Node embeddings, shape ``(N, d)``.
    labels : np.ndarray
        Node labels, shape ``(N,)``.
    class_names : sequence of str, optional
        Human-readable class names for the legend.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size in inches.
    n_neighbors : int
        UMAP n_neighbors parameter.
    min_dist : float
        UMAP min_dist parameter.
    random_state : int
        Random seed for reproducibility.
    **kwargs
        Forwarded to ``umap.UMAP``.

    Returns
    -------
    plt.Figure
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for UMAP.  Install with: "
            "pip install umap-learn"
        )

    embeddings = to_numpy(embeddings)
    labels = to_numpy(labels).astype(int)

    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        **kwargs,
    )
    coords = reducer.fit_transform(embeddings)

    return _plot_embedding_scatter(
        coords, labels, class_names, title, save_path, figsize,
        method_label="UMAP",
    )
