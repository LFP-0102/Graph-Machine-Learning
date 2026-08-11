#!/usr/bin/env python
# @File       : tsne.py
# @Path       : vis_tool/embedding/tsne.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:32:49
# @Version    : v1.0.0
# @Description:
#   t-SNE 降维可视化：将高维节点嵌入通过 t-SNE 降到 2D 并绘图。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建
#   2026/8/11 | 刘赋平 | v1.1.0 | 从 embeddings.py 拆分独立

"""t-SNE node embedding visualization."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from vis_tool.config import save_or_show, to_numpy
from vis_tool.embedding._common import _plot_embedding_scatter


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Node Embeddings — t-SNE",
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (9, 7),
    perplexity: float = 30,
    random_state: int = 42,
    **kwargs,
) -> plt.Figure:
    """Reduce embeddings with t-SNE and plot 2D scatter.

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
    perplexity : float
        t-SNE perplexity parameter.
    random_state : int
        Random seed for reproducibility.
    **kwargs
        Forwarded to ``sklearn.manifold.TSNE``.

    Returns
    -------
    plt.Figure
    """
    embeddings = to_numpy(embeddings)
    labels = to_numpy(labels).astype(int)

    reducer = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        **kwargs,
    )
    coords = reducer.fit_transform(embeddings)

    return _plot_embedding_scatter(
        coords, labels, class_names, title, save_path, figsize,
        method_label="t-SNE",
    )
