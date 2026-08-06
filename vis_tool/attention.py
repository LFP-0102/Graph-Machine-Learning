#!/usr/bin/env python
# @File       : attention.py
# @Path       : vis_tool/attention.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:33:32
# @Version    : v1.0.0
# @Description:
#   图注意力权重可视化：将图中每条边按注意力权重着色和调整粗细。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建

"""Graph attention weight visualizations."""

from __future__ import annotations

from typing import Optional, Mapping, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from vis_tool.config import save_or_show, to_numpy


# ---------------------------------------------------------------------------
# 注意力权重图
# ---------------------------------------------------------------------------
def plot_attention_weights(
    edge_index: np.ndarray,
    attention_weights: np.ndarray,
    node_pos: Optional[Mapping[int, Tuple[float, float]]] = None,
    node_size: int = 300,
    head_idx: Optional[int] = None,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "YlOrRd",
    edge_vmin: float = 0.0,
    edge_vmax: float = 1.0,
    node_color: str = "#4C72B0",
    edge_scale: float = 3.0,
) -> plt.Figure:
    """Visualise attention weights over graph edges.

    Edge colour and thickness are proportional to the attention weight.

    Parameters
    ----------
    edge_index : np.ndarray
        Edge list, shape ``(2, E)``.
    attention_weights : np.ndarray
        Attention scores, shape ``(E,)`` or ``(E, heads)``.
    node_pos : dict, optional
        ``{node_id: (x, y)}``.  If None, a spring layout is computed via
        networkx.
    node_size : int
        Base node size for ``nx.draw_networkx_nodes``.
    head_idx : int, optional
        Which attention head to display when *attention_weights* is
        ``(E, heads)``.  If None, the mean across heads is used.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size in inches.
    cmap : str
        Matplotlib colormap name for attention weights.
    edge_vmin, edge_vmax : float
        Normalisation range for the colormap.
    node_color : str
        Uniform node fill colour.
    edge_scale : float
        Multiplier for edge line-width.

    Returns
    -------
    plt.Figure
    """
    edge_index = to_numpy(edge_index).astype(int)
    attn = to_numpy(attention_weights)

    # 处理多头
    if attn.ndim == 2 and attn.shape[1] > 1:
        if head_idx is not None:
            attn = attn[:, head_idx]
        else:
            attn = attn.mean(axis=1)

    attn = attn.ravel()

    num_edges = edge_index.shape[1]
    if len(attn) != num_edges:
        raise ValueError(
            f"Edge count mismatch: edge_index has {num_edges} edges, "
            f"but attention_weights has {len(attn)} values."
        )

    # 构建 networkx 图
    G = nx.Graph()
    nodes = set()
    edges = []
    for i in range(num_edges):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        G.add_edge(u, v, weight=float(attn[i]))
        nodes.add(u)
        nodes.add(v)
        edges.append((u, v))

    # 布局
    if node_pos is None:
        pos = nx.spring_layout(G, seed=42, k=0.3, iterations=50)
    else:
        pos = {int(k): tuple(v) for k, v in node_pos.items()}

    # 归一化注意力权重
    norm = Normalize(vmin=edge_vmin, vmax=edge_vmax)
    edge_colors = [attn[i] for i in range(num_edges)]
    edge_widths = [edge_scale * float(w) + 0.5 for w in attn]

    fig, ax = plt.subplots(figsize=figsize)

    # 画节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_color,
        alpha=0.9,
        linewidths=1,
        edgecolors="white",
        ax=ax,
    )

    # 画边
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        width=edge_widths,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap(cmap),
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        alpha=0.7,
        ax=ax,
    )

    # colorbar
    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=10)

    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()
    return save_or_show(fig, save_path)
