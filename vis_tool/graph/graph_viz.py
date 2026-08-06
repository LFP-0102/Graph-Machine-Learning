#!/usr/bin/env python
# @File       : graph_viz.py
# @Path       : vis_tool/graph_viz.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:33:06
# @Version    : v1.0.0
# @Description:
#   图结构可视化：按节点颜色/标签展示图拓扑。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建

"""Graph structure visualizations."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable

from vis_tool.config import save_or_show, to_numpy


# ---------------------------------------------------------------------------
# 布局名称 → networkx 布局函数
# ---------------------------------------------------------------------------
_LAYOUT_FUNCTIONS = {
    "spring": nx.spring_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "circular": nx.circular_layout,
    "spectral": nx.spectral_layout,
    "random": nx.random_layout,
    "shell": nx.shell_layout,
    "spiral": nx.spiral_layout,
}


# ---------------------------------------------------------------------------
# 图结构可视化
# ---------------------------------------------------------------------------
def plot_graph(
    edge_index: np.ndarray,
    node_colors: Optional[np.ndarray] = None,
    node_labels: Optional[Sequence[str]] = None,
    layout: str = "spring",
    figsize: tuple[float, float] = (10, 8),
    title: str = "Graph Structure",
    save_path: Optional[str] = None,
    node_size: Union[int, np.ndarray] = 200,
    cmap: str = "tab10",
    label_max_nodes: int = 500,
    **layout_kwargs,
) -> plt.Figure:
    """Visualise graph topology with optional node colours and labels.

    Parameters
    ----------
    edge_index : np.ndarray
        Edge list, shape ``(2, E)``.
    node_colors : np.ndarray, optional
        - If ``(N,)`` of integers → categorical, one colour per label.
        - If ``(N,)`` of floats  → continuous colormap + colorbar.
        - If ``(N, 3)`` or ``(N, 4)`` → RGB(A) values directly.
        - If None → uniform default colour.
    node_labels : sequence of str, optional
        Per-node label strings (e.g. class names).
    layout : str
        Layout algorithm.  One of ``'spring'``, ``'kamada_kawai'``,
        ``'circular'``, ``'spectral'``, ``'random'``, ``'shell'``,
        ``'spiral'``.
    figsize : tuple
        Figure size in inches.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    node_size : int or np.ndarray
        Uniform node size, or per-node sizes ``(N,)``.
    cmap : str
        Colormap for continuous *node_colors*.
    label_max_nodes : int
        When the graph has more than this many nodes, vertex labels are
        suppressed to avoid clutter.
    **layout_kwargs
        Extra arguments forwarded to the networkx layout function
        (e.g. ``k=0.3`` for spring layout, ``scale=2``).

    Returns
    -------
    plt.Figure
    """
    edge_index = to_numpy(edge_index).astype(int)
    num_edges = edge_index.shape[1]

    # 构建 networkx 图
    G = nx.Graph()
    edge_list = []
    for i in range(num_edges):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        G.add_edge(u, v)
        edge_list.append((u, v))

    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())

    # 布局
    if layout not in _LAYOUT_FUNCTIONS:
        raise ValueError(
            f"Unknown layout '{layout}'.  Choose from: "
            f"{list(_LAYOUT_FUNCTIONS.keys())}"
        )
    if layout == "spring":
        layout_kwargs.setdefault("seed", 42)
    pos = _LAYOUT_FUNCTIONS[layout](G, **layout_kwargs)

    # --- 处理节点颜色 ---
    if node_colors is None:
        # 统一颜色
        vmin = vmax = None
        cmap_used = None
        cvals = "#4C72B0"
    elif node_colors.ndim == 2 and node_colors.shape[1] in (3, 4):
        # RGB(A) 直接使用
        vmin = vmax = None
        cmap_used = None
        cvals = [to_rgba(node_colors[i]) for i in range(node_colors.shape[0])]
    elif node_colors.ndim == 1:
        cvals = node_colors.astype(float)
        if np.issubdtype(node_colors.dtype, np.integer) or len(np.unique(cvals)) <= 20:
            # 离散类别
            vmin = vmax = None
            cmap_used = cmap
        else:
            # 连续值
            vmin, vmax = float(cvals.min()), float(cvals.max())
            cmap_used = cmap
    else:
        cvals = "#4C72B0"
        vmin = vmax = cmap_used = None

    # --- 处理 node_size ---
    if isinstance(node_size, np.ndarray):
        node_size_arr = to_numpy(node_size).ravel()
    else:
        node_size_arr = np.full(num_nodes, node_size)

    fig, ax = plt.subplots(figsize=figsize)

    # 画节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size_arr,
        node_color=cvals if cmap_used is None else cvals,
        cmap=plt.get_cmap(cmap_used) if cmap_used else None,
        vmin=vmin,
        vmax=vmax,
        alpha=0.9,
        linewidths=0.5,
        edgecolors="#333333",
        ax=ax,
    )

    # 画边
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edge_list,
        width=0.5,
        alpha=0.3,
        edge_color="#888888",
        ax=ax,
    )

    # 画标签（大图自动跳过）
    if node_labels is not None and num_nodes <= label_max_nodes:
        label_dict = {
            nodes[i]: str(node_labels[i])
            for i in range(min(len(nodes), len(node_labels)))
        }
        nx.draw_networkx_labels(
            G, pos,
            labels=label_dict,
            font_size=7,
            font_color="#333333",
            ax=ax,
        )

    # 连续值的 colorbar
    if cmap_used is not None and vmin is not None:
        sm = ScalarMappable(
            norm=Normalize(vmin=vmin, vmax=vmax),
            cmap=plt.get_cmap(cmap_used),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Node Value", fontsize=10)

    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()
    return save_or_show(fig, save_path)
