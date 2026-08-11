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

from vis_tool.config import CATEGORICAL_PALETTE, save_or_show, to_numpy


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


# ---------------------------------------------------------------------------
# 稠密 attention → 边级 scores
# ---------------------------------------------------------------------------
def _dense_attn_to_edge_scores(avg_attn, edge_index):
    """从稠密 attention 矩阵 [N, N] 提取每条边的 attention 分数.

    avg_attn: np.ndarray [N, N]  多头平均 attention
    edge_index: np.ndarray [2, E]
    返回: np.ndarray [E]
    """
    return avg_attn[edge_index[0], edge_index[1]]


# ---------------------------------------------------------------------------
# 焦点节点邻居注意力柱状图
# ---------------------------------------------------------------------------
def plot_node_attention_bars(
    avg_attn,
    edge_index,
    focal_node,
    top_k=8,
    title=None,
    save_path=None,
    figsize=(10, 5),
    class_names=None,
    node_labels=None,
):
    """展示单个焦点节点对其邻居的注意力权重（降序柱状图）。

    avg_attn:    np.ndarray [N, N]  多头平均 attention
    edge_index:  np.ndarray [2, E]
    focal_node:  int  焦点节点 ID
    top_k:       展示前 k 个邻居
    """
    avg_attn = to_numpy(avg_attn)
    edge_index = to_numpy(edge_index).astype(int)

    N = avg_attn.shape[0]
    # 找 focal_node 的所有出边
    src_mask = edge_index[0] == focal_node
    dsts = edge_index[1][src_mask]
    scores = avg_attn[focal_node, dsts]

    # 按 attention 降序排列
    order = np.argsort(scores)[::-1]
    if top_k > 0:
        order = order[:top_k]
    neighbors = dsts[order]
    values = scores[order]

    # 标签
    if node_labels is not None:
        labels = [f"#{n} {node_labels[n]}" if n < len(node_labels) else f"Node {n}"
                  for n in neighbors]
    elif class_names is not None:
        labels = [f"Node {n}" for n in neighbors]
    else:
        labels = [f"Neighbor {n}" for n in neighbors]

    colors = [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
              for i in range(len(neighbors))]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(neighbors)), values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(neighbors)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Attention Weight")
    ax.set_xlim(0, max(values) * 1.2 if len(values) else 1.0)
    ax.invert_yaxis()

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    self_val = avg_attn[focal_node, focal_node]
    title = title or f"Node #{focal_node} — Top-{len(neighbors)} Neighbor Attention  (self: {self_val:.3f})"
    ax.set_title(title)
    fig.tight_layout()
    return save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 多节点注意力摘要
# ---------------------------------------------------------------------------
def plot_attention_summary(
    avg_attn,
    edge_index,
    edge_index_full=None,
    focal_nodes=None,
    num_nodes=3,
    top_k=8,
    title="GAT — Multi-Node Attention Summary",
    save_path=None,
    class_names=None,
):
    """选取多个焦点节点，以子图网格展示各自的邻居注意力分布.

    avg_attn:        np.ndarray [N, N]
    edge_index:      np.ndarray [2, E]  用于提取边级 scores 画图
    edge_index_full: np.ndarray [2, E]  用于图拓扑可视化（可选）
    focal_nodes:     指定焦点节点列表, 为 None 则自动选高度节点
    """
    avg_attn = to_numpy(avg_attn)
    edge_index = to_numpy(edge_index).astype(int)

    if focal_nodes is None:
        # 自动选: 出度最高的 num_nodes 个测试节点（或用全部）
        degrees = np.bincount(edge_index[0], minlength=avg_attn.shape[0])
        focal_nodes = np.argsort(degrees)[::-1][:num_nodes]

    n_cols = min(3, len(focal_nodes))
    n_rows = int(np.ceil(len(focal_nodes) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for idx, (node, ax) in enumerate(zip(focal_nodes, axes)):
        src_mask = edge_index[0] == node
        dsts = edge_index[1][src_mask]
        scores = avg_attn[node, dsts]

        order = np.argsort(scores)[::-1][:top_k]
        neighbors = dsts[order]
        values = scores[order]

        colors = [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
                  for i in range(len(neighbors))]

        ax.barh(range(len(neighbors)), values[::-1], color=colors[::-1])
        ax.set_yticks(range(len(neighbors)))
        ax.set_yticklabels([f"#{n}" for n in neighbors[::-1]], fontsize=8)
        ax.set_xlim(0, max(values) * 1.2 if len(values) else 1.0)
        ax.invert_yaxis()
        ax.set_xlabel("Attention", fontsize=9)
        self_val = avg_attn[node, node]
        ax.set_title(f"Node #{node}  (self α={self_val:.3f})", fontsize=10)

    # 隐藏多余子图
    for ax in axes[len(focal_nodes):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return save_or_show(fig, save_path)
