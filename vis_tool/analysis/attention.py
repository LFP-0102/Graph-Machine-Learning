"""GAT 注意力权重可视化。"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from vis_tool.config import CATEGORICAL_PALETTE, save_or_show, to_numpy


def plot_attention_weights(
    edge_index: np.ndarray, attention_weights: np.ndarray,
    node_pos: Optional[Mapping[int, Tuple[float, float]]] = None, node_size: int = 300,
    head_idx: Optional[int] = None, title: str = "Attention Weights", save_path: Optional[str] = None,
    figsize: tuple[float, float] = (10, 8), cmap: str = "YlOrRd",
    edge_vmin: float = 0.0, edge_vmax: float = 1.0, node_color: str = "#4C72B0",
    edge_scale: float = 3.0,
) -> plt.Figure:
    """按边的注意力权重绘制图结构，权重同时控制边的颜色与宽度。"""
    edge_index, attention = to_numpy(edge_index).astype(int), to_numpy(attention_weights)
    if attention.ndim == 2 and attention.shape[1] > 1:
        attention = attention[:, head_idx] if head_idx is not None else attention.mean(axis=1)
    attention = attention.ravel()
    num_edges = edge_index.shape[1]
    if len(attention) != num_edges:
        raise ValueError(f"边数量不一致：edge_index 有 {num_edges} 条边，attention_weights 有 {len(attention)} 个值。")
    graph = nx.Graph()
    edges = []
    for index in range(num_edges):
        source, target = int(edge_index[0, index]), int(edge_index[1, index])
        graph.add_edge(source, target, weight=float(attention[index]))
        edges.append((source, target))
    positions = nx.spring_layout(graph, seed=42, k=0.3, iterations=50) if node_pos is None else {int(key): tuple(value) for key, value in node_pos.items()}
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(graph, positions, node_size=node_size, node_color=node_color, alpha=0.9, linewidths=1, edgecolors="white", ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=edges, width=[edge_scale * float(value) + 0.5 for value in attention],
                           edge_color=attention.tolist(), edge_cmap=plt.get_cmap(cmap), edge_vmin=edge_vmin,
                           edge_vmax=edge_vmax, alpha=0.7, ax=ax)
    mapper = ScalarMappable(norm=Normalize(vmin=edge_vmin, vmax=edge_vmax), cmap=plt.get_cmap(cmap))
    mapper.set_array([])
    fig.colorbar(mapper, ax=ax, shrink=0.7, pad=0.02).set_label("Attention Weight", fontsize=10)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return save_or_show(fig, save_path)


def _dense_attn_to_edge_scores(avg_attn, edge_index):
    """从稠密注意力矩阵 ``[N, N]`` 提取每条边的注意力分数。"""
    return avg_attn[edge_index[0], edge_index[1]]


def plot_node_attention_bars(
    avg_attn, edge_index, focal_node, top_k=8, title=None, save_path=None,
    figsize=(10, 5), class_names=None, node_labels=None,
):
    """绘制焦点节点对其邻居的 Top-k 注意力权重条形图。"""
    avg_attn, edge_index = to_numpy(avg_attn), to_numpy(edge_index).astype(int)
    destinations = edge_index[1][edge_index[0] == focal_node]
    scores = avg_attn[focal_node, destinations]
    order = np.argsort(scores)[::-1]
    if top_k > 0:
        order = order[:top_k]
    neighbors, values = destinations[order], scores[order]
    if node_labels is not None:
        labels = [f"#{node} {node_labels[node]}" if node < len(node_labels) else f"Node {node}" for node in neighbors]
    elif class_names is not None:
        labels = [f"Node {node}" for node in neighbors]
    else:
        labels = [f"Neighbor {node}" for node in neighbors]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(neighbors)), values[::-1], color=[CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i in range(len(neighbors))][::-1])
    ax.set_yticks(range(len(neighbors)), labels[::-1])
    ax.set_xlabel("Attention Weight")
    ax.set_xlim(0, max(values) * 1.2 if len(values) else 1.0)
    ax.invert_yaxis()
    for bar, value in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    self_value = avg_attn[focal_node, focal_node]
    ax.set_title(title or f"Node #{focal_node} Top-{len(neighbors)} Neighbor Attention (self: {self_value:.3f})")
    fig.tight_layout()
    return save_or_show(fig, save_path)


def plot_attention_summary(
    avg_attn, edge_index, edge_index_full=None, focal_nodes=None, num_nodes=3, top_k=8,
    title="GAT Multi-Node Attention Summary", save_path=None, class_names=None,
):
    """以子图形式展示多个焦点节点的邻居注意力分布。"""
    avg_attn, edge_index = to_numpy(avg_attn), to_numpy(edge_index).astype(int)
    if focal_nodes is None:
        degree = np.bincount(edge_index[0], minlength=avg_attn.shape[0])
        focal_nodes = np.argsort(degree)[::-1][:num_nodes]
    if len(focal_nodes) == 0:
        raise ValueError("焦点节点列表不能为空。")
    columns = min(3, len(focal_nodes))
    rows = int(np.ceil(len(focal_nodes) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for node, axis in zip(focal_nodes, axes):
        destinations = edge_index[1][edge_index[0] == node]
        values = avg_attn[node, destinations]
        order = np.argsort(values)[::-1][:top_k]
        neighbors, values = destinations[order], values[order]
        axis.barh(range(len(neighbors)), values[::-1], color=[CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i in range(len(neighbors))][::-1])
        axis.set_yticks(range(len(neighbors)), [f"#{value}" for value in neighbors[::-1]], fontsize=8)
        axis.set_xlim(0, max(values) * 1.2 if len(values) else 1.0)
        axis.invert_yaxis()
        axis.set_xlabel("Attention", fontsize=9)
        axis.set_title(f"Node #{node} (self α={avg_attn[node, node]:.3f})", fontsize=10)
    for axis in axes[len(focal_nodes):]:
        axis.set_visible(False)
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return save_or_show(fig, save_path)
