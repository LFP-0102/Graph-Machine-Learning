"""图拓扑结构可视化。"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba

from vis_tool.config import save_or_show, to_numpy

_LAYOUT_FUNCTIONS = {
    "spring": nx.spring_layout, "kamada_kawai": nx.kamada_kawai_layout,
    "circular": nx.circular_layout, "spectral": nx.spectral_layout,
    "random": nx.random_layout, "shell": nx.shell_layout, "spiral": nx.spiral_layout,
}


def plot_graph(
    edge_index: np.ndarray, node_colors: Optional[np.ndarray] = None,
    node_labels: Optional[Sequence[str]] = None, layout: str = "spring",
    figsize: tuple[float, float] = (10, 8), title: str = "Graph Structure",
    save_path: Optional[str] = None, node_size: Union[int, np.ndarray] = 200,
    cmap: str = "tab10", label_max_nodes: int = 500, **layout_kwargs,
) -> plt.Figure:
    """绘制图拓扑，可按节点标签、数值或 RGB(A) 颜色着色。"""
    edge_index = to_numpy(edge_index).astype(int)
    graph = nx.Graph()
    edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]
    graph.add_edges_from(edges)
    nodes = list(graph.nodes())
    num_nodes = graph.number_of_nodes()
    if layout not in _LAYOUT_FUNCTIONS:
        raise ValueError(f"未知的布局方式：{layout}。可选值：{list(_LAYOUT_FUNCTIONS)}")
    if layout == "spring":
        layout_kwargs.setdefault("seed", 42)
    positions = _LAYOUT_FUNCTIONS[layout](graph, **layout_kwargs)

    cmap_used = None
    vmin = vmax = None
    if node_colors is None:
        color_values = "#4C72B0"
    else:
        node_colors = to_numpy(node_colors)
        if node_colors.ndim == 2 and node_colors.shape[1] in (3, 4):
            color_values = [to_rgba(node_colors[i]) for i in range(node_colors.shape[0])]
        elif node_colors.ndim == 1:
            color_values = node_colors.astype(float)
            cmap_used = cmap
            if not (np.issubdtype(node_colors.dtype, np.integer) or len(np.unique(color_values)) <= 20):
                vmin, vmax = float(color_values.min()), float(color_values.max())
        else:
            color_values = "#4C72B0"
    sizes = to_numpy(node_size).ravel() if isinstance(node_size, np.ndarray) else np.full(num_nodes, node_size)
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(graph, positions, node_size=sizes, node_color=color_values,
                           cmap=plt.get_cmap(cmap_used) if cmap_used else None, vmin=vmin, vmax=vmax,
                           alpha=0.9, linewidths=0.5, edgecolors="#333333", ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=edges, width=0.5, alpha=0.3, edge_color="#888888", ax=ax)
    if node_labels is not None and num_nodes <= label_max_nodes:
        labels = {nodes[i]: str(node_labels[i]) for i in range(min(len(nodes), len(node_labels)))}
        nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, font_color="#333333", ax=ax)
    if cmap_used is not None and vmin is not None:
        mapper = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=plt.get_cmap(cmap_used))
        mapper.set_array([])
        fig.colorbar(mapper, ax=ax, shrink=0.7, pad=0.02).set_label("Node Value", fontsize=10)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return save_or_show(fig, save_path)
