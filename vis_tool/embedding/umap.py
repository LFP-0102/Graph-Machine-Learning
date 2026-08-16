"""使用 UMAP 可视化节点嵌入。"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from vis_tool.config import to_numpy
from vis_tool.embedding._common import _plot_embedding_scatter


def plot_umap(
    embeddings: np.ndarray, labels: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Node Embeddings", save_path: Optional[str] = None,
    figsize: tuple[float, float] = (9, 7), n_neighbors: int = 15,
    min_dist: float = 0.1, random_state: int = 42, **kwargs,
) -> plt.Figure:
    """将节点嵌入经 UMAP 降至二维并绘制散点图。"""
    try:
        import umap
    except ImportError as error:
        raise ImportError("使用 UMAP 需要安装 umap-learn，请执行：pip install umap-learn") from error
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist, **kwargs)
    coords = reducer.fit_transform(to_numpy(embeddings))
    return _plot_embedding_scatter(coords, to_numpy(labels).astype(int), class_names, title, save_path, figsize, "UMAP")
