"""使用 t-SNE 可视化节点嵌入。"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from vis_tool.config import to_numpy
from vis_tool.embedding._common import _plot_embedding_scatter


def _stratified_sample_indices(labels: np.ndarray, max_samples: int, random_state: int) -> np.ndarray:
    """Return a reproducible class-balanced sample without replacing nodes."""
    if len(labels) <= max_samples:
        return np.arange(len(labels))
    generator = np.random.default_rng(random_state)
    classes, counts = np.unique(labels, return_counts=True)
    quotas = np.maximum(1, np.floor(max_samples * counts / counts.sum()).astype(int))
    quotas = np.minimum(quotas, counts)
    while quotas.sum() < max_samples:
        candidates = np.where(quotas < counts)[0]
        if not len(candidates):
            break
        index = candidates[np.argmax(counts[candidates] - quotas[candidates])]
        quotas[index] += 1
    indices = [
        generator.choice(np.flatnonzero(labels == label), size=quota, replace=False)
        for label, quota in zip(classes, quotas)
    ]
    return np.sort(np.concatenate(indices))


def plot_tsne(
    embeddings: np.ndarray, labels: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Node Embeddings", save_path: Optional[str] = None,
    figsize: tuple[float, float] = (9, 7), perplexity: float = 30,
    max_samples: Optional[int] = 3000,
    random_state: int = 42, **kwargs,
) -> plt.Figure:
    """将节点嵌入经 t-SNE 降至二维并绘制散点图。"""
    values = to_numpy(embeddings)
    label_values = to_numpy(labels).astype(int)
    if len(values) < 2:
        raise ValueError("t-SNE requires at least two node embeddings.")
    if max_samples is not None and max_samples < 2:
        raise ValueError("max_samples must be at least 2 when provided.")
    indices = _stratified_sample_indices(label_values, max_samples, random_state) if max_samples else np.arange(len(values))
    sampled_values, sampled_labels = values[indices], label_values[indices]
    effective_perplexity = min(perplexity, len(sampled_values) - 1)
    reducer = TSNE(n_components=2, random_state=random_state, perplexity=effective_perplexity, **kwargs)
    coords = reducer.fit_transform(sampled_values)
    if len(indices) < len(values):
        title = f"{title} (stratified sample: {len(indices)}/{len(values)} nodes)"
    return _plot_embedding_scatter(coords, sampled_labels, class_names, title, save_path, figsize, "t-SNE")
