#!/usr/bin/env python
# @File       : __init__.py
# @Path       : vis_tool/embedding/__init__.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:31:40
# @Version    : v1.1.0
# @Description:
#   embedding — 节点嵌入降维可视化子包
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6  | 刘赋平 | v1.0.0 | 初始化创建
#   2026/8/11 | 刘赋平 | v1.1.0 | 拆分为 tsne.py / umap.py

"""Node embedding visualisation subpackage."""

from vis_tool.embedding.tsne import plot_tsne
from vis_tool.embedding.umap import plot_umap


def plot_embeddings(embeddings, labels, method="tsne", **kwargs):
    """Dispatch to ``plot_tsne`` or ``plot_umap`` based on *method*.

    This is a convenience wrapper kept for backward compatibility.
    Prefer calling ``plot_tsne()`` / ``plot_umap()`` directly.
    """
    if method.lower() == "umap":
        return plot_umap(embeddings, labels, **kwargs)
    elif method.lower() == "tsne":
        return plot_tsne(embeddings, labels, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'.  Choose 'tsne' or 'umap'.")


__all__ = ["plot_embeddings", "plot_tsne", "plot_umap"]
