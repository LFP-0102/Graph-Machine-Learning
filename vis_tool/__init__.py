#!/usr/bin/env python
# @File       : __init__.py
# @Path       : vis_tool/__init__.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:31:40
# @Version    : v1.0.0
# @Description:
#   vis_tool — 图机器学习可视化工具包
#
#   提供以下子模块：
#   - config       : 全局样式、配色方案、通用工具函数
#   - curves       : 训练/验证曲线（Loss / Accuracy / …）
#   - embeddings   : 节点嵌入降维可视化（t-SNE / UMAP）
#   - graph_viz    : 图拓扑结构可视化
#   - attention    : 图注意力权重可视化
#   - classification: 混淆矩阵、ROC 曲线、PR 曲线
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建

"""Graph Machine Learning visualisation toolkit."""

from vis_tool.attention import plot_attention_weights
from vis_tool.classification import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
)
from vis_tool.config import (
    save_or_show,
    set_style,
    get_cmap,
    to_numpy,
)
from vis_tool.curves import plot_train_curves
from vis_tool.embeddings import plot_embeddings
from vis_tool.graph_viz import plot_graph

__all__ = [
    # config
    "save_or_show",
    "set_style",
    "get_cmap",
    "to_numpy",
    # curves
    "plot_train_curves",
    # embeddings
    "plot_embeddings",
    # graph_viz
    "plot_graph",
    # attention
    "plot_attention_weights",
    # classification
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
]
