#!/usr/bin/env python
# @File       : __init__.py
# @Path       : vis_tool/__init__.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:31:40
# @Version    : v1.0.0
# @Description:
#   vis_tool — 图机器学习可视化工具包
#
#   提供以下子包：
#   - config      : 全局样式、配色方案、通用工具函数
#   - graph       : 图拓扑结构可视化
#   - embedding   : 节点嵌入降维可视化（t-SNE / UMAP）
#   - analysis    : 注意力权重可视化 / 分类评估（混淆矩阵、ROC、PR）
#   - plots       : 训练/验证曲线（Loss / Accuracy / …）
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建

"""Graph Machine Learning visualisation toolkit."""

from vis_tool.analysis import (
    plot_attention_weights,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from vis_tool.config import (
    get_cmap,
    save_or_show,
    set_style,
    to_numpy,
)
from vis_tool.embedding import plot_embeddings
from vis_tool.graph import plot_graph
from vis_tool.plots import plot_train_curves

__all__ = [
    # config
    "save_or_show",
    "set_style",
    "get_cmap",
    "to_numpy",
    # plots
    "plot_train_curves",
    # embedding
    "plot_embeddings",
    # graph
    "plot_graph",
    # analysis
    "plot_attention_weights",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
]
