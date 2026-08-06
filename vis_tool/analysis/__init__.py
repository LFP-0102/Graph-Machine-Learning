#!/usr/bin/env python
# @File       : __init__.py
# @Path       : vis_tool/analysis/__init__.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:31:40
# @Version    : v1.0.0
# @Description:
#   analysis — 图注意力与分类评估可视化子包
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建

"""Attention and classification visualisation subpackage."""

from vis_tool.analysis.attention import plot_attention_weights
from vis_tool.analysis.classification import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
)

__all__ = [
    "plot_attention_weights",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
]
