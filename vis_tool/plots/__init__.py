#!/usr/bin/env python
# @File       : __init__.py
# @Path       : vis_tool/plots/__init__.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:31:40
# @Version    : v1.1.0
# @Description:
#   plots — 训练曲线与分类评估可视化子包
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6  | 刘赋平 | v1.0.0 | 初始化创建
#   2026/8/11 | 刘赋平 | v1.1.0 | 拆分为 train_curve / roc / pr

"""Training curve and evaluation visualisation subpackage."""

from vis_tool.plots.train_curve import plot_train_curves
from vis_tool.plots.roc import plot_roc_curves
from vis_tool.plots.pr import plot_precision_recall_curves

__all__ = [
    "plot_train_curves",
    "plot_roc_curves",
    "plot_precision_recall_curves",
]
