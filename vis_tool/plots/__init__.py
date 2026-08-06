#!/usr/bin/env python
# @File       : __init__.py
# @Path       : vis_tool/plots/__init__.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:31:40
# @Version    : v1.0.0
# @Description:
#   plots — 训练曲线可视化子包
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建

"""Training curve visualisation subpackage."""

from vis_tool.plots.curves import plot_train_curves

__all__ = ["plot_train_curves"]
