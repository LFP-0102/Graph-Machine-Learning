#!/usr/bin/env python
# @File       : __init__.py
# @Path       : vis_tool/embedding/__init__.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 12:31:40
# @Version    : v1.0.0
# @Description:
#   embedding — 节点嵌入降维可视化子包
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建

"""Node embedding visualisation subpackage."""

from vis_tool.embedding.embeddings import plot_embeddings

__all__ = ["plot_embeddings"]
