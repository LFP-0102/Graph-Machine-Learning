#!/usr/bin/env python
# @File       : graph_conv.py.py
# @Path       : layers/graph_conv.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 15:33:46
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建
import torch
from torch import nn


class GraphConv(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim
    ):
        super().__init__()

        self.linear = nn.Linear(
            in_dim,
            out_dim
        )


    def forward(
        self,
        x,
        adj
    ):

        x = torch.matmul(
            adj,
            x
        )

        x = self.linear(x)

        return x
