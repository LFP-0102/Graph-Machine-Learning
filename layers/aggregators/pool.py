#!/usr/bin/env python
# @File       : pool.py
# @Path       : layers/ aggregators/pool.py
# @Author     : 刘赋平
# @Date       : 2026-08-09 09:44:18
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/9 | 刘赋平 | v1.0.0 | 初始化创建
import torch
import torch.nn as nn
class PoolAggregator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU()
        )
    def forward(
        self,
        neighbor_features
    ):
        if len(neighbor_features)==0:

            return torch.zeros(
                self.mlp[0].out_features
            )
        h = self.mlp(
            neighbor_features
        )
        h = torch.max(
            h,
            dim=0
        )[0]
        return h
