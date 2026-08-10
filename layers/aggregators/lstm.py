#!/usr/bin/env python
# @File       : lstm.py
# @Path       : layers/ aggregators/lstm.py
# @Author     : 刘赋平
# @Date       : 2026-08-09 09:45:04
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/9 | 刘赋平 | v1.0.0 | 初始化创建
import torch
import torch.nn as nn

class LSTMAggregator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )
    def forward(
        self,
        neighbor_features
    ):
        if len(neighbor_features)==0:
            return torch.zeros(
                self.lstm.hidden_size
            )
        # 增加batch维
        x = neighbor_features.unsqueeze(0)
        _,(h,c)=self.lstm(x)
        return h.squeeze(0)