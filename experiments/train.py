#!/usr/bin/env python
# @File       : train.py
# @Path       : experiments/train.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 15:37:05
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建
import torch


from datasets.loader import load_dataset

from models.gcn import GCN

from trainers.trainer import Trainer



dataset = load_dataset(
    "Cora"
)


data = dataset[0]


model = GCN(
    dataset.num_features,
    16,
    dataset.num_classes
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


trainer = Trainer(
    model,
    data,
    optimizer
)


trainer.train(
    200
)
