#!/usr/bin/env python
# @File       : trainer.py.py
# @Path       : trainers/trainer.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 15:34:31
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建
import torch


class Trainer:


    def __init__(
        self,
        model,
        data,
        optimizer
    ):

        self.model = model
        self.data = data
        self.optimizer = optimizer



    def train_epoch(self):

        self.model.train()

        self.optimizer.zero_grad()


        out = self.model(
            self.data.x,
            self.data.edge_index
        )


        loss = torch.nn.functional.cross_entropy(
            out[self.data.train_mask],
            self.data.y[self.data.train_mask]
        )


        loss.backward()

        self.optimizer.step()


        return loss.item()



    def train(
        self,
        epochs
    ):

        for epoch in range(epochs):

            loss = self.train_epoch()


            if epoch % 10 == 0:

                print(
                    epoch,
                    loss
                )
