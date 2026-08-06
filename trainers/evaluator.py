#!/usr/bin/env python
# @File       : evaluator.py
# @Path       : trainers/evaluator.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 15:34:41
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建
import torch


def accuracy(
    pred,
    label
):

    return (
        pred == label
    ).sum().item() / len(label)



def evaluate(
    model,
    data
):

    model.eval()


    with torch.no_grad():

        out = model(
            data.x,
            data.edge_index
        )


    pred = out.argmax(dim=1)


    acc = accuracy(
        pred[data.test_mask],
        data.y[data.test_mask]
    )


    return acc