#!/usr/bin/env python
# @File       : test_dataset.py.py
# @Path       : tests/test_dataset.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 16:11:48
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建
from datasets.loader import load_dataset


dataset = load_dataset("Cora")


print(dataset)


data = dataset[0]


print(data)
print("----------------")


print("节点数量:")
print(data.num_nodes)


print("特征维度:")
print(data.num_features)


print("类别数量:")
print(dataset.num_classes)
