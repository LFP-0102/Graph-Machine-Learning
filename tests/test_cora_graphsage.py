#!/usr/bin/env python
# @File       : test_cora_graphsage.py
# @Path       : tests/test_cora_graphsage.py
# @Author     : 刘赋平
# @Date       : 2026-08-09 09:36:32
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/9 | 刘赋平 | v1.0.0 | 初始化创建
from datasets.cora_graphsage import load_cora_graphsage
from utils.paths import DATA_DIR


data = load_cora_graphsage(
    DATA_DIR
)

print(data["features"].shape)

print(data["labels"].shape)

print(data["adj_lists"][0])

print(len(data["train_nodes"]))
