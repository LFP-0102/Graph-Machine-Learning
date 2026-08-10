#!/usr/bin/env python
# @File       : test_graphsage.py
# @Path       : tests/test_graphsage.py
# @Author     : 刘赋平
# @Date       : 2026-08-09 09:28:26
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/9 | 刘赋平 | v1.0.0 | 初始化创建
"""
测试GraphSAGE模型
功能:
1. 加载Cora数据
2. 创建GraphSAGE
3. 测试forward
"""
import torch
from datasets.cora import load_cora
from models.graphsage import GraphSAGE

def test_graphsage():
    # =========================
    # 1. 加载数据
    # =========================
    data = load_cora()
    print("数据:")
    print(data)
    # 节点特征维度
    in_channels = data.x.shape[1]
    # 类别数量
    out_channels = int(
        data.y.max()
    ) + 1
    print(
        "输入特征维度:",
        in_channels
    )
    print(
        "类别数量:",
        out_channels
    )
    # =========================
    # 2. 创建模型
    # =========================
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=out_channels
    )
    print(model)
    # =========================
    # 3. forward测试
    # =========================
    out = model(
        data.x,
        data.edge_index
    )
    print(
        "输出shape:",
        out.shape
    )
    # =========================
    # 4. 检查结果
    # =========================
    assert out.shape == (
        data.num_nodes,
        out_channels
    )
    print(
        "GraphSAGE测试通过!"
    )
if __name__ == "__main__":
    test_graphsage()