#!/usr/bin/env python
# @File       : graph_utils.py
# @Path       : utils/graph_utils.py
# @Author     : 刘赋平
# @Date       : 2026-08-07 12:35:31
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/7 | 刘赋平 | v1.0.0 | 初始化创建
import numpy as np
def normalize_adj(adj):
    """
    计算GCN需要的归一化邻接矩阵
    论文公式：
    A_hat =
    D^(-1/2)(A+I)D^(-1/2)
    参数:
    adj:
        原始邻接矩阵
    返回:
    A_hat:
        归一化后的邻接矩阵
    """
    # --------------------------------
    # 1. 添加自环
    # --------------------------------
    # 原始邻接矩阵:
    # A
    # 加入单位矩阵:
    # A+I
    # 表示节点也接收自己的信息
    adj=adj+np.eye(
        adj.shape[0]
    )
    # --------------------------------
    # 2. 计算节点度
    # --------------------------------
    # 度:
    # d_i=sum_j A_ij
    degree=np.array(
        adj.sum(1)
    )
    # --------------------------------
    # 3. 计算D^(-1/2)
    # --------------------------------
    degree_inv=np.power(
        degree,
        -0.5
    )
    # 防止孤立节点导致:
    # 1/0 = inf
    degree_inv[
        np.isinf(degree_inv)
    ]=0
    # 构造:
    # D^(-1/2)
    D=np.diag(
        degree_inv
    )
    # --------------------------------
    # 4. 得到最终GCN邻接矩阵
    # --------------------------------
    # A_hat=D^(-1/2)*(A+I)*D^(-1/2)
    A_hat=D@adj@D
    return A_hat


def add_self_loops(adj):
    """
    添加自环 A+I，用于 GAT 等需要原始二值邻接矩阵的模型。

    参数:
        adj: 原始邻接矩阵 [N, N] (numpy)
    返回:
        adj_loop: 添加自环后的二值矩阵 [N, N] (numpy)
    """
    adj = adj + np.eye(adj.shape[0])
    return (adj > 0).astype(np.float32)