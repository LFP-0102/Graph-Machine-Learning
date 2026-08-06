#!/usr/bin/env python
# @File       : result_table.py
# @Path       : vis_tool/ statistics/result_table.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 15:44:24
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建
import pandas as pd


def create_result_table(results):

    """
    results格式:

    [
        {
            "model":"GCN",
            "dataset":"Cora",
            "accuracy":0.815
        },

        {
            "model":"GAT",
            "dataset":"Cora",
            "accuracy":0.832
        }
    ]

    """

    df = pd.DataFrame(results)

    return df



def save_result_table(
        results,
        path="./outputs/results.csv"
):

    df = create_result_table(results)

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Results saved to {path}"
    )