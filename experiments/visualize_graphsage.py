#!/usr/bin/env python
# @File       : visualize_graphsage.py
# @Path       : experiments/visualize_graphsage.py
# @Author     : 刘赋平
# @Date       : 2026-08-09 10:16:37
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/9 | 刘赋平 | v1.0.0 | 初始化创建
"""
GraphSAGE 论文图表复现 + 自定义可视化。

用法：
PYTHONPATH=. python experiments/visualize_graphsage.py

输出：
outputs/visualizations/graphsage/
outputs/runs/graphsage/
"""

from experiments.runner import run_experiment
from models.graphsage import GraphSAGE


if __name__ == "__main__":

    run_experiment(
        model_class=GraphSAGE,
        model_name="GraphSAGE",
        dataset_name="cora",

        model_kwargs={
            "hidden_dim": 16,
            "dropout": 0.5,
        },

        use_normalized_adj=False,

        lr=0.01,
        weight_decay=5e-4,

        epochs=200,
        patience=100,

        seed=42,
    )