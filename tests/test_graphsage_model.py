#!/usr/bin/env python
# @File       : test_graphsage_model.py
# @Path       : tests/test_graphsage_model.py
# @Author     : 刘赋平
# @Date       : 2026-08-09 09:53:54
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/9 | 刘赋平 | v1.0.0 | 初始化创建
from datasets.cora_graphsage import load_cora_graphsage

from layers.sampler import NeighborSampler

from layers.aggregators.mean import MeanAggregator

from models.graphsage import GraphSAGE

from utils.paths import DATA_DIR



# =====================
# 加载数据
# =====================

data = load_cora_graphsage(
    str(DATA_DIR)
)


features = data["features"]

adj_lists = data["adj_lists"]



# =====================
# 创建 sampler
# =====================

sampler1 = NeighborSampler(
    adj_lists,
    num_samples=5
)


sampler2 = NeighborSampler(
    adj_lists,
    num_samples=5
)



# =====================
# 创建 aggregator
# =====================

agg1 = MeanAggregator(
    input_dim=1433
)


agg2 = MeanAggregator(
    input_dim=64
)



# =====================
# 创建模型
# =====================

model = GraphSAGE(

    input_dim=1433,

    hidden_dim=64,

    output_dim=7,

    aggregator1=agg1,

    aggregator2=agg2,

    sampler1=sampler1,

    sampler2=sampler2

)



# =====================
# 测试forward
# =====================


nodes = [
    0,
    1,
    2,
    3
]


output = model(

    nodes,

    features,

    adj_lists

)


print(
    "输出:"
)

print(output.shape)
