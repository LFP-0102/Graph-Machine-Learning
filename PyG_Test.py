#!/usr/bin/env python
# @File       : PyG_Test.py
# @Path       : /PyG_Test.py
# @Author     : 刘赋平
# @Date       : 2026-08-03 10:49:56
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/3 | 刘赋平 | v1.0.0 | 初始化创建
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


# 加载Cora数据集
dataset = Planetoid(
    root="./data",
    name="Cora"
)

data = dataset[0]


class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(
            dataset.num_features,
            16
        )

        self.conv2 = GCNConv(
            16,
            dataset.num_classes
        )


    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x



# 使用GPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


model = GCN().to(device)
data = data.to(device)


out = model(
    data.x,
    data.edge_index
)


print("数据集:", dataset)
print("节点数:", data.num_nodes)
print("边数:", data.num_edges)
print("输出:", out.shape)
print("设备:", out.device)