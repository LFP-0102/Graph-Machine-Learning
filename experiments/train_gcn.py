#!/usr/bin/env python3
"""Train a GCN model on the Cora dataset."""

import torch
from torch_geometric.datasets import Planetoid

import sys
sys.path.insert(0, "..")

from models.gcn import GCN


def main():
    # Load Cora dataset
    dataset = Planetoid(root="../data", name="Cora")
    data = dataset[0]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GCN(dataset.num_features, dataset.num_classes).to(device)
    data = data.to(device)

    out = model(data.x, data.edge_index)

    print("数据集:", dataset)
    print("节点数:", data.num_nodes)
    print("边数:", data.num_edges)
    print("输出维度:", out.shape)
    print("设备:", out.device)


if __name__ == "__main__":
    main()
