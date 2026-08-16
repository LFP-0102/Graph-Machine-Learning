"""兼容早期 GraphSAGE Cora 数据格式的加载函数。"""

import os
import pickle

import numpy as np
import torch
from scipy import sparse


def load_cora_graphsage(root):
    """读取 Cora 的 Planetoid 文件，返回早期采样器所需的邻居字典与节点划分。"""
    path = os.path.join(root, "Cora", "raw")
    objects = []
    for name in ["x", "y", "tx", "ty", "allx", "ally", "graph"]:
        with open(os.path.join(path, f"ind.cora.{name}"), "rb") as file:
            objects.append(pickle.load(file, encoding="latin1"))
    _, _, tx, ty, allx, ally, graph = objects
    features = torch.FloatTensor(sparse.vstack((allx, tx)).toarray())
    labels = torch.LongTensor(np.vstack((ally, ty)).argmax(axis=1))
    test_index = np.loadtxt(os.path.join(path, "ind.cora.test.index"), dtype=np.int32)
    return {
        "features": features, "labels": labels, "adj_lists": graph,
        "train_nodes": list(range(ally.shape[0])),
        "val_nodes": list(range(ally.shape[0] - 500, ally.shape[0])),
        "test_nodes": list(test_index),
    }
