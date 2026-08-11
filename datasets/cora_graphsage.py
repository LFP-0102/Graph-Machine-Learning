import os
import pickle
import torch
import numpy as np
from scipy import sparse

def load_cora_graphsage(root):
    path = os.path.join(root, "Cora", "raw")
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for name in names:
        with open(os.path.join(path, f"ind.cora.{name}"), "rb") as f:
            objects.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objects

    # 特征合并
    features = sparse.vstack((allx, tx))
    features = torch.FloatTensor(features.toarray())

    # 标签
    labels = np.vstack((ally, ty))
    labels = torch.LongTensor(labels.argmax(axis=1))

    # 邻居列表 & 节点划分
    adj_lists = graph
    train_nodes = list(range(ally.shape[0]))
    test_index = np.loadtxt(os.path.join(path, "ind.cora.test.index"), dtype=np.int32)
    test_nodes = list(test_index)
    val_nodes = list(range(ally.shape[0] - 500, ally.shape[0]))

    return {
        "features": features,
        "labels": labels,
        "adj_lists": adj_lists,
        "train_nodes": train_nodes,
        "val_nodes": val_nodes,
        "test_nodes": test_nodes,
    }
