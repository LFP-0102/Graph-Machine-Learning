"""供 GCN/GAT 复现使用的 Planetoid 引文网络数据加载器。"""
import pickle
from pathlib import Path

import numpy as np
import torch


_SPLITS = {
    "cora": {"n_train": 140, "n_val": 500},
    "citeseer": {"n_train": 120, "n_val": 500},
    "pubmed": {"n_train": 60, "n_val": 500},
}

_DATASET_DIR = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


class DataDict(dict):
    """支持属性访问的字典数据容器。"""

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"DataDict 中不存在属性：'{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def row_normalize(features):
    """执行 Planetoid GCN/GAT 基线使用的逐行归一化。"""
    row_sum = np.asarray(features.sum(axis=1), dtype=np.float32).reshape(-1)
    inv = np.zeros_like(row_sum)
    nonzero = row_sum != 0
    inv[nonzero] = 1.0 / row_sum[nonzero]
    return features * inv[:, None]


def _expand_test_block(values, test_idx_range):
    """为 CiteSeer 缺失的测试节点编号补充全零行。"""
    start, stop = int(test_idx_range[0]), int(test_idx_range[-1])
    full_range = np.arange(start, stop + 1, dtype=np.int64)
    dense = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    if len(full_range) == len(test_idx_range):
        return dense, full_range

    expanded = np.zeros((len(full_range), dense.shape[1]), dtype=dense.dtype)
    expanded[test_idx_range - start] = dense
    return expanded, full_range


def restore_planetoid_test_rows(allx, tx, test_idx_reorder, num_nodes):
    """恢复 ``tx`` 测试行在 Planetoid 数据中的原始节点顺序。

    数据文件的 ``tx`` 按节点编号排序，但 ``test.index`` 被刻意打乱。
    直接按 ``test.index`` 写入会造成测试节点与特征、标签错位。
    """
    test_idx_reorder = np.asarray(test_idx_reorder, dtype=np.int64)
    test_idx_range = np.sort(test_idx_reorder)
    tx_dense, test_idx_range_full = _expand_test_block(tx, test_idx_range)
    features = np.zeros((num_nodes, allx.shape[1]), dtype=np.float32)
    features[:allx.shape[0]] = allx.toarray()
    tx_start = allx.shape[0]
    features[tx_start:tx_start + len(tx_dense)] = tx_dense
    features[test_idx_reorder] = features[test_idx_range]
    return features


def _load_raw_objects(root, name):
    objects = []
    for suffix in ("x", "y", "tx", "ty", "allx", "ally", "graph"):
        with (root / f"ind.{name}.{suffix}").open("rb") as handle:
            objects.append(pickle.load(handle, encoding="latin1"))
    with (root / f"ind.{name}.test.index").open("r", encoding="utf-8") as handle:
        objects.append([int(line.strip()) for line in handle if line.strip()])
    return objects


def load_kipf_data(root, dataset_name):
    """正确加载 Planetoid 格式的 Cora、CiteSeer 或 PubMed 数据集。"""
    name = dataset_name.lower()
    if name not in _SPLITS:
        raise ValueError(f"不支持的数据集：{dataset_name}")

    root = Path(root)
    x, y, tx, ty, allx, ally, graph_dict, test_idx = _load_raw_objects(root, name)
    num_nodes = len(graph_dict)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    test_idx_range = np.sort(test_idx)

    features = row_normalize(
        restore_planetoid_test_rows(allx, tx, test_idx, num_nodes)
    )

    ty_dense, _ = _expand_test_block(ty, test_idx_range)
    labels_one_hot = np.zeros((num_nodes, ally.shape[1]), dtype=np.float32)
    labels_one_hot[:ally.shape[0]] = ally
    ty_start = ally.shape[0]
    labels_one_hot[ty_start:ty_start + len(ty_dense)] = ty_dense
    labels_one_hot[test_idx] = labels_one_hot[test_idx_range]
    labels = labels_one_hot.argmax(axis=1)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    edges = set()
    for source, neighbors in graph_dict.items():
        for target in neighbors:
            adj[source, target] = 1.0
            adj[target, source] = 1.0
            edges.add((source, target))
            edges.add((target, source))
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()

    split = _SPLITS[name]
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[:split["n_train"]] = True
    val_start = split["n_train"]
    val_mask[val_start:val_start + split["n_val"]] = True
    test_mask[test_idx] = True

    return DataDict(
        features=torch.from_numpy(features),
        labels=torch.from_numpy(labels).long(),
        adj=torch.from_numpy(adj),
        edge_index=edge_index,
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
        num_classes=int(labels.max() + 1),
        num_features=features.shape[1],
    )


def load_dataset(name):
    """加载项目内的一个 Planetoid 引文网络数据集。"""
    from utils.paths import DATA_DIR

    normalized = name.lower()
    if normalized not in _DATASET_DIR:
        raise ValueError(f"不支持的数据集：{name}。可选值：{list(_DATASET_DIR)}")
    return load_kipf_data(DATA_DIR / _DATASET_DIR[normalized] / "raw", normalized)
