"""加载 williamleif/GraphSAGE 使用的 JSON/NumPy 输入格式。"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from datasets.base import DataDict


def _read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _node_key(value):
    return str(value)


def _label_matrix(class_map, node_ids):
    labels = [class_map[_node_key(node_id)] for node_id in node_ids]
    if isinstance(labels[0], list):
        return np.asarray(labels, dtype=np.float32)

    num_classes = max(labels) + 1
    result = np.zeros((len(labels), num_classes), dtype=np.float32)
    result[np.arange(len(labels)), labels] = 1.0
    return result


def _sampled_adjacency(num_nodes, neighbors, eligible_nodes, max_degree, rng):
    """构建官方迭代器使用的固定宽度邻接表。"""
    sentinel = num_nodes
    adjacency = np.full((num_nodes + 1, max_degree), sentinel, dtype=np.int64)
    degree = np.zeros(num_nodes, dtype=np.int64)
    for node in eligible_nodes:
        node_neighbors = neighbors[node]
        degree[node] = len(node_neighbors)
        if not node_neighbors:
            continue
        replace = len(node_neighbors) < max_degree
        adjacency[node] = rng.choice(node_neighbors, max_degree, replace=replace)
    return adjacency, degree


def load_graphsage_json(train_prefix, max_degree=128, seed=123):
    """严格按官方文件格式加载监督式 GraphSAGE 数据集。

    ``train_prefix`` 对应 ``-G.json``、``-id_map.json``、
    ``-class_map.json`` 与 ``-feats.npy``。与官方加载器一致，特征只用
    训练节点拟合标准化器。返回的训练邻接表移除了验证/测试节点的消息传递边，
    评估邻接表则保留完整图。
    """
    prefix = Path(train_prefix)
    graph_json = _read_json(Path(f"{prefix}-G.json"))
    id_map_raw = _read_json(Path(f"{prefix}-id_map.json"))
    class_map = _read_json(Path(f"{prefix}-class_map.json"))
    features = np.load(Path(f"{prefix}-feats.npy")).astype(np.float32)
    id_map = {_node_key(key): int(value) for key, value in id_map_raw.items()}
    num_nodes = len(id_map)
    if features.shape[0] != num_nodes:
        raise ValueError("特征数量与 id_map 中的节点数量不一致")

    node_attrs = {
        id_map[_node_key(node["id"])]: node
        for node in graph_json["nodes"]
        if _node_key(node["id"]) in id_map
    }
    if len(node_attrs) != num_nodes:
        raise ValueError("图中的节点与 id_map 不一致")

    train_neighbors = [[] for _ in range(num_nodes)]
    test_neighbors = [[] for _ in range(num_nodes)]
    for edge in graph_json["links"]:
        source = id_map[_node_key(edge["source"])]
        target = id_map[_node_key(edge["target"])]
        test_neighbors[source].append(target)
        test_neighbors[target].append(source)
        source_attrs = node_attrs[source]
        target_attrs = node_attrs[target]
        # 除了标为 train_removed 的边，官方迭代器还会将验证/测试节点
        # 从训练阶段的邻接表中排除。
        if (
            not edge.get("train_removed", False)
            and not source_attrs.get("val", False)
            and not source_attrs.get("test", False)
            and not target_attrs.get("val", False)
            and not target_attrs.get("test", False)
        ):
            train_neighbors[source].append(target)
            train_neighbors[target].append(source)

    train_nodes = [
        node for node, attrs in node_attrs.items()
        if not attrs.get("val", False) and not attrs.get("test", False)
    ]
    val_nodes = [node for node, attrs in node_attrs.items() if attrs.get("val", False)]
    test_nodes = [node for node, attrs in node_attrs.items() if attrs.get("test", False)]

    # graphsage.utils.load_data(normalize=True) 只在训练节点上拟合
    # StandardScaler，再转换所有节点特征；这不同于 Planetoid 加载器的逐行归一化。
    scaler = StandardScaler()
    scaler.fit(features[np.asarray(train_nodes, dtype=np.int64)])
    features = scaler.transform(features).astype(np.float32)

    rng = np.random.RandomState(seed)
    train_adj, train_degree = _sampled_adjacency(
        num_nodes, train_neighbors, train_nodes, max_degree, rng
    )
    test_adj, _ = _sampled_adjacency(
        num_nodes, test_neighbors, range(num_nodes), max_degree, rng
    )
    train_nodes = [node for node in train_nodes if train_degree[node] > 0]

    node_ids = [None] * num_nodes
    for raw_id, index in id_map.items():
        node_ids[index] = raw_id
    labels = _label_matrix(class_map, node_ids)
    padded_features = np.vstack([features, np.zeros((1, features.shape[1]), dtype=np.float32)])

    return DataDict(
        features=torch.from_numpy(padded_features),
        labels=torch.from_numpy(labels),
        train_adj=torch.from_numpy(train_adj),
        test_adj=torch.from_numpy(test_adj),
        train_nodes=torch.tensor(train_nodes, dtype=torch.long),
        val_nodes=torch.tensor(val_nodes, dtype=torch.long),
        test_nodes=torch.tensor(test_nodes, dtype=torch.long),
        num_features=features.shape[1],
        num_classes=labels.shape[1],
        num_nodes=num_nodes,
        max_degree=max_degree,
    )
