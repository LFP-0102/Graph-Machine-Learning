"""
通用 Kipf 格式 pickle 数据加载器。

支持 Cora / CiteSeer / PubMed 三个 Planetoid 引用网络数据集。
每个数据集返回统一的 DataDict 格式。
"""
import pickle
import warnings
import numpy as np
import torch

# scipy.sparse 矩阵 pickle 反序列化时产生无害的 VisibleDeprecationWarning
warnings.filterwarnings("ignore", message=".*align.*")

# ---------------------------------------------------------------------------
# 各数据集的标准 Planetoid 划分
# ---------------------------------------------------------------------------
_SPLITS = {
    "cora":     {"n_train": 140, "n_val": 500},
    "citeseer": {"n_train": 120, "n_val": 500},
    "pubmed":   {"n_train":  60, "n_val": 500},
}


class DataDict(dict):
    """统一数据容器。

    支持字典和属性两种访问方式：
        data["features"]  ←→  data.features

    包含的字段：
        features:   Tensor [N, D]  节点特征
        labels:     Tensor [N]     整数标签 (0 .. C-1)
        adj:        Tensor [N, N]  原始邻接矩阵（无自环，未归一化）
        train_mask: Tensor [N]     bool
        val_mask:   Tensor [N]     bool
        test_mask:  Tensor [N]     bool
        num_classes: int
        num_features: int
    """

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'DataDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def load_kipf_data(root, dataset_name):
    """加载 Kipf 格式的 Planetoid 数据集。

    读取 7 个 pickle 文件：
        ind.{name}.x, .y, .tx, .ty, .allx, .ally, .graph

    参数：
        root:         数据目录路径（包含上述 7 个文件的目录）
        dataset_name: "cora" | "citeseer" | "pubmed"

    返回：
        DataDict
    """
    name = dataset_name.lower()

    # ── 1. 读取 pickle 文件 ──────────────────────────────────
    filenames = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]
    objects = []
    for fname in filenames:
        path = f"{root}/ind.{name}.{fname}"
        with open(path, "rb") as f:
            if fname == "test.index":
                # test.index 是文本文件，每行一个整数
                objects.append([int(line.strip()) for line in f])
            else:
                objects.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph, test_idx = objects

    # ── 2. 合并特征矩阵 ─────────────────────────────────────
    # allx: scipy.sparse.csr_matrix (train+val 节点)
    # tx:   scipy.sparse.csr_matrix (test 节点, 按 test.index 顺序)
    features = np.vstack((allx.toarray(), tx.toarray()))          # [N, D]

    # ── 3. 重排序 test 节点 ─────────────────────────────────
    # test.index[i] 指定 tx[i] 在原始图中的真实索引
    # 将 tx[i] 放回图中位置 test_idx[i]
    features[test_idx] = tx.toarray()

    # ── 4. 合并标签 ─────────────────────────────────────────
    # ally, ty: numpy one-hot
    labels = np.vstack((ally, ty))                                # [N, C]
    labels[test_idx] = ty
    labels = np.argmax(labels, axis=1)                            # [N]

    # ── 4. 构造邻接矩阵 & edge_index ──────────────────────────
    num_nodes = len(graph)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    edges = set()
    for i in graph:
        for j in graph[i]:
            adj[i][j] = 1.0
            edges.add((i, j))
            edges.add((j, i))                     # 无向边

    edge_index = torch.tensor(
        list(edges), dtype=torch.long
    ).t().contiguous()                            # [2, E]

    # ── 5. 划分 train / val / test mask ─────────────────────
    split = _SPLITS[name]
    n_train = split["n_train"]
    n_val = split["n_val"]
    n_allx = allx.shape[0]                      # 带标签节点数（train+val 在其中）
    n_test = tx.shape[0]                        # 测试节点数

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[test_idx] = True

    # ── 6. 组装返回 ────────────────────────────────────────
    return DataDict(
        features=torch.FloatTensor(features),
        labels=torch.LongTensor(labels),
        adj=torch.FloatTensor(adj),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
        edge_index=edge_index,
        num_classes=int(labels.max() + 1),
        num_features=features.shape[1],
    )


# ── 数据集名称 → 目录名映射 ────────────────────────────────
_DATASET_DIR = {
    "cora":     "Cora",
    "citeseer": "CiteSeer",
    "pubmed":   "PubMed",
}


def load_dataset(name):
    """统一数据集加载入口。

    参数：
        name: "cora" | "citeseer" | "pubmed"

    返回：
        DataDict
    """
    from utils.paths import DATA_DIR

    normalized = name.lower()
    if normalized in _DATASET_DIR:
        return load_kipf_data(
            DATA_DIR / _DATASET_DIR[normalized] / "raw", normalized
        )

    raise ValueError(
        f"不支持的数据集: '{name}'。可选: {list(_DATASET_DIR.keys())}"
    )
