"""
Cora 数据集加载。

Cora 引用网络：
    - 2708 个节点（机器学习论文）
    - 1433 维特征（词袋表示）
    - 7 个类别
    - 标准划分：140 train / 500 val / 1000 test
"""
from datasets.base import load_kipf_data
from utils.paths import DATA_DIR


def load_cora(path=None):
    """加载 Cora 数据集。

    参数：
        path: 数据目录路径，默认为 data/Cora/raw

    返回：
        DataDict
    """
    if path is None:
        path = DATA_DIR / "Cora" / "raw"
    return load_kipf_data(str(path), "cora")
