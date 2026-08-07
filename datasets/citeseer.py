"""
CiteSeer 数据集加载。

CiteSeer 引用网络：
    - 3327 个节点（计算机科学论文）
    - 3703 维特征（词袋表示）
    - 6 个类别
    - 标准划分：120 train / 500 val / 1000 test
"""
from datasets.base import load_kipf_data
from utils.paths import DATA_DIR


def load_citeseer(path=None):
    """加载 CiteSeer 数据集。

    参数：
        path: 数据目录路径，默认为 data/CiteSeer/raw

    返回：
        DataDict
    """
    if path is None:
        path = DATA_DIR / "CiteSeer" / "raw"
    return load_kipf_data(str(path), "citeseer")
