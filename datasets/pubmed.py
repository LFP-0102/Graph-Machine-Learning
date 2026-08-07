"""
PubMed 数据集加载。

PubMed 引用网络：
    - 19717 个节点（糖尿病相关论文）
    - 500 维特征（TF-IDF 词袋表示）
    - 3 个类别
    - 标准划分：60 train / 500 val / 1000 test
"""
from datasets.base import load_kipf_data
from utils.paths import DATA_DIR


def load_pubmed(path=None):
    """加载 PubMed 数据集。

    参数：
        path: 数据目录路径，默认为 data/PubMed/raw

    返回：
        DataDict
    """
    if path is None:
        path = DATA_DIR / "PubMed" / "raw"
    return load_kipf_data(str(path), "pubmed")
