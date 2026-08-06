#!/usr/bin/env python
# @File       : citation.py
# @Path       : datasets/citation.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 15:30:21
# @Version    : v1.0.0
# @Description:
#   加载 Planetoid 引用网络数据集（Cora / CiteSeer / PubMed）。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建
from torch_geometric.datasets import Planetoid

from utils.paths import DATA_DIR


def load_citation(name):
    """Load a citation network dataset (Cora, CiteSeer, PubMed).

    注意：PyG 的 Planetoid 会自动在 root 后追加 name 作为子目录，
    因此 root 只需要指向 data/ 目录，raw_dir 会解析为 data/{name}/raw/。
    """
    dataset = Planetoid(
        root=str(DATA_DIR),
        name=name,
    )
    return dataset
