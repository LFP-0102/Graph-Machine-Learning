"""
GCN 论文图表复现 + 自定义可视化。

用法：
    PYTHONPATH=. python experiments/visualize_gcn.py

输出：
    outputs/visualizations/gcn/   （所有图表）
    outputs/runs/gcn/             （训练历史 + 结果表）
"""
from experiments.runner import run_experiment
from models.gcn import GCN

if __name__ == "__main__":
    run_experiment(
        model_class=GCN,
        model_name="GCN",
        dataset_name="cora",
        model_kwargs={"hidden_dim": 16, "dropout": 0.5},
        use_normalized_adj=True,     # GCN 用归一化邻接矩阵
        lr=0.01,
        weight_decay=5e-4,
        epochs=200,
        patience=100,
        seed=100,
    )
