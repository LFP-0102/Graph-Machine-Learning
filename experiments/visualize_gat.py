"""
GAT 论文图表复现 + 自定义可视化。

用法：
    PYTHONPATH=. python experiments/visualize_gat.py

输出：
    outputs/visualizations/gat/   （所有图表）
    outputs/runs/gat/             （训练历史 + 结果表）
"""
from experiments.runner import run_experiment
from models.gat import GAT

if __name__ == "__main__":
    run_experiment(
        model_class=GAT,
        model_name="GAT",
        dataset_name="cora",
        model_kwargs={"hidden_dim": 8, "n_heads": 8, "dropout": 0.6},
        use_normalized_adj=False,    # GAT 用二值邻接矩阵
        lr=0.01,
        weight_decay=5e-4,
        epochs=200,
        seed=42,
    )
