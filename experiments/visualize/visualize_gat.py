"""
GAT 论文图表复现 + 自定义可视化。

用法：
    PYTHONPATH=. python experiments/visualize/visualize_gat.py

输出：
    outputs/visualizations/gat/
    outputs/runs/gat/
"""
from experiments.runner import run_experiment
from models.gat import GAT

if __name__ == "__main__":
    run_experiment(
        model_class=GAT,
        model_name="GAT",
        dataset_name="cora",
        model_kwargs={"hidden_dim": 8, "n_heads": 8, "dropout": 0.6},
        use_normalized_adj=False,
        lr=0.005,
        weight_decay=5e-4,
        epochs=200,
        patience=100,
        seed=42,
    )
