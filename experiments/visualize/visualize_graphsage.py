"""
GraphSAGE 论文图表复现 + 自定义可视化。

用法：
    PYTHONPATH=. python experiments/visualize/visualize_graphsage.py

输出：
    outputs/visualizations/graphsage/
    outputs/runs/graphsage/
"""
from experiments.runner import run_experiment
from models.graphsage import GraphSAGE

if __name__ == "__main__":
    run_experiment(
        model_class=GraphSAGE,
        model_name="GraphSAGE",
        dataset_name="cora",
        model_kwargs={"hidden_dim": 16, "dropout": 0.5},
        use_normalized_adj=False,
        lr=0.01,
        weight_decay=5e-4,
        epochs=200,
        patience=100,
        seed=42,
    )
