"""
All Datasets Experiment

GCN / GraphSAGE / GAT × Cora / CiteSeer / PubMed

每个模型在每个数据集上训练一次（seed=42），
结果自动追加到 outputs/results/benchmark.csv。
"""
from experiments.runner import run_experiment
from models.gcn import GCN
from models.graphsage import GraphSAGE
from models.gat import GAT

DATASETS = ["cora", "citeseer", "pubmed"]

MODELS = [
    (GCN,        "GCN",        {"hidden_dim": 16, "dropout": 0.5},                        0.01),
    (GraphSAGE,  "GraphSAGE",  {"hidden_dim": 16, "dropout": 0.5},                        0.01),
    (GAT,        "GAT",        {"hidden_dim":  8, "n_heads": 8, "dropout": 0.6},          0.005),
]

for dataset in DATASETS:
    for model_class, name, kwargs, lr in MODELS:
        model_key = f"{name}_{dataset}"
        print(f"\n{'='*60}")
        print(f"  {model_key}")
        print(f"{'='*60}")

        result = run_experiment(
            model_class=model_class,
            model_name=model_key,
            dataset_name=dataset,
            model_kwargs=kwargs,
            lr=lr,
            weight_decay=5e-4,
            epochs=200,
            patience=100,
            seed=42,
            skip_graph=True,
        )
        print(f"\n  >>> {model_key}: Test Acc = {result['test_acc']:.4f}  "
              f"({result['parameters']} params, {result['train_time']:.1f}s)")

print(f"\n{'='*60}")
print("  All done.  See outputs/results/benchmark.csv")
print(f"{'='*60}")
