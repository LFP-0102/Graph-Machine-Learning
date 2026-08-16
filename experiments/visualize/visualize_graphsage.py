"""生成 GraphSAGE 在 Cora 上消融实验的组会图表。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.runner import run_experiment
from models.graphsage import GraphSAGE

if __name__ == "__main__":
    run_experiment(GraphSAGE, "GraphSAGE", "cora", model_kwargs={"hidden_dim": 128, "dropout": 0.0, "sample_sizes": (25, 10)}, lr=0.01, epochs=200, patience=100, seed=42)
