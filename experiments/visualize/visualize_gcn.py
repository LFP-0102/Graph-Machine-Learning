"""生成 GCN 在 Cora 上复现结果的组会图表。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.runner import run_experiment
from models.gcn import GCN

if __name__ == "__main__":
    run_experiment(GCN, "GCN", "cora", model_kwargs={"hidden_dim": 16, "dropout": 0.5}, lr=0.01, epochs=200, patience=100, seed=100)
