"""生成 GAT 在 Cora 上复现结果的组会图表。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.runner import run_experiment
from models.gat import GAT

if __name__ == "__main__":
    run_experiment(GAT, "GAT", "cora", model_kwargs={"hidden_dim": 8, "n_heads": 8, "dropout": 0.6}, lr=0.005, epochs=1000, patience=100, seed=42)
