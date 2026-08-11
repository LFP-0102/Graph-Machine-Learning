"""
GCN / GraphSAGE / GAT 对比实验

统计: Test Accuracy / Parameters / Training Time
"""
import time
import torch
import pandas as pd

from datasets.base import load_dataset
from models.gcn import GCN
from models.graphsage import GraphSAGE
from models.gat import GAT
from utils.checkpoint import load_checkpoint
from utils.graph_utils import normalize_adj
from utils.seed import set_seed

# ── 配置 ──────────────────────────────────────────────────
set_seed(42)

# ── 加载数据 ──────────────────────────────────────────────────
data = load_dataset("cora")
x, y = data.features, data.labels
edge_index = data.edge_index
adj = torch.FloatTensor(normalize_adj(data.adj.numpy()))


# ── 参数统计 ──────────────────────────────────────────────────
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e3:.2f}K"


# ── 测试函数 ──────────────────────────────────────────────────
def test_model(model, mode):
    model.eval()
    start = time.time()
    with torch.no_grad():
        if mode == "GCN":
            out = model(x, adj)
        else:
            out = model(x, edge_index)
        pred = out.argmax(dim=1)
        acc = (pred[data.test_mask] == y[data.test_mask]).float().mean().item()
    return acc, time.time() - start


# ── 创建模型 ──────────────────────────────────────────────────
models = {
    "GCN": GCN(input_dim=data.num_features, hidden_dim=16, output_dim=data.num_classes, dropout=0.5),
    "GraphSAGE": GraphSAGE(input_dim=data.num_features, hidden_dim=16, output_dim=data.num_classes, dropout=0.5),
    "GAT": GAT(input_dim=data.num_features, hidden_dim=8, output_dim=data.num_classes, n_heads=8, dropout=0.6),
}

# ── 加载 checkpoint ──────────────────────────────────────────
checkpoints = {
    "GCN": "outputs/checkpoints/best_gcn.pt",
    "GraphSAGE": "outputs/checkpoints/best_graphsage.pt",
    "GAT": "outputs/checkpoints/best_gat.pt",
}

results = []
for name, model in models.items():
    ckpt = load_checkpoint(checkpoints[name])
    model.load_state_dict(ckpt["model_state_dict"])
    acc, t = test_model(model, name)
    params = count_parameters(model)
    results.append([name, acc, params, t])

# ── 输出表格 ──────────────────────────────────────────────────
print("\nModel Comparison")
print("-" * 60)
print(f"{'Model':12}{'Accuracy':12}{'Parameters':12}{'Inference':10}")
for r in results:
    print(f"{r[0]:12}{r[1]:11.2%}{format_params(r[2]):>12}{r[3]:10.4f}")

df = pd.DataFrame(results, columns=["Model", "Accuracy", "Parameters", "Inference_Time"])
df.to_csv("outputs/model_comparison.csv", index=False)
