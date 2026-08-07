"""
GAT 训练入口 —— Cora 数据集。

复现论文：Graph Attention Networks (Veličković et al., ICLR 2018)

论文中 Cora 上的结果（Transductive）：
    - GAT (K=8): 83.0 ± 0.7%
"""
import torch

from datasets.base import load_dataset
from models.gat import GAT
from utils.graph_utils import add_self_loops
from utils.seed import set_seed
from utils.training import train_epoch, evaluate


# ── 配置 ────────────────────────────────────────────────────
set_seed(42)

# ── 1. 加载数据 ────────────────────────────────────────────
data = load_dataset("cora")
print(f"Cora: {data.num_features} features, "
      f"{data.num_classes} classes, "
      f"{data.features.shape[0]} nodes")

# ── 2. 预处理邻接矩阵 ─────────────────────────────────────
# GAT 只需要二值邻接矩阵（含自环）作为 attention mask
adj = add_self_loops(data.adj.numpy())
adj = torch.FloatTensor(adj)

# ── 3. 创建模型 ────────────────────────────────────────────
# 论文超参数: hidden_dim=8, n_heads=8, dropout=0.6
model = GAT(
    input_dim=data.num_features,
    hidden_dim=8,            # 每头 8 维
    output_dim=data.num_classes,
    n_heads=8,
    dropout=0.6,
)
print(model)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)

# ── 4. 训练 ────────────────────────────────────────────────
best_val_acc = 0.0
epochs = 200

for epoch in range(epochs):
    loss = train_epoch(
        model, data.features, adj,
        data.labels, data.train_mask, optimizer
    )

    if epoch % 10 == 0 or epoch == epochs - 1:
        val_acc = evaluate(
            model, data.features, adj,
            data.labels, data.val_mask
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

# ── 5. 测试 ────────────────────────────────────────────────
test_acc = evaluate(
    model, data.features, adj,
    data.labels, data.test_mask
)
print(f"\nTest Acc: {test_acc:.4f}")
