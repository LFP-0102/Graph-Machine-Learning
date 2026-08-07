"""
GCN 训练入口 —— Cora 数据集。

复现论文：Semi-Supervised Classification with Graph Convolutional Networks
            (Kipf & Welling, ICLR 2017)

两层 GCN：
    Z = softmax(Â ReLU(Â X W₀) W₁)
    其中 Â = D^(-1/2)(A+I)D^(-1/2)
"""
import torch

from datasets.base import load_dataset
from models.gcn import GCN
from utils.graph_utils import normalize_adj
from utils.paths import CHECKPOINT_DIR, ensure_dirs
from utils.seed import set_seed
from utils.training import train_epoch, evaluate

# ── 配置 ────────────────────────────────────────────────────
set_seed(42)
# ── 1. 加载数据 ────────────────────────────────────────────
data = load_dataset("cora")
print(f"Cora: {data.num_features} features, "
      f"{data.num_classes} classes, "
      f"{data.features.shape[0]} nodes")
print(f"Features: {list(data.features.shape)}")
print(f"Labels:   {list(data.labels.shape)}")
print(f"Adj:      {list(data.adj.shape)}")

# ── 2. 预处理邻接矩阵 ─────────────────────────────────────
# GCN 公式: Â = D^(-1/2)(A+I)D^(-1/2)
adj = normalize_adj(data.adj.numpy())
adj = torch.FloatTensor(adj)

# ── 3. 创建模型 ────────────────────────────────────────────
model = GCN(
    input_dim=data.num_features,
    hidden_dim=16,
    output_dim=data.num_classes,
    dropout=0.5,
)
print(model)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)

# ── 4. 训练 ────────────────────────────────────────────────
best_val_acc = 0.0
best_state = None
patience = 100
counter = 0
epochs = 200

for epoch in range(1, epochs + 1):
    loss = train_epoch(
        model, data.features, adj,
        data.labels, data.train_mask, optimizer
    )

    val_acc = evaluate(
        model, data.features, adj,
        data.labels, data.val_mask
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        counter = 0
    else:
        counter += 1

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# ── 5. 保存最佳模型 ────────────────────────────────────────
ensure_dirs(CHECKPOINT_DIR)
ckpt_path = CHECKPOINT_DIR / "best_gcn.pt"
torch.save(
    {
        "model_state_dict": best_state,
        "val_acc": best_val_acc,
        "model_kwargs": {
            "input_dim": data.num_features,
            "hidden_dim": 16,
            "output_dim": data.num_classes,
            "dropout": 0.5,
        },
    },
    ckpt_path,
)
print(f"Checkpoint saved: {ckpt_path}")

# ── 6. 测试（使用验证集最佳模型）────────────────────────────
model.load_state_dict(best_state)
test_acc = evaluate(
    model, data.features, adj,
    data.labels, data.test_mask
)
print(f"\nTest Acc: {test_acc:.4f}")
