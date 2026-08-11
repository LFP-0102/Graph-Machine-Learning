"""
GAT 训练入口 —— Cora 数据集

复现论文：Graph Attention Networks (Veličković et al., ICLR 2018)

输出：Loss / Val Acc / Best Val Acc / Test Acc / Best checkpoint
"""
import torch

from datasets.base import load_dataset
from models.gat import GAT
from utils.checkpoint import save_checkpoint
from utils.paths import CHECKPOINT_DIR, ensure_dirs
from utils.seed import set_seed
from utils.training import train_epoch, evaluate

# ── 配置 ──────────────────────────────────────────────────────
set_seed(42)

# ── 1. 加载数据 ───────────────────────────────────────────────
data = load_dataset("cora")
print(f"Cora: {data.num_features} features, {data.num_classes} classes, "
      f"{data.features.shape[0]} nodes")
print(f"Features: {list(data.features.shape)}")
print(f"Labels:   {list(data.labels.shape)}")
print(f"Edges:    {list(data.edge_index.shape)}")

# ── 2. 创建模型 ───────────────────────────────────────────────
model = GAT(input_dim=data.num_features, hidden_dim=8,
            output_dim=data.num_classes, n_heads=8, dropout=0.6)
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# ── 3. 训练 ───────────────────────────────────────────────────
best_val_acc, best_state, counter = 0.0, None, 0
patience, epochs = 100, 200

for epoch in range(1, epochs + 1):
    loss = train_epoch(model, data.features, data.edge_index,
                       data.labels, data.train_mask, optimizer)
    val_acc = evaluate(model, data.features, data.edge_index,
                       data.labels, data.val_mask)

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

# ── 4. 保存 checkpoint ────────────────────────────────────────
ensure_dirs(CHECKPOINT_DIR)
ckpt_path = CHECKPOINT_DIR / "best_gat.pt"
save_checkpoint(ckpt_path, best_state, best_val_acc, {
    "input_dim": data.num_features, "hidden_dim": 8,
    "output_dim": data.num_classes, "n_heads": 8, "dropout": 0.6,
})
print(f"Checkpoint saved: {ckpt_path}")

# ── 5. 测试 ───────────────────────────────────────────────────
model.load_state_dict(best_state)
test_acc = evaluate(model, data.features, data.edge_index,
                    data.labels, data.test_mask)
print(f"\nTest Acc: {test_acc:.4f}")
