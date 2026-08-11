"""
GraphSAGE 训练入口 —— Cora 数据集

Paper: Inductive Representation Learning on Large Graphs (Hamilton et al., NeurIPS 2017)

输出：Loss / Train Acc / Val Acc / Best Val Acc / Test Acc / Parameters / Training Time / Best checkpoint
"""
import time
import torch
import torch.nn.functional as F

from datasets.base import load_dataset
from models.graphsage import GraphSAGE
from utils.checkpoint import save_checkpoint
from utils.paths import CHECKPOINT_DIR, ensure_dirs
from utils.seed import set_seed

# ── 工具函数 ──────────────────────────────────────────────────
def accuracy(pred, label):
    return (pred == label).float().mean().item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── 配置 ──────────────────────────────────────────────────────
set_seed(42)

# ── 1. 加载数据 ───────────────────────────────────────────────
data = load_dataset("cora")
print("=" * 50)
print("Model: GraphSAGE")
print("Dataset: Cora\n")
print(f"Features: {data.features.shape[0]} × {data.features.shape[1]}")
print(f"Classes: {data.num_classes}")
print(f"Edges: {data.edge_index.shape[1]}")
print("=" * 50)

# ── 2. 创建模型 ───────────────────────────────────────────────
model = GraphSAGE(input_dim=data.num_features, hidden_dim=16, output_dim=data.num_classes, dropout=0.5)
print(model)
print(f"Parameters: {count_parameters(model)}")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# ── 3. 训练 ───────────────────────────────────────────────────
start_time = time.time()
best_val_acc, best_epoch, best_state = 0.0, 0, None
patience, counter, epochs = 100, 0, 200

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.features, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.labels[data.train_mask])
    loss.backward()
    optimizer.step()

    # evaluation
    model.eval()
    with torch.no_grad():
        pred = out.argmax(dim=1)
        train_acc = accuracy(pred[data.train_mask], data.labels[data.train_mask])
        val_acc = accuracy(pred[data.val_mask], data.labels[data.val_mask])

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        counter = 0
    else:
        counter += 1

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

train_time = time.time() - start_time

# ── 4. 测试最佳模型 ────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    out = model(data.features, data.edge_index)
    pred = out.argmax(dim=1)
    test_acc = accuracy(pred[data.test_mask], data.labels[data.test_mask])

# ── 5. 保存 checkpoint ────────────────────────────────────────
ensure_dirs(CHECKPOINT_DIR)
ckpt_path = CHECKPOINT_DIR / "best_graphsage.pt"
save_checkpoint(ckpt_path, best_state, best_val_acc, {
    "input_dim": data.num_features, "hidden_dim": 16,
    "output_dim": data.num_classes, "dropout": 0.5,
}, model_name="GraphSAGE", dataset="Cora", epoch=best_epoch,
   best_val_acc=best_val_acc, test_acc=test_acc,
   parameters=count_parameters(model), training_time=train_time)

# ── 6. 最终输出 ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("Training Finished\n")
print(f"Best Epoch: {best_epoch}")
print(f"Best Val Acc: {best_val_acc:.4f}")
print(f"Test Acc: {test_acc:.4f}")
print(f"Parameters: {count_parameters(model)}")
print(f"Training Time: {train_time:.2f}s\n")
print(f"Checkpoint: {ckpt_path}")
print("=" * 50)
