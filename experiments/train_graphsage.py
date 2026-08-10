"""
GraphSAGE 训练入口 —— Cora 数据集

GraphSAGE:
Inductive Representation Learning on Large Graphs
(Hamilton et al., NeurIPS 2017)

输出：
- Loss
- Validation Accuracy
- Best Validation Accuracy
- Test Accuracy
- Best checkpoint
"""

import torch
import torch.nn.functional as F

from datasets.base import load_dataset
from models.graphsage import GraphSAGE
from utils.paths import CHECKPOINT_DIR, ensure_dirs
from utils.seed import set_seed


# -------------------------------------------------
# 配置
# -------------------------------------------------

set_seed(42)


# -------------------------------------------------
# 1. 加载数据
# -------------------------------------------------

data = load_dataset("cora")


print(
    f"Cora: {data.num_features} features, "
    f"{data.num_classes} classes, "
    f"{data.features.shape[0]} nodes"
)


print(f"Features: {list(data.features.shape)}")
print(f"Labels:   {list(data.labels.shape)}")
print(f"Edges:    {list(data.edge_index.shape)}")


# -------------------------------------------------
# 2. 创建模型
# -------------------------------------------------

model = GraphSAGE(
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



# -------------------------------------------------
# 3. 训练
# -------------------------------------------------

best_val_acc = 0.0
best_state = None

patience = 100
counter = 0

epochs = 200


for epoch in range(1, epochs + 1):

    model.train()

    optimizer.zero_grad()


    # GraphSAGE 输入：
    # 节点特征 + 边

    out = model(
        data.features,
        data.edge_index
    )


    loss = F.nll_loss(
        out[data.train_mask],
        data.labels[data.train_mask]
    )


    loss.backward()

    optimizer.step()



    # -----------------------------
    # validation
    # -----------------------------

    model.eval()

    with torch.no_grad():

        pred = out.argmax(dim=1)


        correct = (
            pred[data.val_mask]
            ==
            data.labels[data.val_mask]
        ).sum().item()


        val_acc = (
            correct /
            data.val_mask.sum().item()
        )



    # -----------------------------
    # 保存最佳模型
    # -----------------------------

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        best_state = {
            k: v.cpu().clone()
            for k, v in model.state_dict().items()
        }

        counter = 0

    else:

        counter += 1



    # -----------------------------
    # 打印
    # -----------------------------

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Best: {best_val_acc:.4f}"
        )


    # -----------------------------
    # Early stopping
    # -----------------------------

    if counter >= patience:

        print(
            f"Early stopping at epoch {epoch}"
        )

        break



# -------------------------------------------------
# 4. 保存checkpoint
# -------------------------------------------------

ensure_dirs(CHECKPOINT_DIR)


ckpt_path = CHECKPOINT_DIR / "best_graphsage.pt"



torch.save(
    {
        "model_state_dict": best_state,

        "val_acc": best_val_acc,

        "model_kwargs":
        {
            "input_dim": data.num_features,
            "hidden_dim":16,
            "output_dim":data.num_classes,
            "dropout":0.5,
        },

    },

    ckpt_path
)



print(
    f"Checkpoint saved: {ckpt_path}"
)



# -------------------------------------------------
# 5. 测试
# -------------------------------------------------

model.load_state_dict(best_state)


model.eval()


with torch.no_grad():

    out = model(
        data.features,
        data.edge_index
    )


    pred = out.argmax(dim=1)


    test_acc = (
        (
            pred[data.test_mask]
            ==
            data.labels[data.test_mask]
        )
        .sum()
        .item()
        /
        data.test_mask.sum().item()
    )


print(
    f"\nTest Acc: {test_acc:.4f}"
)