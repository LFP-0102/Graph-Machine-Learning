"""
轻量训练 / 评估辅助函数。

不引入 Trainer 类 —— 每个实验脚本直接调用 train_epoch / evaluate，
保持训练过程透明、可修改。
"""
import torch
import torch.nn.functional as F


def train_epoch(model, features, graph, labels, train_mask, optimizer):
    """单轮训练。

    参数：
        model:       nn.Module，forward(x, graph) → log_softmax
        features:    Tensor [N, D]
        graph:       Tensor [N, N] adj 或 [2, E] edge_index
        labels:      Tensor [N]     整数标签
        train_mask:  Tensor [N]     bool
        optimizer:   torch.optim.Optimizer

    返回：
        loss: float
    """
    model.train()
    optimizer.zero_grad()

    output = model(features, graph)
    loss = F.nll_loss(output[train_mask], labels[train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, features, graph, labels, mask):
    """评估准确率。

    返回：
        acc: float (0 ~ 1)
    """
    model.eval()
    output = model(features, graph)
    pred = output.argmax(dim=1)
    correct = (pred[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()


@torch.no_grad()
def predict(model, features, graph):
    """返回模型预测的类别标签 [N] 和概率 [N, C]"""
    model.eval()
    output = model(features, graph)
    pred = output.argmax(dim=1)
    prob = torch.exp(output)
    return pred, prob
