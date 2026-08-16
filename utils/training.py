"""轻量训练与评估函数，供各实验脚本直接调用。"""

import torch
import torch.nn.functional as F


def train_epoch(model, features, graph, labels, train_mask, optimizer):
    """完成一个训练轮次，返回训练损失。"""
    model.train()
    optimizer.zero_grad()
    output = model(features, graph)
    loss = F.nll_loss(output[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, features, graph, labels, mask):
    """计算指定节点掩码上的分类准确率。"""
    model.eval()
    prediction = model(features, graph).argmax(dim=1)
    return (prediction[mask] == labels[mask]).sum().item() / mask.sum().item()


@torch.no_grad()
def predict(model, features, graph):
    """返回预测类别 ``[N]`` 与对应概率 ``[N, C]``。"""
    model.eval()
    output = model(features, graph)
    return output.argmax(dim=1), torch.exp(output)
