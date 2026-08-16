"""统一保存与加载模型 checkpoint 的工具函数。"""

import torch


def save_checkpoint(path, model_state_dict, val_acc, model_kwargs, **extra):
    """保存模型权重、最佳验证集准确率、模型参数及额外实验信息。"""
    torch.save({"model_state_dict": model_state_dict, "val_acc": val_acc, "model_kwargs": model_kwargs, **extra}, path)


def load_checkpoint(path):
    """加载并返回完整 checkpoint 字典。"""
    return torch.load(path, weights_only=False)
