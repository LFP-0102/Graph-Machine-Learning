"""统一 checkpoint 保存 / 加载.

所有训练脚本和 runner 都通过这两个函数操作 checkpoint,
保证格式一致，避免散落 torch.save/load 调用。
"""
import torch


def save_checkpoint(path, model_state_dict, val_acc, model_kwargs, **extra):
    """保存 checkpoint.

    path:             保存路径 (str | Path)
    model_state_dict: model.state_dict() 或 {k: v.cpu().clone()}
    val_acc:          最佳验证准确率
    model_kwargs:     模型构造参数 dict (用于复现)
    **extra:          额外字段, 如 model_name, dataset, test_acc, train_time 等
    """
    checkpoint = {
        "model_state_dict": model_state_dict,
        "val_acc": val_acc,
        "model_kwargs": model_kwargs,
        **extra,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path):
    """加载 checkpoint, 返回完整 dict.

    包含: model_state_dict, val_acc, model_kwargs, [model_name, test_acc, ...]
    """
    return torch.load(path, weights_only=False)
