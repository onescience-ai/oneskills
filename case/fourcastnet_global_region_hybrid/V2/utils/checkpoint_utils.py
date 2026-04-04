import os
import torch
from typing import Optional, Dict, Any


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    checkpoint_path: str,
    is_best: bool = False,
    **kwargs,
):
    """
    保存模型 checkpoint。

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        loss: 当前 loss
        checkpoint_path: checkpoint 保存路径
        is_best: 是否为最佳模型
        **kwargs: 其他需要保存的信息
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    checkpoint.update(kwargs)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    if is_best:
        best_path = checkpoint_path.replace(".pth", "_best.pth")
        torch.save(checkpoint, best_path)
        print(f"Best checkpoint saved to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    strict: bool = False,
) -> Dict[str, Any]:
    """
    加载模型 checkpoint。

    Args:
        checkpoint_path: checkpoint 路径
        model: 模型
        optimizer: 优化器（可选）
        device: 设备
        strict: 是否严格加载权重

    Returns:
        checkpoint: checkpoint 字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "loss" in checkpoint:
        print(f"  Loss: {checkpoint['loss']}")

    return checkpoint


def save_checkpoint_early(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    filename: str = "checkpoint_early.pth",
    **kwargs,
):
    """
    快速保存 checkpoint（用于训练中断）。

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        loss: 当前 loss
        checkpoint_dir: checkpoint 保存目录
        filename: checkpoint 文件名
        **kwargs: 其他需要保存的信息
    """
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=loss,
        checkpoint_path=checkpoint_path,
        **kwargs,
    )


def load_global_pretrained(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = False,
    prefix: str = "global_",
):
    """
    加载全球模型预训练权重。

    Args:
        model: 混合模型
        checkpoint_path: checkpoint 路径
        device: 设备
        strict: 是否严格加载权重
        prefix: 全球分支参数前缀
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Global checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)

    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    global_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            global_state_dict[k] = v
        elif k in ["patch_embed", "blocks", "pos_embed", "head"]:
            global_state_dict[f"{prefix}{k}"] = v

    model.load_state_dict(global_state_dict, strict=strict)
    print(f"Global pretrained weights loaded from {checkpoint_path}")
    print(f"  Loaded {len(global_state_dict)} parameters with prefix '{prefix}'")


def freeze_parameters(
    model: torch.nn.Module,
    prefix: str = "global_",
):
    """
    冻结指定前缀的参数。

    Args:
        model: 模型
        prefix: 参数前缀
    """
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            param.requires_grad = False
    print(f"Frozen parameters with prefix '{prefix}'")


def unfreeze_parameters(
    model: torch.nn.Module,
    prefix: str = "global_",
):
    """
    解冻指定前缀的参数。

    Args:
        model: 模型
        prefix: 参数前缀
    """
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            param.requires_grad = True
    print(f"Unfrozen parameters with prefix '{prefix}'")


def get_parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    """
    获取模型参数统计。

    Args:
        model: 模型

    Returns:
        stats: 参数统计字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    stats = {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
    }

    return stats


def print_parameter_stats(model: torch.nn.Module):
    """
    打印模型参数统计。

    Args:
        model: 模型
    """
    stats = get_parameter_count(model)

    print("\n" + "=" * 60)
    print("Model Parameter Statistics")
    print("=" * 60)
    print(f"Total parameters:     {stats['total']:,}")
    print(f"Trainable parameters: {stats['trainable']:,}")
    print(f"Frozen parameters:    {stats['frozen']:,}")
    print("=" * 60 + "\n")
