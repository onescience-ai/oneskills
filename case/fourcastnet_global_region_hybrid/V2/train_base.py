import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys

# 添加当前 case 目录到路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.fourcastnet_global_region_hybrid import FourCastNetBase
from utils.synthetic_data import create_dataloaders
from utils.checkpoint_utils import save_checkpoint, save_checkpoint_early, print_parameter_stats


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    writer=None,
):
    """
    训练一个 epoch。

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
        writer: TensorBoard writer

    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        inputs = batch["global_input"].to(device)
        targets = batch["global_target"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}], Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.6f}")

            if writer is not None:
                global_step = epoch * num_batches + batch_idx
                writer.add_scalar("train/loss", loss.item(), global_step)

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model,
    dataloader,
    criterion,
    device,
    epoch,
    writer=None,
):
    """
    验证模型。

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前 epoch
        writer: TensorBoard writer

    Returns:
        avg_loss: 平均损失
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["global_input"].to(device)
            targets = batch["global_target"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    avg_loss = total_loss / num_batches

    if writer is not None:
        writer.add_scalar("val/loss", avg_loss, epoch)

    return avg_loss


def train_base_model(
    img_size=(720, 1440),
    patch_size=(8, 8),
    in_chans=19,
    out_chans=19,
    embed_dim=768,
    depth=12,
    mlp_ratio=4.0,
    drop_rate=0.0,
    drop_path_rate=0.0,
    num_blocks=8,
    sparsity_threshold=0.01,
    hard_thresholding_fraction=1.0,

    batch_size=4,
    num_epochs=10,
    learning_rate=1e-4,
    weight_decay=1e-4,

    num_train_samples=800,
    num_val_samples=200,

    checkpoint_dir="./checkpoints/base_model",
    log_dir="./logs/base_model",
    save_interval=2,
    device=None,
):
    """
    训练基座 FourCastNet 模型。

    Args:
        img_size: 图像尺寸
        patch_size: patch 尺寸
        in_chans: 输入变量通道数
        out_chans: 输出变量通道数
        embed_dim: embedding 维度
        depth: trunk 层数
        mlp_ratio: MLP 隐层放大倍数
        drop_rate: dropout 比例
        drop_path_rate: Stochastic Depth 比例
        num_blocks: AFNO 通道分块数
        sparsity_threshold: AFNO soft shrink 阈值
        hard_thresholding_fraction: AFNO 保留频率模式比例

        batch_size: batch size
        num_epochs: 训练 epoch 数
        learning_rate: 学习率
        weight_decay: 权重衰减

        num_train_samples: 训练样本数量
        num_val_samples: 验证样本数量

        checkpoint_dir: checkpoint 保存目录
        log_dir: 日志保存目录
        save_interval: checkpoint 保存间隔
        device: 设备
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print("Training Base FourCastNet Model")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Image size: {img_size}")
    print(f"Patch size: {patch_size}")
    print(f"Embed dim: {embed_dim}")
    print(f"Depth: {depth}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'=' * 60}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = FourCastNetBase(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        num_blocks=num_blocks,
        sparsity_threshold=sparsity_threshold,
        hard_thresholding_fraction=hard_thresholding_fraction,
    ).to(device)

    print_parameter_stats(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_loader, val_loader = create_dataloaders(
        batch_size=batch_size,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        in_chans=in_chans,
        out_chans=out_chans,
        global_img_size=img_size,
        region_img_size=(1000, 1000),
        num_workers=0,
        seed=42,
    )

    writer = SummaryWriter(log_dir)

    best_val_loss = float("inf")

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                writer=writer,
            )

            val_loss = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                writer=writer,
            )

            scheduler.step()

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"  * New best model! (val_loss: {best_val_loss:.6f})")

            if (epoch + 1) % save_interval == 0 or is_best:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_epoch_{epoch + 1}.pth"
                )
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    loss=val_loss,
                    checkpoint_path=checkpoint_path,
                    is_best=is_best,
                    scheduler=scheduler.state_dict(),
                )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving early checkpoint...")
        save_checkpoint_early(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=val_loss if 'val_loss' in locals() else 0.0,
            checkpoint_dir=checkpoint_dir,
            filename="checkpoint_interrupted.pth",
            scheduler=scheduler.state_dict(),
        )
        print("Early checkpoint saved. Exiting gracefully.")

    finally:
        writer.close()

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train base FourCastNet model")
    parser.add_argument("--img_size", type=int, nargs=2, default=[720, 1440], help="Image size (H W)")
    parser.add_argument("--patch_size", type=int, nargs=2, default=[8, 8], help="Patch size (H W)")
    parser.add_argument("--in_chans", type=int, default=19, help="Input channels")
    parser.add_argument("--out_chans", type=int, default=19, help="Output channels")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=12, help="Trunk depth")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/base_model", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="./logs/base_model", help="Log directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    train_base_model(
        img_size=tuple(args.img_size),
        patch_size=tuple(args.patch_size),
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        embed_dim=args.embed_dim,
        depth=args.depth,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
