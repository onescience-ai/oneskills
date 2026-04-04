"""
PanguAFNOHybrid 模型使用示例

本文件展示如何使用 PanguAFNOHybrid 模型进行推理。
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from pangu_afno_hybrid import PanguAFNOHybrid


def create_sample_input(batch_size=1):
    """
    创建示例输入数据。

    Args:
        batch_size (int):
            批大小。

    Returns:
        torch.Tensor:
            输入张量，形状为 `(Batch, 4 + 3 + 5 * 13, Height, Width)`。
    """
    Height, Width = 721, 1440
    Channels = 4 + 3 + 5 * 13

    x = torch.randn(batch_size, Channels, Height, Width)
    return x


def main():
    """
    主函数：演示 PanguAFNOHybrid 模型的使用。
    """
    print("Initializing PanguAFNOHybrid model...")
    model = PanguAFNOHybrid(
        img_size=(721, 1440),
        patch_size=(2, 4, 4),
        embed_dim=192,
        afno_dim=768,
        afno_depth=12,
        pressure_levels=8,
        afno_mlp_ratio=4.0,
        afno_num_blocks=8,
        afno_sparsity_threshold=0.01,
        afno_hard_thresholding_fraction=1.0,
    )

    model.eval()

    print(f"\nModel architecture:")
    print(model)

    print("\nCreating sample input...")
    batch_size = 1
    x = create_sample_input(batch_size)
    print(f"Input shape: {x.shape}")

    print("\nRunning forward pass...")
    with torch.no_grad():
        output_surface, output_upper_air = model(x)

    print(f"\nOutput shapes:")
    print(f"  Surface output: {output_surface.shape}")
    print(f"  Upper-air output: {output_upper_air.shape}")

    print("\nExpected shapes:")
    print(f"  Surface output: ({batch_size}, 4, 721, 1440)")
    print(f"  Upper-air output: ({batch_size}, 5, 13, 721, 1440)")

    print("\nModel parameter count:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    main()
