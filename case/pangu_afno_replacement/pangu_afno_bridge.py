import torch
import torch.nn as nn
import math


class Pangu3DTo2DBridge(nn.Module):
    """
    将 Pangu 的 3D token 转换为 2D patch 网格，以便输入到 AFNO 主干。

    该模块接收 Pangu embedding 输出的 3D 特征 `(Batch, Channels, PressureLevels, Height, Width)`，
    将其转换为 2D patch 网格 `(Batch, Height, Width, dim)`。

    转换策略：
    1. 将 PressureLevels 维展平到通道维
    2. 使用 1x1 卷积调整特征维度到目标维度

    Args:
        in_channels (int):
            输入特征通道数，通常为 Pangu 的 embed_dim=192。
        out_dim (int):
            输出特征维度，建议使用 768（FourCastNet 默认）或保持 192。
        pressure_levels (int):
            气压层数，Pangu 中为 8（1 个 surface + 7 个 upper-air）。

    Shape:
        输入:
            `(Batch, in_channels, PressureLevels, Height, Width)`
        输出:
            `(Batch, Height, Width, out_dim)`
    """

    def __init__(self, in_channels=192, out_dim=768, pressure_levels=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.pressure_levels = pressure_levels

        self.proj = nn.Conv2d(
            in_channels * pressure_levels,
            out_dim,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                输入张量，形状为 `(Batch, in_channels, PressureLevels, Height, Width)`

        Returns:
            torch.Tensor:
                输出张量，形状为 `(Batch, Height, Width, out_dim)`
        """
        Batch, Channels, PressureLevels, Height, Width = x.shape

        x = x.reshape(Batch, Channels * PressureLevels, Height, Width)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)

        return x


class Pangu2DTo3DBridge(nn.Module):
    """
    将 2D patch 网格转换回 Pangu 的 3D token 格式。

    该模块接收 AFNO 主干输出的 2D patch 网格 `(Batch, Height, Width, dim)`，
    将其转换回 3D 特征 `(Batch, Channels, PressureLevels, Height, Width)`。

    转换策略：
    1. 使用 1x1 卷积调整特征维度
    2. 将通道维拆分回 PressureLevels 维

    Args:
        in_dim (int):
            输入特征维度，应与 AFNO 主干的输出维度一致。
        out_channels (int):
            输出特征通道数，通常为 Pangu 的 embed_dim=192。
        pressure_levels (int):
            气压层数，Pangu 中为 8（1 个 surface + 7 个 upper-air）。

    Shape:
        输入:
            `(Batch, Height, Width, in_dim)`
        输出:
            `(Batch, out_channels, PressureLevels, Height, Width)`
    """

    def __init__(self, in_dim=768, out_channels=192, pressure_levels=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_channels = out_channels
        self.pressure_levels = pressure_levels

        self.proj = nn.Conv2d(
            in_dim,
            out_channels * pressure_levels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                输入张量，形状为 `(Batch, Height, Width, in_dim)`

        Returns:
            torch.Tensor:
                输出张量，形状为 `(Batch, out_channels, PressureLevels, Height, Width)`
        """
        Batch, Height, Width, Dim = x.shape

        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.reshape(Batch, self.out_channels, self.pressure_levels, Height, Width)

        return x
