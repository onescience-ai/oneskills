import math
import torch
import numpy as np

from torch import nn
from dataclasses import dataclass

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from onescience.models.meta import ModelMetaData

from onescience.modules import (
    OneEmbedding,
    OneFuser,
    OneRecovery,
)

from pangu_afno_bridge import Pangu3DTo2DBridge, Pangu2DTo3DBridge


@dataclass
class MetaData(ModelMetaData):
    name: str = "PanguAFNOHybrid"
    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
    onnx_cpu: bool = False
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class PanguAFNOHybrid(nn.Module):
    """
    Pangu-Weather 与 FourCastNet AFNO 的混合模型。

    该模型使用 Pangu 的输入输出定义和 embedding/recovery 模块，
    但将主干特征提取部分替换为 FourCastNet 风格的 AFNO 主干。

    结构特点：
    - 使用 `PanguEmbedding` 分别编码 surface 和 upper-air 变量
    - 通过 `Pangu3DTo2DBridge` 将 3D token 转换为 2D patch 网格
    - 使用多层 `FourCastNetFuser` (AFNO) 完成主干特征提取
    - 通过 `Pangu2DTo3DBridge` 将 2D patch 网格转换回 3D token
    - 使用 `PanguPatchRecovery` 恢复 surface 和 upper-air 输出

    与原 Pangu 模型的区别：
    - 主干特征提取从 3D Transformer block 替换为 2D AFNO block
    - 不再使用下采样/上采样结构
    - 通过桥接层实现 3D token 和 2D patch 网格之间的转换

    Args:
        img_size (tuple[int, int]):
            输入空间尺寸 `(Height, Width)`。
        patch_size (tuple[int, int, int]):
            patch 切分尺寸 `(PatchPressureLevels, PatchHeight, PatchWidth)`。
        embed_dim (int):
            Pangu embedding 后的特征维度，默认为 192。
        afno_dim (int):
            AFNO 主干的特征维度，默认为 768（FourCastNet 默认）。
        afno_depth (int):
            AFNO block 的堆叠层数，默认为 12。
        pressure_levels (int):
            气压层数，Pangu 中为 8（1 个 surface + 7 个 upper-air）。
        afno_mlp_ratio (float):
            AFNO block 中 MLP 的隐层放大倍数，默认为 4.0。
        afno_num_blocks (int):
            AFNO 的通道分块数，默认为 8。
        afno_sparsity_threshold (float):
            AFNO soft shrink 阈值，默认为 0.01。
        afno_hard_thresholding_fraction (float):
            AFNO 保留的频率模式比例，默认为 1.0。
    """

    def __init__(
        self,
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
    ):
        super().__init__()
        drop_path = np.linspace(0, 0.2, afno_depth).tolist()

        self.patchembed2d = OneEmbedding(
            style="PanguEmbedding",
            img_size=img_size,
            patch_size=patch_size[1:],
            Variables=7,
            embed_dim=embed_dim,
        )
        self.patchembed3d = OneEmbedding(
            style="PanguEmbedding",
            img_size=(13, *img_size),
            patch_size=patch_size,
            Variables=5,
            embed_dim=embed_dim,
        )

        self.bridge_3d_to_2d = Pangu3DTo2DBridge(
            in_channels=embed_dim,
            out_dim=afno_dim,
            pressure_levels=pressure_levels,
        )

        self.afno_blocks = nn.ModuleList([
            OneFuser(
                style="FourCastNetFuser",
                dim=afno_dim,
                mlp_ratio=afno_mlp_ratio,
                drop_path=drop_path[i],
                double_skip=True,
                num_blocks=afno_num_blocks,
                sparsity_threshold=afno_sparsity_threshold,
                hard_thresholding_fraction=afno_hard_thresholding_fraction,
            )
            for i in range(afno_depth)
        ])

        self.bridge_2d_to_3d = Pangu2DTo3DBridge(
            in_dim=afno_dim,
            out_channels=embed_dim,
            pressure_levels=pressure_levels,
        )

        patched_input_shape = (
            pressure_levels,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )

        self.patchrecovery2d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(721, 1440),
            patch_size=(4, 4),
            in_chans=embed_dim * 2,
            out_chans=4,
        )
        self.patchrecovery3d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(13, 721, 1440),
            patch_size=(2, 4, 4),
            in_chans=embed_dim * 2,
            out_chans=5,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                Input tensor with shape `(Batch, 4 + 3 + 5 * 13, Height, Width)`.

                Channel layout:
                - first 4 channels: prognostic surface variables
                - next 3 channels: static masks
                - remaining `5 * 13` channels: upper-air variables flattened over pressure levels

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - surface output:
                  `(Batch, 4, Height, Width)`
                - upper-air output:
                  `(Batch, 5, 13, Height, Width)`
        """
        SurfaceInput = x[:, :7, :, :]
        UpperAirInput = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])

        SurfaceFeatures = self.patchembed2d(SurfaceInput)
        UpperAirFeatures = self.patchembed3d(UpperAirInput)

        CombinedFeatures = torch.concat(
            [SurfaceFeatures.unsqueeze(2), UpperAirFeatures], dim=2
        )
        Batch, Channels, PressureLevels, Height, Width = CombinedFeatures.shape

        SkipFeatures = CombinedFeatures.clone()

        PatchGrid = self.bridge_3d_to_2d(CombinedFeatures)

        for block in self.afno_blocks:
            PatchGrid = block(PatchGrid)

        CombinedFeatures = self.bridge_2d_to_3d(PatchGrid)

        OutputFeatures = torch.concat([CombinedFeatures, SkipFeatures], dim=1)

        output_surface = OutputFeatures[:, :, 0, :, :]
        output_upper_air = OutputFeatures[:, :, 1:, :, :]

        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)

        return output_surface, output_upper_air
