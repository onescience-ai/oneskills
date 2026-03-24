component:
  meta:
    name: PanguPatchRecovery2D
    中文名称: Pangu-Weather 二维 Patch 恢复模块
    别名: PanguRecovery2D, PatchRecovery2D, 二维面片重建层
    version: 1.0
    领域: 气象, 深度学习
    分类: 神经网络组件
    子类: 重建层/上采样模块
    作者: OneScience
    tags:
      - Pangu-Weather
      - PatchRecovery
      - 上采样
      - 二维场重建
      - ERA5

  concept:
    描述: >
      PanguPatchRecovery2D 是 Pangu-Weather 输出头中的二维 Patch 恢复模块，
      使用一次二维转置卷积将地表 Patch 特征图上采样回 ERA5 分辨率的地表气象场，
      并在纬度和经度方向对称裁剪补零边界，使输出与 0.25°×0.25° 经纬网格逐点对齐。
    直觉理解: >
      可以把 3DEST 主干输出的地表特征看成把全球“经度×纬度”的二维天气地图切成小瓷砖后压缩存储，
      PanguPatchRecovery2D 就像一台二维投影仪，按固定放大比例把这些瓷砖重新铺回完整地图，
      再把四周多出来的黑边在南北、东西方向各剪一半，最终恢复到与原始 ERA5 地表网格逐点对应的二维场。
    解决的问题:
      - 将 3DEST 主干输出的低分辨率地表 Patch 特征精确还原为 ERA5 分辨率的地表二维场 (721 × 1440)。
      - 统一处理 Patch 划分与零填充带来的边界冗余，保证恢复后的二维场在经纬方向与输入再分析场完全对齐。
      - 在从潜在特征空间到物理变量空间的映射中提供可学习的空间重建能力，而非简单插值。

  theory:
    核心公式:
      二维反卷积恢复:
        表达式: >
          给定输入特征 x ∈ R^{B×C_in×H'×W'}，采用 kernel_size = stride = (k_h,k_w)
          的二维转置卷积算子 ConvTranspose2d 得到中间输出
          x̃ = ConvTranspose2d(x; W) ∈ R^{B×C_out×Ĥ×Ŵ}。
          设目标输出大小为 img_size = (H,W)，则裁剪量为
          h_pad = Ĥ - H, w_pad = Ŵ - W。
          采用对称裁剪得到最终输出
          y = x̃[:, :, h_pad//2 : Ĥ - (h_pad - h_pad//2),
                         w_pad//2 : Ŵ - (w_pad - w_pad//2)]，
          其中 y ∈ R^{B×C_out×H×W} 对应目标经纬网格上的二维场。
        变量说明:
          B:
            名称: 批大小
            形状: [1]
            描述: 一个批次中同时处理的样本数量。
          C_in:
            名称: 输入通道数
            形状: [1]
            描述: Patch 特征的通道数，典型为 192×2，对应 3DEST 解码器输出通道。
          C_out:
            名称: 输出通道数
            形状: [1]
            描述: 恢复后二维气象场的物理变量通道数，典型为 4（MSLP/T2M/U10/V10）。
          H':
            名称: Patch 特征高度
            形状: [1]
            描述: Patch 网格在纬度方向的尺寸，例如 181。
          W':
            名称: Patch 特征宽度
            形状: [1]
            描述: Patch 网格在经度方向的尺寸，例如 360。
          Ĥ:
            名称: 反卷积输出高度
            形状: [1]
            描述: ConvTranspose2d 放大后的中间高度，在裁剪前略大于或等于目标高度 H。
          Ŵ:
            名称: 反卷积输出宽度
            形状: [1]
            描述: ConvTranspose2d 放大后的中间宽度，在裁剪前略大于或等于目标宽度 W。
          H:
            名称: 目标高度
            形状: [1]
            描述: 恢复后二维场在纬向的网格数，ERA5 中为 721。
          W:
            名称: 目标宽度
            形状: [1]
            描述: 恢复后二维场在经向的网格数，ERA5 中为 1440。
          x:
            名称: 输入 Patch 特征图
            形状: [B, C_in, H', W']
            描述: 由 Patch Embedding 与 3DEST 主干输出的地表 Patch 特征图。
          y:
            名称: 恢复后二维场
            形状: [B, C_out, H, W]
            描述: 对应 ERA5 原始经纬度网格的地表物理变量二维场。

  structure:
    架构类型: 二维 ConvTranspose PatchRecovery 模块
    计算流程:
      - 根据 img_size 与 patch_size 构造 ConvTranspose2d，kernel_size 与 stride 等于 patch_size。
      - 在 Patch 网格 (H', W') 上对输入特征 x 执行 ConvTranspose2d 上采样，得到中间输出 output，空间尺寸为 (Ĥ, Ŵ)。
      - 计算 h_pad = Ĥ - H, w_pad = Ŵ - W，将多余尺寸在纬向和经向上对称分配。
      - 沿高度与宽度方向对 output 做对称裁剪，去除补零边界，得到精确匹配 (H, W) 的二维输出。
      - 将输出作为地表变量的最终预报场或中间表示，输入到后续损失计算与评价指标模块。
    计算流程图: |
      输入 x (B, C_in, H', W')
                  |
                  v
            ConvTranspose2d
                  |
                  v
        中间输出 (B, C_out, Ĥ, Ŵ)
                  |
                  v
        计算 h_pad, w_pad
                  |
                  v
            二维对称裁剪
                  |
                  v
        输出 y (B, C_out, H, W)

  interface:
    参数:
      img_size:
        类型: Tuple[int, int]
        默认值: (721, 1440)
        描述: 输出目标二维场尺寸 (H, W)，分别对应纬度和经度网格数。
      patch_size:
        类型: Tuple[int, int]
        默认值: (4, 4)
        描述: 二维 Patch 大小 (patch_h, patch_w)，同时作为 ConvTranspose2d 的 kernel_size 与 stride。
      in_chans:
        类型: int
        默认值: 384
        描述: 输入特征通道数，典型为 192×2，对应 3DEST 解码器输出通道。
      out_chans:
        类型: int
        默认值: 4
        描述: 输出地表物理变量通道数，对应 MSLP/T2M/U10/V10。
    输入:
      x:
        类型: Tensor
        形状: [B, in_chans, H', W']
        描述: 来自 3DEST 解码器的地表 Patch 特征图，空间尺寸约为 img_size/patch_size。
    输出:
      y:
        类型: Tensor
        形状: [B, out_chans, img_size[0], img_size[1]]
        描述: 恢复到原始 ERA5 分辨率的地表二维物理量场。

  types:
    PatchFeatureVolume2D:
      形状: [B, C_in, H', W']
      描述: 在纬度×经度 Patch 网格上的特征图，每个格点对应一个二维 Patch 嵌入。
    RecoveredField2D:
      形状: [B, C_out, H, W]
      描述: 通过 PanguPatchRecovery2D 从 Patch 特征恢复出的地表物理变量二维场。

  constraints:
    shape_constraints:
      - 规则: Ĥ ≥ H 且 Ŵ ≥ W，裁剪量均为非负整数
        描述: 反卷积输出尺寸必须不小于目标尺寸，才能通过二维对称裁剪精确恢复原始网格。
      - 规则: H'·patch_size[0] ≈ H 且 W'·patch_size[1] ≈ W
        描述: Patch 网格尺寸与 patch_size 需要与目标分辨率大致匹配，否则补零与裁剪范围过大会导致信息浪费。 [基于通用知识补充]
    parameter_constraints:
      - 规则: img_size 中的两个维度均为正整数
        描述: 输出高度与宽度必须为正整数网格数，对应经纬度栅格。 [基于通用知识补充]
      - 规则: patch_size 中的两个维度为正整数
        描述: Patch 大小必须为正整数，才能作为 ConvTranspose2d 的 kernel_size 与 stride。 [基于通用知识补充]
      - 规则: in_chans, out_chans > 0 且为整数
        描述: 通道数必须为正整数，与上游特征维度和目标变量数保持一致。 [基于通用知识补充]
    compatibility_rules:
      - 输入类型: PatchFeatureVolume2D
        输出类型: RecoveredField2D
        描述: 仅当 PatchFeatureVolume2D 的空间尺寸与 patch_size 和 img_size 一致时才能保证无缝恢复。 [基于通用知识补充]

  implementation:
    框架: pytorch
    示例代码: |
      import torch
      from torch import nn


      class PanguPatchRecovery2D(nn.Module):
          def __init__(
              self,
              img_size=(721, 1440),
              patch_size=(4, 4),
              in_chans=192 * 2,
              out_chans=4,
          ):
              super().__init__()
              self.img_size = img_size
              self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

          def forward(self, x: torch.Tensor) -> torch.Tensor:
              output = self.conv(x)
              _, _, H, W = output.shape

              h_pad = H - self.img_size[0]
              w_pad = W - self.img_size[1]

              padding_top = h_pad // 2
              padding_bottom = int(h_pad - padding_top)

              padding_left = w_pad // 2
              padding_right = int(w_pad - padding_left)

              return output[
                  :,
                  :,
                  padding_top : H - padding_bottom,
                  padding_left : W - padding_right,
              ]
    Attention结构诊断: null

  usage_examples:
    - 描述: 使用默认配置将 (181, 360) Patch 特征恢复为 (721, 1440) 四通道地表场
      示例代码: |
        import torch
        from onescience.modules.recovery.pangupatchrecovery2d import PanguPatchRecovery2D

        recovery = PanguPatchRecovery2D(
            img_size=(721, 1440),
            patch_size=(4, 4),
            in_chans=384,
            out_chans=4,
        )

        x = torch.randn(2, 384, 181, 360)
        y = recovery(x)

  knowledge:
    应用说明: >
      PanguPatchRecovery2D 所属的 Patch Recovery/上采样模块类别，
      位于“3D 体建模策略”与“U-Net 式下采样/上采样范式”的解码端，
      负责将统一 3D 体或 2D Patch 网格上的潜在特征张量重建回原始经纬网格上的物理变量场，
      在领域建模中充当“从潜在表示回到可观测物理空间”的桥梁，有助于在保持高分辨率输出的同时控制主干网络的计算规模。
    热点模型:
      - 模型: Pangu-Weather
        年份: 未知
        场景: >
          基于 ERA5 再分析数据的全球中期（1 h–7 d）天气预报，需要在 3D 体主干中下采样建模后恢复出 0.25°×0.25° 高分辨率的上空与地表场。
        方案: >
          采用 3D 体建模策略，将上空与地表变量通过 3D/2D Patch Embedding 统一为体张量 [8,360,181,C]，
          在 3DEST 编码–解码主干与 U-Net 式多尺度结构中完成特征提取，
          最后通过 Patch Recovery（上空使用逆 Patch 操作、地表使用二维上采样与裁剪）重建为原始 13 层上空场和地表场。
        作用: >
          保证网络在压缩空间进行建模的同时，输出仍精确对齐 ERA5 网格，
          使得损失函数与 RMSE/ACC 等指标可以在统一经纬网格上计算，并支撑极端天气与集合预报等下游分析。
        创新: >
          将 Patch Embedding/Recovery 嵌入到 3D 体建模和 Earth-specific Transformer 主干中，
          结合多尺度 U-Net 结构，实现“体内压缩建模 + 网格级高分辨率重建”的统一框架。
      - 模型: FuXi
        年份: 未知
        场景: >
          在基于 Swin Transformer V2 的 U-Transformer 主干中，对 2D 经纬网格上的多变量场做多尺度下采样建模后恢复到原始分辨率，用于全球天气预报。
        方案: >
          通过 Cube embedding 将输入场映射到特征张量 [C,180,360]，
          使用 Down Block 下采样到更低分辨率后在主干中建模，
          再用 Up Block 与跳跃连接上采样回 [C,180,360]，
          形成与 Patch Recovery 类似的“压缩编码–上采样重建”结构。
        作用: >
          在有限计算预算下统一处理多尺度空间结构，同时保持与输入经纬网格一致的输出分辨率，方便与 ERA5 等数据集进行比较与评估。
        创新: >
          将 Swin V2 的窗口注意力与 U-Net 下/上采样结构结合，
          在 2D 通道堆叠范式下延续 Patch Embedding/Recovery 思想，增强多尺度建模能力。
    最佳实践:
      - 在修改上游 Patch 划分或 3DEST 主干分辨率时，同步更新 img_size 与 patch_size，保持 H、W 与 Patch 网格之间的整数倍关系，避免过大的补零与裁剪。 [基于通用知识补充]
      - 训练与评估阶段统一使用恢复后的二维场作为损失与指标计算的输入，确保与纬度/面积加权等几何先验在同一经纬网格上作用。 [基于通用知识补充]
      - 在实验不同输出变量组合（仅地表/上空+地表）时，保持 out_chans 与损失权重配置一致，避免变量遗漏或重复。 [基于通用知识补充]
    常见错误:
      - 误设 patch_size 或 H'、W'，导致反卷积输出尺寸与 img_size 差异过大，裁剪后有效信息严重丢失。 [基于通用知识补充]
      - 调整上游 Patch Embedding 或 3DEST 主干配置后未同步更新 img_size/patch_size，造成恢复场与输入场在经纬度方向上错位。 [基于通用知识补充]
    论文参考:
      - 标题: Pangu-Weather 模型论文
        作者: 未在当前归纳文档中给出
        年份: 未知
        摘要: ""

  graph:
    类型关系:
      - 神经网络层
      - 上采样模块
      - Patch 恢复模块
    所属结构:
      - 2D Patch Embedding 与 Patch Recovery 子系统
      - Pangu-Weather 输出头
    依赖组件:
      - 3D Patch Embedding 与 Patch Recovery
      - 3D Earth-specific Transformer 主干
    变体组件:
      - PanguPatchRecovery3D
    使用模型:
      - Pangu-Weather
    类型兼容:
      输入:
        - PatchFeatureVolume2D
      输出:
        - RecoveredField2D
