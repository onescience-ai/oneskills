component:

  meta:
    name: XihePatchRecovery
    alias: XiheRecovery
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: decoder_component
    author: OneScience
    license: Apache-2.0
    tags:
      - xihe
      - patch_recovery
      - deconvolution
      - conv_transpose2d
      - high_resolution
      - asymmetric_patch

  concept:

    description: >
      XihePatchRecovery 是专门为羲和（Xihe）高分辨率全球气象大模型设计的二维 Patch 恢复模块。
      虽然其内部的计算逻辑（反卷积+中心裁剪）与盘古（Pangu-Weather）的二维恢复模块几乎完全一致，
      但它针对羲和模型的超高分辨率网格（如 2041 $\times$ 4320）和非对称的 Patch 划分策略（如 6 $\times$ 12）进行了特定的默认参数适配。

    intuition: >
      在处理极高分辨率的地球物理场时，经纬度的物理距离比例在投影到二维网格时可能会发生变化。
      羲和模型采用了非对称的 Patch 大小（高为 6，宽为 12），以更好地捕捉特定投影下的局部空间关联。
      该模块通过 `ConvTranspose2d` 利用非对称的 kernel 和 stride 将隐层特征放大，
      最后像裁剪相片白边一样，精确去除在编码阶段为了满足 $(6, 12)$ 整数倍而额外补齐的边缘网格点，还原真实的地球网格。

    problem_it_solves:
      - 将潜空间（Latent Space）中的隐层特征逆向解构回超高分辨率的真实物理状态空间
      - 支持非对称尺寸（Asymmetric）的 Patch 到 Grid 的上采样还原
      - 解决 $(2041, 4320)$ 这样特殊的超大网格尺寸无法被 $(6, 12)$ 整除而导致的边缘对齐问题

  theory:

    formula:

      deconvolution_recovery:
        expression: y_{raw} = \text{ConvTranspose2d}(x, \text{weight}, \text{stride}=(P_h, P_w), \text{kernel\_size}=(P_h, P_w))

      cropping:
        expression: y = y_{raw}[:, :, \text{pad}_{top} : H - \text{pad}_{bottom}, \text{pad}_{left} : W - \text{pad}_{right}]

    variables:

      x:
        name: LatentFeature
        shape: [B, in_chans, H', W']
        description: 编码器输出的低分辨率二维特征图

      P_h, P_w:
        name: AsymmetricPatchSize
        description: Patch 的高度和宽度（如 6 和 12），作为转置卷积的 kernel_size 和 stride

      y_{raw}:
        name: UncroppedOutput
        description: 经过转置卷积放大后，包含边缘 Padding 的初步输出特征

      y:
        name: TargetOutput
        shape: [B, out_chans, img_size[0], img_size[1]]
        description: 裁剪对齐后，与目标超高分辨率图像尺寸完全一致的输出

  structure:

    architecture: patch_decoder

    pipeline:

      - name: AsymmetricSpatialUpsampling
        operation: conv_transpose2d

      - name: PaddingCalculation
        operation: compute_difference_from_target_size

      - name: CenterCrop
        operation: slice_tensor_to_match_target

  interface:

    parameters:

      img_size:
        type: tuple[int, int]
        default: (2041, 4320)
        description: 最终输出的目标图像/网格尺寸 (H, W)，对应羲和模型的超高分辨率

      patch_size:
        type: tuple[int, int]
        default: (6, 12)
        description: 恢复时的非对称 Patch 尺寸，对应转置卷积的核大小和步长

      in_chans:
        type: int
        default: 192
        description: 输入隐层特征的通道数

      out_chans:
        type: int
        default: 96
        description: 恢复的物理变量或中间解码通道数

    inputs:

      x:
        type: Tensor
        shape: [B, in_chans, H', W']
        dtype: float32
        description: 待恢复的二维隐层张量

    outputs:

      output:
        type: Tensor
        shape: [B, out_chans, img_size[0], img_size[1]]
        description: 恢复并裁剪到目标尺寸的二维气象场

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn


      class XihePatchRecovery(nn.Module):

          """
              Pangu-Weather 模型中的二维 Patch 恢复模块，用反卷积将 Patch 特征还原为原始空间分辨率的二维场，并裁剪掉补零边界。
              
              Args:
                  img_size (tuple[int, int]): 输出目标图像尺寸 (H, W)
                  patch_size (tuple[int, int]): Patch 大小 (patch_h, patch_w)，即反卷积的 kernel_size 与 stride
                  in_chans (int): 输入特征通道数
                  out_chans (int): 输出图像通道数

              形状:
                  输入:  x 形状为 (B, in_chans, H', W')
                  输出:  y 形状为 (B, out_chans, img_size[0], img_size[1])

              Example:
                  >>> recovery = PanguPatchRecovery2D(
                  ...     img_size=(721, 1440),
                  ...     patch_size=(4, 4),
                  ...     in_chans=384,
                  ...     out_chans=4,
                  ... )
                  >>> x = torch.randn(2, 384, 181, 360)
                  >>> y = recovery(x)
                  >>> y.shape
                  torch.Size([2, 4, 721, 1440])
          """
          def __init__(self, 
                      img_size = (2041, 4320),
                      patch_size = (6, 12),
                      in_chans = 192,
                      out_chans = 96):
              super().__init__()
              self.img_size = img_size
              self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

          def forward(self, x):
              output = self.conv(x)
              _, _, H, W = output.shape
              h_pad = H - self.img_size[0]
              w_pad = W - self.img_size[1]

              padding_top = h_pad // 2
              padding_bottom = int(h_pad - padding_top)

              padding_left = w_pad // 2
              padding_right = int(w_pad - padding_left)

              return output[
                  :, :, padding_top : H - padding_bottom, padding_left : W - padding_right
              ]

  skills:

    build_xihe_recovery:

      description: 为超高分辨率气象数据构建基于非对称转置卷积的 Patch 恢复层

      inputs:
        - img_size
        - patch_size
        - in_chans
        - out_chans

      prompt_template: |
        构建一个 XihePatchRecovery 模块。
        目标空间分辨率为 {{img_size}}，非对称 Patch 大小为 {{patch_size}}。
        输入通道数为 {{in_chans}}，输出通道数为 {{out_chans}}。

    diagnose_asymmetric_recovery_issues:

      description: 分析非对称重构网络输出张量维度异常的问题

      checks:
        - asymmetric_kernel_size_mismatch_with_input_tensor_ratio
        - memory_overflow_due_to_ultra_high_resolution_output

  knowledge:

    usage_patterns:

      ultra_high_res_decoding:

        pipeline:
          - Encoder (Asymmetric Patch Embed)
          - Transformer Blocks
          - XihePatchRecovery (映射到超大规模 2041x4320 网格)

    design_patterns:

      asymmetric_unpatchify:

        structure:
          - 由于地球经纬度网格在某些投影或特殊分辨率下，相邻像素代表的实际物理距离存在差异。
          - 采用非对称的 $(6, 12)$ 步长和卷积核，使得特征在不同维度上以不同的速率放大，以更好地适应数据的内在物理结构。

    hot_models:

      - model: Xihe (羲和)
        year: 2023-2024
        role: 高分辨率数据驱动天气预报模型
        architecture: transformer_based
        attention_type: None (此处为解码端)

    model_usage_details:

      Xihe Global Model:

        img_size: (2041, 4320) （极高分辨率，可能对应 ~0.083 度或相似的超细粒度网格）
        patch_size: (6, 12) （非对称还原）

    best_practices:

      - 由于目标尺寸高达 $2041 \times 4320$，`ConvTranspose2d` 生成的特征图会占用极其庞大的显存（VRAM）。在模型训练或推理部署时，需要采用激活重计算（Activation Checkpointing）、混合精度（FP16/BF16）或模型并行策略来避免 OOM。
      - 当替换模型时，务必注意该模块需要与输入端的 Patch Embedding 的形状（stride/kernel）严格对称。

    anti_patterns:

      - （代码遗留问题警告）在源码的 Docstring 注释中，完全复制粘贴了 `PanguPatchRecovery2D` 的注释（包括 Example 中的类名和形状 `721, 1440`）。这种拷贝遗留（Copy-paste artifact）在维护大型代码库时极易误导后续开发者，建议在工程实践中修正 Docstring 以匹配实际的默认参数。

    paper_references:

      - title: "Xihe: A Data-Driven Model for Global Weather Forecasting" (或羲和团队相关公开文献)
        authors: Xihe Research Team
        year: 2024

  graph:

    is_a:
      - NeuralNetworkComponent
      - DecoderComponent
      - UpsamplingLayer

    part_of:
      - XiheWeatherModel
      - UltraHighResDecoder

    depends_on:
      - nn.ConvTranspose2d

    variants:
      - PanguPatchRecovery2D (对称 Patch 变体)

    used_in_models:
      - Xihe (羲和)

    compatible_with:

      inputs:
        - Tensor

      outputs:
        - Tensor