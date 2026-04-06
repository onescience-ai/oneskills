component:

  meta:
    name: PanguPatchRecovery2D
    alias: PatchRecovery2D
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: decoder_component
    author: OneScience
    license: Apache-2.0
    tags:
      - pangu_weather
      - patch_recovery
      - deconvolution
      - conv_transpose2d
      - spatial_cropping

  concept:

    description: >
      PanguPatchRecovery2D 是盘古气象大模型（Pangu-Weather）中的二维 Patch 恢复模块。
      它的主要功能是通过二维转置卷积（ConvTranspose2d）将低分辨率的隐层 Patch 特征还原为高分辨率的原始物理空间（二维场，如地表变量）。
      同时，该模块内置了自动裁剪（Crop）逻辑，用于去除输入端为了适配 Patch 划分而额外填充（Padding）的边界。

    intuition: >
      在基于 Vision Transformer 的架构中，输入的图像（或气象场）首先被切分为一个个互不重叠的 Patch（如 4x4）。
      在网络末端，我们需要将这些高维的特征“反向”解构回原始的像素/网格点。
      转置卷积利用学习到的权重，将每一个 Patch 特征重新“放大并涂抹”成 4x4 的局部网格；
      由于地球网格的纬度（如 721）往往无法被 Patch 大小（如 4）整除，编码时通常会进行补零。因此解码的最后一步就是像裁切照片白边一样，精准切掉多余的边界，恢复 $721 \times 1440$ 的原始尺寸。

    problem_it_solves:
      - 将潜空间（Latent Space）中的 2D 特征图逆向映射回真实的物理状态空间
      - 解决气象网格数据分辨率（如 $721 \times 1440$）无法被 Patch Size（如 4）完美整除导致的尺寸不匹配问题
      - 利用步长等于卷积核大小（stride = kernel_size）的转置卷积实现无重叠的 Un-patchify 操作

  theory:

    formula:

      deconvolution_recovery:
        expression: y_{raw} = \text{ConvTranspose2d}(x, \text{weight}, \text{stride}=P, \text{kernel\_size}=P)

      cropping:
        expression: y = y_{raw}[:, :, \text{pad}_{top} : H - \text{pad}_{bottom}, \text{pad}_{left} : W - \text{pad}_{right}]

    variables:

      x:
        name: LatentFeature
        shape: [B, in_chans, H', W']
        description: 编码器输出的低分辨率二维特征图

      P:
        name: PatchSize
        description: Patch 的大小，作为转置卷积的 kernel_size 和 stride

      y_{raw}:
        name: UncroppedOutput
        description: 经过转置卷积放大后，包含边缘 Padding 的初步输出特征

      y:
        name: TargetOutput
        shape: [B, out_chans, img_size[0], img_size[1]]
        description: 裁剪对齐后，与目标图像尺寸完全一致的输出

  structure:

    architecture: patch_decoder

    pipeline:

      - name: SpatialUpsampling
        operation: conv_transpose2d

      - name: PaddingCalculation
        operation: compute_difference_from_target_size

      - name: CenterCrop
        operation: slice_tensor_to_match_target

  interface:

    parameters:

      img_size:
        type: tuple[int, int]
        default: (721, 1440)
        description: 最终输出的目标图像/网格尺寸 (H, W)

      patch_size:
        type: tuple[int, int]
        default: (4, 4)
        description: 恢复时的 Patch 尺寸，对应转置卷积的核大小和步长

      in_chans:
        type: int
        default: 384 (192*2)
        description: 输入隐层特征的通道数

      out_chans:
        type: int
        default: 4
        description: 恢复的物理变量通道数（如地表模型的 4 个气象要素）

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

      class PanguPatchRecovery2D(nn.Module):
          """
              Pangu-Weather 模型中的二维 Patch 恢复模块，用反卷积将 Patch 特征还原为原始空间分辨率的二维场，并裁剪掉补零边界。
          """
          def __init__(self, 
                      img_size = (721, 1440),
                      patch_size = (4, 4),
                      in_chans = 192*2,
                      out_chans = 4):
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

    build_pangu_recovery_2d:

      description: 为 2D 气象数据构建基于转置卷积的 Patch 恢复层

      inputs:
        - img_size
        - patch_size
        - in_chans
        - out_chans

      prompt_template: |
        构建一个 PanguPatchRecovery2D 模块。
        目标空间分辨率为 {{img_size}}，Patch大小为 {{patch_size}}。
        输入通道数为 {{in_chans}}，目标物理通道数为 {{out_chans}}。

    diagnose_recovery_crop_issues:

      description: 分析重构网络输出张量维度与目标 Label 无法对齐的问题

      checks:
        - incorrect_crop_logic_causing_off_by_one_error
        - mismatch_between_img_size_parameter_and_actual_ground_truth

  knowledge:

    usage_patterns:

      surface_variable_prediction:

        pipeline:
          - Encoder (2D Patch Embed)
          - Transformer Blocks (2D Earth-Specific)
          - PanguPatchRecovery2D (输出 4 个地表变量，如 MSLP, U10, V10, T2M)
          - Loss Computation

    design_patterns:

      unpatchify_with_crop:

        structure:
          - 使用无重叠的转置卷积（stride == kernel_size）快速且均匀地放大特征图
          - 通过动态计算 `H - img_size[0]` 并在四周对称裁剪，完美抵消了编码时 `Pad` 模块引入的边缘像素，确保数据空间强一致性

    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 高分辨率数据驱动全球天气预报模型
        architecture: 3D_earth_specific_transformer
        attention_type: None (此处作为模型的 Decoder 最后一层)

    model_usage_details:

      Pangu-Weather 2D Surface Sub-model:

        img_size: (721, 1440) 对应 0.25 度的全球经纬度网格
        out_chans: 4
        patch_size: (4, 4)

    best_practices:

      - 必须保证 `PanguPatchRecovery2D` 初始化的 `img_size` 与物理空间训练数据（Ground Truth）的尺寸严格一致，否则会导致后续 Loss 计算因形状不匹配而崩溃。
      - `h_pad // 2` 结合 `h_pad - padding_top` 的设计能够健壮地处理需要裁剪的像素数为奇数的情况（虽然在 4x4 patch 和 721 分辨率下，通常是两端各裁剪多余的一小部分）。

    anti_patterns:

      - 在编码端（Patch Embedding）使用了不对称的 Padding，但在解码端（Recovery）使用对称裁剪，会导致恢复的物理场在经纬度空间上发生平移错位（Shifted grid）。应确保编码器和解码器的 Padding/Cropping 逻辑严格互逆。

    paper_references:

      - title: "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast"
        authors: Bi et al.
        year: 2023

  graph:

    is_a:
      - NeuralNetworkComponent
      - DecoderComponent
      - UpsamplingLayer

    part_of:
      - PanguWeatherModel
      - VisionTransformerDecoder

    depends_on:
      - nn.ConvTranspose2d

    variants:
      - PanguPatchRecovery3D (引入高度/压力层维度的变体)
      - PixelShuffle (另一种常用的分辨率恢复方法)

    used_in_models:
      - Pangu-Weather

    compatible_with:

      inputs:
        - Tensor

      outputs:
        - Tensor