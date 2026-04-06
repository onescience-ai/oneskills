component:

  meta:
    name: PanguPatchRecovery3D
    alias: PatchRecovery3D
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: decoder_component
    author: OneScience
    license: Apache-2.0
    tags:
      - pangu_weather
      - patch_recovery
      - 3d_deconvolution
      - conv_transpose3d
      - spatial_cropping

  concept:

    description: >
      PanguPatchRecovery3D 是盘古气象大模型（Pangu-Weather）中的三维 Patch 恢复模块。
      与 2D 版本类似，它主要负责将低分辨率的隐层 3D Patch 特征（通常对应高空多压力层的大气数据）通过三维转置卷积（ConvTranspose3d）还原为原始的高分辨率三维物理空间网格（层数、纬度、经度）。
      同时，该模块对输出张量在深度（层数）、高度（纬度）和宽度（经度）三个维度上执行自动裁剪，以消除编码时为了适配 Patch 大小而引入的补零边界。

    intuition: >
      地球的高空大气是一个三维空间。在 3D Earth-Specific Transformer 架构中，数据被划分为三维的立体块（Voxels/Patches，如 2x4x4）。
      在解码器末端，我们需要将这些高维隐空间特征重新放大回物理空间。转置卷积在三个维度上同时进行步长放缩，将其恢复为完整的 3D 气象场；
      由于实际的物理网格尺寸（如 13 个压力层、721 纬度、1440 经度）通常无法被三维 Patch 大小完美整除，因此必须在深度、高度和宽度三个方向上将多余的边缘像“削苹果皮”一样精准裁剪掉。

    problem_it_solves:
      - 将潜空间中的 3D 特征图（Latent Voxels）逆向映射回真实的三维高空物理状态空间
      - 解决 3D 气象网格数据（如 13x721x1440）无法被 3D Patch Size（如 2x4x4）完美整除所导致的维度不匹配问题
      - 实现无重叠的三维 Un-patchify 操作并保证物理场坐标系的绝对对齐

  theory:

    formula:

      deconvolution_recovery_3d:
        expression: y_{raw} = \text{ConvTranspose3d}(x, \text{weight}, \text{stride}=P, \text{kernel\_size}=P)

      cropping_3d:
        expression: y = y_{raw}[:, :, \text{pad}_{front} : Pl - \text{pad}_{back}, \text{pad}_{top} : Lat - \text{pad}_{bottom}, \text{pad}_{left} : Lon - \text{pad}_{right}]

    variables:

      x:
        name: LatentFeature
        shape: [B, in_chans, L', H', W']
        description: 编码器输出的低分辨率三维特征张量

      P:
        name: PatchSize3D
        description: 3D Patch 的大小（元组），作为三维转置卷积的 kernel_size 和 stride

      y_{raw}:
        name: UncroppedOutput
        description: 经过转置卷积放大后，包含三个维度边缘 Padding 的初步输出张量

      y:
        name: TargetOutput
        shape: [B, out_chans, img_size[0], img_size[1], img_size[2]]
        description: 裁剪对齐后，与目标三维网格尺寸完全一致的输出

  structure:

    architecture: patch_decoder_3d

    pipeline:

      - name: SpatialUpsampling3D
        operation: conv_transpose3d

      - name: PaddingCalculation3D
        operation: compute_difference_from_target_size_for_all_three_dims

      - name: CenterCrop3D
        operation: slice_tensor_to_match_target_in_3d

  interface:

    parameters:

      img_size:
        type: tuple[int, int, int]
        default: (13, 721, 1440)
        description: 最终输出的目标三维场尺寸 (Pl, Lat, Lon)，分别对应垂直层数、纬度和经度

      patch_size:
        type: tuple[int, int, int]
        default: (2, 4, 4)
        description: 恢复时的 3D Patch 尺寸 (patch_l, patch_h, patch_w)，对应转置卷积的核大小和步长

      in_chans:
        type: int
        default: 384 (192*2)
        description: 输入隐层特征的通道数

      out_chans:
        type: int
        default: 5
        description: 恢复的高空物理变量通道数（如 Z, Q, T, U, V 五个核心变量）

    inputs:

      x:
        type: Tensor
        shape: "[B, in_chans, L', H', W']"
        dtype: float32
        description: 待恢复的三维隐层张量

    outputs:

      output:
        type: Tensor
        shape: "[B, out_chans, img_size[0], img_size[1], img_size[2]]"
        description: 恢复并裁剪到目标尺寸的三维高空气象场

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn

      class PanguPatchRecovery3D(nn.Module):
          """
              Pangu-Weather 模型中的三维 Patch 恢复模块，用三维反卷积将 Patch 特征还原为原始层数与空间分辨率的三维场，并裁剪掉补零边界。
          """
          def __init__(self, img_size = (13, 721, 1440), 
                       patch_size = (2, 4, 4),
                       in_chans = 192*2, 
                       out_chans = 5):
              super().__init__()
              self.img_size = img_size
              self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

          def forward(self, x: torch.Tensor):
              output = self.conv(x)
              _, _, Pl, Lat, Lon = output.shape

              pl_pad = Pl - self.img_size[0]
              lat_pad = Lat - self.img_size[1]
              lon_pad = Lon - self.img_size[2]

              padding_front = pl_pad // 2
              padding_back = pl_pad - padding_front

              padding_top = lat_pad // 2
              padding_bottom = lat_pad - padding_top

              padding_left = lon_pad // 2
              padding_right = lon_pad - padding_left

              return output[
                  :,
                  :,
                  padding_front : Pl - padding_back,
                  padding_top : Lat - padding_bottom,
                  padding_left : Lon - padding_right,
              ]

  skills:

    build_pangu_recovery_3d:

      description: 为 3D 高空气象数据构建基于三维转置卷积的 Patch 恢复层

      inputs:
        - img_size
        - patch_size
        - in_chans
        - out_chans

      prompt_template: |
        构建一个 PanguPatchRecovery3D 模块。
        目标三维网格分辨率为 {{img_size}}，3D Patch大小为 {{patch_size}}。
        输入特征通道数为 {{in_chans}}，目标高空物理变量通道数为 {{out_chans}}。

    diagnose_recovery_crop_issues_3d:

      description: 分析重构网络输出张量的三维维度（深度/纬度/经度）与目标 Label 无法对齐的问题

      checks:
        - incorrect_depth_crop_logic
        - mismatch_between_pl_dimension_and_actual_pressure_levels

  knowledge:

    usage_patterns:

      upper_air_variable_prediction:

        pipeline:
          - Encoder (3D Patch Embed)
          - Transformer Blocks (3D Earth-Specific)
          - PanguPatchRecovery3D (输出 5 个高空变量分布在 13 个压力层上)
          - Loss Computation

    design_patterns:

      unpatchify_with_crop_3d:

        structure:
          - 使用无重叠的三维转置卷积（stride == kernel_size）放大 3D 立体特征图
          - 针对垂直方向（Pressure Levels）和水平方向（Lat/Lon）分别计算 Padding 并进行三维对称裁剪，保证数据还原在三维空间中的绝对物理中心对齐

    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 高分辨率数据驱动全球天气预报模型
        architecture: 3D_earth_specific_transformer
        attention_type: None (此处作为 3D 高空子模型的 Decoder 最后一层)

    model_usage_details:

      Pangu-Weather 3D Upper-Air Sub-model:

        img_size: (13, 721, 1440) 对应 13 个标准气压层和 0.25 度的全球水平网格
        out_chans: 5 (对应位势高度 Z、比湿 Q、温度 T、经向风 U、纬向风 V)
        patch_size: (2, 4, 4)

    best_practices:

      - 由于 3D 转置卷积的显存占用和计算量极大，在调用该模块时需要密切关注 Batch Size，以免触发 OOM (Out Of Memory)。
      - 必须保证 `PanguPatchRecovery3D` 初始化的 `img_size` 与三维物理空间训练数据严格一致。特别是高度层维度（Pl），如果有 13 层数据，编码端的 padding 可能将其补齐到 14，此处必须精准还原回 13。
      - 与 2D 相同，`pl_pad - padding_front` 的逻辑保证了在需要截断奇数层时代码的健壮性。

    anti_patterns:

      - 误将 `patch_size` 设置为各个方向上非对称比例（如 (1, 4, 4) 但原模型期待 (2, 4, 4)），会导致三维转置卷积后维度无法对应，引发 `IndexError`。
      - 在处理 3D 切片时，忽略了深度维度的坐标系顺序（`[Pl, Lat, Lon]` vs `[Lat, Lon, Pl]`），导致输出张量形状错乱。

    paper_references:

      - title: "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast"
        authors: Bi et al.
        year: 2023

  graph:

    is_a:
      - NeuralNetworkComponent
      - DecoderComponent
      - UpsamplingLayer3D

    part_of:
      - PanguWeatherModel
      - 3DVisionTransformerDecoder

    depends_on:
      - nn.ConvTranspose3d

    variants:
      - PanguPatchRecovery2D (2D 表面参数变体)

    used_in_models:
      - Pangu-Weather (Upper-Air Model)

    compatible_with:

      inputs:
        - Tensor (3D Volume)

      outputs:
        - Tensor (3D Volume)