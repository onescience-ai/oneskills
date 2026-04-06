component:

  meta:
    name: PanguEmbedding3D
    alias: Pangu3DPatchEmbedding
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: embedding
    author: OneScience
    license: Apache-2.0
    tags:
      - embedding
      - 3d_patch_embedding
      - volumetic_embedding
      - earth_system_modeling
      - pangu_weather

  concept:

    description: >
      PanguEmbedding3D 旨在处理气象预报中包含了多个压力高度层（即 z 轴 / 垂直气压面）和平面 (lat, lon) 特征的三维地球大气状态输入变量。
      它借助带填充 (Padding) 支持的 3D 卷积核，将庞大的空间立方体划分成粗颗粒度的 3D 隐维体素。此层主要应用于 Pangu-Weather 的高空大尺度特征提取。

    intuition: >
      地球上的云系并不只是二维的一张薄饼，而是一个多垂直层次交织的巨大立体构造（在盘古气象里常见划分为13层）。
      PanguEmbedding3D 就像是在切一块很大的积木，如果边缘的高层数或是经纬边界无法完美划分，它会自动先给这块大积木涂满一圈透明包边（3D Zero Padding），然后用一个三维的模型模具将整块特征拓印出来，从而实现包含高空深度的完整特征嵌入。

    problem_it_solves:
      - 提供对于立体多气压高度层气象信息的同步联合嵌入方案
      - 处理垂直维度 (z-axis/level) 以及平面维度不均等导致的维度失配与边缘丢失效应


  theory:

    formula:

      padded_conv3d_projection:
        expression: |
          x_{pad} = \text{ZeroPad3d}_{\text{auto\_calculate}}(x_{in})
          x_{patch} = \text{Conv3D}(x_{pad}, \text{kernel\_size}=P_{3d}, \text{stride}=P_{3d})

    variables:

      x_{in}:
        name: UpperAirMeteorologicalField
        shape: [B, C, P, H, W]
        description: 包含了垂直气压层 P 的的高空气象特征体素

      P_{3d}:
        name: PatchSize3D
        shape: [3]
        description: 对应高度轴层 (level)、纬高 (H)、经宽 (W) 三个方面的池化降裁切分辨率


  structure:

    architecture: padded_volumetric_patch_embedding

    pipeline:

      - name: VolumetricBoundaryPadding
        operation: calculate_3D_remainder_and_zero_pad

      - name: VolumetricProjection
        operation: non_overlapping_conv3d

      - name: Normalization
        operation: permute_for_channel_layernorm


  interface:

    parameters:

      img_size:
        type: tuple[int, int, int]
        description: 输入预期的立体大小，即 (Level数, 纬度数, 经度数)，盘古模型中这往往是 (13, 721, 1440)

      patch_size:
        type: tuple[int, int, int]
        description: 使用 3D Patch 的长宽高降维参数，推荐如 (2, 4, 4) 取两个气压层进行打组

      embed_dim:
        type: int
        description: 打块后每一个组合像素映射的高维特征维数，如 192

      in_chans:
        type: int
        description: 高空气压层的气象参数类型信道总数，常见如 5 个（温、高、湿、东西风）

      norm_layer:
        type: nn.Module
        description: 附加在提取体素特征之后的标准特征稳定缩放层 (如 LayerNorm)

    inputs:

      x:
        type: Tensor
        shape: [B, C, P, H, W]
        dtype: float32
        description: 高空气象 3D 传感器域体素堆叠集合

    outputs:

      x_out:
        type: Tensor
        shape: [B, embed_dim, P', H', W']
        description: 输出提取完的隐藏态特征空间（自动向上取齐处理）


  types:

    UpperAirVolumes:
      shape: [B, C, P, H, W]
      description: 批次的具备多个垂直深层次的大气气压平面合集特征


  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn
      
      class PanguEmbedding3D(nn.Module):
          """
              将三维图像分割为不重叠的 patch 并嵌入到向量空间。
      
              Args:
                  img_size (tuple[int, int, int]): 输入图像尺寸 (P, H, W)
                  patch_size (tuple[int, int, int]): 每个 patch 的大小 (patch_p, patch_h, patch_w)
                  in_chans (int): 输入图像通道数
                  embed_dim (int): 每个 patch 嵌入后的向量维度
                  norm_layer (nn.Module, optional): 归一化层，默认为 None
      
              形状:
                  输入: (B, C, P, H, W)
                  输出: (B, embed_dim, P', H', W'), 其中 P' = ⌈P / patch_p⌉, H' = ⌈H / patch_h⌉, W' = ⌈W / patch_w⌉
      
              Example:
                  >>> patch_embed = PatchEmbed3D(
                  ...     img_size=(13, 128, 256),
                  ...     patch_size=(1, 4, 4),
                  ...     in_chans=5,
                  ...     embed_dim=192
                  ... )
                  >>> x = torch.randn(4, 5, 13, 128, 256)
                  >>> out = patch_embed(x)
                  >>> out.shape
                  torch.Size([4, 192, 13, 32, 64])
      
          """
      
          def __init__(self, 
                      img_size = (13, 721, 1440),
                      patch_size = (2, 4, 4),
                      in_chans = 5,
                      embed_dim = 192,
                      norm_layer = None):
      
              super().__init__()
      
              level, height, width = img_size
              l_patch_size, h_patch_size, w_patch_size = patch_size
      
              padding_left = (
                  padding_right
              ) = padding_top = padding_bottom = padding_front = padding_back = 0
      
              l_remainder = level % l_patch_size
              h_remainder = height % l_patch_size
              w_remainder = width % w_patch_size
      
              if l_remainder:
                  l_pad = l_patch_size - l_remainder
                  padding_front = l_pad // 2
                  padding_back = l_pad - padding_front
              if h_remainder:
                  h_pad = h_patch_size - h_remainder
                  padding_top = h_pad // 2
                  padding_bottom = h_pad - padding_top
              if w_remainder:
                  w_pad = w_patch_size - w_remainder
                  padding_left = w_pad // 2
                  padding_right = w_pad - padding_left
      
              self.pad = nn.ZeroPad3d(
                  (
                      padding_left,
                      padding_right,
                      padding_top,
                      padding_bottom,
                      padding_front,
                      padding_back,
                  )
              )
              self.proj = nn.Conv3d(
                  in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
              )
              if norm_layer is not None:
                  self.norm = norm_layer(embed_dim)
              else:
                  self.norm = None
      
          def forward(self, x: torch.Tensor):
              B, C, L, H, W = x.shape
              x = self.pad(x)
              x = self.proj(x)
              if self.norm:
                  x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
              return x

  skills:

    build_pangu_3d_embedding:

      description: 构建可以容错 3D 边界问题的高空气压三维特征起步映射器

      inputs:
        - img_size
        - patch_size

      prompt_template: |

        请构建能够将 3D 大气层状输入切分成 3维 Patch 的入口模块。
        重点针对 Z 轴（即 P/Level轴）产生不能整除的不完全特征引入了三方向 6 端点的 ZeroPad3D，并通过对应的 Permute 实现 LayerNorm 的安全接驳。




    diagnose_panguembedding3d:

      description: 分析 PanguEmbedding3D 模块在使用中的常见失效模式

      checks:
        - padding_or_resolution_errors
        - bad_device_alignment
        - output_shape_collapse
  knowledge:

    usage_patterns:

      pangu_upper_air_backbone:

        pipeline:
          - Gather P Layers Data (Geopotential, Temp, Humidity, Wind U/V)
          - PanguEmbedding3D (Volume Mapping)
          - Add 3D Position Bias
          - EarthSwin3D_Attention

    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 树立了物理 Z 轴也可以如图像一样进行深度降维切分的标杆工程实现
        architecture: 3D Earth Specific Swin Transformer

    best_practices:

      - 复杂的 Permute : `permute(0, 2, 3, 4, 1)` 后接 `LayerNorm` 后再换回通道优先排列（`0, 4, 1, 2, 3`），因为PyTorch提供的常规归一化很多是在最后一个维度求得极值方差。


    anti_patterns:

      - 忽略 Z 轴的不平衡性而采取普通剪切或直接插值缩放，破坏了预训练好的数值模拟预定义层内大气动力关系结构。

    paper_references:

      - title: "Accurate medium-range global weather forecasting with 3D neural networks"
        authors: Bi et al. (Nature)
        year: 2023


  graph:

    is_a:
      - 3DPatchEmbedding
      - DimensionalCompression

    part_of:
      - PanguWeatherModel
      - AtmosphericNeuralNets

    depends_on:
      - Conv3d
      - ZeroPad3d
      - Tensors Permutation

    variants:
      - FuxiEmbedding (针对 T 时序轴切分，而非 Z 气压轴)

    used_in_models:
      - Pangu-Weather

    compatible_with:

      inputs:
        - MultiplePressureLevelsField

      outputs:
        - Flattened3DFeatureBlock