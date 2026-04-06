component:

  meta:
    name: FourCastNetEmbedding
    alias: FourCastNetPatchEmbedding
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: embedding
    author: OneScience
    license: Apache-2.0
    tags:
      - embedding
      - patch_embedding
      - vision_transformer
      - earth_system_modeling
      - fourcastnet

  concept:

    description: >
      FourCastNetEmbedding 是 FourCastNet 气象预测模型的 2D Patch 嵌入模块。
      它使用标准的 2D 卷积操作，将高分辨率的气象要素网格（例如 720x1440）划分为不重叠的小图块 (Patches)。
      随后，将这些空间维度的特征映射并展平为一维序列形式，以便直接馈入后续的 Vision Transformer 编码器。

    intuition: >
      处理高分辨率气象变量，如果依然用 CNN 处理全局那感受野受限；如果要用 Transformer 逐像素注意力会导致算力爆炸。
      此处的 2D 卷积就像是一个“分屏器”，把一张极大的全球气象场图片均匀地裁剪成“小格块（Patch）”，然后将每个小格块压缩为代表其特征的一维向量，就像是把图像翻译成了一句包含许多“词”的空间句子。

    problem_it_solves:
      - 解决输入高分辨率地球科学数据给 Transformer 处理时序列过长的问题
      - 将不同通道（如温度、风速等多个气象物理量）的原始特征在局域块内聚合为一个高维语义表示


  theory:

    formula:

      patch_projection:
        expression: |
          x_{patch} = \text{Conv2D}(x_{in}, \text{kernel\_size}=P, \text{stride}=P)
          x_{seq} = \text{Flatten}(x_{patch}, \text{start\_dim}=2)

    variables:

      P:
        name: PatchSize
        shape: [2]
        description: Patch 的高宽 (Plat, Plon)，决定特征降采样比例

      x_{in}:
        name: MeteorologicalField
        shape: [B, C, lat, lon]
        description: 气象场的原始二维网格图像

      C:
        name: VariableChannels
        description: 多变量气象信道数量（如 19个预报变量）


  structure:

    architecture: patch_embedding

    pipeline:

      - name: SpatialPatching
        operation: non_overlapping_strided_conv2d

      - name: SequenceFormatting
        operation: flatten_and_transpose


  interface:

    parameters:

      img_size:
        type: tuple[int, int]
        description: 预期的输入气象场分辨率，默认 (720, 1440) 对应 0.25°

      patch_size:
        type: tuple[int, int]
        description: 给定裁剪块的高宽配置，例如 (8, 8)

      in_chans:
        type: int
        description: 输入变量通道数，如 FourCastNet 典型值为 19

      embed_dim:
        type: int
        description: Patch 投射目标的隐层维度，如 768

    inputs:

      x:
        type: Tensor
        shape: [B, C, lat, lon]
        dtype: float32
        description: 单帧多通道高空与地表融合气象场

    outputs:

      x_seq:
        type: Tensor
        shape: [B, num_patches, embed_dim]
        description: 投射和展平后的 Patch 序列（此处的 num_patches=(lat//Plat) * (lon//Plon)）


  types:

    LatLonGrid:
      shape: [B, C, lat, lon]
      description: 通用经纬度网格的 2D 数据排列张量


  implementation:

    framework: pytorch

    code: |
      import torch
      import torch.nn as nn
      
      class FourCastNetEmbedding(nn.Module):
          """
              FourCastNet 的 2D Patch Embedding 模块。
      
              使用 2D 卷积将气象场图像划分为不重叠的 Patch 并投影到嵌入空间，
              是 FourCastNet 编码器的入口层。与 FuxiEmbedding 的 3D 卷积不同，
              该模块仅处理单帧二维气象场，输出展平为序列形式供后续 Transformer 使用。
      
              Args:
                  img_size (tuple[int, int], optional): 输入气象场的空间分辨率 (lat, lon)，
                      默认为 (720, 1440)。
                  patch_size (tuple[int, int], optional): Patch 大小 (Plat, Plon)，
                      默认为 (8, 8)。
                  in_chans (int, optional): 输入气象变量的通道数，默认为 19。
                  embed_dim (int, optional): Patch 嵌入维度，默认为 768。
      
              形状:
                  - 输入 x: (B, C, lat, lon)，其中 C = in_chans
                  - 输出:   (B, num_patches, embed_dim)
                      其中 num_patches = (lat // Plat) * (lon // Plon)
      
              Examples:
                  >>> # 典型 FourCastNet 配置
                  >>> # 分辨率 720×1440，Patch 大小 8×8
                  >>> # num_patches = (720//8) * (1440//8) = 90 * 180 = 16200
                  >>> embedding = FourCastNetEmbedding(
                  ...     img_size=(720, 1440),
                  ...     patch_size=(8, 8),
                  ...     in_chans=19,
                  ...     embed_dim=768,
                  ... )
                  >>> x = torch.randn(2, 19, 720, 1440)  # (B, C, lat, lon)
                  >>> out = embedding(x)
                  >>> out.shape
                  torch.Size([2, 16200, 768])
          """
          def __init__(self, 
                       img_size=(720, 1440), 
                       patch_size=(8, 8), 
                       in_chans=19, 
                       embed_dim=768):
              super().__init__()
              num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
              self.img_size = img_size
              self.patch_size = patch_size
              self.num_patches = num_patches
              self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
      
          def forward(self, x):
              B, C, H, W = x.shape
              assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
              x = self.proj(x).flatten(2).transpose(1, 2)
              return x
      
      

  skills:

    build_fourcastnet_embedding:

      description: 构建针对高分辨率经纬网格图像切割和处理的特征提取入口

      inputs:
        - img_size
        - patch_size
        - in_chans
        - embed_dim

      prompt_template: |

        构建 2D Conv Patch Embedding，步长等于核大小，并提供输出空间维展平为 Transformer 标准 Sequence 表示。




    diagnose_fourcastnetembedding:

      description: 分析 FourCastNetEmbedding 模块在使用中的常见失效模式

      checks:
        - padding_or_resolution_errors
        - bad_device_alignment
        - output_shape_collapse
  knowledge:

    usage_patterns:

      fourcastnet_vit_backbone:

        pipeline:
          - Concat 2D meteorological fields (Surface + Pressure levels)
          - FourCastNetEmbedding (Transform to sequence)
          - Add Absolute Positional Embedding
          - Vision Transformer Blocks (including AFNO)

    hot_models:

      - model: FourCastNet
        year: 2022
        role: 最早将 Vision Transformer 及 AFNO 应用于中长期高分辨率全球气象预测
        architecture: AFNO / ViT

    best_practices:

      - 由于气象场经纬度大小不一是常态，在前向网络计算中加入了 `assert` 判断捕捉非法输入是构建工程组件的良性实践。
      - 当模型需要在更大分辨率上推理时应基于原有的位置编码作插值，但 `img_size` 的严格约束在此起到保护作用。


    anti_patterns:

      - 未展平特征使得其不匹配标准 Transformer 或 AFNO 接受的 `[B, N, C]` 的 3D 张量形状。


    paper_references:

      - title: "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators"
        authors: Pathak et al.
        year: 2022


  graph:

    is_a:
      - PatchEmbedding
      - DimensionalityReduction

    part_of:
      - FourCastNet
      - EarthSystemModels

    depends_on:
      - Conv2d

    variants:
      - ViTPatchEmbed
      - 3DPatchEmbed (如 FuxiEmbedding)

    used_in_models:
      - FourCastNet

    compatible_with:

      inputs:
        - LatLonGrid

      outputs:
        - SequenceEmbedding