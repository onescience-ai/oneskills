component:

  meta:
    name: FengWuEncoder
    alias: FengWu Weather Encoder
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: encoder
    author: OneScience
    license: Apache-2.0
    tags:
      - weather_forecasting
      - transformer
      - multi_scale
      - earth_science
      - patch_embedding

  concept:

    description: >
      FengWu编码器是专门用于气象预报的二维层次化编码器，对高分辨率气象场进行编码并输出中分辨率特征与高分辨率跳跃连接。
      采用"Patch嵌入 → 高分辨率处理 → 下采样 → 中分辨率处理"的流水线结构，通过窗口注意力机制高效处理地球球面数据，
      为解码器提供多尺度特征表示。

    intuition: >
      就像气象预报员先看高分辨率的卫星云图（高分辨率处理），然后关注大尺度天气系统（中分辨率处理），
      同时保留原始观测的细节信息（跳跃连接）。Patch嵌入将连续的气象场分割成可处理的区域块，
      窗口注意力确保计算效率的同时保持局部相关性。

    problem_it_solves:
      - 高分辨率气象数据的高效编码
      - 地球球面数据的空间建模
      - 多尺度特征的层次化提取
      - 编码器-解码器架构中的信息传递
      - 气象变量的时空特征提取

  theory:

    formula:

      patch_embedding:
        expression: |
          x_{patch} = \text{PatchEmbed2D}(x_{input})
          x_{seq} = \text{ReshapeAndTranspose}(x_{patch})

      hierarchical_processing:
        expression: |
          x_{high} = \text{TransformerBlocks}_{high}(x_{seq})
          skip = \text{ReshapeToSpatial}(x_{high})
          x_{middle} = \text{DownSample}(x_{high})
          x_{encoded} = \text{TransformerBlocks}_{middle}(x_{middle})

      window_attention:
        expression: |
          \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
          M: \text{window mask for local attention}

    variables:

      x_{input}:
        name: WeatherField
        shape: [batch, in_chans, H, W]
        description: 输入的气象场数据

      x_{patch}:
        name: PatchEmbeddings
        shape: [batch, dim, H_patch, W_patch]
        description: Patch嵌入后的特征

      skip:
        name: SkipConnection
        shape: [batch, input_resolution[0], input_resolution[1], dim]
        description: 高分辨率跳跃连接特征

      x_{encoded}:
        name: EncodedFeatures
        shape: [batch, middle_resolution[0] * middle_resolution[1], 2 * dim]
        description: 最终编码的中分辨率特征

  structure:

    architecture: hierarchical_transformer_encoder

    pipeline:

      - name: PatchEmbedding
        operation: pangu_embedding_2d

      - name: HighResolutionProcessing
        operation: earth_transformer_blocks (dim)

      - name: SkipConnectionExtraction
        operation: reshape_to_spatial

      - name: Downsampling
        operation: pangu_downsample_2d

      - name: MiddleResolutionProcessing
        operation: earth_transformer_blocks (dim * 2)

  interface:

    parameters:

      input_resolution:
        type: tuple[int, int]
        description: 高分辨率编码阶段的空间分辨率 (H1, W1)

      middle_resolution:
        type: tuple[int, int]
        description: 下采样后中分辨率编码阶段的空间分辨率 (Hm, Wm)

      in_chans:
        type: int
        description: 输入气象变量通道数

      img_size:
        type: tuple[int, int]
        description: 原始输入场尺寸 (H, W)

      patch_size:
        type: tuple[int, int]
        description: Patch大小 (patch_h, patch_w)

      dim:
        type: int
        description: 高分辨率阶段的特征维度

      depth:
        type: int
        description: 高分辨率Transformer块层数

      depth_middle:
        type: int
        description: 中分辨率Transformer块层数

      num_heads:
        type: int or tuple[int, int]
        description: 多头自注意力头数配置（单个或(high, middle)）

      window_size:
        type: int or tuple[int, int]
        description: 窗口注意力窗口大小

      mlp_ratio:
        type: float
        description: 前馈网络隐藏层与特征维度的比例

      qkv_bias:
        type: bool
        description: 是否在QKV投影中使用偏置

      qk_scale:
        type: float or None
        description: QK点积缩放因子

      drop:
        type: float
        description: 特征上的dropout比例

      attn_drop:
        type: float
        description: 注意力权重上的dropout比例

      drop_path:
        type: float or Sequence[float]
        description: DropPath / Stochastic Depth比例

      norm_layer:
        type: nn.Module
        description: 归一化层类型

    inputs:

      x:
        type: WeatherField
        shape: [batch, in_chans, H, W]
        description: 输入气象场

    outputs:

      x:
        type: EncodedFeatures
        shape: [batch, middle_resolution[0] * middle_resolution[1], 2 * dim]
        description: 编码的中分辨率特征

      skip:
        type: SkipConnection
        shape: [batch, input_resolution[0], input_resolution[1], dim]
        description: 高分辨率跳跃连接特征

  types:

    WeatherField:
      shape: [batch, channels, height, width]
      description: 气象场数据

    EncodedFeatures:
      shape: [batch, seq_len, feature_dim]
      description: 编码后的序列特征

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules.embedding.oneembedding import OneEmbedding
      from onescience.modules.sample.onesample import OneSample
      from onescience.modules.transformer.onetransformer import OneTransformer

      class FengWuEncoder(nn.Module):
          def __init__(self, input_resolution=(181, 360), middle_resolution=(91, 180),
                       in_chans=37, img_size=(721, 1440), patch_size=(4, 4),
                       dim=192, depth=2, depth_middle=6, num_heads=(6, 12),
                       window_size=(6, 12), **kwargs):
              super().__init__()
              
              # Patch嵌入
              self.patchembed2d = OneEmbedding(
                  style="PanguEmbedding2D",
                  img_size=img_size, patch_size=patch_size,
                  in_chans=in_chans, embed_dim=dim
              )
              
              # 高分辨率处理
              self.blocks = nn.ModuleList([
                  OneTransformer(style="EarthTransformer2DBlock", dim=dim,
                               input_resolution=input_resolution, num_heads=num_heads,
                               window_size=window_size)
                  for _ in range(depth)
              ])
              
              # 下采样
              self.downsample = OneSample(
                  style="PanguDownSample2D", in_dim=dim,
                  input_resolution=input_resolution, output_resolution=middle_resolution
              )
              
              # 中分辨率处理
              self.blocks_middle = nn.ModuleList([
                  OneTransformer(style="EarthTransformer2DBlock", dim=dim * 2,
                               input_resolution=middle_resolution, num_heads=num_heads_middle,
                               window_size=window_size)
                  for _ in range(depth_middle)
              ])

          def forward(self, x):
              x = self.patchembed2d(x)
              B, C, Lat, Lon = x.shape
              x = x.reshape(B, C, -1).transpose(1, 2)
              
              # 高分辨率处理
              for blk in self.blocks:
                  x = blk(x)
              
              # 提取跳跃连接
              skip = x.reshape(B, Lat, Lon, C)
              
              # 下采样和中分辨率处理
              x = self.downsample(x)
              for blk in self.blocks_middle:
                  x = blk(x)
              
              return x, skip

  skills:

    build_weather_encoder:

      description: 构建气象预报专用的层次化编码器

      inputs:
        - input_resolution
        - middle_resolution
        - in_chans
        - dim
        - depth
        - window_size

      prompt_template: |

        构建FengWu气象预报编码器，支持多尺度处理和窗口注意力。

        参数：
        input_resolution = {{input_resolution}}
        middle_resolution = {{middle_resolution}}
        in_chans = {{in_chans}}
        dim = {{dim}}
        depth = {{depth}}
        window_size = {{window_size}}

        要求：
        1. 支持高分辨率到中分辨率的层次化处理
        2. 窗口注意力适配地球球面数据
        3. 输出跳跃连接供解码器使用
        4. Patch嵌入处理连续气象场

    optimize_weather_encoding:

      description: 优化气象编码器的计算效率

      checks:
        - window_size_compatibility (窗口大小与分辨率的兼容性)
        - memory_efficiency (大分辨率下的显存使用)
        - feature_preservation (跳跃连接的信息保持)

  knowledge:

    usage_patterns:

      global_weather_encoding:

        pipeline:
          - Input: 高分辨率气象场
          - Patch Embedding: 区域块嵌入
          - High Resolution: 局部天气模式
          - Down Sample: 分辨率降低
          - Middle Resolution: 全球天气系统
          - Output: 多尺度特征

      multi_scale_analysis:

        pipeline:
          - Fine Scale: 细粒度特征提取
          - Coarse Scale: 粗粒度特征提取
          - Skip Connections: 跨层级信息传递
          - Hierarchical Features: 层次化特征表示

    hot_models:

      - model: FengWu
        year: 2022
        role: 华为云盘古气象大模型编码器
        architecture: multi-scale transformer

      - model: Pangu-Weather
        year: 2023
        role: 盘古气象模型，类似架构
        architecture: earth transformer

      - model: FourCastNet
        year: 2023
        role: NVIDIA的气象预报模型
        architecture: vision transformer

    best_practices:

      - Patch大小应根据气象数据的物理特性选择
      - 窗口大小需考虑地球曲率和计算效率的平衡
      - 跳跃连接对保持细节信息至关重要
      - 中分辨率层数通常多于高分辨率，以充分处理全局信息

    anti_patterns:

      - Patch大小选择不当导致信息丢失或计算冗余
      - 忽略地球球面几何特性
      - 跳跃连接维度不匹配
      - 窗口注意力设置不当影响计算效率

    paper_references:

      - title: "FengWu: Pushing the Skillful Forecast Horizon Beyond 10 days Using Global Forecast Data"
        authors: Huawei Cloud Team
        year: 2023

      - title: "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Weather Forecasting"
        authors: Bi et al.
        year: 2023

      - title: "FourCastNet: Forecasting at Scale with High-Resolution Neural Networks"
        authors: Pathak et al.
        year: 2022

  graph:

    is_a:
      - NeuralNetworkEncoder
      - MultiScaleProcessor
      - WeatherForecastingComponent

    part_of:
      - EncoderDecoderArchitecture
      - WeatherPredictionSystem
      - EarthScienceModels

    depends_on:
      - PanguEmbedding2D
      - EarthTransformer2DBlock
      - PanguDownSample2D
      - WindowAttention

    variants:
      - StandardEncoder (无多尺度处理)
      - CNNEncoder (卷积版本)
      - GraphEncoder (图神经网络版本)

    used_in_models:
      - FengWu
      - Pangu-Weather
      - 其他气象预报模型

    compatible_with:

      inputs:
        - WeatherField
        - ClimateVariables

      outputs:
        - EncodedFeatures
        - SkipConnection
        - MultiScaleFeatures
