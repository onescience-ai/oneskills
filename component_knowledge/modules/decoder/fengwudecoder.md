component:

  meta:
    name: FengWuDecoder
    alias: FengWu Weather Decoder
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: decoder
    author: OneScience
    license: Apache-2.0
    tags:
      - weather_forecasting
      - transformer
      - multi_scale
      - earth_science
      - patch_recovery

  concept:

    description: >
      FengWu解码器是专门用于气象预报的多尺度Transformer解码器模块。
      它采用"中间层处理 → 上采样 → 精细层处理 → 跳跃连接融合 → Patch恢复"的流水线结构，
      能够将编码器输出的中间分辨率特征逐步恢复到原始高分辨率气象场。
      该模块特别适用于全球气象预报系统，通过窗口注意力机制高效处理地球球面数据。

    intuition: >
      就像气象预报员先看大尺度天气系统（中间层处理），然后关注局部细节（精细层处理），
      最后结合历史观测数据（跳跃连接）给出最终预报。上采样过程相当于从低分辨率卫星云图
      推断高分辨率天气细节，Patch恢复则是将网格数据还原为真实的地理坐标场。

    problem_it_solves:
      - 气象预报中多尺度特征融合的需求
      - 地球球面数据的高效处理（窗口注意力）
      - 从低分辨率隐空间恢复高分辨率物理场
      - 编码器-解码器架构中的信息传递与细节恢复
      - 气象变量的时空连续性保持

  theory:

    formula:

      multi_scale_processing:
        expression: |
          x_{middle} = \text{TransformerBlocks}_{middle}(x_{input})
          x_{upsampled} = \text{Upsample}(x_{middle})
          x_{refined} = \text{TransformerBlocks}_{output}(x_{upsampled})
          x_{fused} = \text{Concat}(x_{refined}, x_{skip})
          output = \text{PatchRecovery}(x_{fused})

      window_attention:
        expression: |
          \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
          M: \text{window mask for local attention}

    variables:

      x_{input}:
        name: MiddleResolutionFeatures
        shape: [batch, middle_lat * middle_lon, dim * 2]
        description: 编码器输出的中间分辨率特征序列

      x_{skip}:
        name: SkipConnectionFeatures
        shape: [batch, output_lat, output_lon, dim]
        description: 来自编码器的跳跃连接特征，保留细节信息

      M:
        name: WindowMask
        description: 窗口注意力掩码，限制注意力在局部地理区域内

  structure:

    architecture: multi_scale_transformer_decoder

    pipeline:

      - name: MiddleResolutionProcessing
        operation: earth_transformer_blocks (dim * 2)

      - name: Upsampling
        operation: pangu_upsample_2d

      - name: OutputResolutionProcessing
        operation: earth_transformer_blocks (dim)

      - name: SkipConnectionFusion
        operation: concatenation + reshape

      - name: PatchRecovery
        operation: pangu_patch_recovery_2d

  interface:

    parameters:

      output_resolution:
        type: tuple[int, int]
        description: 输出特征图分辨率 (lat, lon)

      middle_resolution:
        type: tuple[int, int]
        description: 中间层特征图分辨率 (lat, lon)

      out_chans:
        type: int
        description: 最终输出的气象变量通道数

      img_size:
        type: tuple[int, int]
        description: 原始输入图像分辨率，用于PatchRecovery还原

      patch_size:
        type: tuple[int, int]
        description: Patch大小，用于PatchRecovery

      dim:
        type: int
        description: 基础嵌入维度，中间层使用dim*2

      depth:
        type: int
        description: 输出分辨率处Transformer Block的层数

      depth_middle:
        type: int
        description: 中间分辨率处Transformer Block的层数

      num_heads:
        type: tuple[int, int] or int
        description: 各阶段注意力头数 (中间层, 输出层)

      window_size:
        type: tuple[int, int]
        description: 窗口注意力的窗口大小 (Wlat, Wlon)

    inputs:

      inp:
        type: List[Tensor]
        description: 包含两个张量的列表 [x, skip]

      x:
        type: SequenceEmbedding
        shape: [batch, middle_lat * middle_lon, dim * 2]
        description: 中间分辨率特征序列

      skip:
        type: SpatialFeatureMap
        shape: [batch, output_lat, output_lon, dim]
        description: 跳跃连接特征图

    outputs:

      output:
        type: WeatherField
        shape: [batch, out_chans, img_lat, img_lon]
        description: 最终恢复的气象预报场

  types:

    SequenceEmbedding:
      shape: [batch, seq_len, dim]
      description: 序列嵌入表示

    WeatherField:
      shape: [batch, channels, height, width]
      description: 气象场数据

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules.transformer.onetransformer import OneTransformer
      from onescience.modules.sample.onesample import OneSample
      from onescience.modules.recovery.onerecovery import OneRecovery

      class FengWuDecoder(nn.Module):
          def __init__(self, output_resolution=(181, 360), middle_resolution=(91, 180),
                       out_chans=37, img_size=(721, 1440), patch_size=(4, 4),
                       dim=192, depth=2, depth_middle=6, num_heads=(6, 12),
                       window_size=(6, 12), **kwargs):
              super().__init__()
              # 中间分辨率处理
              self.blocks_middle = nn.ModuleList([
                  OneTransformer(style="EarthTransformer2DBlock", 
                               dim=dim * 2, input_resolution=middle_resolution,
                               num_heads=num_heads_middle, window_size=window_size)
                  for _ in range(depth_middle)
              ])
              # 上采样
              self.upsample = OneSample(style="PanguUpSample2D",
                                      in_dim=dim * 2, out_dim=dim,
                                      input_resolution=middle_resolution,
                                      output_resolution=output_resolution)
              # 输出分辨率处理
              self.blocks = nn.ModuleList([
                  OneTransformer(style="EarthTransformer2DBlock",
                               dim=dim, input_resolution=output_resolution,
                               num_heads=num_heads, window_size=window_size)
                  for _ in range(depth)
              ])
              # Patch恢复
              self.patchrecovery2d = OneRecovery(style="PanguPatchRecovery2D",
                                                img_size=img_size, patch_size=patch_size,
                                                in_chans=2 * dim, out_chans=out_chans)

          def forward(self, inp):
              x, skip = inp[0], inp[1]
              B, Lat, Lon, C = skip.shape
              
              # 中间层处理
              for blk in self.blocks_middle:
                  x = blk(x)
              
              # 上采样
              x = self.upsample(x)
              
              # 输出层处理
              for blk in self.blocks:
                  x = blk(x)
              
              # 跳跃连接融合和Patch恢复
              output = torch.concat([x, skip.reshape(B, -1, C)], dim=-1)
              output = output.transpose(1, 2).reshape(B, -1, Lat, Lon)
              output = self.patchrecovery2d(output)
              return output

  skills:

    build_weather_decoder:

      description: 构建气象预报专用的多尺度解码器

      inputs:
        - output_resolution
        - middle_resolution
        - out_chans
        - dim
        - depth
        - window_size

      prompt_template: |

        构建FengWu气象预报解码器，支持多尺度处理和窗口注意力。

        参数：
        output_resolution = {{output_resolution}}
        middle_resolution = {{middle_resolution}}
        out_chans = {{out_chans}}
        dim = {{dim}}
        depth = {{depth}}
        window_size = {{window_size}}

        要求：
        1. 中间层使用dim*2特征维度
        2. 窗口注意力适配地球球面数据
        3. 支持跳跃连接融合
        4. Patch恢复到原始地理分辨率

    optimize_weather_forecasting:

      description: 优化气象预报解码器的计算效率

      checks:
        - window_size_compatibility (窗口大小与分辨率的兼容性)
        - memory_efficiency (大分辨率下的显存使用)
        - temporal_consistency (时间序列预报的连续性)

  knowledge:

    usage_patterns:

      global_weather_forecasting:

        pipeline:
          - Input: 编码器中间特征
          - Middle Processing: 全球尺度天气模式
          - Upsampling: 分辨率增强
          - Local Processing: 区域天气细节
          - Output: 高分辨率气象场

      regional_downscaling:

        pipeline:
          - Coarse Features: GCM输出
          - Multi-scale Fusion: 多尺度特征融合
          - High-resolution Output: 区域细化预报

    hot_models:

      - model: FengWu
        year: 2022
        role: 华为云盘古气象大模型解码器
        architecture: multi-scale transformer + patch recovery

      - model: Pangu-Weather
        year: 2023
        role: 盘古气象模型，类似架构
        architecture: earth transformer

      - model: GraphCast
        year: 2023
        role: DeepMind气象预报模型
        architecture: graph neural network

    best_practices:

      - 窗口大小应根据地理分辨率调整，高分辨率使用较小窗口
      - 中间层深度通常大于输出层，以充分处理全局信息
      - 跳跃连接对保持细节信息至关重要，不应省略
      - Patch恢复时注意边界效应，可采用重叠策略

    anti_patterns:

      - 窗口大小设置不当导致计算效率低下或信息丢失
      - 忽略地球球面几何特性，直接使用平面窗口
      - 上采样方法选择不当导致伪影
      - Patch恢复时分辨率不匹配造成信息扭曲

    paper_references:

      - title: "FengWu: Pushing the Skillful Forecast Horizon Beyond 10 days Using Global Forecast Data"
        authors: Huawei Cloud Team
        year: 2023

      - title: "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Weather Forecasting"
        authors: Bi et al.
        year: 2023

      - title: "GraphCast: Learning Skillful Medium-Range Global Weather Forecasting"
        authors: Lam et al.
        year: 2023

  graph:

    is_a:
      - NeuralNetworkDecoder
      - MultiScaleProcessor
      - WeatherForecastingComponent

    part_of:
      - EncoderDecoderArchitecture
      - WeatherPredictionSystem
      - EarthScienceModels

    depends_on:
      - EarthTransformer2DBlock
      - PanguUpSample2D
      - PanguPatchRecovery2D
      - WindowAttention

    variants:
      - StandardDecoder (无多尺度处理)
      - CNNDecoder (卷积版本)
      - GraphDecoder (图神经网络版本)

    used_in_models:
      - FengWu
      - Pangu-Weather
      - 其他气象预报模型

    compatible_with:

      inputs:
        - SequenceEmbedding
        - SpatialFeatureMap

      outputs:
        - WeatherField
        - ClimateVariables
