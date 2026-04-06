component:

  meta:
    name: PanguFuser
    alias: Pangu-Weather Feature Fuser
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: feature_fusion
    author: OneScience
    license: Apache-2.0
    tags:
      - weather_forecasting
      - transformer
      - feature_fusion
      - spatiotemporal
      - 3d_attention

  concept:

    description: >
      Pangu-Weather模型的三维特征融合模块，在给定三维网格上堆叠多层3D Transformer块以融合多时刻、多高度和空间信息。
      通过3D窗口注意力机制在时间、高度、经度三个维度上进行局部注意力计算，实现时空特征的深度融合。

    intuition: >
      就像气象预报员同时分析不同高度层次的天气图，不仅关注地表天气，还关注高空大气状况。
      3D窗口注意力就像在每个时空高度局部区域内进行"立体关联分析"，理解天气系统在垂直和水平方向上的发展规律。

    problem_it_solves:
      - 多时刻、多高度气象信息的时空融合
      - 三维大气数据的联合建模
      - 垂直方向和水平方向的特征交互
      - 天气系统的立体结构建模

  theory:

    formula:

      spatiotemporal_attention:
        expression: |
          \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M_{3D}\right)V
          M_{3D}: \text{3D window mask for spatiotemporal local attention}

      feature_fusion:
        expression: |
          x_{fused} = \text{Transformer3DBlocks}(x_{input})
          x_{output} = x_{fused}

  structure:

    architecture: spatiotemporal_transformer_fusion

    pipeline:

      - name: InputFeatures
        operation: spatiotemporal_grid_features

      - name: 3DTransformerBlocks
        operation: earth_transformer_3d_blocks

      - name: OutputFeatures
        operation: fused_spatiotemporal_features

  interface:

    parameters:

      dim:
        type: int
        description: 输入与输出特征的通道维度

      input_resolution:
        type: tuple[int, int, int]
        description: 三维输入特征的网格尺寸 (T, H, W)

      depth:
        type: int
        description: 3D Transformer块的层数

      num_heads:
        type: int
        description: 多头自注意力的头数

      window_size:
        type: tuple[int, int, int]
        description: 三维窗口注意力的窗口大小 (Wt, Wh, Ww)

      mlp_ratio:
        type: float
        description: 前馈网络隐藏层与特征维度的比例

    inputs:

      x:
        type: SpatiotemporalFeatures
        shape: [batch, T * H * W, dim]
        description: 已展平的三维网格特征

    outputs:

      x:
        type: FusedFeatures
        shape: [batch, T * H * W, dim]
        description: 融合后的时空特征

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules.transformer.onetransformer import OneTransformer

      class PanguFuser(nn.Module):
          def __init__(self, dim=256, input_resolution=(10, 181, 360), depth=4,
                       num_heads=8, window_size=(2, 6, 12), **kwargs):
              super().__init__()
              self.dim = dim
              self.input_resolution = input_resolution
              self.depth = depth

              self.blocks = nn.ModuleList([
                  OneTransformer(
                      style="EarthTransformer3DBlock",
                      dim=dim, input_resolution=input_resolution,
                      num_heads=num_heads, window_size=window_size,
                      shift_size=(0, 0, 0) if i % 2 == 0 else None,
                  )
                  for i in range(depth)
              ])

          def forward(self, x):
              for blk in self.blocks:
                  x = blk(x)
              return x

  skills:

    build_pangu_fuser:

      description: 构建Pangu气象预报专用的时空特征融合器

  knowledge:

    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 盘古气象模型融合器
        architecture: 3D transformer

    best_practices:

      - 三维窗口大小应考虑时间、高度、水平方向的特性
      - 垂直方向窗口通常较小以体现大气分层特性

    paper_references:

      - title: "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Weather Forecasting"
        authors: Bi et al.
        year: 2023

  graph:

    is_a:
      - FeatureFusionModule
      - SpatiotemporalProcessor
      - TransformerModule

    used_in_models:
      - Pangu-Weather

    compatible_with:

      inputs:
        - SpatiotemporalFeatures
        - WeatherFields

      outputs:
        - FusedFeatures
        - IntegratedRepresentations
