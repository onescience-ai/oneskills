component:

  meta:
    name: FengWuFuser
    alias: FengWu Weather Feature Fuser
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
      FengWu模型的三维特征融合模块，在中分辨率的时空网格上堆叠多层3D Transformer块以融合多时刻、多变量信息。
      通过3D窗口注意力机制在时间、纬度、经度三个维度上进行局部注意力计算，实现时空特征的深度融合。

    intuition: >
      就像气象预报员同时分析多个时间步的天气图，不仅关注每个时刻的空间模式，还关注时间上的演变趋势。
      3D窗口注意力就像在每个时空局部区域内进行"时空关联分析"，理解天气系统在时间和空间上的发展规律。

    problem_it_solves:
      - 多时刻气象信息的时空融合
      - 时空数据的联合建模
      - 高维特征的高效融合
      - 天气系统的时空演化建模

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

      input_resolution:
        type: tuple[int, int, int]
        description: 三维输入特征的网格尺寸 (T, H, W)

      dim:
        type: int
        description: 输入与输出特征的通道维度

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

      class FengWuFuser(nn.Module):
          def __init__(self, input_resolution=(6, 91, 180), dim=192*2, depth=6,
                       num_heads=12, window_size=(2, 6, 12), **kwargs):
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

    build_weather_fuser:

      description: 构建气象预报专用的时空特征融合器

  knowledge:

    hot_models:

      - model: FengWu
        year: 2022
        role: 华为云盘古气象大模型融合器
        architecture: 3D transformer

    best_practices:

      - 3D窗口大小应根据时空特性平衡
      - 时间维度窗口通常小于空间维度
      - 多层融合有助于复杂时空模式学习

    paper_references:

      - title: "FengWu: Pushing the Skillful Forecast Horizon Beyond 10 days Using Global Forecast Data"
        authors: Huawei Cloud Team
        year: 2023

  graph:

    is_a:
      - FeatureFusionModule
      - SpatiotemporalProcessor
      - TransformerModule

    used_in_models:
      - FengWu
      - Pangu-Weather

    compatible_with:

      inputs:
        - SpatiotemporalFeatures
        - WeatherFields

      outputs:
        - FusedFeatures
        - IntegratedRepresentations
