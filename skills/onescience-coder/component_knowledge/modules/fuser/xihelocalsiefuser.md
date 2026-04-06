component:

  meta:
    name: XiheLocalSIEFuser
    alias: Xihe Local SIE Fuser
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: local_attention
    author: OneScience
    license: Apache-2.0
    tags:
      - ocean_modeling
      - local_attention
      - 3d_windows
      - sea_ice
      - weather_learn

  concept:

    description: >
      基于WeatherLearn的3D局部SIEFuser模块，用于Xihe模型的多层局部特征融合。
      通过3D窗口注意力机制在压力层、纬度、经度三个维度上进行局部信息交互，
      特别适合处理海洋系统的局部动力学特征。

    intuition: >
      就像海洋学家分析局部海域：关注特定压力层内的涡旋、温度梯度等局部现象，
      通过3D窗口同时考虑垂直和水平方向的相互影响，理解局部海洋动力学。

    problem_it_solves:
      - 局部海洋特征的精细建模
      - 3D局部窗口的高效计算
      - 海洋系统的局部动力学
      - 多尺度局部特征的提取

  theory:

    formula:

      local_3d_attention:
        expression: |
          \text{Attention}_{3D}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M_{3D}\right)V
          M_{3D}: \text{3D window mask for local attention}

      window_partition:
        expression: |
          x_{windows} = \text{WindowPartition3D}(x, window\_size)
          x_{attended} = \text{Attention}(x_{windows})
          x_{restored} = \text{WindowReverse3D}(x_{attended})

    variables:

      window_size:
        name: WindowSize
        shape: [Wpl, Wlat, Wlon]
        description: 3D窗口大小

      M_{3D}:
        name: LocalWindowMask
        description: 3D局部窗口注意力掩码

  structure:

    architecture: local_3d_attention

    pipeline:

      - name: InputProcessing
        operation: reshape_to_3d + apply_mask

      - name: WindowPartition
        operation: 3d_window_partitioning

      - name: LocalAttention
        operation: multi_head_attention_in_windows

      - name: WindowReverse
        operation: 3d_window_reverse

      - name: OutputReshape
        operation: reshape_to_sequence

  interface:

    parameters:

      dim:
        type: int
        description: 输入通道数

      input_resolution:
        type: tuple[int, int, int]
        description: 输入空间分辨率 (Pl, Lat, Lon)

      depth:
        type: int
        description: Transformer块的数量

      num_heads:
        type: int
        description: 注意力头数量

      window_size:
        type: tuple[int, int, int]
        description: 3D局部窗口大小

      mlp_ratio:
        type: float
        description: MLP隐层扩展比例

    inputs:

      obj:
        type: OceanData
        description: 包含x和mask的对象

      x:
        type: Features
        shape: [batch, L, C] 或 [batch, C, Pl, Lat, Lon]
        description: 输入特征

      mask:
        type: AttentionMask
        shape: [batch, 1, Pl, Lat, Lon]
        description: 注意力掩码

    outputs:

      x:
        type: LocalFeatures
        shape: 与输入相同
        description: 局部融合后的特征

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules.transformer.onetransformer import OneTransformer

      class XiheLocalSIEFuser(nn.Module):
          def __init__(self, dim=192, input_resolution=(13, 128, 256), depth=2,
                       num_heads=6, window_size=(1, 6, 12), **kwargs):
              super().__init__()
              self.dim = dim
              self.input_resolution = input_resolution
              
              # 3D Transformer块
              self.blocks = nn.ModuleList([
                  OneTransformer(
                      style="EarthTransformer3DBlock",
                      dim=dim, input_resolution=input_resolution,
                      num_heads=num_heads, window_size=window_size,
                      shift_size=(0, 0, 0) if i % 2 == 0 else None,
                  )
                  for i in range(depth)
              ])

          def forward(self, obj):
              x = obj.x
              mask = getattr(obj, 'mask', None)
              
              # 如果输入是序列格式，转换为3D格式
              if x.dim() == 3:
                  B, L, C = x.shape
                  Pl, Lat, Lon = self.input_resolution
                  x = x.transpose(1, 2).view(B, C, Pl, Lat, Lon)
              
              # 应用掩码
              if mask is not None and mask.dim() == 2:
                  mask = mask.view(B, 1, *self.input_resolution)
              
              # 3D Transformer处理
              for blk in self.blocks:
                  x = blk(x, mask=mask)
              
              # 转换回原始格式
              if obj.x.dim() == 3:
                  x = x.view(B, C, -1).transpose(1, 2)
              
              return x

  skills:

    build_local_sie_fuser:

      description: 构建局部SIE融合器

  knowledge:

    hot_models:

      - model: Xihe
        year: 2023
        role: 海洋模型局部注意力模块
        architecture: 3D window attention

      - model: WeatherLearn
        year: 2022
        role: 局部窗口注意力的基础架构
        architecture: local attention

    best_practices:

      - 窗口大小应考虑海洋现象的空间尺度
      - 压力层窗口通常较小（1层）
      - 水平窗口应覆盖典型涡流尺度

    paper_references:

      - title: "WeatherLearn: A Deep Learning-based Weather Forecasting Model"
        authors: Chen et al.
        year: 2022

  graph:

    is_a:
      - LocalAttentionModule
      - OceanModelingComponent
      - 3DWindowAttention

    used_in_models:
      - Xihe
      - WeatherLearn-based models

    compatible_with:

      inputs:
        - OceanFeatures
        - ClimateData
        - 3DSpatialData

      outputs:
        - LocalOceanFeatures
        - RegionalRepresentations
