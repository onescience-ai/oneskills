component:

  meta:
    name: FuxiFeedForward
    alias: FuxiFC
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: linear_projection
    author: OneScience
    license: Apache-2.0
    tags:
      - linear_layer
      - projection
      - fuxi
      - weather_forecasting
      - fluid_dynamics

  concept:

    description: >
      FuxiFC 是针对伏羲（FuXi）气象大模型设计的一个线性全连接投影层。
      它主要用于在特征通道（Channel）维度上进行线性变换，将高维的隐藏特征（如 1536 维）
      映射回特定的物理状态空间或预测目标空间（如 70 个气象变量在 4x4 空间 patch 上的展开）。
      该模块直接处理具有空间维度（纬度 Lat、经度 Lon）的网格数据。

    intuition: >
      在处理地球科学或流体力学中的 2D 网格数据时，该模块相当于一个逐点（Point-wise）的线性变换。
      输入形状被假定为 (Batch, Lat, Lon, Channels)，PyTorch 的 nn.Linear 会自动作用于最后一个维度（Channels）。
      这使得模型可以在每个地理坐标网格点上独立地进行特征的降维或解码，而不需要显式地重塑（Reshape）空间维度。

    problem_it_solves:
      - 气象大模型潜空间特征到物理变量空间的解码映射
      - 在保持地理空间结构（Lat, Lon）不变的情况下，进行高效的通道级线性组合
      - 适配具有特定 Patch 划分策略（如 4x4）的物理场重构需求

  theory:

    formula:

      fuxi_fc_output:
        expression: y_{b, lat, lon, c_{out}} = \sum (x_{b, lat, lon, c_{in}} * W_{c_{in}, c_{out}}) + b_{c_{out}}

    variables:

      x:
        name: Input
        shape: [B, Lat, Lon, in_channels]
        description: 包含批次、纬度、经度和输入通道数的特征张量

      W:
        name: Weight
        shape: [out_channels, in_channels]
        description: 线性变换的权重矩阵

      b:
        name: Bias
        shape: [out_channels]
        description: 偏置项向量

      y:
        name: Output
        shape: [B, Lat, Lon, out_channels]
        description: 映射后的输出特征张量

  structure:

    architecture: linear_projection

    pipeline:

      - name: ChannelProjection
        operation: linear_transformation

  interface:

    parameters:

      in_channels:
        type: int
        default: 1536
        description: 输入特征的通道数，通常对应模型深层的隐藏维度

      out_channels:
        type: int
        default: 1120 (70 * 4 * 4)
        description: 输出特征的通道数，通常对应物理变量数乘以空间 Patch 大小

    inputs:

      x:
        type: Tensor
        shape: [B, Lat, Lon, in_channels]
        dtype: float32
        description: 保持空间结构的输入网格特征

    outputs:

      output:
        type: Tensor
        shape: [B, Lat, Lon, out_channels]
        description: 经过通道映射后的输出特征

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量（需保证最后一维为 Channels）

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn
      from torch.nn import functional as F
      from typing import Sequence


      class FuxiFC(nn.Module):

          def __init__(self,
                       in_channels=1536,
                       out_channels=70*4*4):
              super().__init__()
              
              self.fc = nn.Linear(in_channels, out_channels)
          
          def forward(self, x: torch.Tensor):
              x = self.fc(x)  # B Lat Lon C
              return x

  skills:

    build_fuxi_fc_layer:

      description: 为网格化的物理特征（如流体场、气象场）构建通道投影层

      inputs:
        - in_channels
        - num_variables (物理变量数，如 70)
        - patch_size (如 4)

      prompt_template: |
        构建一个 FuxiFC 投影层。
        参数：
        in_channels = {{in_channels}}
        out_channels = {{num_variables}} * {{patch_size}} * {{patch_size}}
        请确保输入张量在传入前已被 permute 为 (B, Lat, Lon, C) 格式。

    diagnose_fuxi_fc:

      description: 分析使用普通 nn.Linear 处理空间特征张量时的维度匹配问题

      checks:
        - input_tensor_is_channels_last
        - mismatch_between_out_channels_and_physical_targets

  knowledge:

    usage_patterns:

      decoder_projection:

        pipeline:
          - LatentFeature
          - FuxiFC (映射到物理维度)
          - PatchUn-embed (重构回全分辨率物理场)

    design_patterns:

      channels_last_processing:

        structure:
          - 将具有空间维度的数据格式化为 `Channels-Last` (例如 B, H, W, C)
          - 直接利用 `nn.Linear` 广播机制跨越空间维度执行逐点运算，避免显式展平和耗时的内存重排

    hot_models:

      - model: FuXi (伏羲)
        year: 2023
        role: 级联机器学习全球天气预报系统
        architecture: swin_transformer_based
        attention_type: None (此处作为特征解码的输出层)

    model_usage_details:

      FuXi:

        in_channels: 1536
        out_channels: 1120 (代表 70 个气象变量在 4x4 的局部下采样 patch 中的信息)

    best_practices:

      - 必须保证输入张量的维度顺序为 `[Batch, Lat, Lon, Channels]`，`nn.Linear` 才能正确作用于通道维。
      - 当 `out_channels` 代表包含多个物理变量（如 u/v 风速、温度等）和空间 Patch 时，输出结果往往需要后续的 `Reshape` 和逆向 Patch 化操作才能还原为标准的物理场数据。

    anti_patterns:

      - 直接输入标准计算机视觉中 `Channels-First` 的张量 `[Batch, Channels, Lat, Lon]`，这会导致全连接层在经度（Lon）维度上错误地执行投影。

    paper_references:

      - title: "FuXi: A cascade machine learning forecasting system for 15-day global weather forecast"
        authors: Chen et al.
        year: 2023

  graph:

    is_a:
      - NeuralNetworkComponent
      - LinearTransformation
      - ProjectionHead

    part_of:
      - FuxiModel
      - WeatherForecastingDecoder

    depends_on:
      - Linear

    variants:
      - Standard nn.Linear
      - Pointwise Conv2d (1x1卷积，作为Channels-First的等效实现)

    used_in_models:
      - FuXi

    compatible_with:

      inputs:
        - Tensor (Channels-Last format)

      outputs:
        - Tensor (Channels-Last format)