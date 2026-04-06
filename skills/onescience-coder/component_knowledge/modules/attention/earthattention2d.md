component:

  meta:
    name: EarthAttention2D
    alias: 2D Earth Position Bias Window Attention
    version: 1.0
    domain: meteorological_ai
    category: neural_network
    subcategory: attention_mechanism
    author: OneScience 
    license: Apache-2.0
    tags:
      - earth_position_bias
      - swin_transformer
      - pangu_weather
      - surface_variables
      - meteorology

  concept:

    description: >
      EarthAttention2D 是一种用于处理 2D 地表变量的地球位置偏置窗口注意力机制。
      它在标准的 Swin-Transformer 窗口注意力基础之上，引入了可学习的地球位置偏置（Earth Position Bias）。
      该模块专门用于捕捉纬度方向上窗口间的特定空间关系，是处理地表气象变量（如 2m 温度、10m 风速、海平面气压等）的理想选择。

    intuition: >
      在处理全球气象网格数据时，标准的计算机视觉注意力机制会把所有像素一视同仁。但地球是一个球体，
      经纬度网格投影到平面后，高纬度地区的网格实际物理面积远小于赤道地区。
      EarthAttention2D 就像是给模型配备了“经纬度雷达”，它通过识别当前窗口所在的具体纬度带（`type_of_windows`），
      在计算注意力分数时加上一个与该纬度强相关的偏置项，从而让模型能够感知并适应地球真实的几何畸变。

    problem_it_solves:
      - 解决传统 2D Transformer 在处理圆柱投影的全球气象数据时，缺乏空间几何畸变感知能力的缺陷。
      - 提升模型对地表单层变量（仅包含经纬度，无高度通道）的特征提取和预测精度。

  theory:

    formula:
      
      qkv_projection:
        expression: Q, K, V = Linear(X)

      attention_with_earth_bias:
        expression: Attention = Softmax( (Q * K^T / sqrt(d_k)) + EarthPositionBias ) * V

    variables:

      EarthPositionBias:
        name: 纬度特异性偏置矩阵
        shape: [nH, num_lat, Wlat*Wlon, Wlat*Wlon]
        description: 这是一个可学习的参数表，针对不同的纬度带（num_lat）提供独立的注意力校正项。

  structure:

    architecture: swin_attention_with_bias

    pipeline:

      - name: QKVGeneration
        operation: linear_projection

      - name: ScaledDotProduct
        operation: matmul_and_scale

      - name: EarthBiasAddition
        operation: broadcast_add_bias (将从偏置表中索引出的纬度偏置加到注意力分数上)

      - name: MaskAddition
        operation: masked_fill (可选，用于处理循环边界等特殊拓扑)

      - name: AttentionWeights
        operation: softmax_and_dropout

      - name: ValueAggregation
        operation: weighted_sum_and_projection


  interface:

    parameters:

      dim:
        type: int
        description: 输入通道数（特征嵌入维度）

      input_resolution:
        type: tuple[int, int]
        description: 输入特征图的整体空间分辨率 (lat, lon)，用于计算纬度方向的窗口总数

      window_size:
        type: tuple[int, int]
        description: 局部注意力计算的窗口大小 (Wlat, Wlon)

      num_heads:
        type: int
        description: 多头注意力的头数

      qkv_bias:
        type: bool
        default: true
        description: 是否为 QKV 线性投影层添加偏置项

      qk_scale:
        type: float
        default: null
        description: QK 点积的缩放系数，若为 null 则默认使用 head_dim ** -0.5

      attn_drop:
        type: float
        default: 0.0
        description: 注意力权重的 Dropout 比例

      proj_drop:
        type: float
        default: 0.0
        description: 最终输出投影层的 Dropout 比例

    inputs:

      x:
        type: Tensor
        shape: [B * num_lon, num_lat, Wlat * Wlon, C]
        dtype: float32
        description: 按照窗口划分并重排后的地表特征输入序列

      mask:
        type: Tensor
        shape: [num_lon, num_lat, Wlat * Wlon, Wlat * Wlon]
        default: null
        description: 注意力掩码，主要用于处理经度方向的循环填充（Wrap-around padding）

    outputs:

      output:
        type: Tensor
        shape: [B * num_lon, num_lat, Wlat * Wlon, C]
        description: 融合了地球位置偏置后的局部注意力输出序列

  types:

    WindowedSurfaceFeatures:
      shape: [B * num_lon, num_lat, N, C]
      description: 经过窗口切分操作后的地表特征格式

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn

      from ..func_utils import (
          get_earth_position_index,
          trunc_normal_,
      )

      class EarthAttention2D(nn.Module):
          """
              用于2D地表变量的地球位置偏置窗口注意力机制。
              
              在标准窗口注意力基础上引入地球位置偏置（Earth Position Bias），
              用于捕捉纬度方向窗口间的空间关系。适用于处理地表气象变量（如
              2m温度、10m风速等）的Swin-Transformer风格注意力计算。
              
              Args:
                  dim (int): 输入通道数（嵌入维度）。
                  input_resolution (tuple[int, int]): 输入特征图的空间分辨率 (lat, lon)，
                      用于计算纬度方向的窗口数量 num_lat = lat // Wlat。
                  window_size (tuple[int, int]): 注意力窗口大小 (Wlat, Wlon)。
                  num_heads (int): 多头注意力的头数。
                  qkv_bias (bool, optional): 是否为QKV投影添加偏置项，默认为True。
                  qk_scale (float, optional): QK点积的缩放系数，默认为None，
                      此时自动使用 head_dim ** -0.5。
                  attn_drop (float, optional): 注意力权重的Dropout比例，默认为0.0。
                  proj_drop (float, optional): 输出投影的Dropout比例，默认为0.0。
              
              形状:
                  - 输入 x: (B * num_lon, num_lat, Wlat * Wlon, C)
                      其中 num_lat = lat // Wlat，num_lon = lon // Wlon
                  - 输入 mask: (num_lon, num_lat, Wlat * Wlon, Wlat * Wlon) 或 None
                  - 输出: (B * num_lon, num_lat, Wlat * Wlon, C)
              
              Examples:
                  >>> # 典型Pangu-Weather地表变量配置
                  >>> # 分辨率: lat=128, lon=256，窗口大小: 8×8
                  >>> # num_lat = 128 // 8 = 16
                  >>> # num_lon = 256 // 8 = 32
                  >>> # B_ = B * num_lon = 4 * 32 = 128
                  >>> # N = Wlat * Wlon = 8 * 8 = 64
                  >>> attn = EarthAttention2D(
                  ...     dim=192,
                  ...     input_resolution=(128, 256),
                  ...     window_size=(8, 8),
                  ...     num_heads=6,
                  ... )
                  >>> x = torch.randn(128, 16, 64, 192)  # (B*num_lon, num_lat, N, C)
                  >>> out = attn(x)
                  >>> out.shape
                  torch.Size([128, 16, 64, 192])
                  
                  >>> # 带mask的前向传播（用于循环边界填充场景）
                  >>> mask = torch.zeros(32, 16, 64, 64)  # (num_lon, num_lat, N, N)
                  >>> out = attn(x, mask=mask)
                  >>> out.shape
                  torch.Size([128, 16, 64, 192])
          """

          def __init__(
              self,
              dim,
              input_resolution,
              window_size,
              num_heads,
              qkv_bias=True,
              qk_scale=None,
              attn_drop=0.0,
              proj_drop=0.0,
          ):
              super().__init__()
              self.dim = dim
              self.window_size = window_size  # Wlat, Wlon
              self.num_heads = num_heads
              head_dim = dim // num_heads
              self.scale = qk_scale or head_dim**-0.5

              self.type_of_windows = input_resolution[0] // window_size[0]

              self.earth_position_bias_table = nn.Parameter(
                  torch.zeros(
                      (window_size[0] ** 2) * (window_size[1] * 2 - 1),
                      self.type_of_windows,
                      num_heads,
                  )
              )  # Wlat**2 * Wlon*2-1, Nlat//Wlat, nH

              earth_position_index = get_earth_position_index(
                  window_size, ndim=2
              )  # Wlat*Wlon, Wlat*Wlon
              self.register_buffer("earth_position_index", earth_position_index)

              self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
              self.attn_drop = nn.Dropout(attn_drop)
              self.proj = nn.Linear(dim, dim)
              self.proj_drop = nn.Dropout(proj_drop)

              self.earth_position_bias_table = trunc_normal_(
                  self.earth_position_bias_table, std=0.02
              )
              self.softmax = nn.Softmax(dim=-1)

          def forward(self, x: torch.Tensor, mask=None):
              """
              Args:
                  x: input features with shape of (B * num_lon, num_lat, N, C)
                  mask: (0/-inf) mask with shape of (num_lon, num_lat, Wlat*Wlon, Wlat*Wlon)
              """
              B_, nW_, N, C = x.shape
              qkv = (
                  self.qkv(x)
                  .reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads)
                  .permute(3, 0, 4, 1, 2, 5)
              )
              q, k, v = qkv[0], qkv[1], qkv[2]

              q = q * self.scale
              attn = q @ k.transpose(-2, -1)

              earth_position_bias = self.earth_position_bias_table[
                  self.earth_position_index.view(-1)
              ].view(
                  self.window_size[0] * self.window_size[1],
                  self.window_size[0] * self.window_size[1],
                  self.type_of_windows,
                  -1,
              )  # Wlat*Wlon, Wlat*Wlon, num_lat, nH
              earth_position_bias = earth_position_bias.permute(
                  3, 2, 0, 1
              ).contiguous()  # nH, num_lat, Wlat*Wlon, Wlat*Wlon
              attn = attn + earth_position_bias.unsqueeze(0)

              if mask is not None:
                  nLon = mask.shape[0]
                  attn = attn.view(
                      B_ // nLon, nLon, self.num_heads, nW_, N, N
                  ) + mask.unsqueeze(1).unsqueeze(0)
                  attn = attn.view(-1, self.num_heads, nW_, N, N)
                  attn = self.softmax(attn)
              else:
                  attn = self.softmax(attn)

              attn = self.attn_drop(attn)

              x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
              x = self.proj(x)
              x = self.proj_drop(x)
              return x


  skills:

    build_pangu_2d_attention:

      description: 构建兼容盘古气象模型的二维地表注意力层

      inputs:
        - dim
        - input_resolution
        - window_size
        - num_heads

      prompt_template: |
        为盘古气象模型的 2D 网络构建一个 EarthAttention2D 模块。
        参数：
        输入分辨率 = {{input_resolution}}
        通道数 = {{dim}}
        请确保纬度切分数 (`type_of_windows`) 能够被输入分辨率和窗口大小整除。

    diagnose_bias_shape_error:

      description: 诊断由于分辨率或窗口大小设置不当导致的偏置矩阵形状匹配错误

      checks:
        - type_of_windows_mismatch (检查 input_resolution[0] 是否能被 window_size[0] 完美整除)
        - external_index_function_missing (确保工程中正确导入了 `get_earth_position_index` 辅助函数)

  knowledge:

    usage_patterns:

      pangu_surface_network:
        pipeline:
          - SurfacePatchEmbedding
          - EarthAttention2D (with window mapping)
          - MLP
          - PatchMerging / Unmerging

    design_patterns:

      latitude_aware_bias:
        structure:
          - 将纬度带数量 (`type_of_windows`) 作为 `earth_position_bias_table` 的一个关键维度。
          - 结合绝对位置索引表 (`earth_position_index`)，通过查表法获取当前窗口内各个 token 的相对位置偏移，并追加到注意力分数矩阵中。

    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 首个预测精度超越传统数值天气预报（NWP）的 3D 神经网络模型
        architecture: 3D Earth-Specific Transformer
        attention_type: Earth Position Bias Attention

    model_usage_details:

      Pangu_Weather_Surface_Variable_Standard:
        input_resolution: (128, 256)
        window_size: (8, 8)
        num_heads: 6

    best_practices:
      - 初始化模型时，请保证 `input_resolution[0] // window_size[0]` 能够得到一个整数，否则模型底层的偏置表维度将会对齐失败。
      - 当涉及到全球经度的物理连贯性时，可以在前向传播时传入设计好的 `mask` 来实现包裹式（Wrap-around）计算。

    anti_patterns:
      - 错误地将三维变量数据（如高空的多层风场或气压场数据）输入此模块。对于 3D 变量，应当使用包含高度维度的 `EarthAttention3D`。
      - 试图在批处理之前就传入展平的 `(B, L, C)` 数据，此模块要求输入在进入 forward 之前已经被手动 reshape 为 `(B_, nW_, N, C)`。

    paper_references:

      - title: "Accurate medium-range global weather forecasting with 3D neural networks"
        authors: Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, Qi Tian
        year: 2023
        journal: Nature

  graph:

    is_a:
      - AttentionMechanism
      - NeuralNetworkComponent

    part_of:
      - PanguWeatherModel
      - SurfaceVariableNetwork

    depends_on:
      - get_earth_position_index
      - trunc_normal_
      - Softmax
      - Linear

    compatible_with:
      inputs:
        - WindowedSurfaceFeatures
      outputs:
        - WindowedSurfaceFeatures