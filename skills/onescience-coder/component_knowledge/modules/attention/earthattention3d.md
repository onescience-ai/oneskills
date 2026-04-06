component:

  meta:
    name: EarthAttention3D
    alias: 3D Earth Position Bias Window Attention
    version: 1.0
    domain: meteorological_ai
    category: neural_network
    subcategory: attention_mechanism
    author: OneScience 
    license: Apache-2.0
    tags:
      - earth_position_bias
      - 3d_attention
      - swin_transformer
      - pangu_weather
      - atmospheric_variables
      - meteorology

  concept:

    description: >
      EarthAttention3D 是一种用于处理 3D 大气变量的地球位置偏置窗口注意力机制。
      作为 2D 版本的立体扩展，它在水平方向（纬度）的基础上，进一步在垂直方向（气压层 pressure level）引入了位置偏置。
      通过联合捕捉三维空间中的物理关联，该模块非常适合处理具有垂直高度维度的气压层数据（如不同等压面上的风场、温度场等）。

    intuition: >
      地球大气不仅在不同纬度受到的太阳辐射和自转偏向力不同，在不同垂直高度（气压层）的物理表现也截然不同。
      如果我们把大气圈看作一个多层的“洋葱”，EarthAttention3D 能够同时定位当前数据块处于哪一层“洋葱皮”（气压高度）以及处于什么纬度带。
      模型通过联合标识高度和纬度 (`type_of_windows = num_pl * num_lat`)，为不同高度和纬度的局部注意力匹配专属的偏置校正，
      从而极大地提升了对三维地球流体动力学的建模能力。

    problem_it_solves:
      - 突破传统 3D Transformer 无法区分地球垂直气压层物理差异的局限。
      - 解决高分辨率多层全球气象数据由于纬度面积畸变和垂直梯度变化带来的双重空间异质性建模难题。

  theory:

    formula:
      
      qkv_projection:
        expression: Q, K, V = Linear(X)

      attention_with_3d_earth_bias:
        expression: Attention = Softmax( (Q * K^T / sqrt(d_k)) + EarthPositionBias3D ) * V

    variables:

      EarthPositionBias3D:
        name: 气压层与纬度特异性偏置矩阵
        shape: [nH, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon]
        description: 这是一个可学习的三维参数表，针对不同的气压层和纬度组合区域（num_pl*num_lat）提供独立的注意力校正项。

  structure:

    architecture: swin_3d_attention_with_bias

    pipeline:

      - name: QKVGeneration
        operation: linear_projection

      - name: ScaledDotProduct
        operation: matmul_and_scale

      - name: EarthBias3DAddition
        operation: broadcast_add_bias (将从偏置表中查出的三维复合位置偏置加到注意力分数上)

      - name: MaskAddition
        operation: masked_fill (可选，用于处理经度方向的循环边界填充)

      - name: AttentionWeights
        operation: softmax_and_dropout

      - name: ValueAggregation
        operation: weighted_sum_and_projection

  interface:

    parameters:

      dim:
        type: int
        description: 输入特征图的通道数（嵌入维度）

      input_resolution:
        type: tuple[int, int, int]
        description: 经过填充(Padding)后的特征图三维空间分辨率 (pl, lat, lon)，用于计算偏置窗口的数量

      window_size:
        type: tuple[int, int, int]
        description: 局部注意力计算的三维窗口大小 (Wpl, Wlat, Wlon)

      num_heads:
        type: int
        description: 多头注意力的并行头数

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
        shape: [B * num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, C]
        dtype: float32
        description: 经过三维切分、纬度与气压层组合排列后的高空大气特征张量

      mask:
        type: Tensor
        shape: [num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, Wpl * Wlat * Wlon]
        default: null
        description: 三维注意力掩码，常用于屏蔽无效填充区域或维持环球物理循环边界

    outputs:

      output:
        type: Tensor
        shape: [B * num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, C]
        description: 融合了 3D 地球位置偏置后的输出张量

  types:

    WindowedAtmosphericFeatures:
      shape: [B * num_lon, nW_, N, C]
      description: 经过预处理重排后，专供三维地球注意力机制计算使用的数据格式

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn

      from ..func_utils import (
          get_earth_position_index,
          trunc_normal_,
      )

      class EarthAttention3D(nn.Module):
          """
              用于3D大气变量的地球位置偏置窗口注意力机制。
              
              EarthAttention2D 的三维扩展版本，在气压层（pressure level）维度上
              额外引入位置偏置，用于同时捕捉垂直方向与水平方向的空间关系。适用于
              处理多层大气变量（如位势高度、温度、风场等等压面数据）的注意力计算。
              
              Args:
                  dim (int): 输入通道数（嵌入维度）。
                  input_resolution (tuple[int, int, int]): 输入特征图的空间分辨率
                      (pl, lat, lon)，用于计算窗口数量：
                      - num_pl = pl // Wpl
                      - num_lat = lat // Wlat
                      其中 type_of_windows = num_pl * num_lat，经度方向折叠进 batch 维。
                  window_size (tuple[int, int, int]): 注意力窗口大小 (Wpl, Wlat, Wlon)。
                  num_heads (int): 多头注意力的头数。
                  qkv_bias (bool, optional): 是否为QKV投影添加偏置项，默认为True。
                  qk_scale (float, optional): QK点积的缩放系数，默认为None，
                      此时自动使用 head_dim ** -0.5。
                  attn_drop (float, optional): 注意力权重的Dropout比例，默认为0.0。
                  proj_drop (float, optional): 输出投影的Dropout比例，默认为0.0。
              
              形状:
                  - 输入 x: (B * num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, C)
                      其中 num_pl = pl // Wpl，num_lat = lat // Wlat，num_lon = lon // Wlon
                  - 输入 mask: (num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, Wpl * Wlat * Wlon) 或 None
                  - 输出: (B * num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, C)
              
                  Examples:
                  >>> # 典型Pangu-Weather大气变量配置
                  >>> # 原始气压层数为13，经 get_pad3d padding 后 pl=14（对齐 Wpl=2）
                  >>> # pad_resolution = (14, 128, 256)，window_size = (2, 8, 8)
                  >>> # num_pl  = 14 // 2 = 7
                  >>> # num_lat = 128 // 8 = 16
                  >>> # num_lon = 256 // 8 = 32
                  >>> # type_of_windows = num_pl * num_lat = 7 * 16 = 112
                  >>> # B_ = B * num_lon = 4 * 32 = 128
                  >>> # N = Wpl * Wlat * Wlon = 2 * 8 * 8 = 128
                  >>> attn = EarthAttention3D(
                  ...     dim=192,
                  ...     input_resolution=(14, 128, 256),  # 传入 padding 后的分辨率
                  ...     window_size=(2, 8, 8),
                  ...     num_heads=6,
                  ... )
                  >>> x = torch.randn(128, 112, 128, 192)  # (B*num_lon, num_pl*num_lat, N, C)
                  >>> out = attn(x)
                  >>> out.shape
                  torch.Size([128, 112, 128, 192])
                  
                  >>> # 带mask的前向传播（经度循环边界填充场景）
                  >>> mask = torch.zeros(32, 112, 128, 128)  # (num_lon, num_pl*num_lat, N, N)
                  >>> out = attn(x, mask=mask)
                  >>> out.shape
                  torch.Size([128, 112, 128, 192])
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
              self.window_size = window_size  # Wpl, Wlat, Wlon
              self.num_heads = num_heads
              head_dim = dim // num_heads
              self.scale = qk_scale or head_dim**-0.5

              self.type_of_windows = (input_resolution[0] // window_size[0]) * (
                  input_resolution[1] // window_size[1]
              )

              self.earth_position_bias_table = nn.Parameter(
                  torch.zeros(
                      (window_size[0] ** 2)
                      * (window_size[1] ** 2)
                      * (window_size[2] * 2 - 1),
                      self.type_of_windows,
                      num_heads,
                  )
              )  # Wpl**2 * Wlat**2 * Wlon*2-1, Npl//Wpl * Nlat//Wlat, nH

              earth_position_index = get_earth_position_index(
                  window_size
              )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
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
              参数:
                  x: 输入特征张量，形状为 (B * num_lon, num_pl*num_lat, N, C)
                  mask: 取值为 0 或 -∞，形状为 (num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon)
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
                  self.window_size[0] * self.window_size[1] * self.window_size[2],
                  self.window_size[0] * self.window_size[1] * self.window_size[2],
                  self.type_of_windows,
                  -1,
              )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon, num_pl*num_lat, nH
              earth_position_bias = earth_position_bias.permute(
                  3, 2, 0, 1
              ).contiguous()  # nH, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
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

    build_pangu_3d_attention:

      description: 构建兼容盘古气象模型的三维高空大气注意力层

      inputs:
        - dim
        - input_resolution
        - window_size
        - num_heads

      prompt_template: |
        为盘古气象模型构建处理高空变量的 EarthAttention3D。
        参数：
        输入分辨率（需提前 Padding 对齐） = {{input_resolution}}
        窗口大小 = {{window_size}}
        请确保高度和纬度的联合块数 (`type_of_windows`) 计算正确无截断。

    diagnose_padding_and_bias_mismatch:

      description: 排查 3D 偏置表初始化失败或运行时维度不匹配的异常

      checks:
        - vertical_padding_missing (盘古高空层为 13 层，若未通过 `get_pad3d` 将其 padding 到偶数 14 层，在计算 `13 // 2` 时会导致边界截断和偏置表索引越界)

  knowledge:

    usage_patterns:

      pangu_atmospheric_network:
        pipeline:
          - Pad3D (例如 13层 -> 14层)
          - 3DPatchEmbedding
          - EarthAttention3D (with 3D window mapping)
          - MLP
          - Unpad3D / PatchUnmerging

    design_patterns:

      pressure_and_latitude_aware_bias:
        structure:
          - 将纬度带与气压层的组合索引数量 (`type_of_windows`) 作为 3D 偏置表的核心维度。
          - 由于经度具有循环周期性（180度经线相连），代码在初始化 `earth_position_bias_table` 时保留了经向的相对跨度 (`window_size[2]*2 - 1`) 以匹配环球几何。

    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 顶尖的三维全球气象数据驱动预报大模型
        architecture: 3D Earth-Specific Transformer
        attention_type: 3D Earth Position Bias Attention

    model_usage_details:

      Pangu_Weather_Atmospheric_Variable_Standard:
        input_resolution: (14, 128, 256) # 原始 13 层经过 Pad 处理
        window_size: (2, 8, 8)
        num_heads: 6

    best_practices:
      - 高度强调 **Padding 的重要性**。真实的气象数据层数可能是一个奇数（如 ERA5 的 13 个标准气压层），但在传入此模块计算 `input_resolution` 前，必须进行诸如 `(14, 128, 256)` 的补齐，以保证窗口划分完美闭合。
      - 与 `EarthAttention2D` 一样，在传入 `forward` 之前，必须在上游做好维度转换和 `permute` 排列。

    anti_patterns:
      - 传入的 `input_resolution` 仍为未 Padding 的真实物理分辨率（如 `(13, 128, 256)`），这会导致后续 `type_of_windows` 丢失最后一部分网格，引发维度张量碰撞。

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
      - AtmosphericVariableNetwork

    depends_on:
      - get_earth_position_index
      - trunc_normal_
      - Softmax
      - Linear

    compatible_with:
      inputs:
        - WindowedAtmosphericFeatures
      outputs:
        - WindowedAtmosphericFeatures