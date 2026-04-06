component:

  meta:
    name: XiHeTransformer3D
    alias: 3D WeatherLearn Transformer Block
    version: 1.0
    domain: meteorological_ai
    category: neural_network
    subcategory: spatial_temporal_block
    author: OneScience (Revised from WeatherLearn)
    license: Apache-2.0
    tags:
      - swin_transformer
      - 3d_attention
      - weatherlearn
      - earth_system_modeling
      - meteorology

  concept:

    description: >
      XiHeTransformer3D 是一个基于三维滑动窗口自注意力机制（3D SW-MSA）的 Transformer 编码器块。
      该模块从气象机器学习开源库 WeatherLearn 改进而来，专门用于处理地球系统科学中具有垂直高度/深度、纬度和经度的三维流体网格数据。
      它支持通过局部三维窗口进行注意力计算，并通过掩码（mask）机制处理陆地-海洋等复杂物理边界。

    intuition: >
      类似于计算机视觉中的视频处理，全球大气或海洋数据不仅在水平面（经纬度）上有相关性，在垂直方向（气压层或水深）也有强烈的物理交换。
      如果对整个三维空间做全局注意力，计算量是不可接受的。因此，模型采用“分块处理+滑动交互”的思想，
      先把地球三维网格切分成一个个像“小魔方”一样的局部窗口（Window Partition）来计算注意力，
      然后在下一层把这些“小魔方”错开半个身位（Shift Window），让相邻区域的信息能够互相传递。

    problem_it_solves:
      - 缓解处理高分辨率三维地球科学数据（如高空风场、温度场的多气压层数据）时面临的维度灾难和显存溢出问题。
      - 提供对无效区域（如气象预测中的地形遮挡区域，或海洋预测中的陆地区域）的有效屏蔽（通过传入特定的掩码张量实现）。

  theory:

    formula:

      window_attention:
        expression: X' = 3D-W-MSA(LayerNorm(X), Mask) + X
        
      shifted_window_attention:
        expression: X'' = 3D-SW-MSA(LayerNorm(X'), Mask_{shift}) + X' (当 roll 激活时)

      feed_forward_network:
        expression: Output = MLP(LayerNorm(X'')) + X''

    variables:

      Pl:
        name: PressureLevels
        description: 气压层数（或海洋模式中的垂直深度分层）

      Lat:
        name: Latitude
        description: 纬度网格点数

      Lon:
        name: Longitude
        description: 经度网格点数

  structure:

    architecture: swin_transformer_3d_variant

    pipeline:

      - name: InputNormalization
        operation: layer_norm

      - name: SpatialPaddingAndShift
        operation: pad_and_roll (补齐边界并根据 shift_size 进行三维空间滚动)

      - name: WindowPartition
        operation: 3d_window_slicing (按 window_size 切分成多个独立的三维小块)

      - name: EarthAttention3D
        operation: masked_3d_attention (执行基于有效性掩码的三维窗口内自注意力计算)

      - name: WindowReverse
        operation: unslice_and_unshift (将小块还原回全局视图并反向滚动)

      - name: FirstResidualConnection
        operation: add_with_drop_path

      - name: MLPLayer
        operation: norm_and_mlp (进行通道特征维度的非线性映射)

      - name: SecondResidualConnection
        operation: add_with_drop_path

  interface:

    parameters:

      dim:
        type: int
        description: 输入特征图的通道维度 C

      input_resolution:
        type: tuple[int, int, int]
        description: 输入网格的三维空间分辨率 (Pl, Lat, Lon)

      num_heads:
        type: int
        description: 并行注意力头的数量

      window_size:
        type: tuple[int, int, int]
        default: (2, 6, 12)
        description: 注意力机制的局部三维窗口大小 (Wpl, Wlat, Wlon)

      shift_size:
        type: tuple[int, int, int]
        default: (1, 3, 6)
        description: 用于建立跨窗口连接的偏移像素步长 (Spl, Slat, Slon)

      mlp_ratio:
        type: float
        default: 4.0
        description: 多层感知机中间隐层宽度的扩张倍率

      qkv_bias:
        type: bool
        default: true
        description: 是否在 Q, K, V 投影时添加偏置项

      qk_scale:
        type: float
        default: null
        description: 覆盖默认的 QK 缩放系数

      drop:
        type: float
        default: 0.0
        description: 输出和 MLP 的 Dropout 比例

      attn_drop:
        type: float
        default: 0.0
        description: 注意力权重的 Dropout 比例

      drop_path:
        type: float
        default: 0.0
        description: DropPath (随机深度) 比例

    inputs:

      x:
        type: Tensor
        shape: [B, Pl * Lat * Lon, C]
        dtype: float32
        description: 三维空间展平后的特征序列

      mask:
        type: Tensor
        shape: "[B, 1, Pl, Lat, Lon] 或 [B, 1, Lat, Lon]"
        default: null
        description: 选填的有效区域掩码张量，0代表忽略，1代表计算

    outputs:

      output:
        type: Tensor
        shape: [B, Pl * Lat * Lon, C]
        description: 经过三维局部自注意力和前馈网络处理后，与输入形状完全一致的特征序列表达

  types:

    EarthSpatialEmbedding:
      shape: [B, L, C]
      description: 包含地球三维网格位置拓扑关系的连续特征序列

  implementation:

    framework: pytorch

    code: |
      import torch

      from torch import nn


      class XiHeTransformer3D(nn.Module):
          """
          Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
          3D Transformer Block
          Args:
              dim (int): Number of input channels.
              input_resolution (tuple[int]): Input resulotion.
              num_heads (int): Number of attention heads.
              window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
              shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
              mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
              qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
              qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
              drop (float, optional): Dropout rate. Default: 0.0
              attn_drop (float, optional): Attention dropout rate. Default: 0.0
              drop_path (float, optional): Stochastic depth rate. Default: 0.0
              act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
              norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
          """

          def __init__(
              self,
              dim,
              input_resolution,
              num_heads,
              window_size=None,
              shift_size=None,
              mlp_ratio=4.0,
              qkv_bias=True,
              qk_scale=None,
              drop=0.0,
              attn_drop=0.0,
              drop_path=0.0,
              act_layer=nn.GELU,
              norm_layer=nn.LayerNorm,
          ):
              super().__init__()
              window_size = (2, 6, 12) if window_size is None else window_size
              shift_size = (1, 3, 6) if shift_size is None else shift_size
              self.dim = dim
              self.input_resolution = input_resolution
              self.num_heads = num_heads
              self.window_size = window_size
              self.shift_size = shift_size
              self.mlp_ratio = mlp_ratio
              self.norm1 = norm_layer(dim)
              padding = get_pad3d(input_resolution, window_size)
              self.pad = nn.ZeroPad3d(padding)
              attn_mask=None

              pad_resolution = list(input_resolution)
              pad_resolution[0] += padding[-1] + padding[-2]
              pad_resolution[1] += padding[2] + padding[3]
              pad_resolution[2] += padding[0] + padding[1]

              self.attn = EarthAttention3D(
                  dim=dim,
                  input_resolution=pad_resolution,
                  window_size=window_size,
                  num_heads=num_heads,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  attn_drop=attn_drop,
                  proj_drop=drop,
              )

              self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
              self.norm2 = norm_layer(dim)
              mlp_hidden_dim = int(dim * mlp_ratio)
              self.mlp = Mlp(
                  in_features=dim,
                  hidden_features=mlp_hidden_dim,
                  act_layer=act_layer,
                  drop=drop,
              )

              shift_pl, shift_lat, shift_lon = self.shift_size
              self.roll = shift_pl and shift_lon and shift_lat

              if self.roll:
                  attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
              else:
                  attn_mask = None

              self.register_buffer("attn_mask", attn_mask)
              

          def forward(self, x: torch.Tensor,mask: torch.Tensor = None):
              Pl, Lat, Lon = self.input_resolution
              B, L, C = x.shape

              shortcut = x
              x = self.norm1(x)
              x = x.view(B, Pl, Lat, Lon, C)
              # start pad
              x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

              _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

              shift_pl, shift_lat, shift_lon = self.shift_size
              
              if self.roll:
                  shifted_x = torch.roll(
                      x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3)
                  )
                  x_windows = window_partition(shifted_x, self.window_size)
              else:        
                  shifted_x = x
                  x_windows = window_partition(shifted_x, self.window_size)
              win_pl, win_lat, win_lon = self.window_size
              
              x_windows = x_windows.view(
                  x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C
              )
       
              attn_mask = None
              if mask is not None:
                  # 期望 mask 是 [B, 1, Lat, Lon] 或 [B, 1, Pl, Lat, Lon]
                  if mask.dim() == 4:                # (B,1,Lat,Lon) -> (B,1,1,Lat,Lon)
                      mask = mask.unsqueeze(2)

                  # 此时 mask: (B, 1, Pl, Lat, Lon) 期望 (N, C, D, H, W)；这里 C=1, D=Pl, H=Lat, W=Lon，直接 pad 即可
                  mask = self.pad(mask)              # (B, 1, Pl_pad, Lat_pad, Lon_pad)

                  # 为了与 window_partition 通用实现对齐，转成 (B, Pl_pad, Lat_pad, Lon_pad, 1)
                  mask5d = mask.permute(0, 2, 3, 4, 1).contiguous()

                  # 与特征 x 完全一致的分块（3D窗口）
                  # mwin: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, 1)
                  mwin = window_partition(mask5d, self.window_size)

                  win_pl, win_lat, win_lon = self.window_size
                  # 计算分块数量
                  # 注意：x 已经 pad 过，这里的 Pl_pad/Lat_pad/Lon_pad 要和上面 x 的 pad 后维度一致
                  _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape               # x 此时是 pad 后的 (B, Pl_pad, Lat_pad, Lon_pad, C)
                  B_eff  = mask5d.shape[0]
                  num_lon   = Lon_pad // win_lon
                  num_pllat = (Pl_pad // win_pl) * (Lat_pad // win_lat)
                  N = win_pl * win_lat * win_lon                         # 每个窗口 token 数

                  # 把 (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, 1) 还原出 (B, num_lon, num_pl*num_lat, N)
                  mwin = mwin.view(B_eff, num_lon, num_pllat, win_pl, win_lat, win_lon, 1)
                  # 取第 0 个 batch
                  mwin = mwin[0]                                         # (num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, 1)
                  mwin = mwin.view(num_lon, num_pllat, N)                # (num_lon, num_pl*num_lat, N)，元素∈{0,1}

                  # 生成注意力掩码 (num_lon, num_pl*num_lat, N, N) 仅允许 海×海，其他（涉及陆地）设为 -inf
                  attn_mask = (mwin.unsqueeze(-1) * mwin.unsqueeze(-2))  # 0/1
                  attn_mask = (attn_mask == 0).float() * -100.0          # 变成 0 / -100

              attn_windows = self.attn(x_windows, mask=attn_mask)
              attn_windows = attn_windows.view(
                  attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C
              )

              if self.roll:
                  shifted_x = window_reverse(
                      attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad
                  )

                  x = torch.roll(
                      shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3)
                  )
              else:
                  shifted_x = window_reverse(
                      attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad
                  )
                  x = shifted_x


              x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(
                  0, 2, 3, 4, 1
              )

              x = x.reshape(B, Pl * Lat * Lon, C)
              #两次残差
              x = shortcut + self.drop_path(x)
              x = x + self.drop_path(self.mlp(self.norm2(x)))

  skills:

    build_xihe_transformer3d:

      description: 构建兼容 WeatherLearn 的三维气象特征提取块

      inputs:
        - dim
        - input_resolution
        - num_heads
        - window_size
        - shift_size

      prompt_template: |
        构建一个基于 XiHeTransformer3D 的气象特征编码层。
        参数：
        输入分辨率 = {{input_resolution}}
        通道数 = {{dim}}
        注意力头数 = {{num_heads}}
        请配置三维窗口大小 {{window_size}} 及偏移量 {{shift_size}}。

    diagnose_mask_tensor:

      description: 排查输入掩码张量维度不一致导致的计算崩溃

      checks:
        - mask_rank_mismatch (检查传入的 mask 维度是否为 4 维或 5 维，以保证内部 unsqueeze 和 pad 操作正常运行)
        - nan_output_due_to_mask (检查是否因为全区域被 mask 设置为 -100.0，导致 softmax 输出 NaN)

  knowledge:

    usage_patterns:

      atmospheric_and_oceanic_modeling:
        pipeline:
          - VariableEmbedding
          - XiHeTransformer3D (Layer 1, shift_size=(0,0,0))
          - XiHeTransformer3D (Layer 2, shift_size=(1,3,6))
          - PredictionHead

    design_patterns:

      3d_masked_window_attention:
        structure:
          - 通过显式传入 `x` 和 `mask` 分离数据与物理属性。
          - 预计算填充边界 (ZeroPad3d) 并切分窗口，配合掩码确保注意力仅在有效物理空间内交互。

    hot_models:

      - model: WeatherLearn
        year: 2023
        role: 致力于气象与气候领域机器学习的开源代码库
        architecture: Swin-Transformer (3D Variants)
        attention_type: EarthAttention3D

      - model: Swin Transformer
        year: 2021
        role: 引入滑动窗口策略的视觉 Transformer

    model_usage_details:

      WeatherLearn_Default_Config:
        window_size: (2, 6, 12)
        shift_size: (1, 3, 6)
        mlp_ratio: 4.0
        qkv_bias: true

    best_practices:
      - 此模块推荐在处理多气压层数据时使用，若仅处理单层表面数据（如海表温度 SST），建议退化 `window_size` 的高度维度。
      - 与上游数据交互时，确保 `mask` 张量使用 `float32` 格式并且仅包含 0 和 1。

    anti_patterns:
      - 传入的 `input_resolution` 与实际展平的序列长度 `L` 不匹配，会导致 `.view(B, Pl, Lat, Lon, C)` 时直接报维度不一致错误。
      - 忘记在连续的 Transformer Block 间交替设置 `shift_size`，导致感受野无法扩大。

    paper_references:

      - title: "WeatherLearn: An open-source library for Machine Learning in Weather and Climate"
        authors: WeatherLearn Contributors / Li Zhuo
        year: 2023 (Github Repository)

      - title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
        authors: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
        year: 2021

  graph:

    is_a:
      - EncoderBlock
      - SpatialTemporalBlock
      - NeuralNetworkComponent

    part_of:
      - 3DEarthTransformer
      - WeatherLearnModels

    depends_on:
      - EarthAttention3D
      - Mlp
      - window_partition
      - window_reverse
      - TorchRoll

    compatible_with:
      inputs:
        - EarthSpatialEmbedding
        - ValidRegionMask
      outputs:
        - EarthSpatialEmbedding