component:

  meta:
    name: XihelocalTransformer
    alias: 3D Earth Swin-Transformer Block
    version: 1.0
    domain: meteorological_ai
    category: neural_network
    subcategory: spatial_temporal_block
    author: OneScience
    license: Apache-2.0
    tags:
      - swin_transformer
      - 3d_attention
      - earth_bias
      - ocean_forecasting
      - meteorology

  concept:

    description: >
      XihelocalTransformer 是一种具有地球位置偏置的三维 Swin-Transformer 模块，专门设计用于处理气象和海洋领域的全球三维网格序列数据（如压力层/深度、纬度、经度）。
      它在标准的三维窗口注意力机制基础上，引入了针对地球几何特性的可选 Shift Window（滑动窗口）操作，并深度集成了针对陆地-海洋边界的特征掩码（Mask）过滤机制。

    intuition: >
      在构建高分辨率的全球海洋或气象预测模型（如 1/12° 分辨率的预测）时，直接对庞大的三维网格（Pl × Lat × Lon）计算全局自注意力会导致计算量和显存需求的爆炸。
      这个模块像是一个高效的“地球网格扫描仪”，它通过 3D 窗口切分（Window Partition）将全球数据划分为一个个局部块进行独立计算，极大地节省了算力。
      同时，由于地球流体（如洋流、大气环流）的运动是不受人为固定窗口限制的，它通过类似 Swin-Transformer 的窗口滑动机制（Shift Window）来促进相邻三维区域的信息交互。
      更巧妙的是，它内置了严格的物理注意力掩码计算（attn_mask），能够精准识别“有效区域×有效区域”（例如海与海），自动以 -100.0 的极小值屏蔽掉涉及陆地等无效区域的计算。

    problem_it_solves:
      - 解决在全球高分辨率网格上进行三维气象或海洋预测时，传统注意力机制带来的二次方计算复杂度与显存开销瓶颈
      - 突破常规计算机视觉 Transformer 无法直接适配真实物理边界的问题，通过自适应的海陆掩码机制（Land-Ocean Mask）排除陆地对海洋环流预测的干扰
      - 增强模型在三维空间中捕捉中尺度涡旋（mesoscale eddies）和大尺度环流（large-scale circulation）等空间强耦合动力学特征的能力

  theory:

    formula:

      window_attention:
        expression: X' = W-MSA(LayerNorm(X), Mask) + X
        
      shifted_window_attention:
        expression: X'' = SW-MSA(LayerNorm(X'), Mask_{shift}) + X' (仅当引入位移参数 shift_size 时生效)

      feed_forward_network:
        expression: Output = MLP(LayerNorm(X'')) + X''

    variables:

      Pl:
        name: PressureLevel_or_Depth
        description: 气象中的垂直气压层数量，或海洋模式中的水深分层数量

      Lat:
        name: Latitude
        description: 纬度方向的空间分辨率点数

      Lon:
        name: Longitude
        description: 经度方向的空间分辨率点数

      Mask:
        name: ValidityMask
        description: 0/1 矩阵，标记某网格点是否为需计算的有效流体区域（例如海洋为1，陆地为0）

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
        operation: masked_one_attention (执行基于有效性掩码的三维窗口内自注意力计算)

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
        default: 6
        description: 并行注意力头的数量

      window_size:
        type: tuple[int, int, int]
        default: (2, 6, 12)
        description: 注意力机制的局部三维窗口大小 (Wpl, Wlat, Wlon)

      shift_size:
        type: tuple[int, int, int]
        default: (0, 0, 0)
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

    inputs:

      obj:
        type: dict_or_object
        description: 包含输入张量和掩码的字典或对象
        properties:
          x: 
            shape: [B, L, C] (其中 L = Pl × Lat × Lon)
            description: 三维空间展平后的特征序列
          mask: 
            shape: [B, 1, Pl, Lat, Lon] 或 [B, 1, Lat, Lon]
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
      from collections.abc import Sequence
      import torch
      import torch.nn as nn
      from onescience.modules.func_utils import Mlp,crop3d,get_pad3d,window_partition,window_reverse,DropPath
      from onescience.modules.attention.oneattention import OneAttention

      class XihelocalTransformer(nn.Module):
          """
          具有地球位置偏置的三维 Swin-Transformer Block（基于窗口注意力 + 可选 Shift Window）。

          Args:
              dim (int): 输入通道数 C。
              input_resolution (tuple[int, int, int]): 输入空间分辨率 (Pl, Lat, Lon)。
              num_heads (int): 注意力头数量。
              window_size (tuple[int, int, int], optional): 窗口大小 (Wpl, Wlat, Wlon)，默认为 (2, 6, 12)。
              shift_size (tuple[int, int, int], optional): Shift Window 偏移大小 (Spl, Slat, Slon)，默认为 (1, 3, 6)。
              mlp_ratio (float, optional): MLP 隐层扩展比例，默认为 4.0。
              qkv_bias (bool, optional): 是否在 QKV 上添加偏置，默认为 True。
              qk_scale (float | None, optional): 覆盖默认 QK 缩放系数 (head_dim ** -0.5)，默认为 None。
              drop (float, optional): 输出/MLP dropout 比例，默认为 0.0。
              attn_drop (float, optional): 注意力权重 dropout 比例，默认为 0.0。
              drop_path (float, optional): DropPath（随机深度）比例，默认为 0.0。
              act_layer (nn.Module, optional): 激活函数层类型，默认为 nn.GELU。
              norm_layer (nn.Module, optional): 归一化层类型，默认为 nn.LayerNorm。

          形状:
              输入 x: (B, L, C)，其中 L = Pl × Lat × Lon
              输入 mask (可选): (B, 1, Pl, Lat, Lon) 或 (B, Pl, Lat, Lon)，值为 0/1（1=有效，0=忽略）
              输出: (B, L, C)

          Example:
              >>> block = TransformerOceanBlock(
              ...     dim=192,
              ...     input_resolution=(13, 128, 256),
              ...     num_heads=6,
              ...     window_size=(1, 8, 8),
              ...     shift_size=(0, 0, 0)
              ... )
              >>> B, Pl, Lat, Lon, C = 2, 13, 128, 256, 192
              >>> x = torch.randn(B, Pl * Lat * Lon, C)
              >>> out = block(x)
              >>> out.shape
              torch.Size([2, 425984, 192])
          """

          def __init__(
              self,
              dim,
              input_resolution,
              num_heads=6,
              window_size=(1,6,12),
              shift_size=(0,0,0),
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
              # shift_size = (1, 3, 6) if shift_size is None else shift_size
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

              self.attn = OneAttention(
                  style="EarthAttention3D",
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

          # def forward(self, x: torch.Tensor,mask: torch.Tensor = None):
          def forward(self, obj):
              # x=obj.x
              # mask=obj.mask
              
              if isinstance(obj, dict):
                  # 字典方式访问
                  x=obj["x"]
                  mask = obj["mask"].clone().detach().float()
          
              # 判断是否为对象（非字典的其他类型）
              else:
                  # 对象方式访问        
                  x=obj.x
                  mask=obj.mask
                  obj={
                      "x":x,
                      "mask":mask,
                  } 
              Pl, Lat, Lon = self.input_resolution
              B, L, C = x.shape
              
              shortcut = x
              x = self.norm1(x)
              x = x.view(B, Pl, Lat, Lon, C)
              # start pad
              x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

              _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

              shift_pl, shift_lat, shift_lon =self.shift_size
              
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

              return x


  skills:

    build_xihe_transformer:

      description: 构建具有特定地球拓扑结构的 3D Swin-Transformer 层

      inputs:
        - dim
        - input_resolution
        - num_heads
        - window_size
        - shift_size

      prompt_template: |
        构建一个针对全分辨率 {{input_resolution}} 气象数据的 XihelocalTransformer 模块。
        参数：
        通道数 = {{dim}}
        注意力头数 = {{num_heads}}
        请配置三维窗口大小 {{window_size}}，并为了加强边界交互设置位移大小 {{shift_size}}。

    diagnose_mask_alignment:

      description: 分析预报模型由于掩码形状不对齐导致的计算异常或穿透问题

      checks:
        - land_ocean_mask_dimension_mismatch (输入的 mask 与数据的 3D 维度 Pl/Lat/Lon 不匹配)
        - boundary_padding_error_on_shift (在执行 Shift Window 并 Padding 后，边缘的陆地节点错误地渗透到海洋交互中)

  knowledge:

    usage_patterns:

      oceanic_circulation_forecasting:
        pipeline:
          - DataSlicingAndPadding
          - LandOceanMasking
          - XihelocalTransformer (Layer n)
          - ShiftedXihelocalTransformer (Layer n+1) 

    design_patterns:

      3d_masked_window_attention:
        structure:
          - 在 3D 空间进行窗口截断并计算自注意力，同时引入 `mwin.unsqueeze(-1) * mwin.unsqueeze(-2)` 以确保只有同时位于有效区（如海洋）的点才会建立互补连接，避免计算资源的浪费和无效数据的干扰。

    hot_models:

      - model: XiHe
        year: 2024
        role: 1/12° 高分辨率的数据驱动全球海洋涡旋分辨预报模型
        architecture: Hierarchical Transformer
        attention_type: 3D Masked Swin-Attention

      - model: Swin Transformer
        year: 2021
        role: 引入滑动窗口策略的开创性视觉特征提取主干

    model_usage_details:

      XiHe_Ocean_Predictor:
        window_size: (2, 6, 12)
        shift_size: (1, 3, 6)
        mlp_ratio: 4.0

    best_practices:
      - 组装模型时，应该交替堆叠常规窗口模块（`shift_size=(0,0,0)`）和滑动窗口模块（如 `shift_size=(1,3,6)`），以保证局部信息可以传递到全局。
      - 当 `roll` 机制激活时，模型边缘的数据会通过环形滚动连接起来，这对处理地球经度的连续性（比如 180°W 和 180°E 的闭环）非常有利。

    anti_patterns:
      - 传入的 Mask 矩阵存在维度缺失（比如仅有经纬度却没有深度通道），导致代码内部在扩充到 5D 张量 `(B, 1, Pl_pad, Lat_pad, Lon_pad)` 时发生严重形变。
      - `shift_size` 的大小超过了 `window_size`，这违背了局部视野交错的初衷。

    paper_references:

      - title: "XiHe: A Data-Driven Model for Global Ocean Eddy-Resolving Forecasting"
        authors: Xiang Wang, Renzhi Wang, Junqiang Song, et al.
        year: 2024

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
      - MeteorologicalForecastingSystem
      - XiHeModel

    depends_on:
      - OneAttention
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