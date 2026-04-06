component:

  meta:
    name: EarthTransformer3DBlock
    alias: EarthSwinBlock3D_Standard
    version: 1.0
    domain: earth_science_deep_learning
    category: neural_network
    subcategory: spatial_temporal_attention
    author: OneScience
    license: Apache-2.0
    tags:
      - 3d_transformer
      - swin_transformer
      - weather_forecasting
      - shifted_window
      - standard_module


  concept:

    description: >
      用于 3D 大气变量的地球感知 Swin Transformer Block（标准非分布式版本）。
      在气压层（Pressure）、纬度（Latitude）和经度（Longitude）三个维度上同时进行窗口注意力计算。
      支持三维循环移位（Shifted Window）以扩大跨窗口感受野，内置 Padding 和 Crop 机制以处理边界不整除问题。

    intuition: >
      与分布式版本一致，它通过在局部 3D 窗口内计算注意力来降低全局注意力的 O(N^2) 计算复杂度。
      在相邻层交替使用常规窗口和半窗口移位，使得高空到低空、全球不同经纬度的气象特征能够充分融合。
      作为标准版本，它不包含跨节点的分布式特征切分，结构更轻量，易于在标准单机多卡环境下进行调试和训练。

    problem_it_solves:
      - 高效处理具有空间和高度三维特性的高分辨率气象数据
      - 解决输入分辨率（如气压层数）无法被窗口大小完美整除的边界处理难题
      - 避免在较小规模训练或推理时引入不必要的分布式计算开销


  theory:

    formula:

      w_msa_block:
        expression: |
          x_hat = x + DropPath(W-MSA(Norm(x)))
          x_out = x_hat + DropPath(MLP(Norm(x_hat)))

      sw_msa_block:
        expression: |
          x_hat = x_out + DropPath(SW-MSA(Norm(x_out)))
          x_final = x_hat + DropPath(MLP(Norm(x_hat)))

    variables:

      x:
        name: InputFeatures
        shape: [batch, pl * lat * lon, dim]
        description: 展平后的三维大气特征向量

      W-MSA:
        name: WindowMultiHeadSelfAttention
        description: 常规 3D 窗口多头自注意力 (shift_size=0,0,0)

      SW-MSA:
        name: ShiftedWindowMultiHeadSelfAttention
        description: 循环移位 3D 窗口多头自注意力


  structure:

    architecture: swin_transformer_3d_block_standard

    pipeline:

      - name: PreNorm
        operation: layer_norm

      - name: BoundaryPadding
        operation: zero_pad_3d

      - name: WindowShift
        operation: torch.roll (根据 shift_size 动态开启)

      - name: WindowPartition
        operation: reshape_to_windows

      - name: 3DWindowAttention
        operation: one_attention (style="EarthAttention3D")

      - name: WindowReverse
        operation: reshape_back

      - name: WindowUnshift
        operation: torch.roll_back

      - name: BoundaryCropping
        operation: crop_3d

      - name: FeedForwardNetwork
        operation: standard_mlp


  interface:

    parameters:

      dim:
        type: int
        description: 输入特征的通道数

      input_resolution:
        type: tuple[int, int, int]
        description: 输入特征图的三维空间分辨率 (pl, lat, lon)

      num_heads:
        type: int
        description: 多头注意力的头数

      window_size:
        type: tuple[int, int, int]
        default: [2, 6, 12]
        description: 3D 注意力窗口大小 (Wpl, Wlat, Wlon)

      shift_size:
        type: tuple[int, int, int]
        default: [1, 3, 6]
        description: 循环移位的偏移量 (shift_pl, shift_lat, shift_lon)

    inputs:

      x:
        type: EarthFeatureMap3D
        shape: [batch, pl * lat * lon, dim]
        dtype: float32

    outputs:

      output:
        type: EarthFeatureMap3D
        shape: [batch, pl * lat * lon, dim]
        description: 融合了 3D 空间交互信息的输出特征


  types:

    EarthFeatureMap3D:
      shape: [batch, pl * lat * lon, dim]
      description: 展平的地球 3D 高空大气特征序列


  implementation:

    framework: pytorch

    code: |

      from collections.abc import Sequence
      import torch
      from timm.layers import to_2tuple
      from timm.models.swin_transformer import SwinTransformerStage
      from torch import nn
      from ..func_utils import DropPath, Mlp, get_pad3d, crop3d, window_partition, window_reverse, get_shift_window_mask
      from ..attention.oneattention import OneAttention

      class EarthTransformer3DBlock(nn.Module):
          """用于3D大气变量的地球感知 Swin Transformer Block"""
          def __init__(
              self, dim, input_resolution, num_heads, window_size=None, shift_size=None,
              mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
              drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
          ):
              super().__init__()
              self.window_size = (2, 6, 12) if window_size is None else window_size
              self.shift_size = (1, 3, 6) if shift_size is None else shift_size
              self.dim = dim
              self.input_resolution = input_resolution
              self.num_heads = num_heads
              
              self.norm1 = norm_layer(dim)
              padding = get_pad3d(input_resolution, self.window_size)
              self.pad = nn.ZeroPad3d(padding)
              
              pad_resolution = list(input_resolution)
              pad_resolution[0] += padding[-1] + padding[-2]
              pad_resolution[1] += padding[2] + padding[3]
              pad_resolution[2] += padding[0] + padding[1]
              
              self.attn = OneAttention(
                  style="EarthAttention3D", dim=dim, input_resolution=pad_resolution,
                  window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
              )
              
              self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
              self.norm2 = norm_layer(dim)
              self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
              
              shift_pl, shift_lat, shift_lon = self.shift_size
              self.roll = shift_pl and shift_lon and shift_lat
              attn_mask = get_shift_window_mask(pad_resolution, self.window_size, self.shift_size) if self.roll else None
              self.register_buffer("attn_mask", attn_mask)

          def forward(self, x: torch.Tensor):
              Pl, Lat, Lon = self.input_resolution
              B, L, C = x.shape
              shortcut = x
              
              x = self.norm1(x).view(B, Pl, Lat, Lon, C)
              x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
              _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape
              
              shift_pl, shift_lat, shift_lon = self.shift_size
              if self.roll:
                  shifted_x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))
              else:
                  shifted_x = x
                  
              x_windows = window_partition(shifted_x, self.window_size)
              win_pl, win_lat, win_lon = self.window_size
              x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C)
              
              attn_windows = self.attn(x_windows, mask=self.attn_mask)
              attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C)
              
              shifted_x = window_reverse(attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad)
              if self.roll:
                  x = torch.roll(shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
              else:
                  x = shifted_x
                  
              x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)
              x = x.reshape(B, Pl * Lat * Lon, C)
              
              x = shortcut + self.drop_path(x)
              x = x + self.drop_path(self.mlp(self.norm2(x)))
              return x


  skills:

    build_standard_3d_block:

      description: 构建成对的标准 3D Swin Transformer Block

      inputs:
        - dim
        - input_resolution
        - num_heads

      prompt_template: |

        实例化 EarthTransformer3DBlock。必须成对配置：
        第一层：shift_size=(0,0,0)
        第二层：shift_size=(win_pl//2, win_lat//2, win_lon//2)


    diagnose_3d_block:

      description: 排查 3D 气象数据的空间不对齐问题

      checks:
        - padding_crop_mismatch
        - attention_mask_error (循环移位遮罩计算错误)



  knowledge:

    usage_patterns:

      standard_3d_weather_encoder:

        pipeline:
          - 3D Patch Embedding
          - [W-MSA Block, SW-MSA Block] 堆叠
          - Downsample (可选)


    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 3D 地球空间网络架构参照模型
        architecture: 3D Earth Specific Transformer (3DEST)


    best_practices:

      - 在无需极大规模多节点并行的实验中，优先使用此标准版本，以避免 `DistributedMlp` 带来的复杂性和通信调试成本。
      - 确保输入的 `input_resolution` 的三维顺序严格为 `(Pressure_Level, Latitude, Longitude)`，否则内置的 Padding 和 Shift 逻辑会发生物理意义上的错位。


    anti_patterns:

      - `shift_size` 设置超过 `window_size` 的一半，会导致邻近窗口间的信息融合效率下降。
      - 忽略 `attn_mask`，会导致位于北极和南极的像素点在循环移位（Roll）后产生错误的注意力交互。


    paper_references:

      - title: "Accurate medium-range global weather forecasting with 3D neural networks"
        authors: Bi et al.
        year: 2023



  graph:

    is_a:
      - TransformerBlock
      - StandardModule

    part_of:
      - 3DEarthEncoder
      - AIWeatherModel

    depends_on:
      - OneAttention
      - Mlp
      - ZeroPad3d

    variants:
      - EarthDistributedTransformer3DBlock

    used_in_models:
      - AI Weather Predictors (High-Altitude Branch)

    compatible_with:

      inputs:
        - EarthFeatureMap3D

      outputs:
        - EarthFeatureMap3D