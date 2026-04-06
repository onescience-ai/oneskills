component:

  meta:
    name: EarthDistributedTransformer3DBlock
    alias: EarthSwinBlock3D
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
      - earth_science


  concept:

    description: >
      用于 3D 大气变量的地球感知 Swin Transformer Block。
      它结合了气压层（Pressure）、纬度（Latitude）和经度（Longitude）三个维度，
      在局部 3D 窗口内进行自注意力计算，并通过三维循环移位（Shifted Window）扩大跨窗口的感受野。

    intuition: >
      全球高分辨率气象数据体量庞大，如果让任意两个空间点都计算注意力（全局 Attention），计算量会爆炸。
      此模块借鉴了 Swin Transformer 的思想：先把全球大气划分为一个个局部的 3D 小方块（Window），
      在方块内部计算注意力；然后在下一层将这些方块的边界“平移”错开（Shift），
      让原本在不同方块的相邻区域能被划分到同一个新方块中进行信息交流。

    problem_it_solves:
      - 解决 3D 全局自注意力计算复杂度随分辨率呈 O(N^2) 增长的内存和算力瓶颈
      - 解决气象数据维度（Pl, Lat, Lon）与窗口大小不整除时的边界处理问题
      - 捕获大气在垂直（气压层）和水平（经纬度）方向上的三维物理耦合关系


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

    architecture: swin_transformer_3d_block

    pipeline:

      - name: PreNorm
        operation: layer_norm

      - name: BoundaryPadding
        operation: zero_pad_3d (解决不整除问题)

      - name: WindowShift
        operation: torch.roll (奇数层开启)

      - name: WindowPartition
        operation: reshape_to_windows

      - name: 3DWindowAttention
        operation: one_attention (带掩码)

      - name: WindowReverseAndUnshift
        operation: reshape_and_roll_back

      - name: BoundaryCropping
        operation: crop_3d (恢复原始分辨率)

      - name: FeedForwardNetwork
        operation: distributed_mlp


  interface:

    parameters:

      dim:
        type: int
        description: 输入 token 的通道数（嵌入维度）

      input_resolution:
        type: tuple[int, int, int]
        description: 输入特征图的空间分辨率 (pl, lat, lon)

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
        type: EarthFeatureMap
        shape: [batch, pl * lat * lon, dim]
        dtype: float32

    outputs:

      output:
        type: EarthFeatureMap
        shape: [batch, pl * lat * lon, dim]
        description: 分辨率与通道数均保持不变的输出特征


  types:

    EarthFeatureMap:
      shape: [batch, pl * lat * lon, dim]
      description: 展平的地球 3D 气象特征序列


  implementation:

    framework: pytorch

    code: |

      import torch
      from torch import nn
      from ..func_utils import DropPath, DistributedMlp, get_pad3d, crop3d, window_partition, window_reverse, get_shift_window_mask
      from ..attention.oneattention import OneAttention

      class EarthDistributedTransformer3DBlock(nn.Module):
          """用于3D大气变量的地球感知 Swin Transformer Block"""
          def __init__(
              self, dim, input_resolution, num_heads, window_size=None, shift_size=None,
              mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
              drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None
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
                  style="EarthDistributedAttention3D", dim=dim, input_resolution=pad_resolution,
                  window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, config=config
              )
              
              self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
              self.norm2 = norm_layer(dim)
              self.mlp = DistributedMlp(
                  in_features=dim, hidden_features=int(dim * mlp_ratio),
                  act_layer=act_layer, drop=drop, config=config
              )
              
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

    build_earth_transformer_layer:

      description: 构建成对的 3D Swin Transformer Block (W-MSA + SW-MSA)

      inputs:
        - dim
        - input_resolution
        - num_heads

      prompt_template: |

        在构建 EarthTransformer 阶段时，必须成对实例化此 Block。
        第一层传入 shift_size=(0,0,0) 作为 W-MSA。
        第二层传入 shift_size=(win_pl//2, win_lat//2, win_lon//2) 作为 SW-MSA。


    diagnose_earth_block:

      description: 分析 3D 地球注意力机制的边界与内存异常

      checks:
        - padding_crop_mismatch (Pad和Crop后分辨率未对齐)
        - attention_mask_error (循环移位后的自注意力泄露到了非相邻的物理空间)
        - distributed_mlp_sync_issue (跨节点 MLP 通信瓶颈)



  knowledge:

    usage_patterns:

      earth_encoder_decoder:

        pipeline:
          - 3D Patch Embedding
          - [EarthTransformer3DBlock (W-MSA), EarthTransformer3DBlock (SW-MSA)] x N
          - 3D Patch Unembedding / Upsampling


    hot_models:

      - model: Pangu-Weather (盘古气象)
        year: 2023
        role: 首个精度超过传统数值天气预报（NWP）的 AI 气象模型
        architecture: 3D Earth Specific Transformer (3DEST)

      - model: Swin Transformer V2
        year: 2021
        role: 提供 Shifted Window 核心理论范式
        architecture: Hierarchical Vision Transformer

      - model: GraphCast / FengWu
        year: 2023
        role: 顶尖 AI 气象预测大模型（作为应用背景参考）


    best_practices:

      - 必须成对使用此 Block：一个是常规窗口（shift=(0,0,0)），紧跟一个移位窗口（如 shift=(1,3,6)），以打通全局感受野。
      - `window_size` 的设置应考虑实际物理意义。例如在经度（Lon）上划分更多的窗口，因为经度的像素通常最多（如 360 或 1440）。
      - 气压层（Pl）的 `window_size` 不宜过大，因为高空和地表的物理规律差异较大，过度平滑垂直特征可能导致低层预测精度下降。


    anti_patterns:

      - 在不需要 3D 拓扑结构的数据上强行使用 3D Block，会引入巨大的 Padding 显存开销。
      - 错误配置 `attn_mask`，导致模型把太平洋东岸的天气和西岸的天气错误地跨边界融合（如果不期望做环球闭合边界的话）。


    paper_references:

      - title: "Accurate medium-range global weather forecasting with 3D neural networks" (Pangu-Weather)
        authors: Bi et al. (Nature)
        year: 2023

      - title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
        authors: Liu et al.
        year: 2021

      - title: "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators"
        authors: Pathak et al.
        year: 2022



  graph:

    is_a:
      - TransformerBlock
      - SpatialTemporalModule

    part_of:
      - AIWeatherModel
      - 3DEarthEncoder

    depends_on:
      - OneAttention
      - DistributedMlp
      - ZeroPad3d
      - DropPath

    variants:
      - EarthDistributedTransformer2DBlock
      - StandardSwinTransformerBlock

    used_in_models:
      - Pangu-Weather (Architecture Reference)

    compatible_with:

      inputs:
        - EarthFeatureMap

      outputs:
        - EarthFeatureMap