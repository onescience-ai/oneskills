component:

  meta:
    name: EarthTransformer2DBlock
    alias: EarthSwinBlock2D
    version: 1.0
    domain: earth_science_deep_learning
    category: neural_network
    subcategory: spatial_attention
    author: OneScience
    license: Apache-2.0
    tags:
      - 2d_transformer
      - swin_transformer
      - weather_forecasting
      - surface_variables
      - shifted_window


  concept:

    description: >
      用于 2D 地表变量的地球感知 Swin Transformer Block。
      它是标准 Swin Transformer Block 的气象场适配版本，结合 EarthAttention2D 
      在二维平面（纬度和经度）上进行带地球位置偏置的窗口自注意力计算。
      支持循环移位（Shifted Window）以扩大局部感受野，并能自动处理分辨率与窗口大小不整除的情况。

    intuition: >
      在处理全球地表气象数据时（如 181x360 的分辨率），如果使用全局注意力会导致计算量过大。
      这个模块将全球地图划分为一个个 2D 矩形窗口（如 6x12），在窗口内部计算注意力。
      为了让相邻窗口的特征能够交互，在下一层计算时会将这些窗口错开半个身位（Shift），
      从而在多层堆叠后实现全局感受野的覆盖。

    problem_it_solves:
      - 降低高分辨率 2D 气象网格数据的自注意力计算复杂度
      - 自动处理纬度和经度维度无法被窗口大小整除的边界情况（通过 Padding 和 Cropping）
      - 有效提取地表气象要素（如地表温度、降水）的空间相关性


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
        name: InputSurfaceFeatures
        shape: [batch, lat * lon, dim]
        description: 展平后的二维地表气象特征向量

      W-MSA:
        name: WindowMultiHeadSelfAttention
        description: 常规 2D 窗口多头自注意力 (shift_size=0,0)

      SW-MSA:
        name: ShiftedWindowMultiHeadSelfAttention
        description: 循环移位 2D 窗口多头自注意力


  structure:

    architecture: swin_transformer_2d_block

    pipeline:

      - name: PreNorm
        operation: layer_norm

      - name: SpatialPadding
        operation: zero_pad_2d

      - name: SpatialShift
        operation: torch.roll (奇数层开启)

      - name: GridPartition
        operation: window_partition_2d

      - name: 2DWindowAttention
        operation: one_attention (带遮罩)

      - name: GridReverse
        operation: window_reverse_2d

      - name: SpatialCrop
        operation: crop_2d

      - name: FeedForward
        operation: mlp


  interface:

    parameters:

      dim:
        type: int
        description: 输入特征的通道数

      input_resolution:
        type: tuple[int, int]
        description: 输入特征图的空间分辨率 (lat, lon)

      num_heads:
        type: int
        description: 多头注意力的头数

      window_size:
        type: tuple[int, int]
        default: [6, 12]
        description: 2D 注意力窗口大小 (Wlat, Wlon)

      shift_size:
        type: tuple[int, int]
        default: [3, 6]
        description: 循环移位的偏移量 (shift_lat, shift_lon)

    inputs:

      x:
        type: SurfaceFeatureMap
        shape: [batch, lat * lon, dim]
        dtype: float32

    outputs:

      output:
        type: SurfaceFeatureMap
        shape: [batch, lat * lon, dim]
        description: 融合了局部到全局空间信息的地表特征


  types:

    SurfaceFeatureMap:
      shape: [batch, lat * lon, dim]
      description: 展平的 2D 地表气象特征序列


  implementation:

    framework: pytorch

    code: |

      import torch
      from torch import nn
      from ..func_utils import DropPath, Mlp, get_pad2d, crop2d, window_partition, window_reverse, get_shift_window_mask
      from onescience.modules.attention.oneattention import OneAttention

      class EarthTransformer2DBlock(nn.Module):
          """用于2D地表变量的地球感知 Swin Transformer Block"""
          def __init__(
              self, dim, input_resolution, num_heads, window_size=None, shift_size=None,
              mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
              drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
          ):
              super().__init__()
              self.window_size = (6, 12) if window_size is None else window_size
              self.shift_size = (3, 6) if shift_size is None else shift_size
              self.dim = dim
              self.input_resolution = input_resolution
              self.num_heads = num_heads
              
              self.norm1 = norm_layer(dim)
              padding = get_pad2d(input_resolution, self.window_size)
              self.pad = nn.ZeroPad2d(padding)
              
              pad_resolution = list(input_resolution)
              pad_resolution[0] += padding[2] + padding[3]
              pad_resolution[1] += padding[0] + padding[1]
              
              self.attn = OneAttention(
                  style="EarthAttention2D", dim=dim, input_resolution=pad_resolution,
                  window_size=self.window_size, num_heads=num_heads,
              )
              
              self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
              self.norm2 = norm_layer(dim)
              self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
              
              shift_lat, shift_lon = self.shift_size
              self.roll = shift_lon and shift_lat
              attn_mask = get_shift_window_mask(pad_resolution, self.window_size, self.shift_size, ndim=2) if self.roll else None
              self.register_buffer("attn_mask", attn_mask)

          def forward(self, x: torch.Tensor):
              Lat, Lon = self.input_resolution
              B, L, C = x.shape
              shortcut = x
              
              x = self.norm1(x).view(B, Lat, Lon, C)
              x = self.pad(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
              _, Lat_pad, Lon_pad, _ = x.shape
              
              shift_lat, shift_lon = self.shift_size
              if self.roll:
                  shifted_x = torch.roll(x, shifts=(-shift_lat, -shift_lat), dims=(1, 2))
              else:
                  shifted_x = x
                  
              x_windows = window_partition(shifted_x, self.window_size, ndim=2)
              win_lat, win_lon = self.window_size
              x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_lat * win_lon, C)
              
              attn_windows = self.attn(x_windows, mask=self.attn_mask)
              attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_lat, win_lon, C)
              
              shifted_x = window_reverse(attn_windows, self.window_size, Lat=Lat_pad, Lon=Lon_pad, ndim=2)
              if self.roll:
                  x = torch.roll(shifted_x, shifts=(shift_lat, shift_lon), dims=(1, 2))
              else:
                  x = shifted_x
                  
              x = crop2d(x.permute(0, 3, 1, 2), self.input_resolution).permute(0, 2, 3, 1)
              x = x.reshape(B, Lat * Lon, C)
              
              x = shortcut + self.drop_path(x)
              x = x + self.drop_path(self.mlp(self.norm2(x)))
              return x


  skills:

    build_surface_transformer_layer:

      description: 构建成对的 2D 地表 Swin Transformer Block

      inputs:
        - dim
        - input_resolution
        - num_heads

      prompt_template: |

        实例化 EarthTransformer2DBlock，必须保证相邻两层配对使用：
        层级 N: shift_size=(0, 0)
        层级 N+1: shift_size=(win_lat//2, win_lon//2)


    diagnose_surface_block:

      description: 排查 2D 气象编码过程中的常见维度和计算错误

      checks:
        - spatial_resolution_mismatch (输入尺寸未被正确 padding/crop 导致形状突变)
        - attention_mask_alignment (Mask 未与 shift_size 对齐导致信息串扰)



  knowledge:

    usage_patterns:

      surface_variable_encoder:

        pipeline:
          - 2D Patch Embedding
          - Multi-Stage EarthTransformer2DBlocks
          - Task Head (如预测 T2m, u10, v10)


    hot_models:

      - model: Swin Transformer
        year: 2021
        role: 基础的 2D 视觉层次化架构
        architecture: Vision Transformer

      - model: FengWu (风乌)
        year: 2023
        role: 结合多模态与多任务学习的全球中期天气预报大模型
        architecture: 包含独立的 2D 地表变量处理分支


    best_practices:

      - 在处理 ERA5 单层数据（如 181x360 分辨率）时，由于 181 无法被通常的 window_size（如 6）整除，模块内置的 `get_pad2d` 和 `crop2d` 是保证网络能顺畅运行的关键。
      - `window_size` 的经纬度比例可以根据地球的实际物理展开形状进行调整，例如经度通常是纬度的两倍，因此窗口常设为 (6, 12)。


    anti_patterns:

      - 将高空 3D 变量强行展平成 2D 送入此模块，会导致垂直方向的物理场信息丢失。
      - 只使用 W-MSA 而不交替使用 SW-MSA，模型将缺乏全局视野，只能看到局部天气。


    paper_references:

      - title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
        authors: Liu et al.
        year: 2021

      - title: "FengWu: Pushing the Global Weather Forecast Beyond 10 Days"
        authors: Chen et al.
        year: 2023



  graph:

    is_a:
      - TransformerBlock
      - SpatialAttentionModule

    part_of:
      - AIWeatherModel
      - 2DSurfaceEncoder

    depends_on:
      - OneAttention
      - Mlp
      - ZeroPad2d

    variants:
      - StandardSwinTransformerBlock

    used_in_models:
      - AI Weather Predictors (Surface Variables Branch)

    compatible_with:

      inputs:
        - SurfaceFeatureMap

      outputs:
        - SurfaceFeatureMap