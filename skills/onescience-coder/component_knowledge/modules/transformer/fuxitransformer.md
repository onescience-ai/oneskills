component:

  meta:
    name: FuxiTransformer
    alias: FuXiBlock
    version: 1.0
    domain: earth_science_deep_learning
    category: neural_network
    subcategory: u_net_transformer
    author: OneScience
    license: Apache-2.0
    tags:
      - fuxi
      - swin_transformer_v2
      - weather_forecasting
      - u_net_architecture
      - high_resolution


  concept:

    description: >
      FuXi 模型的核心 Transformer 处理模块。
      采用经典的 "下采样 → 深层注意力 → 上采样" U 形架构。
      在降低分辨率后的特征图上应用深层 Swin Transformer V2 进行特征提取，
      随后通过跳跃连接（Skip Connection）与下采样前的特征拼接，并进行上采样以恢复原始分辨率。

    intuition: >
      直接在 180x360 甚至更高分辨率的气象图上进行数十层 Transformer 计算，计算量是难以承受的。
      FuXi 模块就像一个“先概括后补充”的处理站：先通过下采样把高斯网格缩小，
      在这个浓缩的网格上用 Swin V2 仔细推演全局的气象演变规律；
      得出推演结论后，再通过跳跃连接结合原始的高清网格细节进行上采样还原。这既保证了极大的空间感受野，又兼顾了显存和计算效率。

    problem_it_solves:
      - 高分辨率全球气象数据在深层 Transformer 中的 O(N^2) 计算复杂度与显存耗尽问题
      - 深层网络推演中高频空间物理细节的丢失（通过跳跃连接弥补）
      - Swin Transformer 窗口分割与地球经纬度网格无法完美整除的边界冲突


  theory:

    formula:

      u_shape_forward:
        expression: |
          x_{down} = DownSample(x)
          x_{pad} = ZeroPad2d(x_{down})
          x_{attn} = SwinTransformerV2(x_{pad})
          x_{crop} = Crop(x_{attn})
          x_{out} = UpSample(Concat(x_{down}, x_{crop}))

    variables:

      x:
        name: InputTensor
        shape: [batch, embed_dim, lat, lon]
        description: 原始的高分辨率气象特征图

      x_{down}:
        name: DownSampledTensor
        shape: [batch, embed_dim, lat // 2, lon // 2]
        description: 空间分辨率降低后的特征图（通常宽高减半，通道数取决于 DownSample 的具体实现）


  structure:

    architecture: swin_unet_hybrid

    pipeline:

      - name: SpatialDownSample
        operation: one_sample (style="FuxiDownSample")

      - name: BoundaryPadding
        operation: zero_pad_2d

      - name: DeepAttentionBlock
        operation: swin_transformer_v2_stage

      - name: BoundaryCrop
        operation: crop_2d

      - name: SkipConnection
        operation: torch.cat (拼接下采样特征与注意力特征)

      - name: SpatialUpSample
        operation: one_sample (style="FuxiUpSample")


  interface:

    parameters:

      embed_dim:
        type: int
        default: 1536
        description: 输入特征图的通道维度

      num_groups:
        type: int | tuple[int, int]
        default: 32
        description: GroupNorm 的分组数量，用于上下采样过程中的归一化处理

      input_resolution:
        type: tuple[int, int]
        default: [90, 180]
        description: 下采样后的特征图空间分辨率 (lat, lon)

      num_heads:
        type: int
        default: 8
        description: Swin Transformer V2 的多头注意力头数

      window_size:
        type: int | tuple[int, int]
        default: 7
        description: 局部自注意力窗口的大小

      depth:
        type: int
        default: 48
        description: SwinTransformerV2Stage 堆叠的 Block 层数（决定网络深度）

    inputs:

      x:
        type: WeatherFeatureMap
        shape: [batch, embed_dim, lat, lon]
        dtype: float32

    outputs:

      output:
        type: WeatherFeatureMap
        shape: [batch, embed_dim, lat, lon]
        description: 融合了多尺度全局演变特征并恢复至原分辨率的输出


  types:

    WeatherFeatureMap:
      shape: [batch, embed_dim, lat, lon]
      description: 包含多通道气象变量的 2D 空间特征张量


  implementation:

    framework: pytorch

    code: |

      import torch
      from torch import nn
      from timm.layers.helpers import to_2tuple
      from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
      from onescience.modules.func_utils.fuxi_utils import get_pad2d
      from onescience.modules.sample.onesample import OneSample

      class FuxiTransformer(nn.Module):
          """FuXi 模型的核心 Transformer 处理模块"""
          def __init__(self, embed_dim=1536, num_groups=32, input_resolution=(90, 180),
                       num_heads=8, window_size=7, depth=48):
              super().__init__()
              
              num_groups = to_2tuple(num_groups)
              window_size = to_2tuple(window_size)
              padding = get_pad2d(input_resolution, window_size)
              padding_left, padding_right, padding_top, padding_bottom = padding
              self.padding = padding
              self.pad = nn.ZeroPad2d(padding)
              
              input_resolution = list(input_resolution)
              input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
              input_resolution[1] = input_resolution[1] + padding_left + padding_right
              
              self.down = OneSample(style="FuxiDownSample", in_chans=embed_dim, out_chans=embed_dim, num_groups=num_groups[0])
              self.layer = SwinTransformerV2Stage(embed_dim, embed_dim, input_resolution, depth, num_heads, window_size)
              self.up = OneSample(style="FuxiUpSample", in_chans=embed_dim*2, out_chans=embed_dim, num_groups=num_groups[0])

          def forward(self, x):
              B, C, Lat, Lon = x.shape
              padding_left, padding_right, padding_top, padding_bottom = self.padding
              
              x = self.down(x)
              shortcut = x

              x = self.pad(x)
              _, _, pad_lat, pad_lon = x.shape

              x = x.permute(0, 2, 3, 1)  # B Lat Lon C
              x = self.layer(x)
              x = x.permute(0, 3, 1, 2)

              x = x[:, :, padding_top: pad_lat - padding_bottom, padding_left: pad_lon - padding_right]

              x = torch.cat([shortcut, x], dim=1)  # B 2*C Lat Lon
              x = self.up(x)
              return x


  skills:

    build_fuxi_transformer:

      description: 构建一个完整的 FuXi 气象预报骨干模块

      inputs:
        - embed_dim
        - input_resolution
        - depth

      prompt_template: |

        实例化 FuxiTransformer。
        注意：传入的 input_resolution 必须是原图经过 FuxiDownSample 之后的尺寸（通常是原分辨率长宽各一半）。
        如果要完全复现论文中的深层主干网络，请将 depth 设为 48。


    diagnose_fuxi_pipeline:

      description: 排查 U-Net 架构在拼贴（Concat）时的形状不匹配问题

      checks:
        - skip_connection_shape_mismatch (Crop 操作后，空间维度未能对齐 shortcut 的维度)
        - downsample_upsample_asymmetry (上下采样的通道倍率计算错误，导致 Cat 后维度不是 embed_dim*2)



  knowledge:

    usage_patterns:

      fuxi_weather_predictor:

        pipeline:
          - CubeEmbedding (时空初步降维编码)
          - FuxiTransformer (深层时空推演与感受野扩大)
          - FC / Linear Head (输出预测时刻的天气图)


    hot_models:

      - model: FuXi (伏羲)
        year: 2023
        role: 15天全球天气预报 AI 大模型，在长时效预报上显著媲美 ECMWF 集合预报
        architecture: Cascade Swin Transformer (U-shaped)

      - model: Swin-Unet
        year: 2021
        role: 类似架构的纯 Transformer 风格 U-Net，多用于医学图像分割


    best_practices:

      - 在使用 FuXi 结构时，SwinV2 阶段的通道数（`embed_dim`）和深度（`depth`）通常设定得非常大（例如 1536 通道，48 层），这需要极高的显存支持。训练时强烈建议配合 Gradient Checkpointing（梯度检查点）技术使用。
      - `input_resolution` 的默认值 (90, 180) 对应的是标准 ERA5 数据 1度分辨率 (180, 360) 下采样后的尺寸。如果你需要处理 0.25 度的超高分辨率数据（721x1440），需要将分辨率参数同步扩大。


    anti_patterns:

      - 去除边界的 `get_pad2d` 和 Crop 修正操作。高斯网格通常包含奇数分辨率（如 Lat=181 等），直接丢进严格要求倍数对齐的 SwinTransformerV2 会导致运行崩溃。


    paper_references:

      - title: "FuXi: A cascade machine learning forecasting system for 15-day global weather forecast"
        authors: Chen et al.
        year: 2023



  graph:

    is_a:
      - TransformerBlock
      - UNetArchitecture

    part_of:
      - FuXiModel
      - AIWeatherModel

    depends_on:
      - SwinTransformerV2Stage
      - OneSample
      - ZeroPad2d

    variants:
      - SwinUNet

    used_in_models:
      - FuXi

    compatible_with:

      inputs:
        - WeatherFeatureMap

      outputs:
        - WeatherFeatureMap