component:

  meta:
    name: FuxiEmbedding
    alias: Fuxi3DPatchEmbedding
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: embedding
    author: OneScience
    license: Apache-2.0
    tags:
      - embedding
      - 3d_patch_embedding
      - vision_transformer
      - earth_system_modeling
      - fuxi

  concept:

    description: >
      FuxiEmbedding 是复旦大学研发的 FuXi 全球大模型三维 Patch Embedding 组件。
      区别于单步或简单的单层预报模块，该模块能够处理包含了时序 (T)、高度 (lat)、宽度 (lon) 三维数据的特征压缩。
      它借助 3D 卷积核操作，将多帧的历史数据连同所有气象变量通道从空间与时间维度协同打成多维切块，极大保留了时空综合关联特征并提供了模型深度降维表示。

    intuition: >
      在做高精度长时序气象预报时（如需要当前和过去 6 小时帧），时间关系至关重要。
      FuxiEmbedding 使用的 `Conv3d` 相当于是一把“时空格栅裁切刀”，它不仅在表面上把全球切片，而是像切果冻一样把两层或多层空间叠成的“时间果冻块”同步切分压缩，形成具有时空意义的隐通道向量块。

    problem_it_solves:
      - 融合处理历史不同时刻输入的气象物理场特征
      - 将庞大的 `(T, Lat, Lon)` 3D 时空立方体转换为低计算复杂度的粗粒度隐特征块
      - 为 3D-Swin Transformer 或时空自注意力机制提供符合形状要求的入站数据流


  theory:

    formula:

      patch_3d_projection:
        expression: |
          x_{patch} = \text{Conv3D}(x_{in}, \text{kernel\_size}=(P_t, P_{lat}, P_{lon}), \text{stride}=(P_t, P_{lat}, P_{lon}))
          x_{norm} = \text{LayerNorm}(x_{patch}\text{._flatten()\_on\_hw()})

    variables:

      x_{in}:
        name: SpatioTemporalField
        shape: [B, C, T, lat, lon]
        description: 具有 C 个特征通道在时间轴上包含 T 个预报状态的高空张量

      (P_t, P_{lat}, P_{lon}):
        name: PatchSize
        description: 对应 Conv3D 在时序和空间经纬方向划分感受野维度的裁剪块大小


  structure:

    architecture: spatio_temporal_patching

    pipeline:

      - name: VolumetricPatching
        operation: Conv3d_projection_and_striding

      - name: ShapeFormatting(Optional)
        operation: flatten_for_LayerNorm

      - name: Normalization
        operation: layer_norm_if_defined

      - name: OutputFormatting
        operation: unflatten_to_3D_grid


  interface:

    parameters:

      img_size:
        type: tuple[int, int, int]
        description: 预期的单条输入形状 (T, Lat, Lon)，默认配置为两帧 0.25度分辨率 (2, 721, 1440)

      patch_size:
        type: tuple[int, int, int]
        description:  三维采样感受野，如默认为 (2, 4, 4) 代表融合 2 个时步与 4x4 的平面像素

      in_chans:
        type: int
        description: 一次传入预测系统气象变量总量，FuXi常见配置为 70

      embed_dim:
        type: int
        description: Patch隐空间总表示通道数量大小，如 1536

      norm_layer:
        type: nn.Module
        description: 用以在投射完成后稳定分布的方差归一化模块

    inputs:

      x:
        type: Tensor
        shape: [B, C, T, Lat, Lon]
        dtype: float32
        description: 经过对齐后的三维时空气象网格数据

    outputs:

      x_out:
        type: Tensor
        shape: [B, embed_dim, nT, nLat, nLon]
        description: 输出的三维时序缩放立方体数据 (例如：[B, 1536, 1, 180, 360])


  types:

    SpatioTemporalGrid:
      shape: [B, C, T, Lat, Lon]
      description: 包含了时序演变的3D物理仿真结果特征

    EmbeddedGrid:
      shape: [B, C', T', L', W']
      description: 隐式的降维三维网络映射张量


  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn
      from torch.nn import functional as F
      from timm.layers.helpers import to_2tuple
      from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
      
      from typing import Sequence
      
      from onescience.modules.func_utils.fuxi_utils import get_pad2d
      
      
      class FuxiEmbedding(nn.Module):
          """
              FuXi 模型的三维 Patch Embedding 模块。
              
              使用 3D 卷积将 (时间步, 纬度, 经度) 三维气象场划分为不重叠的 Patch 并投影到
              嵌入空间，是 FuXi 模型编码器的入口层。与 Pangu-Weather 将气压层和地表变量
              分开处理不同，FuXi 将多帧气象场沿时间轴堆叠后统一做三维 Patch 划分。
              
              Args:
                  img_size (tuple[int, int, int], optional): 输入数据的空间尺寸 (T, lat, lon)，
                      其中 T 为时间步数（通常为 2，对应当前时刻与前一时刻），默认为 (2, 721, 1440)。
                  patch_size (tuple[int, int, int], optional): 3D Patch 大小 (Pt, Plat, Plon)，
                      默认为 (2, 4, 4)，时间维度通常设为与 T 相同以合并时间步。
                  in_chans (int, optional): 输入气象变量通道数，默认为 70。
                  embed_dim (int, optional): Patch 嵌入维度，默认为 1536。
                  norm_layer (nn.Module 或 None, optional): 嵌入后的归一化层类型，
                      为 None 时跳过归一化，默认为 nn.LayerNorm。
                  **kwargs: 额外参数（忽略，兼容统一接口）。
              
              形状:
                  - 输入 x:  (B, C, T, lat, lon)
                      其中 C = in_chans，T = img_size[0]
                  - 输出:    (B, embed_dim, T//Pt, lat//Plat, lon//Plon)
                      即 (B, embed_dim, nT, nLat, nLon)
              
              Examples:
                  >>> # 典型 FuXi 配置：2帧输入，70个气象变量
                  >>> # patch_size=(2,4,4)，时间维度完全合并
                  >>> # nT   = 2 // 2 = 1
                  >>> # nLat = 721 // 4 = 180
                  >>> # nLon = 1440 // 4 = 360
                  >>> embedding = FuxiEmbedding(
                  ...     img_size=(2, 721, 1440),
                  ...     patch_size=(2, 4, 4),
                  ...     in_chans=70,
                  ...     embed_dim=1536,
                  ... )
                  >>> x = torch.randn(2, 70, 2, 721, 1440)  # (B, C, T, lat, lon)
                  >>> out = embedding(x)
                  >>> out.shape
                  torch.Size([2, 1536, 1, 180, 360])
          """
          def __init__(self, 
                       img_size=(2, 721, 1440), 
                       patch_size=(2, 4, 4), 
                       in_chans=70, 
                       embed_dim=1536, 
                       norm_layer=nn.LayerNorm, **kwargs):
              super().__init__()
              patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
      
              self.img_size = img_size
              self.patches_resolution = patches_resolution
              self.embed_dim = embed_dim
              self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
              if norm_layer is not None:
                  self.norm = norm_layer(embed_dim)
              else:
                  self.norm = None
      
          def forward(self, x: torch.Tensor):
              B, C, T, Lat, Lon = x.shape
              assert T == self.img_size[0] and Lat == self.img_size[1] and Lon == self.img_size[2], \
                  f"Input image size ({T}*{Lat}*{Lon}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
              x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
              if self.norm is not None:
                  x = self.norm(x)
              x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
              return x
      
      

  skills:

    build_fuxi_embedding:

      description: 构建兼顾时域聚合及空间切片的气象物理系统初始降维处理模块

      inputs:
        - img_size
        - patch_size
        - in_chans
        - norm_layer

      prompt_template: |

        请提供具备时空聚合特性的 Fuxi 3D Patch embedding。
        运用 Conv3d 以及一个后接的展平 LayerNorm 归一化步骤，以改善随后的 3D SWIN 注意力传播的协方差表现，最后确保输出维度仍保持空间3D布局。




    diagnose_fuxiembedding:

      description: 分析 FuxiEmbedding 模块在使用中的常见失效模式

      checks:
        - padding_or_resolution_errors
        - bad_device_alignment
        - output_shape_collapse
  knowledge:

    usage_patterns:

      video_or_spatials_model_inbound:

        pipeline:
          - Concatenate T time slices
          - FuxiEmbedding(Conv3D + Norm)
          - SwinTransformerV2_3D

    hot_models:

      - model: FuXi
        year: 2023
        role: 大幅度扩展多参数中长期预测稳定度
        architecture: 3D-SwinTransformer Based

    best_practices:

      - `patch_size` 的时间刻度(T)设为等同于 `img_size` 中的时步 T 可以进行所谓的 "Temporal Token Merging"，有效降低之后 Transformer 网络计算量。
      - 需要在 `LayerNorm` 等模块前通过 transpose/reshape 实现展平至 2D 以兼容 pytorch 层，完成后再映射回原形。


    anti_patterns:

      - 没有对切出的隐空间结果做 Normalization 便输入深的类 Transformer 结构，这在极端长尾天气的梯度回传中易引致溢出（使用 `norm_layer` 防范这种风险）。


    paper_references:

      - title: "FuXi: a cascade machine learning forecasting system for 15-day global weather forecast"
        authors: Chen et al.
        year: 2023


  graph:

    is_a:
      - PatchEmbedding
      - SpatioTemporalEncoding

    part_of:
      - FuXi
      - MultimodalVisionModels

    depends_on:
      - Conv3d
      - LayerNorm

    variants:
      - VideoPatchEmbed
      - TubletEmbedding

    used_in_models:
      - FuXi
      - VideoSwinTransformer

    compatible_with:

      inputs:
        - SpatioTemporalGrid

      outputs:
        - EmbeddedGrid