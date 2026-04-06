component:

  meta:
    name: PanguEmbedding2D
    alias: Pangu2DPatchEmbedding
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: embedding
    author: OneScience
    license: Apache-2.0
    tags:
      - embedding
      - feature_extraction
      - patch_embedding
      - earth_system_modeling
      - pangu_weather

  concept:

    description: >
      PanguEmbedding2D 是华为盘古气象 (Pangu-Weather) 大模型对于地表信息 (Surface variables) 进行编码处理的 2D 嵌入层。
      它使用了支持自动填充 (ZeroPadding) 和下采样的 `Conv2D` 操作，专门用来将高分辨率的全球地表二维气象特征空间降维并打包为基于 Patch 的维度，作为类似 Swin Transformer 这类注意力模块的独立输入。

    intuition: >
      气象经纬网格有时难以被给定的分块大小整除（例如 721 不除以 4）。传统的 Embedding 层这可能直接报错或丢弃边缘像素。
      因此，PanguEmbedding2D 像是一个非常严谨的“相框师傅”，如果图片尺寸稍微大一点或者小一点，它会自动给四周补上黑边（Zero Padding），使得原始气象空间信号完美适配随后块操作(patching)的方格要求。

    problem_it_solves:
      - 解决气象场中含有奇数或不规则分辨率（如 721纬度, 即经纬零度等特殊采样分辨率）带来的网状划块难题
      - 独立分离地表气象数据输入进行降维特征提取
      - 为下游自注意力结构提供带有平移稳定性的局域聚合特征


  theory:

    formula:

      padded_conv2d_projection:
        expression: |
          x_{pad} = \text{ZeroPad2d}(x_{in})
          x_{patch} = \text{Conv2D}(x_{pad}, \text{kernel\_size}=P, \text{stride}=P)
          x_{out} = \text{LayerNorm}_{[embed\_dim]}(x_{patch})

    variables:

      x_{in}:
        name: SurfaceMeteorologicalField
        shape: [B, C, H, W]
        description: 二维的地表气象数据（包括地表压、温等组合输入）

      P:
        name: PatchSize
        shape: [2]
        description: (patch_h, patch_w)，通常选用如 (4, 4)


  structure:

    architecture: padded_patch_embedding

    pipeline:

      - name: BoundaryPadding
        operation: calculate_remainder_and_zero_pad

      - name: PatchProjection
        operation: non_overlapping_conv2d

      - name: DimensionalNormalization
        operation: permute_layernorm_and_restore


  interface:

    parameters:

      img_size:
        type: tuple[int, int]
        description: 预期气象原图分辨率。默认常用盘古气象分辨率例如 (721, 1440)

      patch_size:
        type: tuple[int, int]
        description: Patch 块截面尺寸，推荐 (4, 4)

      embed_dim:
        type: int
        description: 模型映射空间的通道数目，通常对应模型宽度

      in_chans:
        type: int
        description: 对于地表嵌入模块常见的特定通道数(例如 4+3 对应盘古特定输入总数)

      norm_layer:
        type: nn.Module
        description: 提供于映射通道数（embed_dim）上的归一化层

    inputs:

      x:
        type: Tensor
        shape: [B, C, H, W]
        dtype: float32
        description: 气象源域特征网格

    outputs:

      x_out:
        type: Tensor
        shape: [B, embed_dim, H', W']
        description: 注意：此时 H', W' 均为已除以 patch_size 并向上取整后的隐维度长宽


  types:

    SurfaceVariablesGrid:
      shape: [B, C, H, W]
      description: 大气模型中的地表二维变量特征层

    PaddedFeatureGrid:
      shape: [B, embed_dim, ceil(H/P_h), ceil(W/P_w)]
      description: 保留二维排列特征但实现深层特征空间的块网络张量


  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn
      
      class PanguEmbedding2D(nn.Module):
          """
              将二维图像分割为不重叠的 patch 并嵌入到向量空间。
      
              Args:
                  img_size (tuple[int, int]): 输入图像尺寸 (H, W)
                  patch_size (tuple[int, int]): 每个 patch 的大小 (patch_h, patch_w)
                  in_chans (int): 输入图像通道数
                  embed_dim (int): 每个 patch 嵌入后的向量维度
                  norm_layer (nn.Module, optional): 归一化层，默认为 None。常用: nn.LayerNorm
      
              形状:
                  输入: (B, C, H, W)
                  输出: (B, embed_dim, H', W')，其中 H' = ⌈H / patch_h⌉, W' = ⌈W / patch_w⌉
      
              Example:
                  >>> patch_embed = PatchEmbed2D(
                  ...     img_size=(128, 256),
                  ...     patch_size=(4, 4),
                  ...     in_chans=3,
                  ...     embed_dim=96
                  ... )
                  >>> x = torch.randn(8, 3, 128, 256)
                  >>> out = patch_embed(x)
                  >>> out.shape
                  torch.Size([8, 96, 32, 64])
          """
              
          def __init__(self, img_size=(721, 1440),
                          patch_size=(4, 4),
                          embed_dim=192,
                          in_chans = 4+3,
                          norm_layer=None,
                          ):
              
              super().__init__()
              height, width = img_size
              h_patch_size, w_path_size = patch_size
              stride = patch_size
              padding_left = padding_right = padding_top = padding_bottom = 0
              h_remainder = height % h_patch_size
              w_remainder = width % w_path_size
      
              if h_remainder:
                  h_pad = h_patch_size - h_remainder
                  padding_top = h_pad // 2
                  padding_bottom = int(h_pad - padding_top)
      
              if w_remainder:
                  w_pad = w_path_size - w_remainder
                  padding_left = w_pad // 2
                  padding_right = int(w_pad - padding_left)
      
              self.pad = nn.ZeroPad2d(
                  (padding_left, padding_right, padding_top, padding_bottom)
              )
              self.proj = nn.Conv2d(
                  in_chans, embed_dim, kernel_size=patch_size, stride=stride
              )
              if norm_layer is not None:
                  self.norm = norm_layer(embed_dim)
              else:
                  self.norm = None
      
          def forward(self, x: torch.Tensor):
              B, C, H, W = x.shape
              x = self.pad(x)
              x = self.proj(x)
              if self.norm is not None:
                  x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
              return x

  skills:

    build_pangu_2d_embedding:

      description: 构建具有鲁棒边界填充支持的高分辨率地表气象 Patch Embedding 层

      inputs:
        - img_size
        - patch_size

      prompt_template: |

        请实现一个 2D 贴片位置嵌入器模块。
        通过计算输入图像长宽对于 patch_size 的余数，使用 ZeroPad2D 加上边缘，再运用 Conv2d 滑动采集映射。
        需保留最终结果输出时的空间 2D 特征对齐形状（不可像 ViT 把维度拉平）。




    diagnose_panguembedding2d:

      description: 分析 PanguEmbedding2D 模块在使用中的常见失效模式

      checks:
        - padding_or_resolution_errors
        - bad_device_alignment
        - output_shape_collapse
  knowledge:

    usage_patterns:

      earth_surface_backbone:

        pipeline:
          - Extract 2D Fields
          - PanguEmbedding2D (Pad & Conv)
          - SwinTransformer/3DEarthSwin

    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 提供 3D 模型处理之前将地表单独分立编码的模型设计方法
        architecture: Earth-Specific 3D Swin

    best_practices:

      - 由于气象域具有环球循环特性，经度（也就是宽方向）严格上讲应当利用 `CircularPadding` 较佳。这部分如果未来进一步优化，可以取代此处的 `ZeroPad2D`。
      - 当 `norm_layer` （如 LayerNorm）应用时，它是在通路上进行操作的，因此 Pytorch 需使用 permute 做暂时的维度搬家。


    anti_patterns:

      - 使用单纯的 `LayerNorm2d` 而没有对高频特征降采样引发的内存及显流过载问题。在此模块，`proj` 先执行大大缓解了后续张量占用的内存。

    paper_references:

      - title: "Accurate medium-range global weather forecasting with 3D neural networks"
        authors: Bi et al. (Nature)
        year: 2023


  graph:

    is_a:
      - PatchEmbedding
      - FeatureEncoder

    part_of:
      - PanguWeatherModel

    depends_on:
      - Conv2d
      - ZeroPad2d

    variants:
      - FuxiEmbedding
      - FourCastNetEmbedding

    used_in_models:
      - Pangu-Weather

    compatible_with:

      inputs:
        - SpatiallyIncompleteGrid

      outputs:
        - DownsampledFeatureMap