component:

  meta:
    name: XiheEmbedding
    alias: Xihe2DPatchEmbedding
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
      - xihe_model

  concept:

    description: >
      XiheEmbedding 是为“羲和”等可能达到极高分辨率的气象大模型设计的 2D Patch 嵌入模块。
      该结构与 Pangu-Weather 的 `PanguEmbedding2D` 极为相似，同样使用具有长宽独立缩放的填充处理与带有对应 stride 步长的 `Conv2d` 。
      通过灵活且稳健的边缘自动 Padding，它可以无损兼容任意像 `(2041, 4320)` 这样的非常规极端大地形气候网格。

    intuition: >
      如果说常规气象模型就像是用常见的 1080P （如 720×1440）屏幕看地球，“羲和”则是采用了 4K 乃至更高分化和纵横比来扫描。
      XiheEmbedding 用一个不对称的矩形透镜（比如 `6x12` 的 Patch_size），先把世界地图严丝合缝地框入边缘修正结构内，再用卷集把它压缩成具备丰富高维通道特征的一叠“地理马赛克”，以便在算力合理的范畴中运行接下来的超级模型结构。

    problem_it_solves:
      - 处理超过两千级别尺寸分辨率的网格切割
      - 支持在特征空间上的非正方形打块池化（如 6高 x 12宽 的矩形感受野）
      - 将大通道数（如包含极多派生气象特征如 96个通导）直接下沉为模型运算隐维表示


  theory:

    formula:

      padded_conv2d_rectangular:
        expression: |
          x_{pad} = \text{ZeroPad2d}_{(left, right, top, bottom)}(x_{in})
          x_{patch} = \text{Conv2D}(x_{pad}, \text{kernel\_size}=(P_h, P_w), \text{stride}=(P_h, P_w))

    variables:

      (P_h, P_w):
        name: AsymmetricPatchSize
        shape: [2]
        description: 在不同气象分辨率上适配的高和宽的不同降采样因数

      x_{in}:
        name: HighResSurfaceField
        shape: [B, 96, 2041, 4320]
        description: 超多通道、高规格地球物理变量网格特征


  structure:

    architecture: asymmetric_padded_patch_embedding

    pipeline:

      - name: DimensionalRemainderEvaluation
        operation: height_width_modulu

      - name: AsymmetricPadding
        operation: zero_pad_edges

      - name: LinearSpatialMapping
        operation: conv2d_downsampling


  interface:

    parameters:

      img_size:
        type: tuple[int, int]
        description: 超高分辨预设参数，常置为 (2041, 4320)

      patch_size:
        type: tuple[int, int]
        description: 进行 Conv 特征融合的打块尺寸范围（如 (6, 12)），以贴合高精经纬向投影变形

      embed_dim:
        type: int
        description: 网络后续通道特征表示数量尺寸，如 192

      in_chans:
        type: int
        description: 数据端预先拼合成的通道容量集（如包含 96 种特征维度）

      norm_layer:
        type: nn.Module
        description: 通道维度的规范化配置结构

    inputs:

      x:
        type: Tensor
        shape: [B, C, H, W]
        dtype: float32
        description: 二维排列的气像物理变量特征矩阵簇

    outputs:

      x_out:
        type: Tensor
        shape: [B, embed_dim, ceil(H/P_h), ceil(W/P_w)]
        description: 下采样之后符合特征块维度的粗结构网格特征矩阵


  types:

    HighResObservationMatrix:
      shape: [B, 96, 2041, 4320]
      description: 富含变量的原始高分辨率全球态气象资料框架


  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn
      
      class XiheEmbedding(nn.Module):
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
              
          def __init__(self, img_size=(2041, 4320),
                          patch_size=(6, 12),
                          embed_dim=192,
                          in_chans =96,
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

    build_xihe_embedding:

      description: 构建兼容非对称方块且高特征输入维度的 Patch Embed 层

      inputs:
        - in_chans
        - patch_size

      prompt_template: |

        构建 2D Conv Patch Embedding 用于气象网格，该网格的 patch_size 在两个维度上可以是非均等的。提供动态 Pad 计算防范被裁剪的丢失。




    diagnose_xiheembedding:

      description: 分析 XiheEmbedding 模块在使用中的常见失效模式

      checks:
        - padding_or_resolution_errors
        - bad_device_alignment
        - output_shape_collapse
  knowledge:

    usage_patterns:

      hi_res_weather_model:

        pipeline:
          - Provide Full C variables (up to 96)
          - Use XiheEmbedding (Drop Resolution with Asymmetric Window)
          - Run Through Multi-stage Transformers

    hot_models:

      - model: Xihe Weather Model
        role: 推动更高分辨率全球或区域化大模型系统级预测能力
        architecture: High-Resolution Extensible Backbone

    best_practices:

      - 当经度宽度（此处为12）远大于纬度块区高度（6）时，实际上暗示了在不同纬圈经线收缩或本身物理映射的比例非等距的地理特性调整。
      - `embed_dim` （隐藏维度）的设定需要兼顾内存和模型表达上限（对于 4320 分辨率其即使通过切片依然有庞大基数）。


    anti_patterns:

      - 如果网格本身并不需要包含边缘保护层（例如已经完美对齐的分辨率数据集），运行无端 `ZeroPad` 会引发冗余的空白张量构建动作引起系统显存峰值抖动。因此这里动态 `%` 判断不可或缺。

    paper_references:

      - title: "Xihe: A Data-Driven High-Resolution Global Weather Forecasting Model"
        authors: Xihe Research Team (Inferred)
        year: 2023

  graph:

    is_a:
      - PatchEmbedding
      - DimensionalityReduction

    part_of:
      - XiheModel

    depends_on:
      - Conv2d
      - ZeroPad2d

    variants:
      - PanguEmbedding2D

    used_in_models:
      - Xihe

    compatible_with:

      inputs:
        - LargeScaleGrid

      outputs:
        - LocalRegionalPatches