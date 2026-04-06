component:

  meta:
    name: PanguUpSample3D
    alias: PanguPixelShuffle3D
    version: 1.0
    domain: physics_informed_machine_learning
    category: neural_network
    subcategory: upsampling
    author: OneScience
    tags:
      - transformer
      - pixel_shuffle
      - pangu_weather
      - 3d_token_upsample

  concept:

    description: >
      PanguUpSample3D 为盘古大模型的解码网络提供立体（3D）的亚像素逆映射架构。
      作为 PanguDownSample3D 的确切逆反流，它只针对地球的表面展开拓扑平铺操作（即将宏观大气隐含特征扩张至精细 2x 的 Lat 与 Lon 分辨率上）， 
      而对于气压层（垂直厚度），由于并未发生过特征体积合并运算，模块只会沿着气压层维度进行对齐剪切（Slicing: `[:out_pl]`），并不涉及内层插值。

    intuition: >
      想象高空气象层信息像一个深层压缩的“千层糕”，其中总层数不变，但是每一层的表面积被平滑地延展放大。在上采样（解码）的过程中，
      依靠可解释的多维映射字典建立出精细化地理坐标格点的表现值，并将边缘多余（前期 DownSample 补进的 Pad）直接切除。

    problem_it_solves:
      - 防止三维物理张量在拉伸时丧失物理海拔守恒定律约束。
      - 取代不稳定的 3D 插值及伴随产生的伪高频网格振荡杂波噪音。
      - 依靠截断矩阵将维度还原至最原原本本的空间维度结构（比如从含有冗界的纬度截开，恢复其极点边界 181 像素的特定宽延）。

  theory:

    formula:

      channel_expansion:
        expression: |
          X_{\text{expand}} = W_{linear} \cdot X_{\text{in}} \text{ (shape: } [B, N, \text{out\_dim} \times 4])
      
      spatial_3d_inflation:
        expression: |
          X_{3D} = X_{\text{expand}} \to \text{Reshape}([B, pl, Lat, Lon, 2, 2, \text{out\_dim}])
          X_{grid} = \text{Permute}(\dots) \to [B, pl, Lat \times 2, Lon \times 2, \text{out\_dim}]

      boundary_stripping:
        expression: |
          Y_{cropped} = X_{grid}[: , \ :out\_pl, \ \text{Pad}_{top} \dots -\text{Pad}_{bot}, \ \text{Pad}_{left} \dots ]

    variables:
      pl:
        name: StaticPressureLevel
        description: 伴随系统运作但无需重整的独立气压海拔维

  structure:

    architecture: horizontal_pixel_shuffle

    pipeline:
      - name: DimensionalExpansion
        operation: Linear(in_dim, out_dim * 4)
      - name: SpaceUnpacking
        operation: Reshape and Permute down to pseudo-3D blocks.
      - name: DimensionalCropping
        operation: Remove redundant Pad buffers and excess Z-axis channels
      - name: Flattening
        operation: Reshape all 3D fields to single contiguous Token array

  interface:

    parameters:

      in_dim:
        type: int
        default: 384
        description: 通过跨级拼凑(Skip-connection)累积而来的宽体流维总计。

      out_dim:
        type: int
        default: 192
        description: 解缩拆解后的标准空间象限节点容量。

      input_resolution:
        type: tuple[int, int, int]
        description: 在压缩隐层空间内的当前模型分层 (pl, lat, lon)。

      output_resolution:
        type: tuple[int, int, int]
        description: 希望扩展还原至的目标天气环境物理场规模。

    inputs:
      x:
        type: Tensor
        shape: [B, pl \times Lat \times Lon, C_{in}]
        description: 未扩展维度的平面长 Token。

    outputs:
      result:
        type: Tensor
        shape: [B, out\_pl \times out\_lat \times out\_lon, out\_dim]
        description: 大规模铺展覆盖面积但维持海拔气流平层的重组序列。

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn
      
      class PanguUpSample3D(nn.Module):
          """
              Pangu-Weather 风格的 3D 大气变量上采样模块。
      
              PanguDownSample3D 的逆操作，同时也是 PanguUpSample2D 的三维扩展版本。
              仅对水平方向（纬度、经度）做 2x 上采样，气压层维度通过切片直接对齐
              output_resolution 的 out_pl，不参与上采样计算。
      
              Args:
                  in_dim (int, optional): 输入 token 的通道数，默认为 192 * 2 = 384。
                  out_dim (int, optional): 输出 token 的通道数，线性层先扩展至 out_dim * 4，
                      拆分后每个子像素通道数恢复为 out_dim，默认为 192。
                  input_resolution (tuple[int, int, int]): 输入特征图的空间分辨率 (pl, lat, lon)。
                  output_resolution (tuple[int, int, int]): 目标输出分辨率 (out_pl, out_lat, out_lon)，
                      水平方向应满足 out_lat ≤ in_lat * 2 且 out_lon ≤ in_lon * 2，
                      超出部分通过中心裁剪去除；气压层直接取前 out_pl 层。
      
              形状:
                  - 输入 x: (B, pl * lat * lon, C)，其中 C = in_dim
                  - 输出:   (B, out_pl * out_lat * out_lon, out_dim)
      
              Examples:
                  >>> # 典型 Pangu-Weather 大气变量配置
                  >>> # 气压层保持不变: pl=8
                  >>> # 水平分辨率 91×180 → 181×360（对应 PanguDownSample3D 的逆操作）
                  >>> # in_lat * 2 = 91 * 2 = 182，裁剪掉多余的1行: pad_h = 182 - 181 = 1
                  >>> # in_lon * 2 = 180 * 2 = 360，无需裁剪: pad_w = 0
                  >>> # 输入 token 数: 8 *  91 * 180 = 131040
                  >>> # 输出 token 数: 8 * 181 * 360 = 521280
                  >>> upsample = PanguUpSample3D(
                  ...     in_dim=384,
                  ...     out_dim=192,
                  ...     input_resolution=(8, 91, 180),
                  ...     output_resolution=(8, 181, 360),
                  ... )
                  >>> x = torch.randn(2, 131040, 384)  # (B, pl*lat*lon, C)
                  >>> out = upsample(x)
                  >>> out.shape
                  torch.Size([2, 521280, 192])
      
                  >>> # 整除情况下无需裁剪（如 pl=13, 64×128 → 128×256）
                  >>> upsample2 = PanguUpSample3D(
                  ...     in_dim=384,
                  ...     out_dim=192,
                  ...     input_resolution=(13, 64, 128),
                  ...     output_resolution=(13, 128, 256),
                  ... )
                  >>> x2 = torch.randn(2, 106496, 384)  # (B, 13*64*128, C)
                  >>> out2 = upsample2(x2)
                  >>> out2.shape
                  torch.Size([2, 425984, 192])
          """
          def __init__(self, 
                       in_dim=192*2, 
                       out_dim=192,
                       input_resolution=None, 
                       output_resolution=None,
                       ):
              super().__init__()
              self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
              self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
              self.norm = nn.LayerNorm(out_dim)
              self.input_resolution = input_resolution
              self.output_resolution = output_resolution
      
          def forward(self, x: torch.Tensor):
              """
              Args:
                  x (torch.Tensor): (B, N, C)
              """
              B, N, C = x.shape
              in_pl, in_lat, in_lon = self.input_resolution
              out_pl, out_lat, out_lon = self.output_resolution
      
              x = self.linear1(x)
              x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(
                  0, 1, 2, 4, 3, 5, 6
              )
              x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)
      
              pad_h = in_lat * 2 - out_lat
              pad_w = in_lon * 2 - out_lon
      
              pad_top = pad_h // 2
              pad_bottom = pad_h - pad_top
      
              pad_left = pad_w // 2
              pad_right = pad_w - pad_left
      
              x = x[
                  :,
                  :out_pl,
                  pad_top : 2 * in_lat - pad_bottom,
                  pad_left : 2 * in_lon - pad_right,
                  :,
              ]
              x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
              x = self.norm(x)
              x = self.linear2(x)
              return x

  skills:

    build_panguupsample3d:

      description: 构建支持物理约束的三维/二维降维或插值组件，整合上下文环境与特征通道

      inputs:
        - input_resolution
        - output_resolution
        - in_dim
        - out_dim

      prompt_template: |

        构建名为 PanguUpSample3D 的气象特征采样模块。

        参数：
        输入分辨率 = {{{input_resolution}}}
        输出分辨率 = {{{output_resolution}}}
        特征维度 = {{{in_dim}}}
        预期输出维度 = {{{out_dim}}}

        要求：
        必须严格遵守该网络结构的物理特征（例如三维气压层分离、拓扑点云近邻不损失等）恒定映射规律。


    diagnose_tensor_shape:

      description: 调试在不同尺度缩放或维数跳转（如 .permute 或 .reshape 等多级重组）时出现的维度不匹配错误

      checks:
        - shape_mismatch_at_boundaries
        - incorrect_permute_strides
        - loss_of_physical_meaning (例如跨气压层或跨时间维的污染混合)


  knowledge:

    usage_patterns:

      spatial_scaling_framework:
        description: 控制多尺度空间特征提取的标准管线
        pipeline:
          - Extract_Macro_Features (利用DownSample提取低频气候抽象)
          - Message_Passing (中心GNN或Transformer处理)
          - Interpolate_Back (利用UpSample恢复至高频物理网格)
          
      multiscale_processor:
        pipeline:
          - Encoder_Block
          - DownSample_Layer
          - Decoder_Block
          - UpSample_Layer


    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 提供基于地球表面的 3D Patch Merging 的空间降维/升维范式
        architecture: 3D Earth-Specific Transformer

      - model: GraphCast
        year: 2023
        role: 提供基于 Mesh 的不规则点云抽象以及边重构拓扑网络
        architecture: Hierarchical Graph Neural Network

      - model: FuXi
        year: 2023
        role: 提供基于经典可变形卷积和群归一化的级联采样结构
        architecture: Cascaded Swin-Transformer + CNN 


    best_practices:

      - 必须使用切片保留操作 [ :out_pl ] 对三维层截取，防范多余缓存残留。
      - 必须保证上采样输出与同层级下采样输出维度与分辨率完全匹配以备特征融合。
      - 所有的池化步幅和卷积核需要匹配地球环面展开特性（极点、赤道），处理由于180°与360°分辨率的不整除造成的尺寸零头。


    anti_patterns:

      - 三维全方向扩大逻辑引发灾难性的海拔层错配污染。
      - 不考虑任何物理约束的简单 AdaptiveAvgPool2d 会让复杂地理特征严重失真。
      - 忽略对多层级气压、高度、地表层的数据异构型差异盲目共用池化算子。


    paper_references:

      - title: "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast"
        authors: Bi et al.
        year: 2023

      - title: "GraphCast: Learning skillful medium-range global weather forecasting"
        authors: Lam et al.
        year: 2023

      - title: "FuXi: A cascade machine learning forecasting system for 15-day global weather forecast"
        authors: Chen et al.
        year: 2023

  graph:
    is_a:
      - TokenExpander
      - UpsampleLayer
      - 3DKernel
    part_of:
      - PanguWeatherModel
      - 3DVisionTransformerDecoder
    depends_on:
      - nn.Linear
      - nn.LayerNorm
    compatible_with:
      - PanguDownSample3D