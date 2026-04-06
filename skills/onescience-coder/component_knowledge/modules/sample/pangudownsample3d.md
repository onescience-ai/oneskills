component:

  meta:
    name: PanguDownSample3D
    alias: PanguPatchMerging3D
    version: 1.0
    domain: physics_informed_machine_learning
    category: neural_network
    subcategory: downsampling
    author: OneScience
    tags:
      - transformer
      - patch_merging
      - pangu_weather
      - 3d_token_downsample

  concept:

    description: >
      PanguDownSample3D 是盘古大模型中用于处理 3D 大气状态变量（如不同高度层的温度、湿度、风速等）的空间下采样模块。
      与 2D 版本不同的是，它仅在水平方向（纬度和经度）执行 $2 \times 2$ 的 Patch Merging 压缩操作。而对于高度/气压层（Pressure Levels, `pl`）维度，
      由于大气垂直方向的物理层级通常较少且各层特性差异巨大（如对流层和同温层物理规律不同），算法选择不进行折叠压缩。
      这保留了完整的重力地转偏向力剖面结构。

    intuition: >
      气象模型在 3D 空间具有显著的各向异性：水平方向（经纬度）往往具有连续和平滑的过渡性（可将相邻 4 个节点压并到一个局部隐变量道中），
      但垂直方向受气压梯度力影响产生断层。故在网络深挖特征时，只沿着地球表面横向抽取粗粒度天气图景，而贯穿地心方向的热力学分层数量维持原样。

    problem_it_solves:
      - 将复杂的 3D 气流数据体积进行有效控制，解决海量网格点的内存上限问题。
      - 保持三维特征中垂直维度（Z轴/Pressure Level）的不变性与独立物理意义保留。
      - 三维矩阵处理时对非偶数水平边界域（如181纬度）的动态 ZeroPad 内存排布缝合机制。

  theory:

    formula:

      horizontal_patch_merging:
        expression: |
          X_{\text{reshaped}} = X_{in}[..., pl, i:i+2, j:j+2, C] \longrightarrow X_{\text{merged}} \text{ (shape: } [B, pl, \frac{Lat \times Lon}{4}, 4C])
      
      channel_contraction:
        expression: |
          Y = W_{linear} \cdot \text{LayerNorm}(X_{\text{merged}}) \text{ (shape: } [B, pl \times \frac{Lat \times Lon}{4}, 2C])

    variables:

      pl:
        name: PressureLevels
        description: 垂直空间的高度层或者气压层（例如 Pangu 常用的 13 层或者 8 层设定）。此维度恒定不缩减。

      C:
        name: TokenDimension
        description: 每个微观 3D 气柱向量的初始多维空间信道深度。

  structure:

    architecture: horizontal_token_merging
    
    pipeline:
      - name: Spatial3DReshape
        operation: "[B, N, C] -> [B, In_{pl}, In_{lat}, In_{lon}, C]"
      - name: ZeroPad3DAlignment
        operation: "Pad on Lat/Lon horizontally. Keep depth(pl) padding=0"
      - name: PatchGrid3DExtraction
        operation: ".reshape([B, In_{pl}, out_{lat}, 2, out_{lon}, 2, C])"
      - name: TokenContraction
        operation: ".permute(...) -> reshape to [B, In_{pl} * out_{lat} * out_{lon}, 4 * C]"
      - name: FeatureCompression
        operation: LayerNorm -> Linear(4*C, 2*C)

  interface:

    parameters:

      input_resolution:
        type: tuple[int, int, int]
        description: 输入 3D 特征块的空间物理分辨率源信息元组合 (pl, lat, lon)。

      output_resolution:
        type: tuple[int, int, int]
        description: 期望缩维之后的最终特征系元坐标 (pl, lat//2, lon//2)。

      in_dim:
        type: int
        default: 192
        description: 初始点云通道深度维度大小。

    inputs:
      x:
        type: Tensor
        shape: [B, pl \times Lat \times Lon, C]
        description: 一维化连续排列的高清 3D 大气序列数据。

    outputs:
      result:
        type: Tensor
        shape: [B, pl \times (Lat//2) \times (Lon//2), C \times 2]
        description: 在保持垂直高度梯度的基础上，被折叠、降温抽取并提升特征抽象通量的后处理张量组。

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn
      
      class PanguDownSample3D(nn.Module):
          """
              Pangu-Weather 风格的 3D 大气变量下采样模块。
              
              PanguDownSample2D 的三维扩展版本，对 (气压层, 纬度, 经度) 三维特征图进行
              空间下采样。注意：仅对水平方向（纬度、经度）做 2x 下采样，气压层维度保持不变，
              因此输出的气压层数与输入相同（out_pl == in_pl）。
              
              Args:
                  input_resolution (tuple[int, int, int]): 输入特征图的空间分辨率 (pl, lat, lon)。
                  output_resolution (tuple[int, int, int]): 输出特征图的空间分辨率
                      (pl, lat//2, lon//2)，气压层维度与输入保持一致，水平方向约为输入的 1/2，
                      模块内部自动计算水平方向所需的 ZeroPad 量。
                  in_dim (int, optional): 输入 token 的通道数，输出通道数为 in_dim * 2，
                      默认为 192。
              
              形状:
                  - 输入 x: (B, pl * lat * lon, C)，其中 C = in_dim
                  - 输出:   (B, pl * out_lat * out_lon, C * 2)
              
              Examples:
                  >>> # 典型 Pangu-Weather 大气变量配置
                  >>> # 气压层保持不变: pl=8
                  >>> # 水平分辨率 181×360 → 91×180
                  >>> # h_pad = 91*2 - 181 = 1（底部补1行）
                  >>> # w_pad = 180*2 - 360 = 0（无需补齐）
                  >>> # 输入 token 数: 8 * 181 * 360 = 521280
                  >>> # 输出 token 数: 8 *  91 * 180 = 131040
                  >>> downsample = PanguDownSample3D(
                  ...     input_resolution=(8, 181, 360),
                  ...     output_resolution=(8, 91, 180),
                  ...     in_dim=192,
                  ... )
                  >>> x = torch.randn(2, 521280, 192)  # (B, pl*lat*lon, C)
                  >>> out = downsample(x)
                  >>> out.shape
                  torch.Size([2, 131040, 384])
                  
                  >>> # 整除情况下无需 padding（如 pl=13, 128×256 → 64×128）
                  >>> downsample2 = PanguDownSample3D(
                  ...     input_resolution=(13, 128, 256),
                  ...     output_resolution=(13, 64, 128),
                  ...     in_dim=192,
                  ... )
                  >>> x2 = torch.randn(2, 425984, 192)  # (B, 13*128*256, C)
                  >>> out2 = downsample2(x2)
                  >>> out2.shape
                  torch.Size([2, 106496, 384])
          """
          def __init__(self, 
                       input_resolution, 
                       output_resolution,
                       in_dim=192):
              super().__init__()
              
              self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
              self.norm = nn.LayerNorm(4 * in_dim)
              self.input_resolution = input_resolution
              self.output_resolution = output_resolution
      
              in_pl, in_lat, in_lon = self.input_resolution
              out_pl, out_lat, out_lon = self.output_resolution
      
              h_pad = out_lat * 2 - in_lat
              w_pad = out_lon * 2 - in_lon
      
              pad_top = h_pad // 2
              pad_bottom = h_pad - pad_top
      
              pad_left = w_pad // 2
              pad_right = w_pad - pad_left
      
              pad_front = pad_back = 0
      
              self.pad = nn.ZeroPad3d(
                  (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
              )
      
          def forward(self, x):
              B, N, C = x.shape
              in_pl, in_lat, in_lon = self.input_resolution
              out_pl, out_lat, out_lon = self.output_resolution
              x = x.reshape(B, in_pl, in_lat, in_lon, C)
      
              # Padding the input to facilitate downsampling
              x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
              x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
              x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)
      
              x = self.norm(x)
              x = self.linear(x)
              return x
      

  skills:

    build_pangudownsample3d:

      description: 构建支持物理约束的三维/二维降维或插值组件，整合上下文环境与特征通道

      inputs:
        - input_resolution
        - output_resolution
        - in_dim
        - out_dim

      prompt_template: |

        构建名为 PanguDownSample3D 的气象特征采样模块。

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

      - 必须且务必保留 pl 气压层的纯净度，仅对经纬平面的 2x2 进行合并压缩。
      - 必须保证上采样输出与同层级下采样输出维度与分辨率完全匹配以备特征融合。
      - 所有的池化步幅和卷积核需要匹配地球环面展开特性（极点、赤道），处理由于180°与360°分辨率的不整除造成的尺寸零头。


    anti_patterns:

      - 像 3D Vision 一样进行 2x2x2 的合并缩放，破坏了气象大气的等压面独立物理规律。
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
      - TokenMerger
      - DownsampleLayer
      - 3DKernel
    part_of:
      - PanguWeatherModel
      - 3DVisionTransformer
    depends_on:
      - nn.ZeroPad3d
      - nn.LayerNorm
      - nn.Linear
    compatible_with:
      - PanguUpSample3D