component:

  meta:
    name: XiheUpSample
    alias: InterpolatingXihePixelShuffle
    version: 1.0
    domain: physics_informed_machine_learning
    category: neural_network
    subcategory: upsampling
    author: OneScience
    tags:
      - transformer
      - pixel_shuffle
      - bilinear_interpolation
      - token_upsample
      - xihe_model

  concept:

    description: >
      XiheUpSample 是受到羲和气象大模型（Xihe）结构特点启发而构建的基于亚像素化操作与显式后插值平滑上采样的 2D 降维恢复算子。
      从流程原理上，前半段它完全借用了盘古大模型的上采思维（Linear扩展4倍通道并借助张量 reshape Permute强行还原为 $2 \times 2$ 晶格），但其最为核心的设计改良在于：
      它没有选择生硬的、甚至略带不可微风险的切边截断（Cropping）机制来对齐被 Padding 搞乱的非规整边界，而是彻底转向了经典的计算机视觉图像重构算法——双线性精准插值（`F.interpolate(mode='bilinear')`）。从而让重建空间直接自发塑型对齐目标长宽。

    intuition: >
      对于如极端天气边界风暴云层来说，盘古模型的“暴烈生切”或许会保留某种物理量数字上的绝对忠诚；但从平滑感受野和气流连贯性上来说，
      羲和网络相信利用神经网络插值的平滑扩散（Bilinear 拉伸或萎缩）能提供一个边界过度最圆滑、毫无阶梯剥离感的数据分布图，从而大大增加短期气象追踪连续预测预报的生成图的保真度感受。

    problem_it_solves:
      - 高频突变以及 Padding 余量处理采用暴力中心点剔除手段导致的天气地图边缘缝合突变误差缺陷。
      - 防止不平衡上界坐标因为舍入逻辑错位引发偏移像素失真效应。
      - 使空间物理缩放的容差变得异常弹性：无论之前差值多多少，都能被丝滑插值进需要的目标界域。

  theory:

    formula:
      channel_spread:
        expression: |
          X_{\text{spread}} = \text{PixelShuffle}(W_{expand} \cdot x_{in})
      
      bilinear_stretching:
        expression: |
          X_{\text{interpolated}}^{u,v} = \sum_{i \in \{-1,1\}} \sum_{j \in \{-1,1\}} w_{i,j} \cdot X_{\text{spread}}(u+i, v+j) \rightarrow \text{Res}_{out\_lat, \ out\_lon}

    variables:
      W_{expand}:
        name: SpaceExpansionMatrix
        description: 承担 4 倍扩容功能的映射底重层。

  structure:

    architecture: interpolation_based_shuffle

    pipeline:
      - name: DimensionalMultiplier
        operation: "Linear(C, out_dim * 4)"
      - name: SpaceUnpacking
        operation: "Reshape & Permute -> Tensor in pseudo-grid state [B, lat*2, lon*2]"
      - name: ResoluteInterpolation
        operation: "F.interpolate(..., size=(target_lat, target_lon), mode='bilinear')"
      - name: OutputAlignment
        operation: LayerNorm -> Final Linear(out_dim) projection.

  interface:

    parameters:

      in_dim:
        type: int
        description: 准备扩容恢复重投影进的深层次信息通道特征维深。

      out_dim:
        type: int
        description: 输出目标特征向量中的重投深。

      input_resolution:
        type: tuple[int, int]
        description: 当下隐含低分辨的拓扑信息图源长高信息对位。

      output_resolution:
        type: tuple[int, int]
        description: 真实期望通过插值所精准捏合造出的物理世界最终分辨目标系结构规模。

    inputs:
      x:
        type: Tensor
        shape: [B, Lat_{in} \times Lon_{in}, C]
        description: 输入流源自更高层的短小精悍的潜表示（Latent Space）。

    outputs:
      result:
        type: Tensor
        shape: [B, Lat_{out} \times Lon_{out}, out\_dim]
        description: 完全被插值匹配到了理想形状空间的气流网络云输出矩阵图样列。

  implementation:

    framework: pytorch

    code: |
      import torch
      import torch.nn.functional as F
      from torch import nn
      
      class XiheUpSample(nn.Module):
          """
              xihe 风格的 2D 空间上采样模块。
              
              XiheDownSample2D 的逆操作。先通过线性层将通道数扩展（C → out_dim * 4），
              再将每个 token 拆分为 2×2 的子像素块（类似 PixelShuffle），空间分辨率扩大为
              原来的 2 倍。当上采样后的分辨率超出目标分辨率时，自动通过裁剪对齐
              output_resolution。
              
              Args:
                  in_dim (int): 输入 token 的通道数。
                  out_dim (int): 输出 token 的通道数，线性层先扩展至 out_dim * 4，
                      拆分后每个子像素通道数恢复为 out_dim。
                  input_resolution (tuple[int, int]): 输入特征图的空间分辨率 (lat, lon)。
                  output_resolution (tuple[int, int]): 目标输出分辨率 (out_lat, out_lon)，
                      应满足 out_lat ≤ in_lat * 2 且 out_lon ≤ in_lon * 2，超出部分通过
                      中心裁剪去除。
              
              形状:
                  - 输入 x: (B, lat * lon, C)，其中 C = in_dim
                  - 输出:   (B, out_lat * out_lon, out_dim)
              
              Examples:
                  >>> # 气象场分辨率 91×180 → 181×360 上采样（对应 PanguDownSample2D 的逆操作）
                  >>> # in_lat * 2 = 91 * 2 = 182，裁剪掉多余的1行: pad_h = 182 - 181 = 1
                  >>> # in_lon * 2 = 180 * 2 = 360，无需裁剪: pad_w = 0
                  >>> # 输入 token 数:  91 * 180 = 16380
                  >>> # 输出 token 数: 181 * 360 = 65160
                  >>> upsample = PanguUpSample2D(
                  ...     in_dim=384,
                  ...     out_dim=192,
                  ...     input_resolution=(91, 180),
                  ...     output_resolution=(181, 360),
                  ... )
                  >>> x = torch.randn(2, 16380, 384)  # (B, lat*lon, C)
                  >>> out = upsample(x)
                  >>> out.shape
                  torch.Size([2, 65160, 192])
                  
                  >>> # 整除情况下无需裁剪（如 64×128 → 128×256）
                  >>> upsample2 = PanguUpSample2D(
                  ...     in_dim=384,
                  ...     out_dim=192,
                  ...     input_resolution=(64, 128),
                  ...     output_resolution=(128, 256),
                  ... )
                  >>> x2 = torch.randn(2, 8192, 384)  # (B, 64*128, C)
                  >>> out2 = upsample2(x2)
                  >>> out2.shape
                  torch.Size([2, 32768, 192])
          """
          def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
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
              in_lat, in_lon = self.input_resolution
              out_lat, out_lon = self.output_resolution
      
              x = self.linear1(x)
              x = x.reshape(B, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 3, 2, 4, 5)
              x = x.reshape(B, in_lat * 2, in_lon * 2, -1)
      
              # ✅ 用插值精确拉伸到目标分辨率
              x = F.interpolate(
                  x.permute(0, 3, 1, 2),
                  size=(out_lat, out_lon),
                  mode="bilinear",
                  align_corners=False
              )
              x = x.permute(0, 2, 3, 1).reshape(B, out_lat * out_lon, -1)
      
              x = self.norm(x)
              x = self.linear2(x)
              return x

  skills:

    build_xiheupsample:

      description: 构建支持物理约束的三维/二维降维或插值组件，整合上下文环境与特征通道

      inputs:
        - input_resolution
        - output_resolution
        - in_dim
        - out_dim

      prompt_template: |

        构建名为 XiheUpSample 的气象特征采样模块。

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

      - 运用 F.interpolate 时必须保留并使用 align_corners=False 来防止极值系统位移。
      - 必须保证上采样输出与同层级下采样输出维度与分辨率完全匹配以备特征融合。
      - 所有的池化步幅和卷积核需要匹配地球环面展开特性（极点、赤道），处理由于180°与360°分辨率的不整除造成的尺寸零头。


    anti_patterns:

      - 不要过度依赖跨越高层级的直接插值而摒弃跳跃连接，否则会导致严重雾化模糊（blur）。
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
      - InterpolatorLayer
      - UpsampleLayer
    part_of:
      - XiheWeatherModel
    depends_on:
      - F.interpolate
      - nn.LayerNorm
      - nn.Linear
    compatible_with:
      - XiheDownSample2D