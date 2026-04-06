component:

  meta:
    name: PanguUpSample2D
    alias: PanguPixelShuffle
    version: 1.0
    domain: physics_informed_machine_learning
    category: neural_network
    subcategory: upsampling
    author: OneScience
    tags:
      - transformer
      - pixel_shuffle
      - pangu_weather
      - token_upsample

  concept:

    description: >
      PanguUpSample2D 是 Pangu-Weather 系统用于大尺度特征逆向重构成高精度物理场的 2D 空间上采样层。
      它作为 PanguDownSample2D 的严格解反向机制存在。它同样不使用任何反卷积，而是利用被称为“PixelShuffle（亚像素卷积提取拓展器）”的维度分割概念：
      首将当前 Token 的抽象深槽空间利用线性映射投射膨胀至 4 倍宽（C -> out_dim * 4），然后将这庞大的通道切割，重新镶嵌、散播回
      空间物理网格阵列，转化为一个宽两倍、高两倍，但降深度的子特征阵列（$2 \times 2$ patch expansion）。如果散开后的尺寸比期望目标尺寸略大，内部将启动物理中心点裁剪裁齐。

    intuition: >
      将一张被揉碎堆入盒子的四开报纸再展平铺开。当隐含知识经过深层的 Transformer 全局运算以后获得了低维但深层抽象的流体力学表达，为了让预测落地，
      不能产生虚假的像素插值（会导致云图发虚），而是将深层通道代表的信息通过学习到的无偏线性层分裂为四个象限，以此获得锐利度极强的高保真气候场反演。

    problem_it_solves:
      - 克服使用双线性插值以及反卷积引发的高解析度生成任务中的视觉模糊与“幻块”问题。
      - 将序列化的长 Token 有序切割回物理经纬度网格空间。
      - 精确去除或修正由于前期下采样在奇数纬度处强行引入补零（Padding）所多出的那一部分背景边距边缘。

  theory:

    formula:

      feature_inflation:
        expression: |
          X_{\text{inflated}} = W_{linear1} \cdot X_{\text{in}}  \text{ (shape: } [B, N, \text{out\_dim} \times 4] )

      spatial_unfolding:
        expression: |
          X_{\text{spatial}} = \text{PixelShuffle}(X_{\text{inflated}})
          X_{\text{spatial}} = X_{\text{inflated}}[C] \to \text{Reshape}([B, H_{out}, W_{out}, \text{out\_dim}])
      
      boundary_cropping:
        expression: |
          Y_{cropped} = X_{\text{spatial}}[\text{Pad}_{top} \dots - \text{Pad}_{bot}, ...]

    variables:

      X_{\text{in}}:
        name: CompactLatentTokens
        description: 取自 Transformer 的全局精简型低分别率信息串维度张量

      P_{crops}:
        name: CroppingOffsets
        description: 基于原始要求输出和强制$2\times2$扩张之间的差余所推算切割偏移率。

  structure:

    architecture: token_pixel_shuffle

    pipeline:
      - name: FeatureExpansion
        operation: "Linear(in_dim, out_dim * 4)"
      - name: DimensionalUnfolding
        operation: ".reshape([B, in_lat, in_lon, 2, 2, out_dim]).permute(...)"
      - name: CenterCropAlignment
        operation: "Crop boundaries based on pad_h / pad_w"
      - name: FormatReshaping
        operation: ".reshape -> [B, Spatial_1D, out_dim]"
      - name: FeatureSmoothing
        operation: LayerNorm -> Linear(out_dim, out_dim)

  interface:

    parameters:

      in_dim:
        type: int
        description: 压缩层态下输入流的通道信度空间宽

      out_dim:
        type: int
        description: 输出目标特征向量中的单一空间节点包含的信度通道量。（通常为 in_dim // 2）

      input_resolution:
        type: tuple[int, int]
        description: 经过压缩运算后给进来的网格分辨率

      output_resolution:
        type: tuple[int, int]
        description: 真实期望反演投射的目标界限物理经纬坐标分辨率。

    inputs:
      x:
        type: Tensor
        shape: [B, Lat_{in} \times Lon_{in}, C]
        description: 展平后的序列态隐层

    outputs:
      result:
        type: Tensor
        shape: [B, Lat_{out} \times Lon_{out}, out\_dim]
        description: 精准展回到物理世界经纬度的像素层列

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn
      
      class PanguUpSample2D(nn.Module):
          """
              Pangu-Weather 风格的 2D 空间上采样模块。
              
              PanguDownSample2D 的逆操作。先通过线性层将通道数扩展（C → out_dim * 4），
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
      
              pad_h = in_lat * 2 - out_lat
              pad_w = in_lon * 2 - out_lon
      
              pad_top = pad_h // 2
              pad_bottom = pad_h - pad_top
      
              pad_left = pad_w // 2
              pad_right = pad_w - pad_left
      
              x = x[
                  :, pad_top : 2 * in_lat - pad_bottom, pad_left : 2 * in_lon - pad_right, :
              ]
              x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
              x = self.norm(x)
              x = self.linear2(x)
              return x

  skills:

    build_panguupsample2d:

      description: 构建支持物理约束的三维/二维降维或插值组件，整合上下文环境与特征通道

      inputs:
        - input_resolution
        - output_resolution
        - in_dim
        - out_dim

      prompt_template: |

        构建名为 PanguUpSample2D 的气象特征采样模块。

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

      - 推荐预先使用 Linear 投影拓展通道数，然后进行 Pixel Shuffle 将特征展平为二维物理尺度。
      - 必须保证上采样输出与同层级下采样输出维度与分辨率完全匹配以备特征融合。
      - 所有的池化步幅和卷积核需要匹配地球环面展开特性（极点、赤道），处理由于180°与360°分辨率的不整除造成的尺寸零头。


    anti_patterns:

      - 对于奇数长宽（如181），直接依靠纯整数倍扩放导致地球赤道与极点数据位移，必须辅以精确的 Pad 切图。
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
    part_of:
      - PanguWeatherModel
      - VisionTransformerDecoder
    depends_on:
      - nn.Linear
      - nn.LayerNorm
    compatible_with:
      - PanguDownSample2D