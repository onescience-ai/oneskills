component:

  meta:
    name: PanguDownSample2D
    alias: PanguPatchMerging
    version: 1.0
    domain: physics_informed_machine_learning
    category: neural_network
    subcategory: downsampling
    author: OneScience
    tags:
      - transformer
      - patch_merging
      - pangu_weather
      - token_downsample

  concept:

    description: >
      PanguDownSample2D 实现了 Pangu-Weather 等气象 Transformer 模型中使用的 2D 空间降维机制。
      它采取了类似 Swin Transformer 的 Patch Merging 策略，摒弃了标准 CNN 的池化层。通过将 $2 \times 2$ 相邻区域内的 Token
      在通道（Channel）维度进行合并拼接展平（使特征量 $C$ 扩展到 $4C$），并在展平后经受结构化的 `LayerNorm` 与 `Linear` 线性压缩（$4C \to 2C$）投影，
      最终实现将有效物理分辨率强行缩小至 1/4 (宽高各除以2)。并且本模块内置了对奇数格点的高级对齐 ZeroPad 填补逻辑。

    intuition: >
      在高频气象（如ERA5 181x360）进入自注意力机制计算时，序列长度达到了极高的水平（O(N^2)复杂度）。如果我们简单的进行池化切除会丢失重要的局部大气微扰信息。
      Patch Merging 巧妙地将相邻 $2 \times 2$ 的四个网格格点像折纸一样折叠“压缩”进一维的通道隐变量里。在此过程中，没有信息被盲目丢弃（只有维度线性压缩）。
      这为后级 Transformer 的全局自注意力场创造出了宽广的高密度信息表达节点。

    problem_it_solves:
      - 几何气象场降维中如何不造成高频信息粗暴受损丢失，保持信息容量密度。
      - Transformer 自注意力序列计算长度的维度灾难 O(N^2) 瓶颈压缩。
      - 不均匀输入分辨率边界（例如纬度 lat=181 不能被 2 整除）导致重组切片越界的问题。

  theory:

    formula:

      token_reassembly:
        expression: |
          X_{\text{reshaped}} = X_{in}[..., i:i+2, j:j+2, C] \longrightarrow X_{\text{merged}} \text{ (shape: } [B, \frac{H \times W}{4}, 4C])
      
      channel_projection:
        expression: |
          Y = W_{linear} \cdot \text{LayerNorm}(X_{\text{merged}}) \text{ (shape: } [B, \frac{H \times W}{4}, 2C])

      padding_mechanism:
        expression: |
          \text{Pad}_{req} = H_{out} \times 2 - H_{in}

    variables:

      X_{in}:
        name: SequenceTokens
        shape: [B, H \times W, C]
        description: 长序列化后的平面 Token 流向量

      W_{linear}:
        name: ReprojectionMatrix
        shape: [4C, \text{out\_dim}]
        description: 将极度宽延的压栈特征融合、提取出有用的隐含气象特征组的线性层映射字典。

  structure:

    architecture: token_patch_merging
    
    pipeline:
      - name: SpatialReshape
        operation: "[B, N, C] -> [B, In_Lat, In_Lon, C]"
      - name: ZeroPaddingAlignment
        operation: nn.ZeroPad2d
      - name: PatchGridExtraction
        operation: ".reshape([B, out_lat, 2, out_lon, 2, C])"
      - name: TokenConcatenation
        operation: ".permute(...) -> reshape to [B, out_lat * out_lon, 4 * C]"
      - name: NormalizationAndSqueeze
        operation: LayerNorm -> Linear(4*C, 2*C)

  interface:

    parameters:

      input_resolution:
        type: tuple[int, int]
        description: 输入特征图的真实空间分辨率原点元组 (lat, lon)。

      output_resolution:
        type: tuple[int, int]
        description: 目标输出层级特征图的空间分辨率 (lat//2, lon//2)。

      in_dim:
        type: int
        default: 192
        description: 输入词块（Token）的单个隐藏维信道总宽，在转换结束时将被转出成 in_dim * 2 。

    inputs:
      x:
        type: Tensor
        shape: [B, \text{Lat}_{in} \times \text{Lon}_{in}, C]
        description: 一维展平特征序列场。

    outputs:
      result:
        type: Tensor
        shape: [B, \text{Lat}_{out} \times \text{Lon}_{out}, C \times 2]
        description: 压栈结合并精压缩后降高并扩张了通道维度的张量段。

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn
      
      class PanguDownSample2D(nn.Module):
          """
              Pangu-Weather 风格的 2D 空间下采样模块。
              
              类似 Swin Transformer 的 Patch Merging 操作，将 2×2 相邻 token 在通道维度拼接
              （C → 4C），再经 LayerNorm 与线性投影还原通道数（4C → 2C），空间分辨率缩小为原来
              的 1/2。当输入分辨率不能被 2 整除时（如 lat=181），自动进行 ZeroPad 补齐。
              
              Args:
                  input_resolution (tuple[int, int]): 输入特征图的空间分辨率 (lat, lon)。
                  output_resolution (tuple[int, int]): 输出特征图的空间分辨率 (lat//2, lon//2)，
                      应满足 output_resolution ≈ input_resolution / 2，模块内部会自动计算所需
                      padding 量以对齐 output_resolution * 2 与 input_resolution 的差值。
                  in_dim (int, optional): 输入 token 的通道数，输出通道数为 in_dim * 2，
                      默认为 192。
              
              形状:
                  - 输入 x: (B, lat * lon, C)，其中 C = in_dim
                  - 输出:   (B, out_lat * out_lon, C * 2)
              
              Examples:
                  >>> # 气象场分辨率 181×360 → 91×180 下采样
                  >>> # h_pad = 91*2 - 181 = 1（底部补1行）
                  >>> # w_pad = 180*2 - 360 = 0（无需补齐）
                  >>> # 输入 token 数: 181 * 360 = 65160
                  >>> # 输出 token 数:  91 * 180 = 16380
                  >>> downsample = PanguDownSample2D(
                  ...     input_resolution=(181, 360),
                  ...     output_resolution=(91, 180),
                  ...     in_dim=192,
                  ... )
                  >>> x = torch.randn(2, 65160, 192)  # (B, lat*lon, C)
                  >>> out = downsample(x)
                  >>> out.shape
                  torch.Size([2, 16380, 384])
                  
                  >>> # 整除情况下无需 padding（如 128×256 → 64×128）
                  >>> downsample2 = PanguDownSample2D(
                  ...     input_resolution=(128, 256),
                  ...     output_resolution=(64, 128),
                  ...     in_dim=192,
                  ... )
                  >>> x2 = torch.randn(2, 32768, 192)  # (B, 128*256, C)
                  >>> out2 = downsample2(x2)
                  >>> out2.shape
                  torch.Size([2, 8192, 384])
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
      
              in_lat, in_lon = self.input_resolution
              out_lat, out_lon = self.output_resolution
      
              h_pad = out_lat * 2 - in_lat
              w_pad = out_lon * 2 - in_lon
      
              pad_top = h_pad // 2
              pad_bottom = h_pad - pad_top
      
              pad_left = w_pad // 2
              pad_right = w_pad - pad_left
      
              self.pad = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
      
          def forward(self, x: torch.Tensor):
              B, N, C = x.shape
              in_lat, in_lon = self.input_resolution
              out_lat, out_lon = self.output_resolution
              x = x.reshape(B, in_lat, in_lon, C)
      
              # Padding the input to facilitate downsampling
              x = self.pad(x.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)
              x = x.reshape(B, out_lat, 2, out_lon, 2, C).permute(0, 1, 3, 2, 4, 5)
              x = x.reshape(B, out_lat * out_lon, 4 * C)
      
              x = self.norm(x)
              x = self.linear(x)
              return x
      

  skills:

    build_pangudownsample2d:

      description: 构建支持物理约束的三维/二维降维或插值组件，整合上下文环境与特征通道

      inputs:
        - input_resolution
        - output_resolution
        - in_dim
        - out_dim

      prompt_template: |

        构建名为 PanguDownSample2D 的气象特征采样模块。

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

      - 将 2x2 维度的 Patch 显式展开合并以降低特征序列长度。
      - 必须保证上采样输出与同层级下采样输出维度与分辨率完全匹配以备特征融合。
      - 所有的池化步幅和卷积核需要匹配地球环面展开特性（极点、赤道），处理由于180°与360°分辨率的不整除造成的尺寸零头。


    anti_patterns:

      - 合并时不考虑经纬度首尾相邻（周期性），导致东西半球的边界出现断层。
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
    part_of:
      - PanguWeatherModel
      - VisionTransformer
    depends_on:
      - nn.Linear
      - nn.LayerNorm
      - nn.ZeroPad2d
    compatible_with:
      - PanguUpSample2D