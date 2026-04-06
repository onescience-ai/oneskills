component:

  meta:
    name: FuxiDownSample
    alias: ConvolutionalDownsampler
    version: 1.0
    domain: physics_informed_machine_learning
    category: neural_network
    subcategory: downsampling
    author: OneScience
    tags:
      - vision_model
      - spatial_downsample
      - fuxi_weather
      - residual_block

  concept:

    description: >
      FuxiDownSample 构成了 FuXi 气象大模型（或者类似的视觉变换网络）架构中的核心空间下采样阶段。
      该模块采用了一种较为传统的基于卷积特征提取的下采样方法，首先通过一个步长（stride）为2的二维卷积将空间分辨率减半（H/2, W/2），
      随后通过多个连续的残差块（Residual Blocks）对下采样后的特征进行深度变换和非线性抽象。

    intuition: >
      类似于图像金字塔的构建，在大气多尺度建模中，高分辨率的网格捕获了细粒度气象特征，而低分辨率网格捕获了宏观气流和天气系统的演变。
      利用步长为 2 的卷积层，可以在压缩信息冗余的同时聚合局部邻域特征；随后的多层残差连接与 GroupNorm 则用于平滑深层网络特征空间并防梯度弥散，
      使得提取的宏观特征更鲁棒。

    problem_it_solves:
      - 高分辨率气象场处理带来的组合爆炸与内存溢出问题。
      - 聚合长尺度上的局域特征，扩大感受野。
      - 优雅处理由于奇数物理经纬度网格产生的不均匀边界对齐（向下截断）。

  theory:

    formula:

      spatial_reduction:
        expression: |
          X_{down} = \text{Conv2d}_{k=3, s=2, p=1}(X_{in})
      
      residual_transform:
        expression: |
          X_{res}^{(i+1)} = X_{res}^{(i)} + \text{SiLU}(\text{GroupNorm}(\text{Conv2d}(X_{res}^{(i)})))
          Y = X_{down} + X_{res}^{(N)}

      bound_truncation:
        expression: |
          Y_{final} = Y_{[:H-H\%2,\ :W-W\%2]} 

    variables:

      X_{in}:
        name: InputTensor
        shape: [B, C_{in}, H, W]
        description: 输入的高分辨率特征场

      X_{down}:
        name: DownsampledTensor
        shape: [B, C_{out}, H/2, W/2]
        description: 经 stride=2 卷积压缩后的低分辨率特征

      \text{SiLU}:
        name: SigmoidLinearUnit
        description: 非线性激活函数，平滑激活利于深层连续训练

  structure:

    architecture: residual_downsample

    pipeline:
      - name: CompressionLayer
        operation: nn.Conv2d(stride=2)
      - name: DeepFeatureExtraction
        operation: N x [Conv2d -> GroupNorm -> SiLU]
      - name: ResidualAddition
        operation: shortcut_addition
      - name: OddBoundaryTruncation
        operation: "remove last column/row if original H/W was odd (e.g. 181 -> 90)"

  interface:

    parameters:

      in_chans:
        type: int
        default: 1536
        description: 输入流的特征通道数。

      out_chans:
        type: int
        default: 1536
        description: 降维后输出流的特征通道数（通常保持不变或翻倍）。

      num_groups:
        type: int
        default: 32
        description: 在特征块中所使用的 GroupNorm 组数，以减少 batch size 对归一化统计量的影响。

      num_residuals:
        type: int
        default: 2
        description: 串联的提取残差块的深度数量。

    inputs:
      x:
        type: Tensor
        shape: [B, in_chans, H, W]
        description: 气象或物理场高维平面图

    outputs:
      result:
        type: Tensor
        shape: [B, out_chans, H_{out}, W_{out}]
        description: 下采样平滑张量，其中 H_out = H // 2，W_out = W // 2

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
      
      
      class FuxiDownSample(nn.Module):
          """
              FuXi 模型的二维下采样模块，先用步长为 2 的卷积将空间分辨率减半，再通过若干残差卷积块进行特征变换。
      
              Args:
                  in_chans (int): 输入特征通道数
                  out_chans (int): 输出特征通道数
                  num_groups (int): GroupNorm 的分组数
                  num_residuals (int): 残差卷积块的数量（每个块包含 Conv2d + GroupNorm + SiLU）
      
              形状:
                  输入:  x 形状为 (B, in_chans, H, W)
                  输出:  y 形状为 (B, out_chans, H_out, W_out)，其中 H_out = H // 2，W_out = W // 2
      
              Example:
                  >>> down = FuxiDownSample(in_chans=1536, out_chans=1536, num_groups=32, num_residuals=2)
                  >>> x = torch.randn(2, 1536, 180, 360)
                  >>> y = down(x)
                  >>> y.shape
                  torch.Size([2, 1536, 90, 180])
          """
          def __init__(self, 
                       in_chans=1536, 
                       out_chans=1536, 
                       num_groups=32, 
                       num_residuals=2):
              super().__init__()
              self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1)
      
              blk = []
              for i in range(num_residuals):
                  blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
                  blk.append(nn.GroupNorm(num_groups, out_chans))
                  blk.append(nn.SiLU())
      
              self.b = nn.Sequential(*blk)
      
          def forward(self, x):
              _, _, h, w = x.shape
              x = self.conv(x)
      
              shortcut = x
      
              x = self.b(x)
      
              res = x + shortcut
              if h % 2 != 0:
                  res = res[:, :, :-1, :]
              if w % 2 != 0:
                  res = res[:, :, :, :-1]
              return res
      

  skills:

    build_fuxidownsample:

      description: 构建支持物理约束的三维/二维降维或插值组件，整合上下文环境与特征通道

      inputs:
        - input_resolution
        - output_resolution
        - in_dim
        - out_dim

      prompt_template: |

        构建名为 FuxiDownSample 的气象特征采样模块。

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

      - Placeholder BP
      - 必须保证上采样输出与同层级下采样输出维度与分辨率完全匹配以备特征融合。
      - 所有的池化步幅和卷积核需要匹配地球环面展开特性（极点、赤道），处理由于180°与360°分辨率的不整除造成的尺寸零头。


    anti_patterns:

      - Placeholder AP
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
      - DownsampleLayer
      - FeatureExtractor
    part_of:
      - FuxiWeatherModel
      - UNetEncoder
    depends_on:
      - nn.Conv2d
      - nn.GroupNorm
      - nn.SiLU
    compatible_with:
      - FuxiUpSample