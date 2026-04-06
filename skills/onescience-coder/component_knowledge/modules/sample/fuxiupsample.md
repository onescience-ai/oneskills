component:

  meta:
    name: FuxiUpSample
    alias: ConvolutionalUpsampler
    version: 1.0
    domain: physics_informed_machine_learning
    category: neural_network
    subcategory: upsampling
    author: OneScience
    tags:
      - vision_model
      - spatial_upsample
      - fuxi_weather
      - transposed_convolution

  concept:

    description: >
      FuxiUpSample 充当了 FuXi 气象大模型（以及类似视觉变换网）中的空间重构阶段。
      它反转了 FuxiDownSample 的进程，利用二维反卷积层（Transposed Convolution）步长为2，直接将局部分辨率扩展了一倍（H*2, W*2）。
      放大后同样需要挂载多个带有残差和正则化的深度卷积层进行特征平滑，以避免反卷积放大经常产生的棋盘效应（Checkerboard Artifacts）。

    intuition: >
      在通过底层编码器理解宏观复杂系统的低频特征变化规律之后，解码器需要将其映射回高清晰度的人类可读物理场/气象场。
      由于逆向解码时往往合并了跳梯连接（Skip Connections），这里使用可学习的反卷积在放大的同时学习特征通道（Channel）层面的融合权重，这比简单的双线性插值更为精准。

    problem_it_solves:
      - 低分辨率隐含特征向量向物理网格空间的逆向降维和网格拓展。
      - 使放大的网格平滑过渡，修复升尺度所导致的棋盘状离散特征缝隙。
      - 在U-Net风格网络中缩减由于特征拼接增加的巨量信道，恢复到目标通道度数。

  theory:

    formula:

      spatial_expansion:
        expression: |
          X_{up} = \text{ConvTranspose2d}_{k=2, s=2}(X_{in})
      
      residual_smoothing:
        expression: |
          X_{res}^{(i+1)} = X_{res}^{(i)} + \text{SiLU}(\text{GroupNorm}(\text{Conv2d}(X_{res}^{(i)})))
          Y = X_{up} + X_{res}^{(N)}

    variables:

      X_{in}:
        name: LatentFeatureMap
        shape: [B, C_{in}, H_{low}, W_{low}]
        description: 位于解码器内部带有高维度信道的低分辨率隐空间特征矩阵

      X_{up}:
        name: ExpandedFeatureMap
        shape: [B, C_{out}, H_{low} \times 2, W_{low} \times 2]
        description: 由于步长反卷积而被铺开的宽广基准图谱

  structure:

    architecture: residual_upsample

    pipeline:
      - name: ExpansionLayer
        operation: nn.ConvTranspose2d(stride=2, kernel_size=2)
      - name: FeatureSmoothing
        operation: N x [Conv2d -> GroupNorm -> SiLU]
      - name: ResidualAddition
        operation: shortcut_addition

  interface:

    parameters:

      in_chans:
        type: int
        default: 3072
        description: 被提取出来的隐状态维度总计。在多数场景下，该维度包含了跳梯传递（Skip-Conn）拼凑过来的层级，使得其一般为 1536 * 2。

      out_chans:
        type: int
        default: 1536
        description: 特征被融合、提取降低之后期望的通道量。

      num_groups:
        type: int
        default: 32
        description: 在特征平滑块中所使用的组归一化拆分数。

      num_residuals:
        type: int
        default: 2
        description: 在分辨率扩张后附加的残差序列块数量。

    inputs:
      x:
        type: Tensor
        shape: [B, in_chans, H, W]
        description: 缩并空间下的融合物理场

    outputs:
      result:
        type: Tensor
        shape: [B, out_chans, H \times 2, W \times 2]
        description: 返回恢复并增加了二维面积的特称场图

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
      
      
      class FuxiUpSample(nn.Module):
          """
              FuXi 模型的二维上采样模块，先用步长为 2 的反卷积将空间分辨率放大一倍，再通过若干残差卷积块细化特征。
      
              Args:
                  in_chans (int): 输入特征通道数
                  out_chans (int): 输出特征通道数
                  num_groups (int): GroupNorm 的分组数
                  num_residuals (int): 残差卷积块的数量（每个块包含 Conv2d + GroupNorm + SiLU）
      
              形状:
                  输入:  x 形状为 (B, in_chans, H, W)
                  输出:  y 形状为 (B, out_chans, H_out, W_out)，其中 H_out = H * 2，W_out = W * 2
      
              Example:
                  >>> up = FuxiUpSample(in_chans=3072, out_chans=1536, num_groups=32, num_residuals=2)
                  >>> x = torch.randn(2, 3072, 90, 180)
                  >>> y = up(x)
                  >>> y.shape
                  torch.Size([2, 1536, 180, 360])
          """
          def __init__(self, 
                       in_chans=1536*2, 
                       out_chans=1536, 
                       num_groups=32, 
                       num_residuals=2):
              super().__init__()
              self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
      
              blk = []
              for i in range(num_residuals):
                  blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
                  blk.append(nn.GroupNorm(num_groups, out_chans))
                  blk.append(nn.SiLU())
      
              self.b = nn.Sequential(*blk)
      
          def forward(self, x):
              x = self.conv(x)
      
              shortcut = x
      
              x = self.b(x)
      
              return x + shortcut
      
      

  skills:

    build_fuxiupsample:

      description: 构建支持物理约束的三维/二维降维或插值组件，整合上下文环境与特征通道

      inputs:
        - input_resolution
        - output_resolution
        - in_dim
        - out_dim

      prompt_template: |

        构建名为 FuxiUpSample 的气象特征采样模块。

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
      - UpsampleLayer
      - FeatureReconstructor
    part_of:
      - FuxiWeatherModel
      - UNetDecoder
    depends_on:
      - nn.ConvTranspose2d
      - nn.Conv2d
      - nn.GroupNorm
    compatible_with:
      - FuxiDownSample