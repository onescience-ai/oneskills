component:

  meta:
    name: UNetComponents
    alias: UNetLayers
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: semantic_segmentation
    author: OneScience
    license: Apache-2.0
    tags:
      - unet
      - multiscale
      - downsampling
      - upsampling
      - skip_connection

  concept:

    description: >
      该模块包含了构建 1D、2D 和 3D U-Net 模型所需的核心多尺度组件集合。
      U-Net 是一种经典的编码器-解码器架构，广泛用于医学图像分割、物理场重建和替代建模。
      这些组件包括：用于提取特征的双卷积块（DoubleConv）、用于降低空间分辨率并增加通道数的下采样块（Down）、
      用于恢复空间分辨率并通过跳跃连接（Skip Connection）融合浅层细节的上采样块（Up），以及最终通道映射层（OutConv）。

    intuition: >
      U-Net 的设计哲学是“局部上下文”与“全局抽象”的结合。
      在编码器阶段（Down），网络像是一个望远镜，视野（感受野）越来越大，捕捉宏观物理规律或图像语义，但丢失了精细的坐标位置；
      在解码器阶段（Up），网络通过上采样放大特征，同时直接从编码器“借用”同层级的浅层特征（跳跃连接拼接），
      以此将宏观语义重新精准定位到微观像素/网格点上。

    problem_it_solves:
      - 解决卷积神经网络在连续空间预测（如图像分割、流场预测）中因下采样导致的空间位置信息丢失问题
      - 提供在 1D（序列/时间序列）、2D（图像/切片）和 3D（体素/时空场）数据上的统一尺度变换方案
      - 处理解码器阶段因池化操作造成的张量尺寸不匹配（如 2D Up 中的自动 padding）

  theory:

    formula:

      double_conv:
        expression: y = \sigma(\text{Norm}(W_2 \cdot \sigma(\text{Norm}(W_1 \cdot x + b_1)) + b_2))

      up_sampling_fusion:
        expression: y = \text{DoubleConv}([x_{\text{skip}} \parallel \text{Upsample}(x_{\text{deep}})])

    variables:

      x:
        name: InputTensor
        description: 卷积块的输入张量

      x_{\text{deep}}:
        name: DeepFeature
        description: 来自上一层解码器的深层抽象特征

      x_{\text{skip}}:
        name: SkipFeature
        description: 来自同层级编码器的浅层空间细节特征

      \sigma:
        name: ReLU
        description: 修正线性单元激活函数

      \parallel:
        name: Concatenation
        description: 在通道维度上的张量拼接

  structure:

    architecture: encoder_decoder_blocks

    pipeline:

      - name: EncoderPath (Down)
        operation: max_pool_and_double_conv

      - name: DecoderPath (Up)
        operation: upsample_concat_and_double_conv

      - name: OutputHead (OutConv)
        operation: pointwise_convolution

  interface:

    parameters:

      in_channels:
        type: int
        description: 输入通道数（注意在 Up 模块中，这是拼接后的总通道数）

      out_channels:
        type: int
        description: 输出通道数

      normtype:
        type: str
        default: 'bn'
        description: 归一化策略，支持 'bn' (BatchNorm) 或 'in' (InstanceNorm)

      kernel_size:
        type: int
        default: 3
        description: 卷积核大小，必须为奇数以保持空间对齐

      bilinear:
        type: bool
        default: true
        description: 在 Up 模块中是否使用插值算法上采样（False 则使用转置卷积）

    inputs:

      x:
        type: Tensor
        shape: "[B, C, L] (1D), [B, C, H, W] (2D), 或 [B, C, D, H, W] (3D)"
        dtype: float32

      x1_x2:
        type: Tuple[Tensor, Tensor]
        description: 仅 Up 模块需要，x1 为深层输入，x2 为跳跃连接输入

    outputs:

      output:
        type: Tensor
        shape: "空间维度根据模块类型 (Down 减半, Up 加倍, 其他不变)"

  types:

    Tensor:
      shape: dynamic
      description: 1D, 2D 或 3D 的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      # 此处仅展示 2D 模块作为核心代表，1D 和 3D 实现结构完全一致，仅替换基础算子
      class DoubleConv2D(nn.Module):
          def __init__(self, in_channels, out_channels, mid_channels=None, normtype="bn", kernel_size=3):
              super().__init__()
              if not mid_channels:
                  mid_channels = out_channels
              padding = kernel_size // 2

              if normtype == "bn":
                  self.double_conv = nn.Sequential(
                      nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
                      nn.BatchNorm2d(mid_channels),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True),
                  )
              # (省略 IN 和 无归一化分支)

          def forward(self, x):
              return self.double_conv(x)

      class Down2D(nn.Module):
          def __init__(self, in_channels, out_channels, normtype="bn", kernel_size=3):
              super().__init__()
              self.maxpool_conv = nn.Sequential(
                  nn.MaxPool2d(2), 
                  DoubleConv2D(in_channels, out_channels, normtype=normtype, kernel_size=kernel_size)
              )

          def forward(self, x):
              return self.maxpool_conv(x)

      class Up2D(nn.Module):
          def __init__(self, in_channels, out_channels, bilinear=True, normtype="bn", kernel_size=3):
              super().__init__()
              if bilinear:
                  self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                  self.conv = DoubleConv2D(in_channels, out_channels, in_channels // 2, normtype=normtype, kernel_size=kernel_size)
              else:
                  self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                  self.conv = DoubleConv2D(in_channels, out_channels, normtype=normtype, kernel_size=kernel_size)

          def forward(self, x1, x2):
              x1 = self.up(x1)
              diffY = x2.size()[2] - x1.size()[2]
              diffX = x2.size()[3] - x1.size()[3]
              x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
              x = torch.cat([x2, x1], dim=1)
              return self.conv(x)

      class OutConv2D(nn.Module):
          def __init__(self, in_channels, out_channels):
              super(OutConv2D, self).__init__()
              self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

          def forward(self, x):
              return self.conv(x)

  skills:

    build_unet_encoder_decoder:

      description: 根据数据的维度 (1D, 2D, 3D) 快速构建 U-Net 主干网络

      inputs:
        - dimension (1, 2, or 3)
        - depth (下采样次数)
        - base_channels

      prompt_template: |
        使用 {{dimension}}D U-Net 组件构建一个编码器-解码器。
        基础通道数为 {{base_channels}}，深度为 {{depth}}。
        请在 Up 模块正确处理跳跃连接。

    diagnose_unet_tensor_shapes:

      description: 解决由于池化导致的跳跃连接张量尺寸无法拼接的问题

      checks:
        - concat_dimension_mismatch_due_to_odd_input_sizes
        - missing_padding_logic_in_1d_and_3d_up_layers

  knowledge:

    usage_patterns:

      standard_unet:

        pipeline:
          - InputConv (DoubleConv)
          - Down (x4)
          - Up (x4, with skip connections)
          - OutConv

    design_patterns:

      skip_connection_with_padding:

        structure:
          - 当输入图像的分辨率不是 2 的整数次幂时，`MaxPool2d` 会向下取整。
          - 在解码阶段通过 `Upsample` 恢复尺寸时，会比对应的跳跃特征小 1 个像素。
          - `Up2D` 模块内置了基于 `F.pad` 的动态补偿逻辑，使其对任意输入尺寸都具有鲁棒性。

    hot_models:

      - model: U-Net (Original)
        year: 2015
        role: 医疗图像分割的开创性模型
        architecture: convolutional_encoder_decoder

      - model: PDE-Surrogates (基于 U-Net)
        year: 2020+
        role: 流体力学偏微分方程的代理模型
        architecture: unet_with_physical_padding

    model_usage_details:

      Physics Surrogates:

        usage: 在流体力学（如 OpenFOAM 数据代理）中，3D U-Net 经常被用来预测 3D 压力场或速度场的演化。InstanceNorm (`in`) 通常比 BatchNorm (`bn`) 在单个物理样本上表现更稳定。

    best_practices:

      - 根据任务选择 `normtype`：对于 Batch Size 较小的 3D 物理场预测，优先使用 `in` (InstanceNorm)；对于大规模图像分割，使用 `bn` (BatchNorm)。
      - `Up` 模块的 `in_channels` 参数是跳跃特征通道数与上采样特征通道数的**总和**。
      - 默认使用 `bilinear=True` (线性插值) 而不是转置卷积，可以有效减少生成图像中的“棋盘伪影” (Checkerboard Artifacts)。

    anti_patterns:

      - 在实例化 `Up` 层时，将 `in_channels` 错误地仅设置为上一层的输出通道数，导致内部 `DoubleConv` 的维度断裂。
      - 在处理 1D 或 3D 边界不规则尺寸时，直接进行 `torch.cat` 拼接而不计算尺寸差异（当前代码在 2D 中提供了 Padding，但在 1D/3D 中可能需要手动补齐）。

    paper_references:

      - title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        authors: Ronneberger et al.
        year: 2015

  graph:

    is_a:
      - NeuralNetworkComponent
      - ConvolutionalBlock
      - MultiScaleModule

    part_of:
      - UNet
      - VNet
      - DiffusionModel (作为其噪声估计网络)

    depends_on:
      - nn.Conv1d/2d/3d
      - nn.MaxPool1d/2d/3d
      - nn.Upsample
      - nn.BatchNorm / nn.InstanceNorm

    variants:
      - Attention U-Net
      - ResUNet

    used_in_models:
      - U-Net
      - 物理场替代模型 (AI Fluid Solvers)

    compatible_with:

      inputs:
        - Tensor

      outputs:
        - Tensor