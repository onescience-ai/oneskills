component:

  meta:
    name: UNetPredictionHeads
    alias: UNetHead
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: prediction_head
    author: OneScience
    license: Apache-2.0
    tags:
      - unet
      - prediction_head
      - 1x1_convolution
      - projection
      - pointwise_convolution

  concept:

    description: >
      该模块包含了 1D、2D 和 3D U-Net 架构的尾部预测头（Prediction Head）组件。
      它们负责将 U-Net 解码器（Decoder）最后一层输出的高维、深层抽象特征，映射为最终的任务目标（如图像分割掩码的类别通道、或流体力学中的速度场/压力场变量）。
      模块默认使用逐点卷积（Pointwise Convolution，即 $1 \times 1$ 或 $1 \times 1 \times 1$ 卷积）进行通道间的线性组合，确保在不破坏空间分辨率的前提下完成维度转换。

    intuition: >
      把 U-Net 的主干网络想象成一个极其复杂的特征提纯车间，数据在里面经过了下采样、上采样和多尺度的跳跃连接，最终变成了几百维的复杂表示。
      UNetHead 则是最后的“翻译官”。在图像分割任务中，它把 64 维的隐性特征“翻译”成 3 个维度的 RGB 或 $N$ 个类别的概率；
      在 AI for Science 求解 PDE 时，它把这些特征转换为网格点上具体的温度或速度数值。
      使用 $1 \times 1$ 卷积，就像是对每一个像素点/网格点独立进行了一次全连接（Fully Connected）投影。

    problem_it_solves:
      - 解决深度潜变量特征图到特定物理量或分类标签的最终维度映射问题
      - 为 1D (序列)、2D (平面场) 和 3D (体素场) 提供统一格式的输出投影层
      - 自动计算并应用 Padding，确保即使 `kernel_size > 1` 时，输出张量的空间尺寸也与输入保持严格一致

  theory:

    formula:

      head_projection:
        expression: y = \text{Conv}(x, \text{kernel\_size}=K, \text{padding}=\lfloor K / 2 \rfloor)

    variables:

      x:
        name: DecoderFeatures
        shape: dynamic
        description: U-Net 解码器最后一层的输出特征

      y:
        name: Predictions
        shape: dynamic
        description: 最终预测目标，其空间维度与输入严格对齐

      K:
        name: KernelSize
        description: 卷积核大小，默认为 1

  structure:

    architecture: linear_projection_head

    pipeline:

      - name: DimensionalityReduction
        operation: convolution_with_dynamic_padding

  interface:

    parameters:

      in_channels:
        type: int
        description: 输入通道数，通常对应 U-Net 解码器最后一次上采样后的基础通道数（base_channels）

      out_channels:
        type: int
        description: 输出通道数，取决于下游任务（如分类的类别数、回归的物理变量数）

      kernel_size:
        type: int
        default: 1
        description: 卷积核大小。如果设置为大于 1 的奇数（如 3），模块会自动添加 padding 以保持分辨率

    inputs:

      x:
        type: Tensor
        shape: "[B, in_channels, L] (1D) 或 [B, in_channels, H, W] (2D) 或 [B, in_channels, D, H, W] (3D)"
        dtype: float32
        description: 深层特征张量

    outputs:

      output:
        type: Tensor
        shape: 空间维度与输入完全相同，通道维度变为 out_channels
        description: 任务特定的预测结果张量

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      import torch.nn as nn

      class UNetHead1D(nn.Module):
          def __init__(self, in_channels, out_channels, kernel_size=1):
              super().__init__()
              padding = kernel_size // 2
              self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

          def forward(self, x):
              return self.conv(x)

      class UNetHead2D(nn.Module):
          def __init__(self, in_channels, out_channels, kernel_size=1):
              super().__init__()
              padding = kernel_size // 2
              self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

          def forward(self, x):
              return self.conv(x)

      class UNetHead3D(nn.Module):
          def __init__(self, in_channels, out_channels, kernel_size=1):
              super().__init__()
              padding = kernel_size // 2
              self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

          def forward(self, x):
              return self.conv(x)

  skills:

    build_unet_head:

      description: 根据数据的维度 (1D/2D/3D) 和任务目标构建 U-Net 预测头

      inputs:
        - dimension (1, 2, or 3)
        - in_channels
        - out_channels

      prompt_template: |
        构建一个 UNetHead{{dimension}}D 模块。
        输入通道为 {{in_channels}}，目标预测变量数量为 {{out_channels}}。

    diagnose_head_resolution_mismatch:

      description: 分析因修改核大小导致的空间维度缩水问题

      checks:
        - verify_odd_kernel_size_to_ensure_symmetric_padding
        - missing_padding_when_replacing_default_head

  knowledge:

    usage_patterns:

      fluid_dynamics_surrogate:

        pipeline:
          - UNetEncoder
          - UNetDecoder (输出如 32 维特征)
          - UNetHead2D (out_channels=3，分别预测 U速度, V速度, 压力P)
          - MSE Loss

    design_patterns:

      pointwise_projection:

        structure:
          - 使用 $1 \times 1$ ($1 \times 1 \times 1$) 卷积代替传统的全连接层（FC）。
          - $1 \times 1$ 卷积本质上是跨通道的线性组合，它在空间上是独立的（Pointwise），这意味着它能够完美保留 U-Net 辛苦恢复出来的逐像素/网格点的空间结构信息。

    hot_models:

      - model: U-Net
        year: 2015
        role: 医疗图像分割标准架构
        architecture: encoder_decoder
        attention_type: None

      - model: AI Fluid Solvers (Based on U-Net)
        year: 2020+
        role: 偏微分方程替代模型
        architecture: 3D_unet

    model_usage_details:

      PDE Surrogates:

        usage: 在预测物理场演化时，最后的 Head 直接输出物理场的增量（$\Delta x$）或下一时间步的绝对状态。由于没有激活函数，它可以拟合任意范围的实数值（回归任务）。

    best_practices:

      - 强烈建议保持 `kernel_size=1`。如果将其修改为 3 或更大，虽然代码中的 `padding = kernel_size // 2` 逻辑保证了输出尺寸不变，但额外的感受野融合会在输出端造成强烈的“平滑（Smoothing）”效应，这可能会导致物理场的激波边缘变模糊，或图像分割的边界不再锐利。
      - 当 `kernel_size` 大于 1 时，务必传入奇数（如 3, 5, 7）。如果传入偶数（如 2），`padding = 1` 会导致输出尺寸实际上变大，引发与 Ground Truth 张量尺寸不匹配的报错崩溃。
      - 如果是分类任务（如语义分割），在此 Head 输出后应连接 `CrossEntropyLoss`（其内部自带 Softmax）；如果是回归任务（物理量预测），则直接将结果送入 `MSELoss`。

    anti_patterns:

      - 试图在 Head 后面再接一个 `nn.ReLU()` 或其他非线性激活函数，这在回归任务中会截断所有的负值（如负的流体速度或温度梯度），导致模型彻底失效。

    paper_references:

      - title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        authors: Ronneberger et al.
        year: 2015

  graph:

    is_a:
      - NeuralNetworkComponent
      - PredictionHead
      - ConvolutionalLayer

    part_of:
      - UNet
      - VNet
      - PDESurrogateModel

    depends_on:
      - nn.Conv1d
      - nn.Conv2d
      - nn.Conv3d

    variants:
      - SegmentationHead (图像分割中通常带 Softmax)
      - RegressionHead (直接输出连续值)

    used_in_models:
      - U-Net
      - Fluid Dynamics Surrogates

    compatible_with:

      inputs:
        - Tensor

      outputs:
        - Tensor