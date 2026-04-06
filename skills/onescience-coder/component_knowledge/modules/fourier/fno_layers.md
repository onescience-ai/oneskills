component:

  meta:
    name: SpectralConvLayers
    alias: FNOLayers
    version: 1.0.0
    domain: deep_learning
    category: neural_operator
    subcategory: spectral_convolution
    author: OneScience
    license: Apache-2.0
    tags:
      - fourier
      - operator
      - convolution
      - fft
      - spatial_mixing

  concept:
    description: >
      标准的傅里叶神经算子（FNO）卷积层体系，包括一维、二维和三维的频域处理模块。
      通过快速傅里叶变换（FFT、FFT2、FFTN）将物理空间特征映射到频域，在此截断高频分量以起到正则化作用，
      并利用复数线性权重对保留的低频模态进行特征加权和混合，最后通过逆傅里叶变换映射回物理空间。

    intuition: >
      如果把普通的卷积看作是用一个“小放大镜”在图像上一点点滑动来提取局部特征，频域卷积则像是站在“全局视角”，
      直接调整整个图像或物理场的“基本波动频率”。它通过修改波形的振幅和相位（在频域与权重相乘）来实现感受野无限大且与分辨率无关的特征提取。

    problem_it_solves:
      - 突破传统CNN固定感受野和强分辨率依赖的限制
      - 能够在大规模连续物理场（如流体力学、气象场）中高效捕捉长程依赖（Long-range dependency）
      - 在Zero-shot Super-resolution（零样本超分辨率）任务中表现出优异泛化能力

  theory:
    formula:
      spectral_convolution:
        expression: |
          \mathcal{K}(a)(x) = \mathcal{F}^{-1}(R_{\phi} \cdot (\mathcal{F}a))
          x_{out} = \sigma(W x_{in} + \mathcal{K}(x_{in}))

    variables:
      \mathcal{F}:
        name: FourierTransform
        description: 快速傅里叶变换（如 FFT 算法）

      R_{\phi}:
        name: ComplexWeights
        shape: [in_channels, out_channels, modes]
        description: 在频域学习复数张量网络权重

      modes:
        name: ModesTrunction
        description: 截断频率的最大数量，起到低通滤波和降低计算复杂度的作用

  structure:
    architecture: fourier_neural_operator_layer

    pipeline:
      - name: ForwardFFT
        operation: torch.fft.rfft/rfft2/rfftn

      - name: LinearTransformFrequency
        operation: complex_tensor_multiplication

      - name: InverseFFT
        operation: torch.fft.irfft/irfft2/irfftn

  interface:
    parameters:
      in_channels:
        type: int
        description: 输入特征层通道数
      out_channels:
        type: int
        description: 输出特征层通道数
      modes1:
        type: int
        description: 在第一个空间维度上保留的频域模态数

    inputs:
      x:
        type: FieldTensor
        shape: [batch, in_channels, x, y, z]
        dtype: float32
        description: 物理空间输入的连续物理场特征张量

    outputs:
      out:
        type: FieldTensor
        shape: [batch, out_channels, x, y, z]
        dtype: float32
        description: 经过频域卷积变换后的输出，网格分辨率不变

  types:
    FieldTensor:
      shape: [batch, channels, spatial_dims]
      description: 连续物理场张量

  implementation:
    framework: pytorch
    code: |
      import torch.nn.functional as F
      import torch.nn as nn
      import torch
      import numpy as np
      import math
      
      ################################################################
      #  1d fourier layer
      ################################################################
      class SpectralConv1d(nn.Module):
          """
          一维傅里叶卷积层。
      
          该层通过快速傅里叶变换（FFT）、频域复数线性变换和逆变换实现全局卷积操作。
          其核心思想是在频域中对低频模态进行线性变换，从而有效地捕捉序列数据中的全局特征和长程依赖。
      
          Args:
              in_channels (int): 输入通道数
              out_channels (int): 输出通道数
              modes1 (int): 截断的傅里叶模态数量（频率分量数）。该层仅保留最低的 `modes1` 个频率分量进行计算。
                            注意：`modes1` 最多为输入长度的 `floor(N/2) + 1`。
      
          形状:
              输入 x: (B, Cin, L)，其中 B 是批量大小，Cin 是输入通道数，L 是序列长度
              输出: (B, Cout, L)，其中 Cout 是输出通道数
      
          Example:
              >>> # 假设序列长度 L=100，保留低频模态数 16
              >>> spec_conv1d = SpectralConv1d(in_channels=64, out_channels=32, modes1=16)
              >>> x = torch.randn(20, 64, 100)
              >>> out = spec_conv1d(x)
              >>> out.shape
              torch.Size([20, 32, 100])
          """
          def __init__(self, in_channels, out_channels, modes1):
              super(SpectralConv1d, self).__init__()
      
              """
              1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
              """
      
              self.in_channels = in_channels
              self.out_channels = out_channels
              self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
      
              self.scale = (1 / (in_channels * out_channels))
              self.weights1 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
      
          # Complex multiplication
          def compl_mul1d(self, input, weights):
              # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
              return torch.einsum("bix,iox->box", input, weights)
      
          def forward(self, x):
              batchsize = x.shape[0]
              # Compute Fourier coeffcients up to factor of e^(- something constant)
              x_ft = torch.fft.rfft(x)
      
              # Multiply relevant Fourier modes
              out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
              out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
      
              # Return to physical space
              x = torch.fft.irfft(out_ft, n=x.size(-1))
              return x
      
      
      ################################################################
      # 2d fourier layer
      ################################################################
      class SpectralConv2d(nn.Module):
          """
          二维傅里叶卷积层。
      
          该层实现了傅里叶神经算子中的二维谱卷积操作。它在二维频域内对输入进行处理。
          为了保持计算效率并捕捉主要特征，该层只对频域矩阵角上的低频分量进行线性变换（复数加权计算）。
          通常用于处理图像、流体切片或二维网格数据。
      
          Args:
              in_channels (int): 输入通道数
              out_channels (int): 输出通道数
              modes1 (int): 第一个空间维度（高度）上保留的傅里叶模态数量
              modes2 (int): 第二个空间维度（宽度）上保留的傅里叶模态数量
      
          形状:
              输入 x: (B, Cin, H, W)，其中 H 和 W 分别是空间维度的高度和宽度
              输出: (B, Cout, H, W)，输出的空间分辨率与输入保持一致
      
          Example:
              >>> # 假设输入为 64x64 的网格
              >>> spec_conv2d = SpectralConv2d(in_channels=32, out_channels=64, modes1=12, modes2=12)
              >>> x = torch.randn(10, 32, 64, 64)
              >>> out = spec_conv2d(x)
              >>> out.shape
              torch.Size([10, 64, 64, 64])
          """
          def __init__(self, in_channels, out_channels, modes1, modes2):
              super(SpectralConv2d, self).__init__()
              """
              2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
              """
              self.in_channels = in_channels
              self.out_channels = out_channels
              self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
              self.modes2 = modes2
      
              self.scale = (1 / (in_channels * out_channels))
              self.weights1 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
              self.weights2 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
      
          # Complex multiplication
          def compl_mul2d(self, input, weights):
              # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
              return torch.einsum("bixy,ioxy->boxy", input, weights)
      
          def forward(self, x):
              batchsize = x.shape[0]
              # Compute Fourier coeffcients up to factor of e^(- something constant)
              x_ft = torch.fft.rfft2(x)
      
              # Multiply relevant Fourier modes
              out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                                   device=x.device)
              out_ft[:, :, :self.modes1, :self.modes2] = \
                  self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
              out_ft[:, :, -self.modes1:, :self.modes2] = \
                  self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
      
              # Return to physical space
              x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
              return x
      
      
      ################################################################
      # 3d fourier layers
      ################################################################
      
      class SpectralConv3d(nn.Module):
          """
          三维傅里叶卷积层。
      
          针对三维体数据或时空数据（例如 2D 空间 + 1D 时间），在三维频域的四个角（低频区域）进行张量收缩计算。
          算法流程包括：对最后三个维度进行 rfftn，在三维频域的不同象限分别进行复数权重收缩，最后通过 irfftn 还原。
      
          Args:
              in_channels (int): 输入通道数
              out_channels (int): 输出通道数
              modes1 (int): 第一维度上保留的傅里叶模态数量
              modes2 (int): 第二维度上保留的傅里叶模态数量
              modes3 (int): 第三维度上保留的傅里叶模态数量
      
          形状:
              输入 x: (B, Cin, D, H, W)，通常对应 (Batch, Channel, X, Y, Z) 或 (Batch, Channel, X, Y, Time)
              输出: (B, Cout, D, H, W)，输出尺寸与输入尺寸相同
      
          Example:
              >>> # 假设输入尺寸为 32x32x32
              >>> spec_conv3d = SpectralConv3d(in_channels=4, out_channels=8, modes1=8, modes2=8, modes3=8)
              >>> x = torch.randn(2, 4, 32, 32, 32)
              >>> out = spec_conv3d(x)
              >>> out.shape
              torch.Size([2, 8, 32, 32, 32])
          """
          def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
              super(SpectralConv3d, self).__init__()
      
              """
              3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
              """
      
              self.in_channels = in_channels
              self.out_channels = out_channels
              self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
              self.modes2 = modes2
              self.modes3 = modes3
      
              self.scale = (1 / (in_channels * out_channels))
              self.weights1 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                          dtype=torch.cfloat))
              self.weights2 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                          dtype=torch.cfloat))
              self.weights3 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                          dtype=torch.cfloat))
              self.weights4 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                          dtype=torch.cfloat))
      
          # Complex multiplication
          def compl_mul3d(self, input, weights):
              # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
              return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
      
          def forward(self, x):
              batchsize = x.shape[0]
              # Compute Fourier coeffcients up to factor of e^(- something constant)
              x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
      
              # Multiply relevant Fourier modes
              out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                                   dtype=torch.cfloat, device=x.device)
              out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
                  self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
              out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
                  self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
              out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
                  self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
              out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
                  self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
      
              # Return to physical space
              x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
              return x
      

  skills:
    build_fno_layer:
      description: 实例化 1D, 2D 或 3D 谱卷积网络层
      inputs:
        - in_channels
        - out_channels
        - modes
      prompt_template: |
        构造 SpectralConv 层，使用 FFT 和设置频域模态截断。
        参数：modes = {{modes}}

    diagnose_spectral_leakage:
      description: 诊断傅里叶谱卷积是否造成别名效应（Aliasing）
      checks:
        - excessive_truncation_loss
        - fft_normalization_issues

  knowledge:
    usage_patterns:
      fno_block:
        pipeline:
          - Linear (Local Convolution) -> Parallel
          - SpectralConv (Global Mixing) -> Parallel
          - Add -> Activation -> Next Block

    hot_models:
      - model: FNO (Fourier Neural Operator)
        year: 2020
        role: 第一个能够直接求解无参数 PDE 的神经网络结构
      - model: FourCastNet
        year: 2022
        role: 高分辨率全球气象预报模型基石

    best_practices:
      - modes 设置不应超过网格分辨率对应维度大小的一半截断 (floor(N/2)+1)
      - Spectral Conv 通常与一个具有同样宽度的局域操作并联使用

    anti_patterns:
      - 将模态数量设置过高，从而引发严重的过拟合

    paper_references:
      - title: "Fourier Neural Operator for Parametric Partial Differential Equations"
        authors: Li et al.
        year: 2020

  graph:
    is_a:
      - NeuralNetworkComponent
      - IntegralTransformLayer
    part_of:
      - FourierNeuralOperator
      - WeatherForecastingModel
    depends_on:
      - torch.fft
      - einsum
    variants:
      - FactorizedFourierLayer
    used_in_models:
      - FourCastNet
    compatible_with:
      inputs:
        - FieldTensor
      outputs:
        - FieldTensor
