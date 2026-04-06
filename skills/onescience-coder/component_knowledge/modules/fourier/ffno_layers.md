component:

  meta:
    name: FactorizedSpectralConvLayers
    alias: FFNOLayers
    version: 1.0.0
    domain: deep_learning
    category: neural_operator
    subcategory: factorized_spectral_convolution
    author: OneScience
    license: Apache-2.0
    tags:
      - fourier
      - operator
      - factorized
      - fft
      - parameter_efficient

  concept:
    description: >
      因式分解傅里叶神经算子（FFNO）卷积层，将传统 FNO 中多维傅里叶变换与昂贵的复数乘法分解为在各个空间轴（X, Y, Z）独立的 1D 傅里叶卷积叠加。

    intuition: >
      与其在二维频域上面进行全网关联叠加，不如把 x 轴和 y 轴的信号分别使用 1D FFT 分解后独立建模再在空间中合并，类似张量低秩分解，大幅度降低复杂度。

    problem_it_solves:
      - 缓解标准高维 FNO 的严重参数冗余和显存占用问题
      - 提高超高分辨率气候、3D流体数据的训练可行性

  theory:
    formula:
      factorized_convolution_2d:
        expression: |
          \mathcal{K}_x(a) = \mathcal{F}^{-1}_x(R_{\phi_x} \cdot (\mathcal{F}_x a))
          \mathcal{K}_y(a) = \mathcal{F}^{-1}_y(R_{\phi_y} \cdot (\mathcal{F}_y a))
          x_{out} = \mathcal{K}_x(x_{in}) + \mathcal{K}_y(x_{in})

    variables:
      \mathcal{F}_x:
        name: Separate1DFFT
        description: 沿相应维度独立进行傅里叶变换

      R_{\phi_x}:
        name: AxisSpecificWeights
        shape: [dim, dim, modes, 2]
        description: 分离后的极低秩频域一维复数参数矩阵

  structure:
    architecture: factorized_fourier_operation

    pipeline:
      - name: AxisSeparateFFT
        operation: torch.fft.rfft(..., dim=-1/-2)
      - name: AxisSeparateModulation
        operation: complex_activation_along_axis
      - name: Aggregation
        operation: addition

  interface:
    parameters:
      in_dim:
        type: int
        description: 输入频带通道
      out_dim:
        type: int
        description: 输出频带通道
      modes_x:
        type: int
        description: X方向预留的低频模式数
      modes_y:
        type: int
        description: Y方向预留的低频模式数

    inputs:
      x:
        type: GridFeature
        shape: [batch, in_dim, M, N]
        dtype: float32
        description: 时空流场输入数据

    outputs:
      out:
        type: GridFeature
        shape: [batch, out_dim, M, N]
        dtype: float32
        description: 参数解耦变换后的空间融合特征

  types:
    GridFeature:
      shape: [batch, channels, spatial_dim_1, spatial_dim_2]
      description: 规则空间坐标张量

  implementation:
    framework: pytorch
    code: |
      import torch.nn.functional as F
      import torch.nn as nn
      import torch
      import numpy as np
      import math
      
      ################################################################
      # 1d fourier layer
      ################################################################
      class SpectralConv1d(nn.Module):
          """
          一维分解傅里叶卷积层 (Factorized Fourier Layer)。
      
          
      
          该层通过一维快速傅里叶变换 (FFT)、频域复数线性变换和逆变换实现卷积操作。
          在 1D 情况下，因子化 FNO (FFNO) 退化为标准的 FNO。其核心思想是在频域中对低频模态进行线性变换，
          从而有效地捕捉序列数据中的全局特征和长程依赖。
      
          Args:
              in_channels (int): 输入通道数
              out_channels (int): 输出通道数
              modes1 (int): 截断的傅里叶模态数量（保留的低频分量数），最多为 floor(L/2) + 1
      
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
          ## FFNO degenerate to FNO in 1D space
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
          二维分解傅里叶卷积层 (Factorized Fourier Layer)。
      
          不同于标准的二维 FNO，该层采用分解策略：分别在 X 维度和 Y 维度上独立进行 1D FFT 操作、频域加权和逆变换，
          最后将两个维度的处理结果相加 (x = x_out_x + x_out_y)。这种方法通常能减少参数量并提高计算效率，
          同时保留了捕捉全局特征的能力。
      
          Args:
              in_dim (int): 输入通道数
              out_dim (int): 输出通道数
              modes_x (int): X 维度（dim=-2）上保留的傅里叶模态数量
              modes_y (int): Y 维度（dim=-1）上保留的傅里叶模态数量
      
          形状:
              输入 x: (B, Cin, H, W)，其中 H 对应 X 维度，W 对应 Y 维度
              输出: (B, Cout, H, W)，输出的空间分辨率与输入保持一致
      
          Example:
              >>> # 假设输入分辨率为 64x64
              >>> spec_conv2d = SpectralConv2d(in_dim=32, out_dim=64, modes_x=12, modes_y=12)
              >>> x = torch.randn(10, 32, 64, 64)
              >>> out = spec_conv2d(x)
              >>> out.shape
              torch.Size([10, 64, 64, 64])
          """
          def __init__(self, in_dim, out_dim, modes_x, modes_y):
              super().__init__()
              self.in_dim = in_dim
              self.out_dim = out_dim
              self.modes_x = modes_x
              self.modes_y = modes_y
      
              self.fourier_weight = nn.ParameterList([])
              for n_modes in [modes_x, modes_y]:
                  weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                  param = nn.Parameter(weight)
                  nn.init.xavier_normal_(param)
                  self.fourier_weight.append(param)
      
          def forward(self, x):
              B, I, M, N = x.shape
      
              # # # Dimesion Y # # #
              x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
              # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]
      
              out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
              # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]
      
              out_ft[:, :, :, :self.modes_y] = torch.einsum(
                  "bixy,ioy->boxy",
                  x_fty[:, :, :, :self.modes_y],
                  torch.view_as_complex(self.fourier_weight[1]))
      
              xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
              # x.shape == [batch_size, in_dim, grid_size, grid_size]
      
              # # # Dimesion X # # #
              x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
              # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]
      
              out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
              # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]
      
              out_ft[:, :, :self.modes_x, :] = torch.einsum(
                  "bixy,iox->boxy",
                  x_ftx[:, :, :self.modes_x, :],
                  torch.view_as_complex(self.fourier_weight[0]))
      
              xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
              # x.shape == [batch_size, in_dim, grid_size, grid_size]
      
              # # Combining Dimensions # #
              x = xx + xy
      
              return x
      
      
      ################################################################
      # 3d fourier layers
      ################################################################
      
      class SpectralConv3d(nn.Module):
          """
          三维分解傅里叶卷积层 (Factorized Fourier Layer)。
      
          该层针对三维数据（或 2D+Time 数据），采用分解策略：分别在 X、Y、Z 三个维度上独立执行谱卷积。
          具体流程为：对每个维度分别进行 RFFT、复数权重收缩、IRFFT，最终将三个分支的输出结果叠加
          (x = x_out_x + x_out_y + x_out_z)。这种设计避免了全维 3D FFT 的高计算成本。
      
          Args:
              in_dim (int): 输入通道数
              out_dim (int): 输出通道数
              modes_x (int): X 维度（dim=-3）上保留的傅里叶模态数量
              modes_y (int): Y 维度（dim=-2）上保留的傅里叶模态数量
              modes_z (int): Z 维度（dim=-1）上保留的傅里叶模态数量
      
          形状:
              输入 x: (B, Cin, D, H, W)，也就是 (Batch, Channel, X, Y, Z)
              输出: (B, Cout, D, H, W)，输出尺寸与输入尺寸相同
      
          Example:
              >>> # 假设输入尺寸为 32x32x32
              >>> spec_conv3d = SpectralConv3d(in_dim=4, out_dim=8, modes_x=8, modes_y=8, modes_z=8)
              >>> x = torch.randn(2, 4, 32, 32, 32)
              >>> out = spec_conv3d(x)
              >>> out.shape
              torch.Size([2, 8, 32, 32, 32])
          """
          def __init__(self, in_dim, out_dim, modes_x, modes_y, modes_z):
              super().__init__()
              self.in_dim = in_dim
              self.out_dim = out_dim
              self.modes_x = modes_x
              self.modes_y = modes_y
              self.modes_z = modes_z
      
              self.fourier_weight = nn.ParameterList([])
              for n_modes in [modes_x, modes_y, modes_z]:
                  weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                  param = nn.Parameter(weight)
                  nn.init.xavier_normal_(param)
                  self.fourier_weight.append(param)
      
          def forward(self, x):
              B, I, S1, S2, S3 = x.shape
      
              # # # Dimesion Z # # #
              x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
              # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]
      
              out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
              # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]
      
              out_ft[:, :, :, :, :self.modes_z] = torch.einsum(
                  "bixyz,ioz->boxyz",
                  x_ftz[:, :, :, :, :self.modes_z],
                  torch.view_as_complex(self.fourier_weight[2]))
      
              xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm='ortho')
              # x.shape == [batch_size, in_dim, grid_size, grid_size]
      
              # # # Dimesion Y # # #
              x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
              # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]
      
              out_ft = x_fty.new_zeros(B, I, S1, S2 // 2 + 1, S3)
              # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]
      
              out_ft[:, :, :, :self.modes_y, :] = torch.einsum(
                  "bixyz,ioy->boxyz",
                  x_fty[:, :, :, :self.modes_y, :],
                  torch.view_as_complex(self.fourier_weight[1]))
      
              xy = torch.fft.irfft(out_ft, n=S2, dim=-2, norm='ortho')
              # x.shape == [batch_size, in_dim, grid_size, grid_size]
      
              # # # Dimesion X # # #
              x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
              # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]
      
              out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2, S3)
              # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]
      
              out_ft[:, :, :self.modes_x, :, :] = torch.einsum(
                  "bixyz,iox->boxyz",
                  x_ftx[:, :, :self.modes_x, :, :],
                  torch.view_as_complex(self.fourier_weight[0]))
      
              xx = torch.fft.irfft(out_ft, n=S1, dim=-3, norm='ortho')
              # x.shape == [batch_size, in_dim, grid_size, grid_size]
      
              # # Combining Dimensions # #
              x = xx + xy + xz
      
              return x
      

  skills:
    build_factorized_fno:
      description: 构建低参 FFNO 层
      inputs:
        - in_dim
        - out_dim
        - modes_list
      prompt_template: |
        参数高效分离构建 Factorized FNO: {in_dim}, {out_dim}

    diagnose_factorization_error:
      description: 排查 FFT 解耦过程对齐错误
      checks:
        - cross_axis_information_loss
        - padding_misalignments

  knowledge:
    usage_patterns:
      ffno_replacement:
        pipeline:
          - 取代原始极高显存的 SpectralConvND，切换至 Factorized 版

    hot_models:
      - model: F-FNO (Factorized FNO)
        year: 2023
        role: 处理千万参数级别分辨率大模型预报

    best_practices:
      - "为了串行保证能量稳定，做FFT时必须填入 `norm='ortho'`"

    anti_patterns:
      - "不必要地在对角依赖极强的情况下完全采用解耦 FNO"

    paper_references:
      - title: "Towards Multi-Spatiotemporal-Scale Generalized PDE Modeling and Zero-Shot Transfer"
        authors: Unknown
        year: 2023

  graph:
    is_a:
      - NeuralNetworkComponent
      - FactorizedMixing
    part_of:
      - LargeScalePDEModels
    depends_on:
      - torch.fft
    variants:
      - SequentialFFNO
    used_in_models:
      - FFNO Solvers
    compatible_with:
      inputs:
        - GridFeature
      outputs:
        - GridFeature
