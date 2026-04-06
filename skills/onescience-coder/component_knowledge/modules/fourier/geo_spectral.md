component:

  meta:
    name: GeometryAwareSpectralConv
    alias: GeoSpectral
    version: 1.0.0
    domain: physics_informed_machine_learning
    category: neural_operator
    subcategory: irregular_geometry_spectral
    author: OneScience
    license: Apache-2.0
    tags:
      - fourier
      - operator
      - geometry
      - PointClouds
      - mapping

  concept:
    description: >
      Geo-FNO 的核心模块，利用变形可学映射网络（IPHI）将非结构化坐标向规则均匀隐空间扭曲投影，并在该空间内实施基于显式直接积分傅里叶求和的高成本距离网络感知谱卷积，突破传统 FNO 对网格整齐的严苛制约。

    intuition: >
      给复杂的几何流域敷设一层橡皮泥拉伸隐射层（IPHI）。通过把各种奇怪机翼边缘展平处理再利用点对点指数复数直接乘法交互取代 FFT，能够做到对任意点云输入的无缝衔接处理。

    problem_it_solves:
      - 解决 FNO 在不规则 PDE 网格上束手无策的问题
      - 保留了谱分解的全局感知视野

  theory:
    formula:
      inverse_mapping:
        expression: |
          \xi = \phi^{-1}(x) = x + \text{Deformation}(x)

      discrete_fourier_summation:
        expression: |
          X_k = \sum_{n=0}^{N-1} x_n e^{-i 2\pi k \xi_n} 

    variables:
      x:
        name: PhysicalCoordinates
        description: 原始物理变形域坐标
      \xi:
        name: ComputationalCoordinates
        description: 通过神经网络拉回的标准方形或均匀平层隐匿集坐标

  structure:
    architecture: explicit_geometry_fourier_operation

    pipeline:
      - name: CoordinateTransformation
        operation: IPHI_Network_Transform
      - name: ExplicitDiscreteFourierTransform
        operation: compute_exponential_kernels_by_point_distances
      - name: LowRankMixing
        operation: complex_weight_multiplication

  interface:
    parameters:
      in_channels:
        type: int
        description: 输入流信道
      out_channels:
        type: int
        description: 输出融合表达通道
      modes1:
        type: int
        description: 最大傅里叶模态保留

    inputs:
      u:
        type: UnstructuredFeature
        shape: [batch, C, N]
        dtype: float32
        description: 随意散布点阵集合的特征信息
      x_in:
        type: Coordinates
        shape: [batch, N, 2]
        dtype: float32
      iphi:
        type: CallableNetwork
        description: 用于坐标回归转换的逆向函数本体

    outputs:
      out:
        type: UnstructuredFeature
        shape: [batch, out_channels, s1, s2]
        dtype: float32
        description: 被提取出潜层标准几何表征的结果场

  types:
    UnstructuredFeature:
      shape: [batch, channels, num_points]
      description: 不规则向量表达

  implementation:
    framework: pytorch
    code: |
      import torch
      import torch.nn as nn
      import torch.fft
      import numpy as np
      from typing import Optional, Callable
      
      # ==========================================
      # 坐标变换网络 (Coordinate Transformation)
      # ==========================================
      
      class IPHI(nn.Module):
          """
              逆映射网络 (Inverse Phi Mapping Network)。
      
              该模块用于学习从物理坐标 x$到计算坐标 xi 的逆映射。
              它结合了极坐标特征工程和 NeRF 风格的傅里叶特征编码，以捕捉高频几何细节。
              通常作为 `GeoSpectralConv` 的 `iphi` 参数传入，用于处理不规则几何。
      
              Args:
                  width (int): 网络隐藏层的宽度。默认值: 32。
      
              形状:
                  输入 x: (B, N, 2)，物理空间坐标。
                  输入 code (可选): (B, N_features)，条件编码（如几何参数、物理参数）。
                  输出: (B, N, 2)，变换后的坐标（残差形式）。
      
              Example:
                  >>> iphi_net = IPHI(width=32).cuda()
                  >>> x = torch.rand(2, 1024, 2).cuda() # Batch=2, Points=1024, Dim=2
                  >>> # 简单的无条件映射
                  >>> x_transformed = iphi_net(x)
                  >>> print(x_transformed.shape)
                  torch.Size([2, 1024, 2])
          """
          def __init__(self, width=32):
              super(IPHI, self).__init__()
              self.width = width
              
              # 特征提取层
              self.fc0 = nn.Linear(4, self.width)
              
              # 条件编码层
              self.fc_code = nn.Linear(42, self.width)
              
              # 融合层
              self.fc_no_code = nn.Linear(3 * self.width, 4 * self.width)
              
              # MLP 主体
              self.fc1 = nn.Linear(4 * self.width, 4 * self.width)
              self.fc2 = nn.Linear(4 * self.width, 4 * self.width)
              self.fc3 = nn.Linear(4 * self.width, 4 * self.width)
              
              # 输出层
              self.fc4 = nn.Linear(4 * self.width, 2)
              
              self.activation = torch.tanh
              
              # 几何中心 (用于计算极坐标)
              self.register_buffer('center', torch.tensor([0.0001, 0.0001]).reshape(1, 1, 2))
      
              # 傅里叶特征频率矩阵 B
              freq_bands = torch.pow(2, torch.arange(0, self.width // 4, dtype=torch.float)) * np.pi
              self.register_buffer('B', freq_bands.reshape(1, 1, 1, self.width // 4))
      
          def forward(self, x, code=None):
              # x: (Batch, N_grid, 2)
              
              # 特征工程: 笛卡尔坐标 -> 极坐标增强
              angle = torch.atan2(x[..., 1] - self.center[..., 1], x[..., 0] - self.center[..., 0])
              radius = torch.norm(x - self.center, dim=-1, p=2)
              xd = torch.stack([x[..., 0], x[..., 1], angle, radius], dim=-1)
      
              # 傅里叶特征编码 (Fourier Features)
              b, n, d = xd.shape
      
              # 计算 sin/cos 特征
              scaled_x = xd.unsqueeze(-1) * self.B
              x_sin = torch.sin(scaled_x).view(b, n, -1) # (B, N, width)
              x_cos = torch.cos(scaled_x).view(b, n, -1) # (B, N, width)
              
              # 基础特征投影
              xd_feat = self.fc0(xd) # (B, N, width)
              
              # 拼接所有特征: [Base, Sin, Cos] -> (B, N, 3*width)
              xd_combined = torch.cat([xd_feat, x_sin, x_cos], dim=-1)
      
              # 条件融合
              if code is not None:
                  cd = self.fc_code(code) # (B, width)
                  cd = cd.unsqueeze(1).repeat(1, n, 1) # (B, N, width)
                  xd = torch.cat([cd, xd_combined], dim=-1)
              else:
                  xd = self.fc_no_code(xd_combined)
      
              xd = self.activation(self.fc1(xd))
              xd = self.activation(self.fc2(xd))
              xd = self.activation(self.fc3(xd))
              
              deformation = self.fc4(xd)
      
              return x + x * deformation
      
      
      # ==========================================
      # 几何感知谱卷积 (Geometry-Aware Spectral Conv)
      # ==========================================
      
      class GeoSpectralConv2d(nn.Module):
          r"""
          几何感知 2D 谱卷积层 (Geometry-Aware Spectral Conv 2D). 
      
          该模块实现了 Geo-FNO 的核心逻辑。与标准 FNO 不同，它不依赖标准 FFT 算法，而是通过显式构造
          DFT 矩阵来处理**不规则网格**或**变形坐标**。
          它支持将物理空间的坐标 $x$ 通过映射 $\phi$ (iphi) 变换到计算空间，在此空间进行谱卷积操作。
      
          **注意**: 由于显式计算 DFT 矩阵，显存占用为 $O(N \cdot K^2)$ (N为点数，K为模式数)，
          适用于点数不是极度庞大的场景。
      
          Args:
              in_channels (int): 输入通道数。
              out_channels (int): 输出通道数。
              modes1 (int): 第一个空间维度的傅里叶模式数。
              modes2 (int): 第二个空间维度的傅里叶模式数。
              s1 (int, optional): 输出网格大小（仅当 x_out 为 None 时使用）。默认值: 32。
              s2 (int, optional): 输出网格大小（仅当 x_out 为 None 时使用）。默认值: 32。
      
          形状:
              输入 u: (B, C_in, N) 或 (B, C_in, H, W)。如果输入是不规则点云，应为 (B, C_in, N)。
              输入 x_in (可选): (B, N, 2)，输入点的坐标。
              输入 x_out (可选): (B, M, 2)，输出点的坐标。如果为 None，则输出到规则网格。
              输入 iphi (可选): 坐标变换函数，$x \to \xi$。通常传入 IPHI 类的实例。
              输入 code (可选): 传入 iphi 的辅助编码。
              输出: (B, C_out, M) 或 (B, C_out, s1, s2)。
      
          Example:
              >>> # 1. 定义坐标变换网络
              >>> iphi = IPHI(width=32).cuda()
              >>> # 2. 定义 Geo 谱卷积
              >>> geo_conv = GeoSpectralConv2d(32, 64, 12, 12).cuda()
              >>> # 3. 输入数据
              >>> u = torch.randn(2, 32, 1024).cuda()
              >>> x_in = torch.rand(2, 1024, 2).cuda()
              >>> # 4. 前向传播 (带坐标变换)
              >>> out = geo_conv(u, x_in=x_in, iphi=iphi)
          """
          def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
              super(GeoSpectralConv2d, self).__init__()
              self.in_channels = in_channels
              self.out_channels = out_channels
              self.modes1 = modes1
              self.modes2 = modes2
              self.s1 = s1
              self.s2 = s2
      
              self.scale = (1 / (in_channels * out_channels))
              # 权重初始化
              self.weights1 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
              self.weights2 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
      
          def compl_mul2d(self, input, weights):
              # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
              return torch.einsum("bixy,ioxy->boxy", input, weights)
      
          def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
              batchsize = u.shape[0]
      
              # 前向傅里叶变换 (Forward Transform)
              if x_in is None:
                  # 规则网格：使用标准 FFT
                  u_ft = torch.fft.rfft2(u)
                  s1, s2 = u.size(-2), u.size(-1)
              else:
                  # 不规则网格：使用显式 DFT
                  u_ft = self.fft2d(u, x_in, iphi, code)
                  s1, s2 = self.s1, self.s2
      
              # 谱卷积 (Spectral Convolution)
              factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
              factor2 = self.compl_mul2d(u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
      
              # 逆傅里叶变换 (Inverse Transform)
              if x_out is None:
                  # 输出到规则网格
                  out_ft = torch.zeros(batchsize, self.out_channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
                  out_ft[:, :, :self.modes1, :self.modes2] = factor1
                  out_ft[:, :, -self.modes1:, :self.modes2] = factor2
                  u = torch.fft.irfft2(out_ft, s=(s1, s2))
              else:
                  # 输出到不规则网格
                  # 拼接模式以准备 IDFT
                  out_ft = torch.cat([factor1, factor2], dim=-2)
                  u = self.ifft2d(out_ft, x_out, iphi, code)
      
              return u
      
          def _get_wavenumbers(self, device):
              # 辅助函数：生成波数 k_x, k_y
              k_x1 = torch.cat((torch.arange(0, self.modes1), 
                                torch.arange(-self.modes1, 0)), dim=0).to(device) # m1
              k_x2 = torch.cat((torch.arange(0, self.modes2), 
                                torch.arange(-(self.modes2 - 1), 0)), dim=0).to(device) # m2
              return k_x1, k_x2
      
          def fft2d(self, u, x_in, iphi=None, code=None):
              batchsize, N, _ = x_in.shape
              device = x_in.device
              
              # 坐标变换
              x = iphi(x_in, code) if iphi is not None else x_in
      
              # 生成波数网格
              k_x1, k_x2 = self._get_wavenumbers(device)
              
              # 计算相位 K = <x, k>
              # x: (B, N, 2), k: (m1, m2)
              K = torch.einsum('bni,mi->bnm', x[..., 0:1], k_x1.view(-1, 1))[:, :, :, None] + \
                  torch.einsum('bni,mi->bnm', x[..., 1:2], k_x2.view(-1, 1))[:, :, None, :]
                  
              # 傅里叶基
              basis = torch.exp(-1j * 2 * np.pi * K)
      
              # 变换: Y = u * basis
              u = u.to(torch.cfloat)
              Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
              return Y
      
          def ifft2d(self, u_ft, x_out, iphi=None, code=None):
              # u_ft: (B, C, m1, m2)
              # x_out: (B, N, 2)
              batchsize, N, _ = x_out.shape
              device = x_out.device
      
              x = iphi(x_out, code) if iphi is not None else x_out
      
              k_x1, k_x2 = self._get_wavenumbers(device)
      
              # K: (B, N, m1, m2)
              K = torch.einsum('bni,mi->bnm', x[..., 0:1], k_x1.view(-1, 1))[:, :, :, None] + \
                  torch.einsum('bni,mi->bnm', x[..., 1:2], k_x2.view(-1, 1))[:, :, None, :]
      
              basis = torch.exp(1j * 2 * np.pi * K)
      
              # 扩展频谱以处理实数信号 (Hermitian Symmetry)
              u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
              u_ft_full = torch.cat([u_ft, u_ft2], dim=-1)
      
              # 逆变换
              Y = torch.einsum("bcxy,bnxy->bcn", u_ft_full, basis)
              return Y.real
      
      
      class GeoSpectralConv3d(nn.Module):
          """
          几何感知 3D 谱卷积层 (Geometry-Aware Spectral Conv 3D).
      
          GeoSpectralConv2d 的 3D 扩展版本，适用于处理不规则 3D 几何体（如 3D 网格、点云、流体体积）上的 PDE 求解。
      
          Args:
              in_channels (int): 输入通道数。
              out_channels (int): 输出通道数。
              modes1 (int): X 轴傅里叶模式数。
              modes2 (int): Y 轴傅里叶模式数。
              modes3 (int): Z 轴傅里叶模式数。
              s1, s2, s3 (int, optional): 输出网格大小（默认 32）。
      
          形状:
              输入 u: (B, C_in, N).
              输入 x_in: (B, N, 3).
              输入 x_out: (B, M, 3).
              输出: (B, C_out, M).
      
          Example:
              >>> geo_conv3d = GeoSpectralConv3d(32, 64, 8, 8, 8).cuda()
              >>> u = torch.randn(2, 32, 5000).cuda()
              >>> x = torch.rand(2, 5000, 3).cuda()
              >>> out = geo_conv3d(u, x_in=x, x_out=x)
          """
          def __init__(self, in_channels, out_channels, modes1, modes2, modes3, s1=32, s2=32, s3=32):
              super(GeoSpectralConv3d, self).__init__()
              self.in_channels = in_channels
              self.out_channels = out_channels
              self.modes1 = modes1
              self.modes2 = modes2
              self.modes3 = modes3
              self.s1 = s1
              self.s2 = s2
              self.s3 = s3
      
              self.scale = (1 / (in_channels * out_channels))
              # 权重维度增加到 5D: (C_in, C_out, m1, m2, m3)
              self.weights1 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
              self.weights2 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
              self.weights3 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
              self.weights4 = nn.Parameter(
                  self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
      
          def compl_mul3d(self, input, weights):
              # (B, C_in, x, y, z), (C_in, C_out, x, y, z) -> (B, C_out, x, y, z)
              return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
      
          def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
              batchsize = u.shape[0]
      
              # Forward Transform
              if x_in is None:
                  u_ft = torch.fft.rfft3(u)
                  s1, s2, s3 = u.size(-3), u.size(-2), u.size(-1)
              else:
                  u_ft = self.fft3d(u, x_in, iphi, code)
                  s1, s2, s3 = self.s1, self.s2, self.s3
      
              # Spectral Conv (Corner Modes Processing)
              factor1 = self.compl_mul3d(u_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
              factor2 = self.compl_mul3d(u_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
              factor3 = self.compl_mul3d(u_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
              factor4 = self.compl_mul3d(u_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
      
              # Inverse Transform
              if x_out is None:
                  out_ft = torch.zeros(batchsize, self.out_channels, s1, s2, s3 // 2 + 1, dtype=torch.cfloat, device=u.device)
                  out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = factor1
                  out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = factor2
                  out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = factor3
                  out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = factor4
                  u = torch.fft.irfft3(out_ft, s=(s1, s2, s3))
              else:
                  out_ft_top = torch.cat([factor1, factor2], dim=-3) 
                  out_ft_bot = torch.cat([factor3, factor4], dim=-3)
                  out_ft = torch.cat([out_ft_top, out_ft_bot], dim=-2) 
                  
                  u = self.ifft3d(out_ft, x_out, iphi, code)
      
              return u
      
          def _get_wavenumbers(self, device):
              k_x1 = torch.cat((torch.arange(0, self.modes1), torch.arange(-self.modes1, 0)), 0).to(device)
              k_x2 = torch.cat((torch.arange(0, self.modes2), torch.arange(-self.modes2, 0)), 0).to(device)
              k_x3 = torch.cat((torch.arange(0, self.modes3), torch.arange(-(self.modes3 - 1), 0)), 0).to(device)
              return k_x1, k_x2, k_x3
      
          def fft3d(self, u, x_in, iphi=None, code=None):
              # x_in: (B, N, 3)
              batchsize, N, _ = x_in.shape
              device = x_in.device
              x = iphi(x_in, code) if iphi is not None else x_in
              
              k_x1, k_x2, k_x3 = self._get_wavenumbers(device)
              
              K = torch.einsum('bni,mi->bnm', x[..., 0:1], k_x1.view(-1, 1))[:, :, :, None, None] + \
                  torch.einsum('bni,mi->bnm', x[..., 1:2], k_x2.view(-1, 1))[:, :, None, :, None] + \
                  torch.einsum('bni,mi->bnm', x[..., 2:3], k_x3.view(-1, 1))[:, :, None, None, :]
      
              basis = torch.exp(-1j * 2 * np.pi * K)
              u = u.to(torch.cfloat)
              Y = torch.einsum("bcn,bnxyz->bcxyz", u, basis)
              return Y
      
          def ifft3d(self, u_ft, x_out, iphi=None, code=None):
              batchsize, N, _ = x_out.shape
              device = x_out.device
              x = iphi(x_out, code) if iphi is not None else x_out
              
              k_x1, k_x2, k_x3 = self._get_wavenumbers(device)
              
              K = torch.einsum('bni,mi->bnm', x[..., 0:1], k_x1.view(-1, 1))[:, :, :, None, None] + \
                  torch.einsum('bni,mi->bnm', x[..., 1:2], k_x2.view(-1, 1))[:, :, None, :, None] + \
                  torch.einsum('bni,mi->bnm', x[..., 2:3], k_x3.view(-1, 1))[:, :, None, None, :]
      
              basis = torch.exp(1j * 2 * np.pi * K)
      
              # 3D Hermitian Symmetry Flip
              u_ft2 = u_ft[..., 1:].flip(-1, -2, -3).conj()
              u_ft_full = torch.cat([u_ft, u_ft2], dim=-1)
      
              Y = torch.einsum("bcxyz,bnxyz->bcn", u_ft_full, basis)
              return Y.real

  skills:
    build_geometry_aware_model:
      description: 构造包含非规律物理网格处理能效的环境
      inputs:
        - in_channels
        - iphi_width
      prompt_template: |
        按照 Geo-FNO 原理搭建带映射回归的偏微分模型架构。

    diagnose_memory_blowup:
      description: 防范显示傅里叶变换产生可怕的 O(N * M^2) 时空占用过载灾难
      checks:
        - out_of_memory_dense_einsum

  knowledge:
    usage_patterns:
      geofno_end_to_end:
        pipeline:
          - Extract Mesh nodes
          - transform = IPHI(nodes)
          - Forward(Features, transform)
        
    hot_models:
      - model: Geo-FNO
        year: 2022
        role: 解决了传统FNO最大的缺陷（网格单一性）

    best_practices:
      - "控制总体采样点数 N，防止巨大的显式密集相乘毁掉训练过程及计算设备。"

    anti_patterns:
      - "本身就是标准的四方阵列强行调用它退回到显式谱展开方法计算，拖慢成败倍时长。"

    paper_references:
      - title: "Geo-FNO: Geometry-Aware Fourier Neural Operator for Point Clouds"
        authors: Li et al.
        year: 2022

  graph:
    is_a:
      - SpectralOperator
      - PointCloudProcessor
    part_of:
      - PDE_Solver
    depends_on:
      - MultiLayerPerceptron
    variants:
      - NUFFT_FNO
    used_in_models:
      - GeoFNO
    compatible_with:
      inputs:
        - UnstructuredFeature
      outputs:
        - StructuredGrid