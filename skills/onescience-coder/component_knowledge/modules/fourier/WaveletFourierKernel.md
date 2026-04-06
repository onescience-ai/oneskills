component:

  meta:
    name: WaveletFourierKernel
    alias: MultiwaveletFourierSparseKernel
    version: 1.0
    domain: physics_informed_machine_learning
    category: neural_network
    subcategory: spectral_operator
    author: OneScience
    tags:
      - multi-wavelet
      - fourier_transform
      - sparse_kernel
      - spectral_convolution

  concept:

    description: >
      WaveletFourierKernel 构成了多小波变换（Multi-Wavelet Transform）的核心频域处理模块。
      该模块在由离散多小波变换得到的隐空间特征上进行操作，通过快速傅里叶变换（FFT）将信号从物理/小波空间转换到频域。
      在频域中，算子截断高频分量（由参数 `alpha`/`modes` 控制），仅保留在物理过程中占主导地位的低频模态，并对保留的频域特征
      施加可学习的复数权重矩阵（Complex Weighting）。最后通过逆傅里叶变换（IRFFT）将其反映射回原始空间。这实质上等效于在
      多小波隐特征层面进行了一次极其巨大的全局感受野卷积。

    intuition: >
      在处理复杂偏微分方程（PDEs）时，流体或波动的宏观低频结构往往主导了能量的级联分布。WaveletFourierKernel 在
      “已经过正交多小波基分解的系数”基础之上，进一步利用傅里叶谱方法的全局聚合能力，抓住了全局尺度上的长期依赖。
      截断高频不仅滤除了不稳定的噪声，更极大地收缩了参数空间，使得 $O(N \log N)$ 的计算负责度成为可能。

    problem_it_solves:
      - 在多小波隐空间中实现全局感受野的特征交互
      - 利用频域稀疏性，显著降低大尺度数据的计算开销与参数量
      - 在高维（1D/2D/3D）动力学系统中捕获长程物理场的守恒律映射

  theory:

    formula:

      fourier_truncation:
        expression: |
          \tilde{x} = \mathcal{F}(x)_{< \alpha}
          \tilde{y} = W \cdot \tilde{x}
          y = \mathcal{F}^{-1}(\tilde{y})

      complex_multiplication:
        expression: |
          (a + bi) \cdot (c + di) = (ac - bd) + (ad + bc)i

    variables:

      x:
        name: MultiWaveletCoefficients
        shape: [batch, ..., c, k^2]
        description: 多小波分解后的隐空间特征矩阵，包含通道与正交基维度

      \mathcal{F}:
        name: RealFastFourierTransform
        description: 实数到复数的快速傅里叶变换，利用共轭对称性仅计算一半频谱以提升计算效率

      \alpha:
        name: Modes
        description: 截断频率，决定了在频域中保留的低频模态数量，超出的高频信息直接置零

      W:
        name: ComplexWeights
        shape: [c \times k^{D}, c \times k^{D}, \alpha, ...]
        description: 针对不同频率分量学习的复数权重张量

  structure:

    architecture: FourierSparseKernel

    pipeline:
      - name: ShapeAlignment
        operation: "[B, N, c, k] -> [B, c * k, N]"
      - name: SpectralTransform
        operation: torch.fft.rfft()
      - name: ModeTruncationAndWeighting
        operation: compl_mul_Nd(x_fft, weights)
      - name: InverseSpectralTransform
        operation: torch.fft.irfft()
      - name: LatentProjection
        operation: Linear(Lo) (Optional in 2D/3D)

  interface:

    parameters:

      k:
        type: int
        description: 多小波基的阶数（或块大小）

      alpha:
        type: int
        description: 傅里叶频域中保留的模态（频率）数量

      c:
        type: int
        default: 1
        description: 通道缩放因子

    inputs:
      x:
        type: Tensor
        shape: [batch, spatial_dims..., c, k_dim]
        description: 物理或隐空间的多小波系数输入，针对2D和3D `k_dim` 通常为 k^2

    outputs:
      out:
        type: Tensor
        shape: [batch, spatial_dims..., c, k_dim]
        description: 经过频域全局混合后的输出张量，维度与输入完全一致

  types:
    Tensor:
      description: PyTorch Tensor
    ComplexTensor:
      dtype: torch.cfloat
      description: 包含复数张量

  implementation:

    framework: pytorch

    code: |
      import torch
      import torch.nn as nn
      import torch.nn.functional as F
      
      from onescience.modules.utils.wavelet_utils import compl_mul1d, compl_mul2d, compl_mul3d
      
      class WaveletFourierKernel1D(nn.Module):
          """
          一维多小波傅里叶稀疏核 (1D Multiwavelet Fourier Sparse Kernel)。
          (原 sparseKernelFT1d)
      
          该模块在多小波变换的隐空间中运行。它将输入的 1D 特征转换到傅里叶频域，
          截断高频分量（由 alpha 控制），并对保留的低频模态应用可学习的复数权重矩阵，
          最后通过逆傅里叶变换返回物理空间。这等效于在物理空间进行全局大卷积。
      
          Args:
              k (int): 多小波基的阶数/块大小。
              alpha (int): 保留的傅里叶模态数量（频率分量数）。
              c (int, optional): 通道缩放因子。默认值: 1。
              nl (int, optional): 保留参数。默认值: 1。
              initializer (callable, optional): 初始化函数（保留接口）。
      
          形状:
              输入 x: (B, N, c, k)，其中 N 是序列长度。
              输出: (B, N, c, k)，形状与输入一致。
      
          Example:
              >>> layer = WaveletFourierKernel1D(k=3, alpha=16, c=1)
              >>> x = torch.randn(10, 128, 1, 3)
              >>> out = layer(x)
              >>> print(out.shape)
              torch.Size([10, 128, 1, 3])
          """
          def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
              super().__init__()
              self.modes1 = alpha
              self.scale = (1 / (c * k * c * k))
              self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat))
              self.weights1.requires_grad = True
              self.k = k
      
          def forward(self, x):
              B, N, c, k = x.shape  
              x = x.view(B, N, -1).permute(0, 2, 1)
              x_fft = torch.fft.rfft(x)
              
              l = min(self.modes1, N // 2 + 1)
              out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
              out_ft[:, :, :l] = compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])
      
              x = torch.fft.irfft(out_ft, n=N)
              x = x.permute(0, 2, 1).view(B, N, c, k)
              return x
      
      
      class WaveletFourierKernel2D(nn.Module):
          """
          二维多小波傅里叶稀疏核 (2D Multiwavelet Fourier Sparse Kernel)。
          (原 sparseKernelFT2d)
      
          处理 2D 多小波特征。利用 2D FFT 将信号转换到频域，提取四个角的低频模态
          （由于共轭对称性，代码中使用两组权重处理正负频率），进行复数加权聚合后，通过逆变换和线性层返回物理空间。
      
          Args:
              k (int): 多小波基阶数/块大小。输入最后一个维度应为 k^2。
              alpha (int): 保留的傅里叶模态数量。
              c (int, optional): 通道缩放因子。默认值: 1。
              nl (int, optional): 保留参数。默认值: 1。
              initializer (callable, optional): 初始化函数。
      
          形状:
              输入 x: (B, N_x, N_y, c, k^2)。
              输出: (B, N_x, N_y, c, k^2)。
      
          Example:
              >>> layer = WaveletFourierKernel2D(k=3, alpha=8, c=1)
              >>> x = torch.randn(2, 64, 64, 1, 9)
              >>> out = layer(x)
              >>> print(out.shape)
              torch.Size([2, 64, 64, 1, 9])
          """
          def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
              super().__init__()
              self.modes = alpha
              self.weights1 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, dtype=torch.cfloat))
              self.weights2 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, dtype=torch.cfloat))
              nn.init.xavier_normal_(self.weights1)
              nn.init.xavier_normal_(self.weights2)
              self.Lo = nn.Linear(c * k ** 2, c * k ** 2)
              self.k = k
      
          def forward(self, x):
              B, Nx, Ny, c, ich = x.shape  
              x = x.view(B, Nx, Ny, -1).permute(0, 3, 1, 2)
              x_fft = torch.fft.rfft2(x)
      
              l1 = min(self.modes, Nx // 2 + 1)
              l2 = min(self.modes, Ny // 2 + 1)
              out_ft = torch.zeros(B, c * ich, Nx, Ny // 2 + 1, device=x.device, dtype=torch.cfloat)
      
              out_ft[:, :, :l1, :l2] = compl_mul2d(x_fft[:, :, :l1, :l2], self.weights1[:, :, :l1, :l2])
              out_ft[:, :, -l1:, :l2] = compl_mul2d(x_fft[:, :, -l1:, :l2], self.weights2[:, :, :l1, :l2])
      
              x = torch.fft.irfft2(out_ft, s=(Nx, Ny)).permute(0, 2, 3, 1)
              x = F.relu(x)
              x = self.Lo(x)
              x = x.view(B, Nx, Ny, c, ich)
              return x
      
      
      class WaveletFourierKernel3D(nn.Module):
          """
          三维多小波傅里叶稀疏核 (3D Multiwavelet Fourier Sparse Kernel)。
          (原 sparseKernelFT3d)
      
          在三维频域内操作（适用于 3D 空间或 2D+Time 时空数据）。使用 rfftn 计算三维频谱，
          针对频谱的关键低频部分使用 4 组复数权重进行特征交互，最后还原回物理空间。能有效捕捉体数据中的全局动态模式。
      
          Args:
              k (int): 多小波参数。输入最后维度应为 k^2。
              alpha (int): 傅里叶模态数。
              c (int, optional): 通道因子。默认值: 1。
              nl (int, optional): 保留参数。默认值: 1。
              initializer (callable, optional): 初始化函数。
      
          形状:
              输入 x: (B, N_x, N_y, T, c, k^2)。
              输出: (B, N_x, N_y, T, c, k^2)。
      
          Example:
              >>> layer = WaveletFourierKernel3D(k=3, alpha=8, c=1)
              >>> x = torch.randn(2, 16, 16, 16, 1, 9)
              >>> out = layer(x)
              >>> print(out.shape)
              torch.Size([2, 16, 16, 16, 1, 9])
          """
          def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
              super().__init__()
              self.modes = alpha
              self.weights1 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
              self.weights2 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
              self.weights3 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
              self.weights4 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
              
              nn.init.xavier_normal_(self.weights1)
              nn.init.xavier_normal_(self.weights2)
              nn.init.xavier_normal_(self.weights3)
              nn.init.xavier_normal_(self.weights4)
      
              self.Lo = nn.Linear(c * k ** 2, c * k ** 2)
              self.k = k
      
          def forward(self, x):
              B, Nx, Ny, T, c, ich = x.shape  
              x = x.view(B, Nx, Ny, T, -1).permute(0, 4, 1, 2, 3)
              x_fft = torch.fft.rfftn(x, dim=[-3, -2, -1])
      
              l1 = min(self.modes, Nx // 2 + 1)
              l2 = min(self.modes, Ny // 2 + 1)
              out_ft = torch.zeros(B, c * ich, Nx, Ny, T // 2 + 1, device=x.device, dtype=torch.cfloat)
      
              out_ft[:, :, :l1, :l2, :self.modes] = compl_mul3d(x_fft[:, :, :l1, :l2, :self.modes], self.weights1[:, :, :l1, :l2, :])
              out_ft[:, :, -l1:, :l2, :self.modes] = compl_mul3d(x_fft[:, :, -l1:, :l2, :self.modes], self.weights2[:, :, :l1, :l2, :])
              out_ft[:, :, :l1, -l2:, :self.modes] = compl_mul3d(x_fft[:, :, :l1, -l2:, :self.modes], self.weights3[:, :, :l1, :l2, :])
              out_ft[:, :, -l1:, -l2:, :self.modes] = compl_mul3d(x_fft[:, :, -l1:, -l2:, :self.modes], self.weights4[:, :, :l1, :l2, :])
      
              x = torch.fft.irfftn(out_ft, s=(Nx, Ny, T))
              x = x.permute(0, 2, 3, 4, 1)
              x = F.relu(x)
              x = self.Lo(x)
              x = x.view(B, Nx, Ny, T, c, ich)
              return x

  skills:

    build_wavelet_fourier_kernel:
      description: 构建在多小波隐空间上操作的高效频域卷积算子
      inputs:
        - k
        - alpha
        - c
      prompt_template: |
        构建多小波傅里叶核算子。
        参数：k={{k}}, 模态上限={{alpha}}, 通道乘子={{c}}
        要求：利用复数乘积保证全频域共轭对称，截断无关高频段以维持系统的稀疏性。

    diagnose_fourier_kernel:
      description: 排查快速傅里叶变换、频域切片或复数维度计算时的隐藏错误
      checks:
        - conjugate_symmetry_violation
        - fft_shift_misalignment
        - gradient_explosion_in_complex_weights

  knowledge:

    usage_patterns:
      multiwavelet_dispatch:
        pipeline:
          - DWT (Discrete Multi-Wavelet Transform)
          - FourierInteraction (WaveletFourierKernel)
          - SpatialInteraction (WaveletSpatialKernel)
          - IWT (Inverse Transform)

    hot_models:
      - model: FNO (Fourier Neural Operator)
        year: 2020
        role: 确立了在偏微分方程领域使用截断频域模态并进行复数相乘的基石标杆
        architecture: Spectral Convolution
      - model: AFNO (Adaptive Fourier Neural Operator)
        year: 2021
        role: 将 ViT 架构与傅里叶神经算子融合的高效 Token 混合模型
        architecture: Transformer + FFT

    best_practices:
      - 必须保证 lpha（modes）小于等于 N // 2 + 1 (Nyquist limit)，避免产生越界切片或频率混叠。
      - 复数张量的初始化应谨慎，使用 scale = 1/(c*k*c*k) 进行缩放或使用 	orch.nn.init.xavier_normal_ 保证数值稳定性。
      - 当涉及到 2D 或 3D 频域时，由于实信号傅里叶变换具有共轭对称性，代码需针对矩阵“四角”相应的正负频率区域定义相互独立的 weights。

    anti_patterns:
      - 在进行 	orch.fft.irfft2 之前没有利用 zeros(...) 模板对高频位置强制填零，导致重构计算引入未定义或悬空向量。
      - 在实空间和频域空间的置换 permute 过程中弄错内存连续性，导致视图（iew）产生严重且隐蔽的交错性错误。

    paper_references:
      - title: "Fourier Neural Operator for Parametric Partial Differential Equations"
        authors: Li et al.
        year: 2020
      - title: "Adaptive Fourier Neural Operators: Efficient Token Mixing for Transformers"
        authors: Guibas et al.
        year: 2021


  graph:
    is_a:
      - SpectralKernel
      - GlobalReceptiveFieldOperator
    part_of:
      - MultiWaveletTransform
      - NeuralOperator
    depends_on:
      - torch.fft
      - compl_mul1d
      - compl_mul2d
      - compl_mul3d
    compatible_with:
      - WaveletSpatialKernel (并行处理)