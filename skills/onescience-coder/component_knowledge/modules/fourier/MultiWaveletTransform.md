component:

  meta:
    name: MultiWaveletTransformLayers
    alias: MWTLayers
    version: 1.0.0
    domain: deep_learning
    category: neural_operator
    subcategory: wavelet_multi_scale
    author: OneScience
    license: Apache-2.0
    tags:
      - wavelet
      - fourier
      - multi_scale
      - spatial_kernel
      - pde

  concept:
    description: >
      正交多小波变换（Multiwavelet Transform）框架。通过正交基对底层栅格逐尺度施加非标准小波深层分解（Non-standard Wavelet Decomposition）。在不同分辨率上，它调度频域的谱计算核来抓取模糊大局信号（Approximate coefficients），同时调用精准狭小的空间算子（Spatial Kernel）捕捉局部断崖异变信息（Detail coefficients），最后分层反向递归还原。

    intuition: >
      这就好像用几张越来越密的纱网筛沙金：巨大的海流变化从粗眼漏下由宏观相机（频域处理）掌控全局；微小漩涡由细眼卡住用微距镜头（空间局域网络）严密聚焦，两方并驾齐驱毫无死角。

    problem_it_solves:
      - 解决单凭 FNO 会忽略极其激烈的断块冲击坡、单凭 CNN 则无法快速连通相隔万里的气象走势之间的极端矛盾。
      - 利用了具有精确数学性质的正交多项组进行展开。

  theory:
    formula:
      mwt_decomposition:
        expression: |
          s_l = \text{Approximate}(x, \phi)
          d_l = \text{Detail}(x, \psi)

      operator_interaction:
        expression: |
          \tilde{s} = A(s) \quad \text{ (WaveletFourier Kernel)}
          \tilde{d} = B(d) + C(s) \quad \text{ (WaveletSpatial Kernels)}

    variables:
      \phi:
        name: ScalingFunction
        description: 逼近原方程本体包络态的尺度集函数
      \psi:
        name: WaveletFunction
        description: 在不同伸缩位移尺度中捕获高频振散的小波基

  structure:
    architecture: multi_resolution_operator

    pipeline:
      - name: IterativeDecomposition
        operation: non_standard_recursive_split_by_2
      - name: MultiScaleProcessing
        operation: parallel_branch_A_B_C
      - name: IterativeReconstruction
        operation: inverse_concatenation_and_mixing

  interface:
    parameters:
      k:
        type: int
        description: "选择小波计算的正交级数（order），比如 k=3 代表二次多项式"
      alpha:
        type: int
        description: "FNO核里面限定处理频数的限制系数或扩增特征流"
      L:
        type: int
        description: "粗粒度层次不分解界限阀值（一般设置在 0~2）"
      base:
        type: str
        description: "基础选型设置（如 legendre 勒让德或 chebyshev 切比雪夫）"

    inputs:
      x:
        type: InputSignal
        shape: [batch, Nx, (Ny), channel, k^2]
        dtype: float32
        description: "包含基展开的流体分布点矩阵，由于是二分叉规则所以 Nx, Ny 等空间轴应该保持在 2 的幂"

    outputs:
      out:
        type: OutputSignal
        shape: [batch, Nx, (Ny), channel, k^2]
        dtype: float32
        description: "完美交融高低频后的重构建流体响应图像"

  types:
    InputSignal:
      shape: [batch, ..., factor]
      description: 指数整齐尺度的输入图卷

  implementation:
    framework: pytorch
    code: |
      import torch
      import torch.nn as nn
      import numpy as np
      import math
      from typing import List
      from torch import Tensor
      
      # --- 导入所需工具与核函数 ---
      from onescience.modules.utils.wavelet_utils import get_filter
      from .WaveletFourierKernel import WaveletFourierKernel1D, WaveletFourierKernel2D, WaveletFourierKernel3D
      from .WaveletSpatialKernel import WaveletSpatialKernel2D, WaveletSpatialKernel3D
      
      class MultiWaveletTransform1D(nn.Module):
          """
          一维多小波变换层 (1D Multiwavelet Transform Layer)。
      
          该模块实现了基于正交多小波变换的非标准分解与重构。它将 1D 信号分解为不同尺度的近似和细节系数。
          在每个尺度上，分别使用 WaveletFourierKernel1D 作为稀疏核对系数进行特征交互和映射，
          有效结合了多尺度局部性与频域的长程依赖捕捉能力。
      
          Args:
              k (int, optional): 多小波基的阶数/块大小。默认值: 3。
              alpha (int, optional): 稀疏核中的模态数量控制参数。默认值: 5。
              L (int, optional): 保持不分解的最粗糙层级数。默认值: 0。
              c (int, optional): 通道缩放因子。默认值: 1。
              base (str, optional): 多小波基类型 (如 'legendre' 或 'chebyshev')。默认值: 'legendre'。
              initializer (callable, optional): 参数初始化函数。
      
          形状:
              输入 x: (B, N, c, k)，其中 N 通常需要是 2 的幂。
              输出: (B, N, c, k)。
      
          Example:
              >>> model = MultiWaveletTransform1D(k=3, alpha=8, L=0, c=1)
              >>> x = torch.randn(8, 256, 1, 3)
              >>> out = model(x)
              >>> print(out.shape)
              torch.Size([8, 256, 1, 3])
          """
          def __init__(self, k=3, alpha=5, L=0, c=1, base='legendre', initializer=None, **kwargs):
              super().__init__()
              self.k = k
              self.L = L
              H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
              
              H0r, G0r = H0 @ PHI0, G0 @ PHI0
              H1r, G1r = H1 @ PHI1, G1 @ PHI1
              H0r[np.abs(H0r) < 1e-8] = 0
              H1r[np.abs(H1r) < 1e-8] = 0
              G0r[np.abs(G0r) < 1e-8] = 0
              G1r[np.abs(G1r) < 1e-8] = 0
      
              self.A = WaveletFourierKernel1D(k, alpha, c)
              self.B = WaveletFourierKernel1D(k, alpha, c)
              self.C = WaveletFourierKernel1D(k, alpha, c)
              self.T0 = nn.Linear(k, k)
      
              self.register_buffer('ec_s', torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
              self.register_buffer('ec_d', torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))
              self.register_buffer('rc_e', torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
              self.register_buffer('rc_o', torch.Tensor(np.concatenate((H1r, G1r), axis=0)))
      
          def forward(self, x):
              B, N, c, ich = x.shape  
              ns = math.floor(np.log2(N))
      
              Ud = torch.jit.annotate(List[Tensor], [])
              Us = torch.jit.annotate(List[Tensor], [])
              
              # decompose
              for i in range(ns - self.L):
                  d, x = self.wavelet_transform(x)
                  Ud += [self.A(d) + self.B(x)]
                  Us += [self.C(d)]
              x = self.T0(x)  # coarsest scale transform
      
              # reconstruct
              for i in range(ns - 1 - self.L, -1, -1):
                  x = x + Us[i]
                  x = torch.cat((x, Ud[i]), -1)
                  x = self.evenOdd(x)
              return x
      
          def wavelet_transform(self, x):
              xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
              d = torch.matmul(xa, self.ec_d)
              s = torch.matmul(xa, self.ec_s)
              return d, s
      
          def evenOdd(self, x):
              B, N, c, ich = x.shape  
              assert ich == 2 * self.k
              x_e = torch.matmul(x, self.rc_e)
              x_o = torch.matmul(x, self.rc_o)
      
              x_out = torch.zeros(B, N * 2, c, self.k, device=x.device)
              x_out[..., ::2, :, :] = x_e
              x_out[..., 1::2, :, :] = x_o
              return x_out
      
      
      class MultiWaveletTransform2D(nn.Module):
          """
          二维多小波变换层 (2D Multiwavelet Transform Layer)。
          (原 MWT_CZ2d)
      
          
      
          该模块实现了二维正交多小波的非标准分解与重构。它将 2D 网格数据分解为多个尺度，
          在每一层级，使用 WaveletFourierKernel2D (频域核) 处理近似信息 (A 分支)，
          使用 WaveletSpatialKernel2D (空域卷积核) 处理高频细节信息 (B, C 分支)。
          这种混合架构有效地兼顾了频域的大感受野与空域的局部特征敏感性。
      
          Args:
              k (int, optional): 多小波基阶数。默认值: 3。
              alpha (int, optional): 核函数模态数或通道倍率。默认值: 5。
              L (int, optional): 不进行分解的最粗糙层级数。默认值: 0。
              c (int, optional): 通道缩放因子。默认值: 1。
              base (str, optional): 多小波基类型。默认值: 'legendre'。
              initializer (callable, optional): 初始化函数。
      
          形状:
              输入 x: (B, N_x, N_y, c, k^2)。Nx 和 Ny 必须是 2 的幂。
              输出: (B, N_x, N_y, c, k^2)。
      
          Example:
              >>> model = MultiWaveletTransform2D(k=3, alpha=8, L=0, c=1)
              >>> x = torch.randn(2, 64, 64, 1, 9)
              >>> out = model(x)
              >>> print(out.shape)
              torch.Size([2, 64, 64, 1, 9])
          """
          def __init__(self, k=3, alpha=5, L=0, c=1, base='legendre', initializer=None, **kwargs):
              super().__init__()
              self.k = k
              self.L = L
              H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
              
              H0r, G0r = H0 @ PHI0, G0 @ PHI0
              H1r, G1r = H1 @ PHI1, G1 @ PHI1
              H0r[np.abs(H0r) < 1e-8] = 0
              H1r[np.abs(H1r) < 1e-8] = 0
              G0r[np.abs(G0r) < 1e-8] = 0
              G1r[np.abs(G1r) < 1e-8] = 0
      
              self.A = WaveletFourierKernel2D(k, alpha, c)
              self.B = WaveletSpatialKernel2D(k, c, c)
              self.C = WaveletSpatialKernel2D(k, c, c)
              self.T0 = nn.Linear(c * k ** 2, c * k ** 2)
      
              if initializer is not None:
                  self.reset_parameters(initializer)
      
              self.register_buffer('ec_s', torch.Tensor(
                  np.concatenate((np.kron(H0, H0).T, np.kron(H0, H1).T, np.kron(H1, H0).T, np.kron(H1, H1).T), axis=0)))
              self.register_buffer('ec_d', torch.Tensor(
                  np.concatenate((np.kron(G0, G0).T, np.kron(G0, G1).T, np.kron(G1, G0).T, np.kron(G1, G1).T), axis=0)))
      
              self.register_buffer('rc_ee', torch.Tensor(np.concatenate((np.kron(H0r, H0r), np.kron(G0r, G0r)), axis=0)))
              self.register_buffer('rc_eo', torch.Tensor(np.concatenate((np.kron(H0r, H1r), np.kron(G0r, G1r)), axis=0)))
              self.register_buffer('rc_oe', torch.Tensor(np.concatenate((np.kron(H1r, H0r), np.kron(G1r, G0r)), axis=0)))
              self.register_buffer('rc_oo', torch.Tensor(np.concatenate((np.kron(H1r, H1r), np.kron(G1r, G1r)), axis=0)))
      
          def forward(self, x):
              B, Nx, Ny, c, ich = x.shape  
              ns = math.floor(np.log2(Nx))
      
              Ud = torch.jit.annotate(List[Tensor], [])
              Us = torch.jit.annotate(List[Tensor], [])
      
              # decompose
              for i in range(ns - self.L):
                  d, x = self.wavelet_transform(x)
                  Ud += [self.A(d) + self.B(x)]
                  Us += [self.C(d)]
              x = self.T0(x.view(B, 2 ** self.L, 2 ** self.L, -1)).view(
                  B, 2 ** self.L, 2 ** self.L, c, ich)  
      
              # reconstruct
              for i in range(ns - 1 - self.L, -1, -1):
                  x = x + Us[i]
                  x = torch.cat((x, Ud[i]), -1)
                  x = self.evenOdd(x)
              return x
      
          def wavelet_transform(self, x):
              xa = torch.cat([x[:, ::2, ::2, :, :], x[:, ::2, 1::2, :, :],
                              x[:, 1::2, ::2, :, :], x[:, 1::2, 1::2, :, :]], -1)
              d = torch.matmul(xa, self.ec_d)
              s = torch.matmul(xa, self.ec_s)
              return d, s
      
          def evenOdd(self, x):
              B, Nx, Ny, c, ich = x.shape  
              assert ich == 2 * self.k ** 2
              x_ee = torch.matmul(x, self.rc_ee)
              x_eo = torch.matmul(x, self.rc_eo)
              x_oe = torch.matmul(x, self.rc_oe)
              x_oo = torch.matmul(x, self.rc_oo)
      
              x_out = torch.zeros(B, Nx * 2, Ny * 2, c, self.k ** 2, device=x.device)
              x_out[:, ::2, ::2, :, :] = x_ee
              x_out[:, ::2, 1::2, :, :] = x_eo
              x_out[:, 1::2, ::2, :, :] = x_oe
              x_out[:, 1::2, 1::2, :, :] = x_oo
              return x_out
      
          def reset_parameters(self, initializer):
              initializer(self.T0.weight)
      
      
      class MultiWaveletTransform3D(nn.Module):
          """
          三维多小波变换层 (3D Multiwavelet Transform Layer)。
          (原 MWT_CZ3d)
      
          针对三维数据（如流体模拟的 3D 场或 2D+T 时空序列）进行多尺度分析。
          通过三维小波变换金字塔分解数据，并在每一层尺度上结合 3D 频域核 (WaveletFourierKernel3D) 
          与 3D 空域核 (WaveletSpatialKernel3D) 进行混合特征演化。
      
          Args:
              k (int, optional): 多小波基阶数。默认值: 3。
              alpha (int, optional): 频域核模态数或空域核膨胀系数。默认值: 5。
              L (int, optional): 粗糙层级保留深度。默认值: 0。
              c (int, optional): 通道缩放因子。默认值: 1。
              base (str, optional): 小波基类型 ('legendre' 或 'chebyshev')。默认值: 'legendre'。
              initializer (callable, optional): 初始化函数。
      
          形状:
              输入 x: (B, N_x, N_y, T, c, k^2)。Nx 和 Ny 必须为 2 的幂。
              输出: (B, N_x, N_y, T, c, k^2)。
      
          Example:
              >>> model = MultiWaveletTransform3D(k=3, alpha=4, L=0, c=1)
              >>> x = torch.randn(1, 32, 32, 16, 1, 9)
              >>> out = model(x)
              >>> print(out.shape)
              torch.Size([1, 32, 32, 16, 1, 9])
          """
          def __init__(self, k=3, alpha=5, L=0, c=1, base='legendre', initializer=None, **kwargs):
              super().__init__()
              self.k = k
              self.L = L
              H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
              
              H0r, G0r = H0 @ PHI0, G0 @ PHI0
              H1r, G1r = H1 @ PHI1, G1 @ PHI1
              H0r[np.abs(H0r) < 1e-8] = 0
              H1r[np.abs(H1r) < 1e-8] = 0
              G0r[np.abs(G0r) < 1e-8] = 0
              G1r[np.abs(G1r) < 1e-8] = 0
      
              self.A = WaveletFourierKernel3D(k, alpha, c)
              self.B = WaveletSpatialKernel3D(k, c, c)
              self.C = WaveletSpatialKernel3D(k, c, c)
              self.T0 = nn.Linear(c * k ** 2, c * k ** 2)
      
              if initializer is not None:
                  self.reset_parameters(initializer)
      
              self.register_buffer('ec_s', torch.Tensor(
                  np.concatenate((np.kron(H0, H0).T, np.kron(H0, H1).T, np.kron(H1, H0).T, np.kron(H1, H1).T), axis=0)))
              self.register_buffer('ec_d', torch.Tensor(
                  np.concatenate((np.kron(G0, G0).T, np.kron(G0, G1).T, np.kron(G1, G0).T, np.kron(G1, G1).T), axis=0)))
      
              self.register_buffer('rc_ee', torch.Tensor(np.concatenate((np.kron(H0r, H0r), np.kron(G0r, G0r)), axis=0)))
              self.register_buffer('rc_eo', torch.Tensor(np.concatenate((np.kron(H0r, H1r), np.kron(G0r, G1r)), axis=0)))
              self.register_buffer('rc_oe', torch.Tensor(np.concatenate((np.kron(H1r, H0r), np.kron(G1r, G0r)), axis=0)))
              self.register_buffer('rc_oo', torch.Tensor(np.concatenate((np.kron(H1r, H1r), np.kron(G1r, G1r)), axis=0)))
      
          def forward(self, x):
              B, Nx, Ny, T, c, ich = x.shape  
              ns = math.floor(np.log2(Nx))
      
              Ud = torch.jit.annotate(List[Tensor], [])
              Us = torch.jit.annotate(List[Tensor], [])
      
              # decompose
              for i in range(ns - self.L):
                  d, x = self.wavelet_transform(x)
                  Ud += [self.A(d) + self.B(x)]
                  Us += [self.C(d)]
              x = self.T0(x.view(B, 2 ** self.L, 2 ** self.L, T, -1)).view(
                  B, 2 ** self.L, 2 ** self.L, T, c, ich)  
      
              # reconstruct
              for i in range(ns - 1 - self.L, -1, -1):
                  x = x + Us[i]
                  x = torch.cat((x, Ud[i]), -1)
                  x = self.evenOdd(x)
      
              return x
      
          def wavelet_transform(self, x):
              xa = torch.cat([x[:, ::2, ::2, :, :, :], x[:, ::2, 1::2, :, :, :],
                              x[:, 1::2, ::2, :, :, :], x[:, 1::2, 1::2, :, :, :]], -1)
              d = torch.matmul(xa, self.ec_d)
              s = torch.matmul(xa, self.ec_s)
              return d, s
      
          def evenOdd(self, x):
              B, Nx, Ny, T, c, ich = x.shape  
              assert ich == 2 * self.k ** 2
              x_ee = torch.matmul(x, self.rc_ee)
              x_eo = torch.matmul(x, self.rc_eo)
              x_oe = torch.matmul(x, self.rc_oe)
              x_oo = torch.matmul(x, self.rc_oo)
      
              x_out = torch.zeros(B, Nx * 2, Ny * 2, T, c, self.k ** 2, device=x.device)
              x_out[:, ::2, ::2, :, :, :] = x_ee
              x_out[:, ::2, 1::2, :, :, :] = x_eo
              x_out[:, 1::2, ::2, :, :, :] = x_oe
              x_out[:, 1::2, 1::2, :, :, :] = x_oo
              return x_out
      
          def reset_parameters(self, initializer):
              initializer(self.T0.weight)

  skills:
    build_mwt_layer:
      description: 为高度剧变的突波流体构造 MWT 层级组
      inputs:
        - order_k
        - truncation_alpha
      prompt_template: |
        按照高频局域、低频全局分拆的理念，建立包含勒让德基的 MultiWavelet 分布解析。

    diagnose_mwt_scale_mismatch:
      description: 诊断尺度递归崩溃（特别是奇数维度被对折卡死）
      checks:
        - padding_mismatch_to_power_of_two

  knowledge:
    usage_patterns:
      mwt_pipeline:
        pipeline:
          - SpatialPadding (使得长宽变成2的指数倍)
          - MultiWaveletTransform (推理演化核心)
          - Cropping (还原真实剪裁尺寸)

    hot_models:
      - model: MWT (Multiwavelet Neural Operator)
        year: 2021
        role: 高精度应对各类激波和瞬变奇异值模型

    best_practices:
      - "请保证你的长宽高维度均满足 $2^N$ ，否则你不得不在外面嵌套一层极其小心的 Padding 掩码来满足正交小波极其死板的降采样限制。"

    anti_patterns:
      - "直接把不规整边界例如 137*215 的不合理尺寸送进去然后看它在切分除二时直接抛出 Shape Dimension 闪退。"

    paper_references:
      - title: "Multiwavelet-based Operator Learning for Differential Equations"
        authors: Gupta et al.
        year: 2021

  graph:
    is_a:
      - OperatorLearningModule
      - HierarchicalNetwork
    part_of:
      - MultiscaleAI_Solvers
    depends_on:
      - WaveletFourierKernel
      - WaveletSpatialKernel
      - get_filter (Base generator)
    variants:
      - 1D/2D/3D MWTs
    used_in_models:
      - WaveletOperator
    compatible_with:
      inputs:
        - PowerOfTwoTensors
      outputs:
        - PowerOfTwoTensors
