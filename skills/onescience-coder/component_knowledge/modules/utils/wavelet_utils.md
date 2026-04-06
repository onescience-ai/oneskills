component:

  meta:
    name: MultiwaveletUtilities
    alias: wavelet_utils
    version: 1.0
    domain: ai_for_science
    category: mathematical_operator
    subcategory: wavelet_transform
    author: OneScience
    license: Apache-2.0
    tags:
      - multiwavelet
      - legendre_polynomials
      - chebyshev_polynomials
      - wavelet_neural_operator
      - pde_solver
      - einsum

  concept:

    description: >
      该模块提供了构建多小波神经算子（Multiwavelet Neural Operator, WNO）所需的核心数学工具集。
      它主要负责生成基于勒让德（Legendre）或切比雪夫（Chebyshev）多项式的正交基函数，并计算用于多尺度分解与重构的滤波器组（H0, H1, G0, G1 等）。
      此外，模块还包含针对复数张量的高维爱因斯坦求和（Einsum）算子，用于在频谱空间进行高效的权重乘法。

    intuition: >
      在求解复杂的偏微分方程（PDE）时，传统的傅里叶变换（FFT）虽然高效，但缺乏局部特征的表达能力。
      多小波变换结合了多项式的高阶拟合能力和小波的分层多尺度特性。
      该模块就像是一个“数学蓝图计算器”，它预先计算出如何将连续的物理场投影到一组优美的正交基上，
      使得神经网络可以在不同的频率和尺度上同时“观察”物理规律。

    problem_it_solves:
      - 解决物理模拟中频谱方法缺乏空间局部化信息的问题。
      - 自动计算不同阶数（k）基函数的系数和正交化滤波器。
      - 提供针对 1D/2D/3D 复数权重学习的高性能计算接口。

  theory:

    formula:

      orthogonality_condition:
        expression: \int_{0}^{1} \phi_i(x) \phi_j(x) dx = \delta_{ij}

      multi_wavelet_decomposition:
        expression: s_{k,n}^{(j-1)} = \sum_{m} H_m s_{k, 2n+m}^{(j)} + \sum_{m} G_m d_{k, 2n+m}^{(j)}

      complex_spectral_multiplication:
        expression: \text{Out}_{b,o,x} = \sum_{i} \text{Input}_{b,i,x} \cdot \text{Weight}_{i,o,x}

    variables:

      phi:
        name: ScalingFunction
        description: 尺度函数基底，用于捕捉信号的低频/平滑部分。

      psi:
        name: WaveletFunction
        description: 小波函数基底，用于捕捉信号的高频/细节部分。

      H0, H1:
        name: LowPassFilters
        description: 对应多小波分解中的低频滤波器组。

      G0, G1:
        name: HighPassFilters
        description: 对应多小波分解中的高频细节滤波器组。

      k:
        name: PolynomialOrder
        description: 使用的多项式阶数，决定了基函数的丰富程度。

  structure:

    architecture: numerical_utility_library

    pipeline:

      - name: BasisGeneration
        operation: compute_legendre_or_chebyshev_coefficients

      - name: Orthogonalization
        operation: gram_schmidt_or_numerical_quadrature

      - name: FilterCalculation
        operation: get_filter_banks

      - name: SpectralMixing
        operation: complex_einsum_multiplication

  interface:

    parameters:

      k:
        type: int
        description: 基多项式的阶数。

      base:
        type: str
        description: 基函数类型，支持 'legendre' 或 'chebyshev'。

    inputs:

      x:
        type: Tensor / Array
        description: 输入的物理场张量或坐标数组。

      weights:
        type: Tensor (Complex)
        description: 在小波谱空间待学习的复数权重。

    outputs:

      filters:
        type: Tuple[ndarray]
        description: (H0, H1, G0, G1, PHI0, PHI1) 滤波器组。

      mul_out:
        type: Tensor
        description: 经过频谱乘法后的输出张量。

  types:

    ComplexTensor:
      shape: [batch, channels, spatial_dims...]
      description: 包含复数实部与虚部的 PyTorch 张量。

  implementation:

    framework: pytorch, numpy, scipy, sympy

    code: |
      import torch
      import numpy as np
      from scipy.special import eval_legendre
      from sympy import Poly, legendre, Symbol, chebyshevt
      from functools import partial

      def legendreDer(k, x):
          """计算勒让德多项式的导数"""
          def _legendre(k, x):
              return (2 * k + 1) * eval_legendre(k, x)
          out = 0
          for i in np.arange(k - 1, -1, -2):
              out += _legendre(i, x)
          return out

      def phi_(phi_c, x, lb=0, ub=1):
          """基于多项式系数和边界计算基函数的值"""
          mask = np.logical_or(x < lb, x > ub) * 1.0
          return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1 - mask)

      def get_phi_psi(k, base):
          """获取尺度函数 (phi) 和小波函数 (psi) 的多项式系数"""
          x = Symbol('x')
          phi_coeff = np.zeros((k, k))
          phi_2x_coeff = np.zeros((k, k))
          
          if base == 'legendre':
              for ki in range(k):
                  coeff_ = Poly(legendre(ki, 2 * x - 1), x).all_coeffs()
                  phi_coeff[ki, :ki + 1] = np.flip(np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))
                  coeff_ = Poly(legendre(ki, 4 * x - 1), x).all_coeffs()
                  phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))

              psi1_coeff = np.zeros((k, k))
              psi2_coeff = np.zeros((k, k))
              for ki in range(k):
                  psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
                  for i in range(k):
                      a = phi_2x_coeff[ki, :ki + 1]
                      b = phi_coeff[i, :i + 1]
                      prod_ = np.convolve(a, b)
                      prod_[np.abs(prod_) < 1e-8] = 0
                      proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                      psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                      psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                  for j in range(ki):
                      a = phi_2x_coeff[ki, :ki + 1]
                      b = psi1_coeff[j, :]
                      prod_ = np.convolve(a, b)
                      prod_[np.abs(prod_) < 1e-8] = 0
                      proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                      psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                      psi2_coeff[ki, :] -= proj_ * psi1_coeff[j, :]

                  a = psi1_coeff[ki, :]
                  prod_ = np.convolve(a, a)
                  prod_[np.abs(prod_) < 1e-8] = 0
                  norm1 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()

                  a = psi2_coeff[ki, :]
                  prod_ = np.convolve(a, a)
                  prod_[np.abs(prod_) < 1e-8] = 0
                  norm2 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * (1 - np.power(0.5, 1 + np.arange(len(prod_))))).sum()
                  norm_ = np.sqrt(norm1 + norm2)
                  psi1_coeff[ki, :] /= norm_
                  psi2_coeff[ki, :] /= norm_

              phi = [np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)]
              psi1 = [np.poly1d(np.flip(psi1_coeff[i, :])) for i in range(k)]
              psi2 = [np.poly1d(np.flip(psi2_coeff[i, :])) for i in range(k)]

          elif base == 'chebyshev':
              # ... (省略部分切比雪夫系数计算以匹配模板简洁性，实际代码包含完整逻辑)
              pass

          return phi, psi1, psi2

      def get_filter(base, k):
          """计算多小波分解与重构滤波器 (H0, H1, G0, G1, PHI0, PHI1)"""
          # 实现基于高斯求积（Legendre）或特定根采样（Chebyshev）的滤波器系数提取
          # 返回值 H0, H1, G0, G1, PHI0, PHI1 为形状为 (k, k) 的矩阵
          pass

      def compl_mul1d(x, weights):
          """一维复数张量乘法"""
          return torch.einsum("bix,iox->box", x, weights)

      def compl_mul2d(x, weights):
          """二维复数张量乘法"""
          return torch.einsum("bixy,ioxy->boxy", x, weights)

      def compl_mul3d(input, weights):
          """三维复数张量乘法"""
          return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

  skills:

    generate_multiwavelet_filters:

      description: 为指定阶数和基函数类型生成正交多小波滤波器组。

      inputs:
        - base ('legendre' / 'chebyshev')
        - order (k)

      prompt_template: |
        使用 get_filter 函数获取 {{base}} 基下的 {{order}} 阶滤波器组。

    perform_spectral_multiplication:

      description: 在多小波频谱空间执行高效的维度变换乘法。

      inputs:
        - input_tensor
        - spectral_weights
        - dimension (1d/2d/3d)

      prompt_template: |
        调用 compl_mul{{dimension}} 函数，将输入的频谱张量与学习到的权重进行 Einsum 压缩。

  knowledge:

    usage_patterns:

      wno_block_logic:

        pipeline:
          - MultiwaveletTransform (MWT)
          - SpectralWeightMultiplication (compl_mulnd)
          - InverseMultiwaveletTransform (IMWT)

    design_patterns:

      orthogonal_basis_projection:

        structure:
          - 利用 SymPy 进行解析多项式的生成。
          - 通过数值积分（Quadrature）确保在 [0, 1] 区间内的严谨正交性，这是 PDE 求解稳定的前提。

    hot_models:

      - model: Wavelet Neural Operator (WNO)
        year: 2021
        role: 处理具有尖锐梯度或不连续解的 PDE 代理模型。
        architecture: spectral_neural_operator
        attention_type: None (使用小波频谱交互代替)

    model_usage_details:

      WNO-Legendre:

        usage: 在流体力学的不连续激波模拟中，Legendre 基由于其在边界上的良好表现而被广泛采用。

    best_practices:

      - 基函数阶数 k 不宜设置过大（通常为 2-4），否则会导致求积过程中的数值不稳定和显著的计算开销。
      - 在执行频谱乘法时，始终使用 `torch.einsum`，它在底层自动优化了访存和计算流。
      - 滤波器系数应在模型初始化阶段预计算一次并缓存为 Buffer，不应在训练循环中动态生成。

    anti_patterns:

      - 使用不支持的基函数类型（如 'fourier' 传入该模块），会导致抛出 'Base not supported' 异常。
      - 忽略了滤波器中极小值的截断（代码中使用了 `1e-8` 截断），这在深度网络多次迭代中可能累积数值噪声。

    paper_references:

      - title: "Wavelet Neural Operator for Solving Parametric Partial Differential Equations"
        authors: Gupta et al.
        year: 2021

      - title: "Multiwavelets for Numerical Solution of Differential Equations"
        authors: Alpert
        year: 1993

  graph:

    is_a:
      - NumericalUtility
      - TransformOperator

    part_of:
      - WaveletNeuralOperator
      - SpectralMethods

    depends_on:
      - sympy.Poly
      - scipy.special.eval_legendre
      - torch.einsum

    variants:
      - FourierTransform (用于 FNO)
      - StandardWavelet (如 Daubechies)

    used_in_models:
      - WNO (Wavelet Neural Operator)
      - MWNO (Multiwavelet Neural Operator)

    compatible_with:

      inputs:
        - ComplexTensor
        - GridData

      outputs:
        - SpectralCoefficients
        - ReconstructedField