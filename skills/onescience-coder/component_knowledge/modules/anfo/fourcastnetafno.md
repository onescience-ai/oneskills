component:

  meta:
    name: FourCastNetAFNO2D
    alias: AFNO
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: spectral_operator
    author: OneScience
    license: Apache-2.0
    tags:
      - afno
      - fourcastnet
      - fourier_transform
      - spectral_mixing
      - weather_forecasting


  concept:

    description: >
      FourCastNetAFNO2D 是 FourCastNet 中的自适应傅里叶神经算子模块。
      模块先将输入 token 从空间域映射到频域，在频域执行按通道块分组的复数线性变换与非线性混合，
      再通过逆傅里叶变换还原到空间域，并与输入残差相加。

    intuition: >
      该模块将“全局 token 混合”放到频域中进行，利用 FFT 把长程依赖转化为频率模式上的局部参数化变换。
      相比显式两两注意力，AFNO 通过频域低频保留 + 软阈值稀疏化，在维持全局感受野的同时降低计算和显存开销。

    problem_it_solves:
      - 用频域混合替代标准自注意力，提供全局建模能力
      - 通过 hard_thresholding_fraction 限制参与计算的频率模式数，降低计算量
      - 通过 softshrink 抑制低幅值频率分量，提升稀疏性与稳定性
      - 在高分辨率网格预测任务中改进效率与可扩展性


  theory:

    formula:

      fft_projection:
        expression: |
          X_f = rFFT2(X)

      spectral_mlp:
        expression: |
          O_1^r = ReLU(X^r W_1^r - X^i W_1^i + b_1^r)
          O_1^i = ReLU(X^i W_1^r + X^r W_1^i + b_1^i)
          O_2^r = O_1^r W_2^r - O_1^i W_2^i + b_2^r
          O_2^i = O_1^i W_2^r + O_1^r W_2^i + b_2^i

      sparsification_and_inverse:
        expression: |
          \hat{X}_f = SoftShrink([O_2^r, O_2^i], \lambda)
          \hat{X} = irFFT2(\hat{X}_f)
          Y = \hat{X} + X

    variables:

      X:
        name: InputTensor
        shape: [batch, H, W, C]
        description: 空间域输入张量

      X_f:
        name: FrequencyTensor
        shape: [batch, H, W/2+1, num_blocks, block_size]
        description: 经过实数 2D FFT 后的复频域表示

      W_1^r:
        name: FirstLayerRealWeight
        shape: [num_blocks, block_size, block_size * hidden_size_factor]
        description: 第一层复线性变换的实部权重

      \lambda:
        name: SparsityThreshold
        description: SoftShrink 阈值，对低幅频率分量进行稀疏化


  structure:

    architecture: afno_2d_block

    pipeline:

      - name: ResidualBuffer
        operation: store_input_as_bias

      - name: FrequencyTransform
        operation: rfft2_and_channel_blocking

      - name: BlockwiseComplexMixing
        operation: complex_linear_relu_linear

      - name: FrequencySparsification
        operation: softshrink

      - name: InverseTransform
        operation: irfft2

      - name: ResidualAdd
        operation: output_plus_bias


  interface:

    parameters:

      hidden_size:
        type: int
        description: 输入通道维度，且必须可被 num_blocks 整除

      num_blocks:
        type: int
        description: 通道分块数量，每个块独立执行频域混合

      sparsity_threshold:
        type: float
        description: 频域软阈值化系数

      hard_thresholding_fraction:
        type: float
        description: 保留频率模式比例，控制低频模式数量

      hidden_size_factor:
        type: int
        description: 频域 MLP 隐层扩展倍率

    inputs:

      x:
        type: GridEmbedding
        shape: [batch, H, W, C]
        dtype: float32
        description: 网格场 token 表示

    outputs:

      output:
        type: GridEmbedding
        shape: [batch, H, W, C]
        description: 频域混合并加残差后的输出


  types:

    GridEmbedding:
      shape: [batch, H, W, C]
      description: 二维网格 embedding 表示


  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      class FourCastNetAFNO2D(nn.Module):
          def __init__(self,
                       hidden_size=768,
                       num_blocks=8,
                       sparsity_threshold=0.01,
                       hard_thresholding_fraction=1,
                       hidden_size_factor=1):
              super().__init__()
              assert hidden_size % num_blocks == 0

              self.hidden_size = hidden_size
              self.sparsity_threshold = sparsity_threshold
              self.num_blocks = num_blocks
              self.block_size = self.hidden_size // self.num_blocks
              self.hard_thresholding_fraction = hard_thresholding_fraction
              self.hidden_size_factor = hidden_size_factor
              self.scale = 0.02

              self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
              self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
              self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
              self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

          def forward(self, x):
              bias = x
              dtype = x.dtype
              x = x.float()
              B, H, W, C = x.shape

              x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
              x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

              o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
              o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
              o2_real = torch.zeros(x.shape, device=x.device)
              o2_imag = torch.zeros(x.shape, device=x.device)

              total_modes = H // 2 + 1
              kept_modes = int(total_modes * self.hard_thresholding_fraction)

              o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
                  torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) -
                  torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) +
                  self.b1[0]
              )

              o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
                  torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) +
                  torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) +
                  self.b1[1]
              )

              o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
                  torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) -
                  torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) +
                  self.b2[0]
              )

              o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
                  torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) +
                  torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) +
                  self.b2[1]
              )

              x = torch.stack([o2_real, o2_imag], dim=-1)
              x = F.softshrink(x, lambd=self.sparsity_threshold)
              x = torch.view_as_complex(x)
              x = x.reshape(B, H, W // 2 + 1, C)
              x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
              x = x.type(dtype)

              return x + bias


  skills:

    build_afno_block:

      description: 构建用于二维网格建模的 AFNO 频域混合模块

      inputs:
        - hidden_size
        - num_blocks
        - sparsity_threshold
        - hard_thresholding_fraction
        - hidden_size_factor

      prompt_template: |

        构建 FourCastNetAFNO2D 模块。

        参数：
        hidden_size = {{hidden_size}}
        num_blocks = {{num_blocks}}
        sparsity_threshold = {{sparsity_threshold}}
        hard_thresholding_fraction = {{hard_thresholding_fraction}}
        hidden_size_factor = {{hidden_size_factor}}

        要求：
        在频域内执行复数线性混合，包含 softshrink 稀疏化和残差连接。


    diagnose_afno:

      description: 分析 AFNO 在训练或推理中的频域稳定性与性能问题

      checks:
        - hidden_size_not_divisible_by_num_blocks
        - instability_due_to_inappropriate_sparsity_threshold
        - too_few_kept_modes_causing_information_loss
        - numerical_drift_between_fft_and_ifft


  knowledge:

    usage_patterns:

      fourcastnet_backbone:

        pipeline:
          - PatchEmbedding
          - AFNOBlocks
          - PredictionHead

      spectral_token_mixing:

        pipeline:
          - SpatialToFrequency
          - BlockwiseComplexMLP
          - FrequencyToSpatial


    hot_models:

      - model: FourCastNet
        year: 2022
        role: 在全球天气预报任务中采用 AFNO 作为核心 token mixing 模块
        architecture: spectral_operator_transformer_hybrid

      - model: AFNO (Adaptive Fourier Neural Operator)
        year: 2021
        role: 用可学习频域块混合实现高效全局建模
        architecture: neural_operator


    best_practices:

      - 保持 hidden_size 可被 num_blocks 整除，避免通道分块失配。
      - 在高分辨率场景中优先调节 hard_thresholding_fraction 以控制频域计算量。
      - 对不同数据变量单独网格搜索 sparsity_threshold 以平衡稀疏性和精度。


    anti_patterns:

      - 在频率模式过少时强行增大稀疏阈值，导致关键信号被过度抑制。
      - 忽略输入 dtype 转换与恢复，导致混合精度训练下数值不一致。


    paper_references:

      - title: "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators"
        authors: Pathak et al.
        year: 2022

      - title: "Efficient Token Mixing for Transformers via Adaptive Fourier Neural Operators"
        authors: Guibas et al.
        year: 2021


  graph:

    is_a:
      - NeuralNetworkComponent
      - SpectralOperator

    part_of:
      - FourCastNet
      - WeatherForecastingModels

    depends_on:
      - rFFT2
      - irFFT2
      - SoftShrink
      - EinsteinSummation

    variants:
      - AFNO1D
      - AFNO3D

    used_in_models:
      - FourCastNet
      - Pangu-Weather
      - AFNO-based Forecasting Models

    compatible_with:

      inputs:
        - GridEmbedding

      outputs:
        - GridEmbedding