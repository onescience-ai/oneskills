component:

  meta:
    name: FourierPositionalEncoding
    alias: FNOFourier
    version: 1.0.0
    domain: deep_learning
    category: positional_encoding
    subcategory: fourier_features
    author: OneScience
    license: Apache-2.0
    tags:
      - fourier
      - features
      - embedding
      - positional_encoding
      - nerf_style

  concept:
    description: >
      输入坐标特征的基于高斯随机的傅里叶前置层，将极低维物理坐标映射为高度周期波动的多维表征矩阵。

    intuition: >
      给模型加上由无数频段组成的高频观测透镜，解决传统MLP看大不看小、面对细微跳变易产生所谓“谱偏移”平滑误差的通病。

    problem_it_solves:
      - 克服神经网络的 Spectral Bias（偏向拟合低频）现象
      - 大幅提升对坐标和局部边缘细节的分辨响应

  theory:
    formula:
      fourier_mapping:
        expression: |
          \hat{x} = x \cdot B
          x_{out} = [\sin(2\pi \hat{x}), \cos(2\pi \hat{x})]

    variables:
      x:
        name: InputCoordinates
        shape: [..., in_features]
        description: 归一化输入端点
      B:
        name: FrequencyMatrix
        description: 特征抽样矩阵

  structure:
    architecture: coordinate_projection_layer

    pipeline:
      - name: MatrixMultiplication
        operation: dot_product_with_sampled_frequency
      - name: TrigonometricActivation
        operation: sin_and_cos_expansion
      - name: Concatenation
        operation: stack_periodic_channels

  interface:
    parameters:
      in_features:
        type: int
        description: 原空间物理特征（x,y）维数
      frequencies:
        type: List_or_Str
        description: 高频参数组或采样类型比如 "gaussian" 或 "full"

    inputs:
      x:
        type: Coordinates
        shape: [batch, N, in_features]
        dtype: float32
        description: 物理坐标位置

    outputs:
      x_i:
        type: EmbeddedFeatures
        shape: [batch, N, in_features * freq]
        dtype: float32
        description: 高维频域位置特征

  types:
    Coordinates:
      shape: [..., dim]
      description: 坐标场向量

  implementation:
    framework: pytorch
    code: |
      import math
      
      import numpy as np
      import torch
      import torch.nn as nn
      from torch import Tensor
      
      
      class FNOFourier(nn.Module):
          """Fourier layer used in the Fourier feature network"""
      
          def __init__(
              self,
              in_features: int,
              frequencies,
          ) -> None:
              super().__init__()
      
              # To do: Need more robust way for these params
              if isinstance(frequencies[0], str):
                  if "gaussian" in frequencies[0]:
                      nr_freq = frequencies[2]
                      np_f = (
                          np.random.normal(0, 1, size=(nr_freq, in_features)) * frequencies[1]
                      )
                  else:
                      nr_freq = len(frequencies[1])
                      np_f = []
                      if "full" in frequencies[0]:
                          np_f_i = np.meshgrid(
                              *[np.array(frequencies[1]) for _ in range(in_features)],
                              indexing="ij",
                          )
                          np_f.append(
                              np.reshape(
                                  np.stack(np_f_i, axis=-1),
                                  (nr_freq**in_features, in_features),
                              )
                          )
                      if "axis" in frequencies[0]:
                          np_f_i = np.zeros((nr_freq, in_features, in_features))
                          for i in range(in_features):
                              np_f_i[:, i, i] = np.reshape(
                                  np.array(frequencies[1]), (nr_freq)
                              )
                          np_f.append(
                              np.reshape(np_f_i, (nr_freq * in_features, in_features))
                          )
                      if "diagonal" in frequencies[0]:
                          np_f_i = np.reshape(np.array(frequencies[1]), (nr_freq, 1, 1))
                          np_f_i = np.tile(np_f_i, (1, in_features, in_features))
                          np_f_i = np.reshape(np_f_i, (nr_freq * in_features, in_features))
                          np_f.append(np_f_i)
                      np_f = np.concatenate(np_f, axis=-2)
      
              else:
                  np_f = frequencies  # [nr_freq, in_features]
      
              frequencies = torch.tensor(np_f, dtype=torch.get_default_dtype())
              frequencies = frequencies.t().contiguous()
              self.register_buffer("frequencies", frequencies)
      
          def out_features(self) -> int:
              return int(self.frequencies.size(1) * 2)
      
          def forward(self, x: Tensor) -> Tensor:
              x_hat = torch.matmul(x, self.frequencies)
              x_sin = torch.sin(2.0 * math.pi * x_hat)
              x_cos = torch.cos(2.0 * math.pi * x_hat)
              x_i = torch.cat([x_sin, x_cos], dim=-1)
              return x_i
          
      

  skills:
    build_fourier_encoder:
      description: 生成周期空间嵌入坐标编码层
      inputs:
        - in_features
        - frequencies
      prompt_template: |
        按照 Fourier Features Let Networks Learn High Frequency Functions 建立高斯位置层。

    diagnose_fourier_collapse:
      description: 指出坐标没有经过合理幅度归一化带来的灾难性崩解失效
      checks:
        - coordinate_not_scaled

  knowledge:
    usage_patterns:
      nerf_embedding:
        pipeline:
          - 解析空间坐标 (x, y)
          - Fourier Embedding
          - 拼接原本的输入，送入 MLP

    hot_models:
      - model: NeRF (Neural Radiance Fields)
        year: 2020
      - model: PINNs / FNO
        year: 2021

    best_practices:
      - "对无序节点用 'gaussian' ，对网格图像使用 'full' 分配法"

    anti_patterns:
      - "不加约束对原本就是潜空间高维度的中间特征作 Fourier 展开"

    paper_references:
      - title: "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
        authors: Tancik et al.
        year: 2020

  graph:
    is_a:
      - PositionalEncoding
      - FeatureExpansion
    part_of:
      - NeuralOperator
    depends_on:
      - torch.sin
      - torch.cos
    variants:
      - HashGridEncoding
    used_in_models:
      - NeRF
      - FNO
    compatible_with:
      inputs:
        - Coordinates
      outputs:
        - EmbeddedFeatures