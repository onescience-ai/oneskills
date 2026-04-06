component:

  meta:
    name: TimestepEmbedding
    alias: SinusoidalEmbedding
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: embedding
    author: OneScience
    license: Apache-2.0
    tags:
      - embedding
      - timestep
      - sinusoidal
      - transformer
      - diffusion

  concept:

    description: >
      Timestep Embedding 模块用于将标量时间步（或噪声水平）转换为高维向量表示。
      类似于 Transformer 中的位置编码（Positional Encoding），它使用不同频率的正弦和余弦函数对输入的标量进行编码，
      使得神经网络能够感知输入数据中时间步（如扩散模型中的噪声水平）的有规律变化。

    intuition: >
      单纯的标量时间步传给神经网络时，模型由于只有一个维度的变化往往很难捕捉到绝对大小的差异。
      Timestep Embedding 就像是一个可以记录不同时间刻度的“多频率时钟”，分别用快表针（高频）和慢表针（低频）的组合方式，将单一的标量转换为丰富的高维结构，让模型可以清晰辨别时间的进度。

    problem_it_solves:
      - 解决标量时间参数（如扩散步骤）在神经网络深层传递中容易消失或表现不突出的问题
      - 赋予模型对绝对或相对时间步的细粒度位置感知能力


  theory:

    formula:

      sinusoidal_encoding:
        expression: |
          PE(t, 2i) = \sin\left(\frac{t}{\text{max\_period}^{\frac{2i}{dim}}}\right)
          PE(t, 2i+1) = \cos\left(\frac{t}{\text{max\_period}^{\frac{2i}{dim}}}\right)

    variables:

      t:
        name: Timesteps
        shape: [N]
        description: 标量时间步的一维张量

      dim:
        name: EmbeddingDimension
        description: 输出嵌入向量的维度大小

      max_period:
        name: MaximumPeriod
        description: 控制嵌入的最小频率对应的最大周期，常见默认值为 10000


  structure:

    architecture: positional_encoding_components

    pipeline:

      - name: FrequencyComputation
        operation: exponential_decay_frequencies

      - name: ArgumentScaling
        operation: timesteps_times_frequencies

      - name: HarmonicEncoding
        operation: concatenate_cos_sin


  interface:

    parameters:

      dim:
        type: int
        description: 输出嵌入的维度
      
      max_period:
        type: int
        description: 控制嵌入的最大周期，默认为 10000
      
      repeat_only:
        type: bool
        description: 代码保留参数，当前实现未使用

    inputs:

      timesteps:
        type: Tensor
        shape: [N]
        dtype: float32 / int
        description: 包含 N 个时间步索引的一维张量

    outputs:

      embedding:
        type: Tensor
        shape: [N, dim]
        description: 转换后的高维正弦时间步表示


  types:

    TimestepsTensor:
      shape: [N]
      description: 时间刻度输入张量


  implementation:

    framework: pytorch

    code: |
      import math
      import torch
      import torch.nn as nn
      from einops import rearrange
      import numpy as np
      
      def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
          """
              创建正弦时间步长嵌入 (Sinusoidal Timestep Embeddings)。
      
              该函数类似于 Transformer 中的位置编码，用于将标量时间步（或噪声水平）转换为高维向量表示。它使用不同频率的正弦和余弦函数对输入进行编码。
              计算公式如下：
              PE(t, 2i) = sin(t / 10000^(2i/dim))
              PE(t, 2i+1) = cos(t / 10000^(2i/dim))
      
              Args:
                  timesteps (Tensor): 一维张量，包含 N 个时间步索引（可以是分数）。形状为 (N,)。
                  dim (int): 输出嵌入的维度。
                  max_period (int, optional): 控制嵌入的最小频率（最大周期）。默认值: 10000。
                  repeat_only (bool, optional): 代码中保留参数，但在当前实现中未使用。默认值: False。
      
              形状:
                  输入: (N,)
                  输出: (N, dim)
      
              Example:
                  >>> t = torch.arange(0, 10)
                  >>> emb = timestep_embedding(t, dim=128)
                  >>> emb.shape
                  torch.Size([10, 128])
          """
      
          half = dim // 2
          freqs = torch.exp(
              -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
          ).to(device=timesteps.device)
          args = timesteps[:, None].float() * freqs[None]
          embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
          if dim % 2:
              embedding = torch.cat([embedding, torch.zeros_like(embedding[:,:,:1])], dim=-1)
          return embedding
      
      

  skills:

    build_timestep_embedding:

      description: 构建扩散模型或 Transformer 中常用的时间步嵌入向量

      inputs:
        - timesteps
        - dim

      prompt_template: |

        构建正弦时间步长嵌入。

        参数：
        dim = {{dim}}
        max_period = {{max_period}}

        要求：
        使用 torch.exp 结合 math.log 计算频率，并基于 cos 和 sin 的连接生成高维嵌入。对于奇数维度进行补零处理。




    diagnose_timestepembedding:

      description: 分析 TimestepEmbedding 模块在使用中的常见失效模式

      checks:
        - padding_or_resolution_errors
        - bad_device_alignment
        - output_shape_collapse
  knowledge:

    usage_patterns:

      diffusion_models:

        pipeline:
          - sample_noise_level (采样扩散时间步 T)
          - timestep_embedding (获得 T 的高维表征)
          - add_to_features (加在模型各个特征层上)

    hot_models:

      - model: DDPM
        year: 2020
        role: 证明了时间步嵌入在去噪扩散概率模型中的关键作用
        architecture: U-Net + Sinusoidal Embeddings

      - model: Transformer
        year: 2017
        role: 首次提出使用基于正弦和余弦的位置编码组合
        architecture: Encoder-Decoder

    best_practices:

      - 使用高维张量操作替代循环以提升在 GPU 端的计算效率。
      - `max_period` 在时间跨度远大于 10000 或需要极高精度时可以增加。


    anti_patterns:

      - 忽略了嵌入维度的奇偶性判断，导致生成错位的大量边界问题（在本代码通过 dim % 2 处理）。
      - 未将 `freqs` 同步到 `timesteps.device`，可能导致跨设备 (CPU/GPU) 类型错误。


    paper_references:

      - title: "Attention Is All You Need"
        authors: Vaswani et al.
        year: 2017

      - title: "Denoising Diffusion Probabilistic Models"
        authors: Ho et al.
        year: 2020


  graph:

    is_a:
      - PositionalEncoding
      - FeatureTransformation

    part_of:
      - DiffusionModels
      - Transformers

    depends_on:
      - TrigonometricFunctions
      - ExponentialDecay

    variants:
      - LearnablePositionalEmbedding
      - RotaryPositionalEmbedding

    used_in_models:
      - DDPM
      - StableDiffusion
      - Transformer

    compatible_with:

      inputs:
        - Timesteps

      outputs:
        - FeatureEmbedding
