component:

  meta:
    name: GMLP
    alias: Gated Multi-Layer Perceptron
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: gated_mlp
    author: OneScience
    license: Apache-2.0
    tags:
      - gmlp
      - gated_mlp
      - attention_free
      - spatial_gating
      - efficient_transformer

  concept:

    description: >
      Gated Multi-Layer Perceptron (GMLP) 是一种无需注意力机制的高效架构。
      通过空间门控单元（Spatial Gating Unit, SGU）控制信息流，在保持线性复杂度的同时实现类似注意力的效果。
      GMLP证明了在许多任务中，简单的门控机制可以替代复杂的注意力计算。

    intuition: >
      就像智能交通系统中的信号灯：不是让每辆车都决定去哪里（注意力），而是通过信号灯（门控）
      控制不同方向的车流。空间门控就像为每个特征维度设置"红绿灯"，控制信息的流动。

    problem_it_solves:
      - 注意力机制的计算复杂度问题
      - 长序列建模的效率问题
      - 注意力-free的替代方案
      - 内存和计算优化

  theory:

    formula:

      spatial_gating_unit:
        expression: |
          u = \text{Split}(x) \quad \text{(split along channel)}
          v = \text{Split}(x)
          s = \sigma(\text{Linear}(v)) \odot \text{Linear}(u)
          \text{SGU}(x) = x + s

      gmlp_block:
        expression: |
          x' = \text{LayerNorm}(x)
          x' = \text{Linear}_1(x')
          x' = \text{GeLU}(x')
          x' = \text{SGU}(x')
          x' = \text{Linear}_2(x')
          output = x + x'

    variables:

      s:
        name: SpatialGate
        description: 空间门控信号

      \sigma:
        name: Sigmoid
        description: 门控激活函数

  structure:

    architecture: gated_mlp

    pipeline:

      - name: InputNorm
        operation: layer_normalization

      - name: ChannelProjection
        operation: linear_projection

      - name: Activation
        operation: gelu_activation

      - name: SpatialGating
        operation: spatial_gating_unit

      - name: OutputProjection
        operation: linear_projection

      - name: ResidualConnection
        operation: addition

  interface:

    parameters:

      input_dim:
        type: int
        description: 输入特征维度

      hidden_dim:
        type: int
        description: 隐藏层维度

      output_dim:
        type: int
        description: 输出特征维度

      num_blocks:
        type: int
        description: GMLP块的数量

      dropout_rate:
        type: float
        description: Dropout概率

    inputs:

      x:
        type: InputFeatures
        shape: [batch, seq_len, input_dim]
        description: 输入特征

    outputs:

      output:
        type: OutputFeatures
        shape: [batch, seq_len, output_dim]
        description: 输出特征

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      class SpatialGatingUnit(nn.Module):
          def __init__(self, dim):
              super().__init__()
              self.dim = dim
              self.proj = nn.Linear(dim, dim)
              
          def forward(self, x):
              x, u = x.chunk(2, dim=-1)
              gate = torch.sigmoid(self.proj(u))
              return x * gate

      class GMLPBlock(nn.Module):
          def __init__(self, dim, hidden_dim, dropout=0.0):
              super().__init__()
              self.norm = nn.LayerNorm(dim)
              self.fc1 = nn.Linear(dim, hidden_dim * 2)  # 2x for split
              self.sgu = SpatialGatingUnit(hidden_dim)
              self.fc2 = nn.Linear(hidden_dim, dim)
              self.dropout = nn.Dropout(dropout)
              
          def forward(self, x):
              residual = x
              x = self.norm(x)
              x = self.fc1(x)
              x = F.gelu(x)
              x = self.sgu(x)
              x = self.fc2(x)
              x = self.dropout(x)
              return x + residual

      class GMLP(nn.Module):
          def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=2, dropout=0.0):
              super().__init__()
              self.input_proj = nn.Linear(input_dim, hidden_dim)
              self.blocks = nn.ModuleList([
                  GMLPBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_blocks)
              ])
              self.output_proj = nn.Linear(hidden_dim, output_dim)
              
          def forward(self, x):
              x = self.input_proj(x)
              for block in self.blocks:
                  x = block(x)
              x = self.output_proj(x)
              return x

  skills:

    build_gmlp:

      description: 构建门控多层感知机

  knowledge:

    hot_models:

      - model: GMLP
        year: 2021
        role: 注意力free的高效架构
        architecture: gated mlp

    best_practices:

      - 隐藏层维度通常是输入维度的2-4倍
      - 空间门控提供类似注意力的效果
      - 适合长序列建模任务

    paper_references:

      - title: "Pay Attention to MLPs"
        authors: Liu et al.
        year: 2021

  graph:

    is_a:
      - MultiLayerPerceptron
      - AttentionFreeNetwork
      - EfficientArchitecture

    used_in_models:
      - GMLP-based models
      - Efficient transformers

    compatible_with:

      inputs:
        - SequenceFeatures
        - TokenEmbeddings

      outputs:
        - TransformedFeatures
        - SequenceRepresentations
