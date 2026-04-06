component:

  meta:
    name: FourCastNetFeedForward
    alias: FourCastNetFC
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: feed_forward_network
    author: OneScience
    license: Apache-2.0
    tags:
      - feed_forward
      - mlp
      - gelu
      - fourcastnet
      - fluid_dynamics
      - weather_forecasting

  concept:

    description: >
      FourCastNetFC 是一个标准的两层前馈神经网络（Feed-Forward Network, FFN）或多层感知机（MLP）模块。
      该模块专为 FourCastNet（一种数据驱动的高分辨率全球天气预报模型）以及类似的 AI 求解流体力学/偏微分方程（PDE）的架构设计。
      它通常紧跟在注意力机制（Attention）或傅里叶神经算子（FNO）之后，用于在特征通道维度上进行非线性特征变换。

    intuition: >
      类似于 Vision Transformer (ViT) 中的 FFN 层，该模块采用“升维-激活-降维”的瓶颈结构。
      它首先将输入特征投影到一个更高维的隐藏空间（默认放大4倍，例如从 768 升至 3072），
      在这个高维空间中应用非线性激活函数（默认使用 GELU）来提取复杂的特征交互，
      然后再将特征投影回原始的输出维度。这种机制赋予了模型在复杂流体动力学和气象物理场中捕捉非线性映射的能力。

    problem_it_solves:
      - 提供特征通道维度上的非线性映射与特征提取能力
      - 增强 AI 模型对复杂物理场（如流场、气象场）演变的拟合和表达能力
      - 通过引入 Dropout 机制，缓解大规模数据驱动物理模型训练时的过拟合问题

  theory:

    formula:

      fourcastnet_fc_output:
        expression: output = Dropout(W_2 * Dropout(GELU(W_1 * x + b_1)) + b_2)

    variables:

      x:
        name: Input
        shape: [..., in_features]
        description: 输入的特征张量

      W_1:
        name: Weight1
        shape: [hidden_features, in_features]
        description: 第一层全连接层的权重矩阵（升维）

      b_1:
        name: Bias1
        description: 第一层全连接层的偏置项

      W_2:
        name: Weight2
        shape: [out_features, hidden_features]
        description: 第二层全连接层的权重矩阵（降维）

      b_2:
        name: Bias2
        description: 第二层全连接层的偏置项

  structure:

    architecture: multilayer_perceptron

    pipeline:

      - name: FC1
        operation: linear_projection (in_features -> hidden_features)

      - name: Activation
        operation: non_linear_activation (default: GELU)

      - name: Dropout1
        operation: feature_dropout

      - name: FC2
        operation: linear_projection (hidden_features -> out_features)

      - name: Dropout2
        operation: feature_dropout

  interface:

    parameters:

      in_features:
        type: int
        default: 768
        description: 输入特征的维度

      hidden_features:
        type: int
        default: 3072
        description: 隐藏层特征的维度，通常为 in_features 的 4 倍

      out_features:
        type: int
        default: null
        description: 输出特征的维度。如果未指定，则默认等于 in_features

      act_layer:
        type: nn.Module
        default: nn.GELU
        description: 隐藏层使用的激活函数

      drop:
        type: float
        default: 0.0
        description: Dropout 的概率值，用于防止过拟合

    inputs:

      x:
        type: Tensor
        shape: "[..., in_features]"
        dtype: float32
        description: 任意维度的输入张量，只要最后一维匹配 in_features 即可

    outputs:

      output:
        type: Tensor
        shape: "[..., out_features]"
        description: 经过两层线性映射和非线性激活后的输出特征

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn

      class FourCastNetFC(nn.Module):
          """FourCastNet 中使用的前馈神经网络 (FFN) 模块"""
          def __init__(self, in_features=768, hidden_features=3072, out_features=None, act_layer=nn.GELU, drop=0.):
              super().__init__()
              out_features = out_features or in_features
              hidden_features = hidden_features or in_features
              self.fc1 = nn.Linear(in_features, hidden_features)
              self.act = act_layer()
              self.fc2 = nn.Linear(hidden_features, out_features)
              self.drop = nn.Dropout(drop)

          def forward(self, x):
              x = self.fc1(x)
              x = self.act(x)
              x = self.drop(x)
              x = self.fc2(x)
              x = self.drop(x)
              return x

  skills:

    build_ffn_layer:

      description: 为 Transformer 或 FNO 架构构建标准的前馈神经网络层

      inputs:
        - in_features
        - hidden_features_ratio (通常为 4)
        - dropout_rate

      prompt_template: |
        构建一个 FourCastNetFC 层。
        参数：
        输入维度 = {{in_features}}
        隐藏维度 = {{in_features}} * {{hidden_features_ratio}}
        Dropout率 = {{dropout_rate}}

    diagnose_ffn:

      description: 分析该模块在气象/流体模型训练中的潜在问题

      checks:
        - over-smoothing_in_deep_layers
        - memory_bloat_due_to_large_hidden_features

  knowledge:

    usage_patterns:

      vision_transformer_block:

        pipeline:
          - LayerNorm
          - SelfAttention / FNO Layer
          - ResidualAdd
          - LayerNorm
          - FourCastNetFC (当前模块)
          - ResidualAdd

    design_patterns:

      inverted_bottleneck:

        structure:
          - 在两层 Linear 之间使用比输入维度大得多的 hidden_features (通常是 4x)
          - 目的是在高维空间中解开特征纠缠，使激活函数更容易分离复杂的物理特征规律

    hot_models:

      - model: FourCastNet
        year: 2022
        role: 高分辨率数据驱动全球天气预报模型
        architecture: vision_transformer + fourier_neural_operator
        attention_type: None (此处作为 Block 的 FFN 部分)

      - model: Vision Transformer (ViT)
        year: 2020
        role: 计算机视觉骨干网络
        architecture: transformer_encoder

    model_usage_details:

      FourCastNet:

        in_features: 768
        hidden_features: 3072 (4倍放大)
        activation: GELU

    best_practices:

      - `hidden_features` 一般建议设置为 `in_features` 的 4 倍，这是兼顾模型容量和计算效率的经验默认值。
      - 在处理流体或气象等连续物理场时，`GELU` 的平滑非线性特性通常比硬截断的 `ReLU` 表现更好，有助于梯度的稳定传播。
      - 如果模型的深度很深，建议合理设置 `drop` 参数以提升模型在未见物理边界条件下的泛化能力。

    anti_patterns:

      - 将 `hidden_features` 设置得过小（如小于 `in_features`），这会导致严重的信息瓶颈，损失流场特征的表达能力。
      - 在不需要正则化的简单拟合任务中滥用过高的 `drop` 值，会导致模型难以收敛。

    paper_references:

      - title: "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators"
        authors: Pathak et al.
        year: 2022

      - title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
        authors: Dosovitskiy et al.
        year: 2020

  graph:

    is_a:
      - NeuralNetworkComponent
      - FeedForwardNetwork
      - MultiLayerPerceptron

    part_of:
      - TransformerBlock
      - FourCastNetBlock

    depends_on:
      - Linear
      - GELU
      - Dropout

    variants:
      - Standard MLP
      - SwiGLU FFN (目前大语言模型常用的变体)

    used_in_models:
      - FourCastNet
      - Vision Transformer
      - 流体力学 AI 代理模型 (AI for Fluid Dynamics)

    compatible_with:

      inputs:
        - Tensor

      outputs:
        - Tensor