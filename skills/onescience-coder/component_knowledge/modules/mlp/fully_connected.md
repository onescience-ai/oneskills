component:

  meta:
    name: FullyConnected
    alias: Fully Connected Layer
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: linear_layer
    author: OneScience
    license: Apache-2.0
    tags:
      - fully_connected
      - linear_layer
      - dense_layer
      - neural_network
      - weight_matrix

  concept:

    description: >
      全连接层（Fully Connected Layer），也称为密集层（Dense Layer）或线性层（Linear Layer）。
      是神经网络中最基础的组件，通过权重矩阵和偏置向量对输入特征进行线性变换，
      实现特征空间的映射和维度的转换。

    intuition: >
      就像一组旋钮和开关：每个输出都是所有输入的加权和，权重就像旋钮控制每个输入的重要性，
      偏置就像基础开关调整输出基准值。通过调整这些参数，可以学习任意的线性变换。

    problem_it_solves:
      - 特征维度变换
      - 线性空间映射
      - 参数化线性变换
      - 神经网络的基础构建块

  theory:

    formula:

      linear_transformation:
        expression: |
          y = Wx + b
          \text{where } W \in \mathbb{R}^{out\_dim \times in\_dim}, b \in \mathbb{R}^{out\_dim}

      matrix_multiplication:
        expression: |
          y_i = \sum_{j=1}^{in\_dim} W_{ij} x_j + b_i \quad \text{for } i = 1, ..., out\_dim

    variables:

      W:
        name: WeightMatrix
        shape: [output_dim, input_dim]
        description: 权重矩阵，控制输入到输出的线性映射

      b:
        name: BiasVector
        shape: [output_dim]
        description: 偏置向量，调整输出基准

      x:
        name: InputVector
        shape: [input_dim]
        description: 输入特征向量

      y:
        name: OutputVector
        shape: [output_dim]
        description: 输出特征向量

  structure:

    architecture: linear_transformation

    pipeline:

      - name: InputFeatures
        operation: feature_vector

      - name: LinearTransformation
        operation: matrix_multiplication + bias_addition

      - name: OutputFeatures
        operation: transformed_vector

  interface:

    parameters:

      input_dim:
        type: int
        description: 输入特征维度

      output_dim:
        type: int
        description: 输出特征维度

      use_bias:
        type: bool
        description: 是否使用偏置

      weight_init:
        type: str
        description: 权重初始化方法

      bias_init:
        type: str
        description: 偏置初始化方法

    inputs:

      x:
        type: InputFeatures
        shape: [batch, input_dim]
        description: 输入特征

    outputs:

      y:
        type: OutputFeatures
        shape: [batch, output_dim]
        description: 输出特征

  types:

    InputFeatures:
      shape: [batch, features]
      description: 输入特征张量

    OutputFeatures:
      shape: [batch, features]
      description: 输出特征张量

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      class FullyConnected(nn.Module):
          def __init__(self, input_dim, output_dim, use_bias=True, 
                       weight_init='kaiming_uniform', bias_init='zeros'):
              super().__init__()
              self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
              
              # 权重初始化
              if weight_init == 'kaiming_uniform':
                  nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
              elif weight_init == 'xavier_uniform':
                  nn.init.xavier_uniform_(self.linear.weight)
              elif weight_init == 'xavier_normal':
                  nn.init.xavier_normal_(self.linear.weight)
              
              # 偏置初始化
              if use_bias:
                  if bias_init == 'zeros':
                      nn.init.zeros_(self.linear.bias)
                  elif bias_init == 'ones':
                      nn.init.ones_(self.linear.bias)

          def forward(self, x):
              return self.linear(x)

          def extra_repr(self):
              return f'in_features={self.linear.in_features}, out_features={self.linear.out_features}, bias={self.linear.bias is not None}'

  skills:

    build_fully_connected:

      description: 构建全连接层

      inputs:
        - input_dim
        - output_dim
        - use_bias

      prompt_template: |

        构建全连接层，实现线性变换。

        参数：
        input_dim = {{input_dim}}
        output_dim = {{output_dim}}
        use_bias = {{use_bias}}

        要求：
        1. 实现标准的线性变换 y = Wx + b
        2. 支持权重和偏置初始化
        3. 高效的矩阵乘法实现

    analyze_linear_transformation:

      description: 分析线性变换的性质

      checks:
        - weight_matrix_properties (权重矩阵性质)
        - gradient_flow (梯度流动)
        - numerical_stability (数值稳定性)

  knowledge:

    usage_patterns:

      dimension_projection:

        pipeline:
          - Input: 高维特征
          - Linear: 降维/升维投影
          - Output: 变换后特征

      classification_head:

        pipeline:
          - Features: 提取的特征
          - Linear: 分类器
          - Output: 类别分数

      embedding_projection:

        pipeline:
          - Embeddings: 嵌入向量
          - Linear: 投影到目标空间
          - Output: 投影后向量

    hot_models:

      - model: Linear Regression
        year: 1800s
        role: 经典统计学习方法
        architecture: single linear layer

      - model: Perceptron
        year: 1957
        role: 最早的神经网络模型
        architecture: linear layer + step function

    best_practices:

      - 选择合适的权重初始化方法
      - 考虑输入输出的尺度匹配
      - 对于大维度使用适当的正则化
      - 注意数值稳定性问题

    anti_patterns:

      - 权重初始化不当导致训练困难
      - 输入输出维度不匹配
      - 忽略偏置的重要性
      - 梯度爆炸/消失问题

    paper_references:

      - title: "Learning representations by back-propagating errors"
        authors: Rumelhart et al.
        year: 1986

      - title: "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification"
        authors: He et al.
        year: 2015

  graph:

    is_a:
      - NeuralNetworkLayer
      - LinearTransformation
      - ParameterizedFunction

    part_of:
      - MultiLayerPerceptron
      - NeuralNetworks
      - DeepLearningModels

    depends_on:
      - MatrixMultiplication
      - BiasAddition
      - WeightParameters

    variants:
      - LinearLayer (基础版本)
      - DenseLayer (Keras风格)
      - AffineTransformation (数学视角)

    used_in_models:
      - 几乎所有神经网络
      - 分类器
      - 回归器
      - 嵌入层

    compatible_with:

      inputs:
        - FeatureVectors
        - Embeddings
        - Activations

      outputs:
        - LinearTransformations
        - Logits
        - Projections
