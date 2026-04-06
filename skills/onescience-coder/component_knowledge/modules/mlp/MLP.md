component:

  meta:
    name: StandardMLP
    alias: Multi-Layer Perceptron
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: mlp
    author: OneScience
    license: Apache-2.0
    tags:
      - mlp
      - fully_connected
      - neural_network
      - feature_transformation
      - universal_approximator

  concept:

    description: >
      标准的多层感知机（MLP）实现，支持多种激活函数、归一化层、Dropout、残差连接等高级特性。
      MLP是深度学习的基础组件，通过多层线性变换和非线性激活的组合，能够逼近任意连续函数，
      是通用函数逼近器的经典实现。

    intuition: >
      就像大脑中的神经元网络：每个神经元接收输入信号，进行加权求和，然后通过激活函数决定是否激活。
      多层神经元网络可以学习复杂的非线性关系，就像多层次的决策过程。

    problem_it_solves:
      - 非线性函数逼近
      - 特征变换和降维
      - 分类和回归任务
      - 作为其他复杂模块的构建块
      - 通用函数学习

  theory:

    formula:

      mlp_forward:
        expression: |
          h_0 = x
          h_{i+1} = \sigma(W_i h_i + b_i) \quad \text{for } i = 0, 1, ..., L-1
          y = W_L h_L + b_L

      residual_connection:
        expression: |
          h_{i+1} = h_i + \sigma(W_i h_i + b_i) \quad \text{(residual)}

      batch_normalization:
        expression: |
          \hat{h}_i = \frac{h_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
          h'_{i} = \gamma \hat{h}_i + \beta

    variables:

      W_i:
        name: WeightMatrix
        shape: [hidden_dims[i+1], hidden_dims[i]]
        description: 第i层权重矩阵

      b_i:
        name: BiasVector
        shape: [hidden_dims[i+1]]
        description: 第i层偏置向量

      \sigma:
        name: ActivationFunction
        description: 非线性激活函数

  structure:

    architecture: multi_layer_perceptron

    pipeline:

      - name: InputLayer
        operation: linear_projection

      - name: HiddenLayers
        operation: linear + activation + normalization + dropout

      - name: SkipConnections
        operation: residual/dense/highway connections (optional)

      - name: OutputLayer
        operation: linear_projection + output_activation

  interface:

    parameters:

      input_dim:
        type: int
        description: 输入特征维度

      hidden_dims:
        type: List[int]
        description: 隐藏层维度列表

      output_dim:
        type: int
        description: 输出特征维度

      activation:
        type: str or Callable
        description: 激活函数类型

      output_activation:
        type: str or Callable
        description: 输出层激活函数

      use_bias:
        type: bool
        description: 是否使用偏置

      dropout_rate:
        type: float
        description: Dropout概率

      norm_layer:
        type: str or None
        description: 归一化层类型

      use_skip_connection:
        type: bool or str
        description: 是否使用跳过连接

      weight_init:
        type: str
        description: 权重初始化方法

    inputs:

      x:
        type: InputFeatures
        shape: [batch, input_dim]
        description: 输入特征

    outputs:

      output:
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

      class StandardMLP(nn.Module):
          def __init__(self, input_dim, hidden_dims, output_dim, activation='relu',
                       output_activation=None, use_bias=True, dropout_rate=0.0,
                       norm_layer=None, use_skip_connection=False, **kwargs):
              super().__init__()
              
              self.layers = nn.ModuleList()
              self.norms = nn.ModuleList() if norm_layer else None
              self.dropouts = nn.ModuleList() if dropout_rate > 0 else None
              
              # 构建隐藏层
              dims = [input_dim] + hidden_dims
              for i in range(len(dims) - 1):
                  self.layers.append(nn.Linear(dims[i], dims[i+1], bias=use_bias))
                  
                  if norm_layer:
                      if norm_layer == 'batch_norm':
                          self.norms.append(nn.BatchNorm1d(dims[i+1]))
                      elif norm_layer == 'layer_norm':
                          self.norms.append(nn.LayerNorm(dims[i+1]))
                  
                  if dropout_rate > 0:
                      self.dropouts.append(nn.Dropout(dropout_rate))
              
              # 输出层
              self.output_layer = nn.Linear(dims[-1], output_dim, bias=use_bias)
              
              # 激活函数
              self.activation = self._get_activation(activation)
              self.output_activation = self._get_activation(output_activation) if output_activation else None
              
              self.use_skip_connection = use_skip_connection

          def forward(self, x):
              residual = x if self.use_skip_connection else None
              
              # 隐藏层前向传播
              for i, layer in enumerate(self.layers):
                  x = layer(x)
                  
                  if self.norms:
                      x = self.norms[i](x)
                  
                  x = self.activation(x)
                  
                  if self.dropouts:
                      x = self.dropouts[i](x)
                  
                  if self.use_skip_connection and residual is not None:
                      x = x + residual
                      residual = x
              
              # 输出层
              x = self.output_layer(x)
              if self.output_activation:
                  x = self.output_activation(x)
              
              return x

          def _get_activation(self, activation):
              if activation == 'relu':
                  return nn.ReLU()
              elif activation == 'gelu':
                  return nn.GELU()
              elif activation == 'tanh':
                  return nn.Tanh()
              elif activation == 'sigmoid':
                  return nn.Sigmoid()
              else:
                  return nn.Identity()

  skills:

    build_mlp:

      description: 构建标准多层感知机

      inputs:
        - input_dim
        - hidden_dims
        - output_dim
        - activation

      prompt_template: |

        构建标准MLP，支持多种配置选项。

        参数：
        input_dim = {{input_dim}}
        hidden_dims = {{hidden_dims}}
        output_dim = {{output_dim}}
        activation = {{activation}}

        要求：
        1. 支持多种激活函数
        2. 可选归一化和Dropout
        3. 支持残差连接
        4. 灵活的权重初始化

    optimize_mlp:

      description: 优化MLP的性能和稳定性

      checks:
        - initialization_consistency (权重初始化一致性)
        - gradient_flow (梯度流动)
        - numerical_stability (数值稳定性)

  knowledge:

    usage_patterns:

      feature_transformation:

        pipeline:
          - Input: 原始特征
          - Hidden Layers: 特征变换
          - Non-linearity: 非线性激活
          - Output: 变换后特征

      classification_head:

        pipeline:
          - Features: 提取的特征
          - MLP: 分类器
          - Softmax: 概率输出
          - Prediction: 分类结果

    hot_models:

      - model: Original MLP
        year: 1957
        role: 神经网络的经典架构
        architecture: multi-layer perceptron

      - model: ResNet MLP Head
        year: 2015
        role: ResNet的分类头
        architecture: mlp with residual connections

    best_practices:

      - 选择合适的激活函数（ReLU/GELU用于深度网络）
      - 使用适当的权重初始化方法
      - 考虑使用BatchNorm提高训练稳定性
      - 残差连接有助于深层网络训练

    anti_patterns:

      - 激活函数选择不当导致梯度消失
      - 权重初始化不当导致训练困难
      - 网络过深而没有残差连接
      - 忽略归一化导致训练不稳定

    paper_references:

      - title: "Learning representations by back-propagating errors"
        authors: Rumelhart et al.
        year: 1986

      - title: "Deep residual learning for image recognition"
        authors: He et al.
        year: 2015

  graph:

    is_a:
      - NeuralNetwork
      - FunctionApproximator
      - FeatureTransformer

    part_of:
      - DeepLearningModels
      - ClassificationNetworks
      - RegressionNetworks

    depends_on:
      - LinearLayer
      - ActivationFunction
      - NormalizationLayer
      - DropoutLayer

    variants:
      - SimpleMLP (基础版本)
      - ResidualMLP (残差版本)
      - HighwayMLP (highway网络)
      - GatedMLP (门控版本)

    used_in_models:
      - 几乎所有深度学习模型
      - 分类器头部
      - 特征变换器

    compatible_with:

      inputs:
        - FeatureVectors
        - Embeddings
        - Tensors

      outputs:
        - TransformedFeatures
        - ClassificationLogits
        - RegressionValues
