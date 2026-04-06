component:

  meta:
    name: FullyConnectedLayers
    alias: FCLayer
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: core_component
    author: OneScience
    license: Apache-2.0
    tags:
      - fully_connected
      - dense_layer
      - 1x1_convolution
      - weight_normalization
      - weight_factorization

  concept:

    description: >
      全连接层及其变体模块（Fully Connected Layer & Conv FC Layer）。
      标准的全连接层（FCLayer）将输入特征进行线性映射并应用可选的激活函数。
      为了处理具有空间或时间维度的数据（如图像、时间序列），模块还实现了基于 $1 \times 1$ 卷积的等效全连接层（Conv1d/2d/3d/Nd FCLayer），
      实现通道维度（Channel-wise）的特征混合，而不破坏其空间/时间结构。该模块还内建了权重归一化（Weight Norm）和权重分解（Weight Fact）机制。

    intuition: >
      标准全连接层就像是对数据进行一次全局的线性变换和投影。
      而基于 $1 \times 1$ 卷积的 FC 层，则相当于在每一个空间像素（或时间步）上独立且共享地应用一个全连接层，
      这在改变通道数（降维或升维）、跨通道整合信息时非常有效。

    problem_it_solves:
      - 实现高维特征空间的线性映射与维度转换（Projection）
      - 通过 $1 \times 1$ 卷积实现序列、图像、体素数据的逐像素/逐时间步通道特征融合
      - 集成了权重归一化（Weight Normalization），加速模型收敛并稳定训练
      - 提供可学习参数的激活函数（Learnable Activations）支持

  theory:

    formula:

      fc_output:
        expression: y = \sigma(\alpha \cdot (W x + b))

      conv_fc_output:
        expression: y_{c_{out}, \dots} = \sigma(\alpha \cdot (\sum_{c_{in}} W_{c_{out}, c_{in}} x_{c_{in}, \dots} + b_{c_{out}}))

    variables:

      x:
        name: Input
        shape: [batch, in_features] or [batch, in_channels, ...]
        description: 输入特征或张量

      W:
        name: WeightMatrix
        description: 权重矩阵或 $1 \times 1$ 卷积核

      b:
        name: Bias
        description: 偏置项

      \sigma:
        name: ActivationFunction
        description: 激活函数，如 Identity, ReLU 等

      \alpha:
        name: ActivationParameter
        description: 激活函数的可学习缩放参数（对应代码中的 activation_par）

  structure:

    architecture: feed_forward

    pipeline:

      - name: LinearOrConvProjection
        operation: linear_or_1x1_conv

      - name: ParameterizedScaling
        operation: scale_by_activation_par (可选)

      - name: Activation
        operation: apply_activation_fn

  interface:

    parameters:

      in_features:
        type: int
        description: 输入特征或通道的数量

      out_features:
        type: int
        description: 输出特征或通道的数量

      activation_fn:
        type: Union[nn.Module, Callable, None]
        default: None
        description: 激活函数，默认为 Identity

      weight_norm:
        type: bool
        default: false
        description: 是否应用权重归一化（Weight Normalization）

      weight_fact:
        type: bool
        default: false
        description: 是否应用权重分解（Weight Factorization）

      activation_par:
        type: Union[nn.Parameter, None]
        default: None
        description: 用于激活函数的附加可学习参数

    inputs:

      x:
        type: Tensor
        shape: "[batch, in_features] 或 [batch, in_channels, spatial_dims...]"
        dtype: float32

    outputs:

      output:
        type: Tensor
        shape: "[batch, out_features] 或 [batch, out_channels, spatial_dims...]"

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      from typing import Callable, Union

      import torch.nn as nn
      from torch import Tensor

      from .activations import Identity
      from .weight_fact import WeightFactLinear
      from .weight_norm import WeightNormLinear


      class FCLayer(nn.Module):
          """Densely connected NN layer
          
          Parameters
          ----------
          in_features : int
              Size of input features
          out_features : int
              Size of output features
          activation_fn : Union[nn.Module, None], optional
              Activation function to use. Can be None for no activation, by default None
          weight_norm : bool, optional
              Applies weight normalization to the layer, by default False
          weight_fact : bool, optional
              Applies weight factorization to the layer, by default False
          activation_par : Union[nn.Parameter, None], optional
              Additional parameters for the activation function, by default None
          """

          def __init__(
              self,
              in_features: int,
              out_features: int,
              activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
              weight_norm: bool = False,
              weight_fact: bool = False,
              activation_par: Union[nn.Parameter, None] = None,
          ) -> None:
              super().__init__()

              if activation_fn is None:
                  self.activation_fn = Identity()
              else:
                  self.activation_fn = activation_fn
              self.weight_norm = weight_norm
              self.weight_fact = weight_fact
              self.activation_par = activation_par

              # Ensure weight_norm and weight_fact are not both True
              if weight_norm and weight_fact:
                  raise ValueError(
                      "Cannot apply both weight normalization and weight factorization together, please select one."
                  )

              if weight_norm:
                  self.linear = WeightNormLinear(in_features, out_features, bias=True)
              elif weight_fact:
                  self.linear = WeightFactLinear(in_features, out_features, bias=True)
              else:
                  self.linear = nn.Linear(in_features, out_features, bias=True)
              self.reset_parameters()

          def reset_parameters(self) -> None:
              """Reset fully connected weights"""
              if not self.weight_norm and not self.weight_fact:
                  nn.init.constant_(self.linear.bias, 0)
                  nn.init.xavier_uniform_(self.linear.weight)

          def forward(self, x: Tensor) -> Tensor:
              x = self.linear(x)

              if self.activation_par is None:
                  x = self.activation_fn(x)
              else:
                  x = self.activation_fn(self.activation_par * x)

              return x


      class ConvFCLayer(nn.Module):
          """Base class for 1x1 Conv layer for image channels"""
          
          def __init__(
              self,
              activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
              activation_par: Union[nn.Parameter, None] = None,
          ) -> None:
              super().__init__()
              if activation_fn is None:
                  self.activation_fn = Identity()
              else:
                  self.activation_fn = activation_fn
              self.activation_par = activation_par

          def apply_activation(self, x: Tensor) -> Tensor:
              if self.activation_par is None:
                  x = self.activation_fn(x)
              else:
                  x = self.activation_fn(self.activation_par * x)
              return x


      class Conv1dFCLayer(ConvFCLayer):
          """Channel-wise FC like layer with 1d convolutions"""
          
          def __init__(
              self,
              in_features: int,
              out_features: int,
              activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
              activation_par: Union[nn.Parameter, None] = None,
              weight_norm: bool = False,
          ) -> None:
              super().__init__(activation_fn, activation_par)
              self.in_channels = in_features
              self.out_channels = out_features
              self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
              self.reset_parameters()

              if weight_norm:
                  raise NotImplementedError("Weight norm not supported for Conv FC layers")

          def reset_parameters(self) -> None:
              nn.init.constant_(self.conv.bias, 0)
              nn.init.xavier_uniform_(self.conv.weight)

          def forward(self, x: Tensor) -> Tensor:
              x = self.conv(x)
              x = self.apply_activation(x)
              return x


      class Conv2dFCLayer(ConvFCLayer):
          """Channel-wise FC like layer with 2d convolutions"""
          
          def __init__(
              self,
              in_channels: int,
              out_channels: int,
              activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
              activation_par: Union[nn.Parameter, None] = None,
          ) -> None:
              super().__init__(activation_fn, activation_par)
              self.in_channels = in_channels
              self.out_channels = out_channels
              self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
              self.reset_parameters()

          def reset_parameters(self) -> None:
              nn.init.constant_(self.conv.bias, 0)
              self.conv.bias.requires_grad = False
              nn.init.xavier_uniform_(self.conv.weight)

          def forward(self, x: Tensor) -> Tensor:
              x = self.conv(x)
              x = self.apply_activation(x)
              return x


      class Conv3dFCLayer(ConvFCLayer):
          """Channel-wise FC like layer with 3d convolutions"""
          
          def __init__(
              self,
              in_channels: int,
              out_channels: int,
              activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
              activation_par: Union[nn.Parameter, None] = None,
          ) -> None:
              super().__init__(activation_fn, activation_par)
              self.in_channels = in_channels
              self.out_channels = out_channels
              self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
              self.reset_parameters()

          def reset_parameters(self) -> None:
              nn.init.constant_(self.conv.bias, 0)
              nn.init.xavier_uniform_(self.conv.weight)

          def forward(self, x: Tensor) -> Tensor:
              x = self.conv(x)
              x = self.apply_activation(x)
              return x


      class ConvNdFCLayer(ConvFCLayer):
          """Channel-wise FC like layer with convolutions of arbitrary dimensions"""
          
          def __init__(
              self,
              in_channels: int,
              out_channels: int,
              activation_fn: Union[nn.Module, None] = None,
              activation_par: Union[nn.Parameter, None] = None,
          ) -> None:
              super().__init__(activation_fn, activation_par)
              self.in_channels = in_channels
              self.out_channels = out_channels
              self.conv = ConvNdKernel1Layer(in_channels, out_channels)
              self.reset_parameters()

          def reset_parameters(self):
              self.conv.apply(self.initialise_parameters) 

          def initialise_parameters(self, model):
              if hasattr(model, "bias"):
                  nn.init.constant_(model.bias, 0)
              if hasattr(model, "weight"):
                  nn.init.xavier_uniform_(model.weight)

          def forward(self, x: Tensor) -> Tensor:
              x = self.conv(x)
              x = self.apply_activation(x)
              return x


      class ConvNdKernel1Layer(nn.Module):
          """Channel-wise FC like layer for convolutions of arbitrary dimensions"""
          
          def __init__(
              self,
              in_channels: int,
              out_channels: int,
          ) -> None:
              super().__init__()
              self.in_channels = in_channels
              self.out_channels = out_channels
              self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

          def forward(self, x: Tensor) -> Tensor:
              dims = list(x.size())
              dims[1] = self.out_channels
              x = self.conv(x.view(dims[0], self.in_channels, -1)).view(dims)
              return x

  skills:

    build_fc_layer:

      description: 根据输入的特征维度和数据类型（1D/2D/3D等）构建合适的 FC 层

      inputs:
        - in_features
        - out_features
        - data_dimension (flat, 1d, 2d, 3d, nd)
        - activation_fn

      prompt_template: |
        构建一个针对 {{data_dimension}} 数据的全连接层。
        输入特征：{{in_features}}
        输出特征：{{out_features}}
        注意：如果不基于特定维度的数据，请使用默认的 FCLayer；如果数据具有空间结构，请选择相应的 Conv*dFCLayer，以确保不破坏空间结构。

    diagnose_fc:

      description: 分析 FC 层或 Conv FC 层配置中的潜在冲突与问题

      checks:
        - conflict_between_weight_norm_and_weight_fact
        - inappropriate_use_of_flat_fc_on_spatial_data

  knowledge:

    usage_patterns:

      standard_mlp:
        pipeline:
          - FCLayer
          - FCLayer (带有不同激活函数)

      bottleneck_layer_in_vision:
        pipeline:
          - Conv2dFCLayer (用于降维，通道数减少)
          - Conv2d (3x3 空间特征提取)
          - Conv2dFCLayer (用于升维，恢复通道数)

    design_patterns:

      channel_mixing_optimization:
        structure:
          - 使用 1x1 卷积（Conv*dFCLayer）替代传统的 Flatten + Linear
          - 在不破坏空间特征（长宽或深度）的情况下，实现跨通道的特征融合与线性映射

    hot_models:

      - model: ResNet (Bottleneck Block)
        year: 2015
        role: 计算机视觉经典架构
        architecture: cnn
        attention_type: None (大量使用 1x1 Conv 作为 FCLayer)

      - model: PointNet
        year: 2017
        role: 3D 点云处理架构
        architecture: mlp_with_shared_weights
        attention_type: None (使用等效的 Conv1d 充当逐点的 FCLayer)

      - model: Network in Network (NiN)
        year: 2013
        role: 早期提出 1x1 卷积用于通道融合的模型
        architecture: cnn

    model_usage_details:

      ResNet:
        usage: 在 Bottleneck 中，使用 1x1 卷积进行降维和升维，以减少 3x3 卷积的计算量。

      PointNet:
        usage: 对 Nx3 的点云输入使用 1x1 Conv1d，相当于对每一个点独立共享使用 MLP。

    best_practices:

      - 默认初始化推荐使用 Xavier Uniform (`nn.init.xavier_uniform_`)，有助于保持信号在层间传播的方差稳定。
      - `weight_norm`（权重归一化）和 `weight_fact`（权重分解）这两种技术不能同时使用。
      - 当处理具有空间维度（图像或体素）或序列维度的数据时，应使用对应的 `Conv*dFCLayer`，避免先展平（Flatten）再送入普通的线性层。

    anti_patterns:

      - 在具有高度空间结构的大尺寸特征图上直接使用普通的 `nn.Linear`（这会破坏空间结构，且产生极其庞大的参数矩阵）。
      - 在同一个 FCLayer 中同时开启 `weight_norm` 和 `weight_fact`。

    paper_references:

      - title: "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks"
        authors: Salimans & Kingma
        year: 2016

      - title: "Network In Network"
        authors: Lin et al.
        year: 2013

  graph:

    is_a:
      - NeuralNetworkComponent
      - LinearTransformation

    part_of:
      - MultiLayerPerceptron
      - ConvolutionalNeuralNetwork
      - TransformerBlock (FFN)

    depends_on:
      - Linear
      - Conv1d
      - Conv2d
      - Conv3d
      - ActivationFunction

    variants:
      - FCLayer
      - Conv1dFCLayer
      - Conv2dFCLayer
      - Conv3dFCLayer
      - ConvNdFCLayer

    used_in_models:
      - ResNet
      - PointNet
      - Transformer (作为 Feed Forward Network)

    compatible_with:

      inputs:
        - Tensor (任意维度)

      outputs:
        - Tensor (与输入同空间/时间维度)