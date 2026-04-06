component:

  meta:
    name: AdvancedActivationFunctions
    alias: Activations
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: activation_function
    author: OneScience
    license: Apache-2.0
    tags:
      - activation
      - non_linearity
      - stan
      - squareplus
      - pinns
      - physics_informed

  concept:

    description: >
      该模块提供了一系列标准和专门设计的非线性激活函数，并包含一个基于字符串标识符的统一工厂接口（`get_activation`）。
      除了集成 PyTorch 原生的激活函数外，还引入了针对物理信息神经网络（PINNs）优化的自适应激活函数 `Stan`、
      平滑且计算高效的 `SquarePlus`，以及防止数值爆炸的带上限截断的激活函数（如 `CappedLeakyReLU` 和 `CappedGELU`）。

    intuition: >
      激活函数的作用是为神经网络引入非线性，使其能够拟合复杂的映射关系。
      标准激活函数（如 ReLU、GELU）在常规任务中表现良好，但在求解偏微分方程（PDE）等物理场景中，常需要高阶导数连续平滑的激活函数。
      `Stan` (Self-scalable Tanh) 引入了可学习参数，动态调整非线性区域以加速收敛；
      `SquarePlus` 是一种极近 Softplus 的平滑函数，但避免了指数计算；
      而截断变体则像是一个“安全阀”，确保网络层输出的绝对值不会在深层或混合精度训练中失控。

    problem_it_solves:
      - 解决物理信息神经网络（PINNs）中传统激活函数（如 Tanh）导致的梯度消失和收敛缓慢问题
      - 提供平滑、无需指数运算的非线性映射（SquarePlus），降低硬件计算开销
      - 限制激活值上限（Capped 变体），防止混合精度（FP16/BF16）训练时发生数值溢出
      - 提供基于字符串字典（Registry）的激活函数统一实例化接口，利于配置驱动开发

  theory:

    formula:

      stan_output:
        expression: y = \tanh(x) \cdot (1.0 + \beta \cdot x)

      squareplus_output:
        expression: y = 0.5 \cdot (x + \sqrt{x^2 + b})

      capped_activation_output:
        expression: y = \min(\text{Activation}(x), \text{cap\_value})

    variables:

      x:
        name: Input
        shape: dynamic
        description: 输入张量

      \beta:
        name: LearnableBeta
        shape: [out_features]
        description: Stan 激活函数中的可学习缩放参数

      b:
        name: SmoothingParameter
        description: SquarePlus 的平滑参数，代码中硬编码为 4

      cap\_value:
        name: CapValue
        description: Capped 变体中的最大截断阈值

  structure:

    architecture: element_wise_operation

    pipeline:

      - name: LookupActivation
        operation: registry_lookup_via_string

      - name: ComputeNonLinearity
        operation: apply_mathematical_function

      - name: ApplyCap
        operation: clamp_max_value (针对 Capped 变体)

  interface:

    parameters:

      activation:
        type: str
        description: 字符串标识，用于在 get_activation 工厂函数中获取激活层（如 "relu", "stan", "capped_gelu"）

      out_features:
        type: int
        default: 1
        description: 仅针对 Stan 有效，指定输入张量最后一维的大小，以初始化可学习参数 beta

      cap_value:
        type: float
        default: 1.0
        description: 仅针对 Capped 变体有效，指定激活的最大阈值

    inputs:

      x:
        type: Tensor
        shape: dynamic
        dtype: float32

    outputs:

      output:
        type: Tensor
        shape: dynamic
        description: 与输入形状相同的非线性映射结果

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      import torch
      import torch.nn as nn

      import onescience  # noqa: F401 for docs

      Tensor = torch.Tensor

      class Identity(nn.Module):
          """Identity activation function"""
          def forward(self, x: Tensor) -> Tensor:
              return x

      class Stan(nn.Module):
          """Self-scalable Tanh (Stan) for 1D Tensors"""
          def __init__(self, out_features: int = 1):
              super().__init__()
              self.beta = nn.Parameter(torch.ones(out_features))

          def forward(self, x: Tensor) -> Tensor:
              if x.shape[-1] != self.beta.shape[-1]:
                  raise ValueError(
                      f"The last dimension of the input must be equal to the dimension of Stan parameters. Got inputs: {x.shape}, params: {self.beta.shape}"
                  )
              return torch.tanh(x) * (1.0 + self.beta * x)

      class SquarePlus(nn.Module):
          """Squareplus activation"""
          def __init__(self):
              super().__init__()
              self.b = 4

          def forward(self, x: Tensor) -> Tensor:
              return 0.5 * (x + torch.sqrt(x * x + self.b))

      class CappedLeakyReLU(torch.nn.Module):
          """Implements a ReLU with capped maximum value."""
          def __init__(self, cap_value=1.0, **kwargs):
              super().__init__()
              self.add_module("leaky_relu", torch.nn.LeakyReLU(**kwargs))
              self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

          def forward(self, inputs):
              x = self.leaky_relu(inputs)
              x = torch.clamp(x, max=self.cap)
              return x

      class CappedGELU(torch.nn.Module):
          """Implements a GELU with capped maximum value."""
          def __init__(self, cap_value=1.0, **kwargs):
              super().__init__()
              self.add_module("gelu", torch.nn.GELU(**kwargs))
              self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

          def forward(self, inputs):
              x = self.gelu(inputs)
              x = torch.clamp(x, max=self.cap)
              return x

      # Dictionary of activation functions
      ACT2FN = {
          "relu": nn.ReLU,
          "leaky_relu": (nn.LeakyReLU, {"negative_slope": 0.1}),
          "gelu": nn.GELU,
          "identity": Identity,
          "stan": Stan,
          "squareplus": SquarePlus,
          "cappek_leaky_relu": CappedLeakyReLU,
          "capped_gelu": CappedGELU,
          # (省略了部分标准 PyTorch 激活函数以保持简洁)
      }

      def get_activation(activation: str) -> nn.Module:
          """Returns an activation function given a string"""
          try:
              activation = activation.lower()
              module = ACT2FN[activation]
              if isinstance(module, tuple):
                  return module[0](**module[1])
              else:
                  return module()
          except KeyError:
              raise KeyError(f"Activation function {activation} not found.")

  skills:

    build_activation:

      description: 使用工厂方法根据配置字符串构建激活函数层

      inputs:
        - activation_string (例如 "stan", "gelu")

      prompt_template: |
        调用 get_activation 构建一个 "{{activation_string}}" 激活层。

    diagnose_activation_errors:

      description: 分析由定制激活函数引起的张量形状或未注册错误

      checks:
        - stan_beta_dimension_mismatch
        - key_error_in_act2fn_registry

  knowledge:

    usage_patterns:

      physics_informed_networks:

        pipeline:
          - LinearLayer
          - Stan (提供平滑二阶导数和自适应非线性)

      stable_fp16_training:

        pipeline:
          - LinearLayer
          - CappedGELU (防止值超过阈值导致 FP16 溢出)

    design_patterns:

      registry_factory:

        structure:
          - 维护一个 `ACT2FN` 字典，将小写字符串映射到类或带有默认 kwargs 的元组
          - 对外暴露 `get_activation` 函数，使得模型构建时可以仅依赖于简单的 YAML/JSON 字符串配置

    hot_models:

      - model: PINNs (Physics-Informed Neural Networks)
        year: 2019+
        role: 偏微分方程的机器学习求解器
        architecture: mlp
        attention_type: None (由于求导需求，高度依赖如 Stan、Tanh 这类无限次可微的激活函数)

    model_usage_details:

      PINNs:

        activation: stan
        reason: 标准 ReLU 的二阶导数为 0，无法求解涉及二阶偏导数的物理方程；Stan 提供了优秀的自适应求导表现。

    best_practices:

      - 在基于配置驱动构建通用 AI 框架时，始终使用 `get_activation(config.act_name)` 替代直接引入特定的模块。
      - 当使用 `Stan` 激活函数时，必须确保其初始化的 `out_features` 参数与传入特征张量的最后一个维度（Channel 维度）精准匹配。
      - 如果在大规模训练时遇到由于激活层导致的 Loss 突刺（Spike），可以尝试无缝替换为对应的 `capped_` 变体。

    anti_patterns:

      - 忽略 `Stan` 激活函数是与通道数绑定的这一事实，在维度会发生动态变化的卷积或线性层输出后错误复用同一个 `Stan` 实例。
      - 字典键名拼写错误（如调用代码中字典注册的 `cappek_leaky_relu` 有拼写失误，使用时需注意对照代码原样拼写）。

    paper_references:

      - title: "Self-scalable Tanh (Stan): Faster Convergence and Better Generalization in Physics-informed Neural Networks"
        authors: Gnanasambandam et al.
        year: 2022

      - title: "Squareplus: A Softplus-Like Algebraic Rectifier"
        authors: Barron
        year: 2021

  graph:

    is_a:
      - NeuralNetworkComponent
      - ActivationFunction
      - ElementWiseOperation

    part_of:
      - MultiLayerPerceptron
      - FeedForwardNetwork
      - FCLayer

    depends_on:
      - PyTorch Math Ops

    variants:
      - Stan
      - SquarePlus
      - CappedLeakyReLU
      - CappedGELU
      - Identity

    used_in_models:
      - Physics-Informed Neural Networks (PINNs)

    compatible_with:

      inputs:
        - Tensor

      outputs:
        - Tensor