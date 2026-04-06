component:

  meta:
    name: ProtenixLinearModules
    alias: ProtenixLinear
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: linear_layer
    author: OneScience
    license: Apache-2.0
    tags:
      - linear
      - initialization
      - precision_control
      - protenix
      - alphafold3


  concept:

    description: >
      ProtenixLinear 系列实现了带定制初始化策略与可选高精度计算路径的线性层。
      主类 ProtenixLinear 支持 default/relu/zeros 三种权重初始化，并可在 forward 中指定 precision
      强制线性计算使用更高精度后再转回输入 dtype；此外还提供无偏置版本和偏置常量初始化版本。

    intuition: >
      该模块将“线性变换 + 初始化策略 + 计算精度控制”打包为统一组件，
      使结构搜索和训练稳定性调优可以通过参数完成，而无需改动主干网络代码。

    problem_it_solves:
      - 提供符合 Protenix/结构建模场景的线性层初始化策略
      - 降低混合精度训练下线性计算数值误差
      - 统一封装无偏置和偏置预设初始化两类常见变体
      - 保持与 torch.nn.Linear 相近接口


  theory:

    formula:

      linear_transform:
        expression: |
          y = xW^T + b

      precision_path:
        expression: |
          y = Linear(cast(x, p), cast(W, p), cast(b, p))
          y_{out} = cast(y, dtype(x))

      init_strategy:
        expression: |
          W \sim TruncNormal(scale \in \{1.0, 2.0\})\ \text{or}\ W=0
          b = 0\ \text{or}\ b=\text{biasinit}

    variables:

      x:
        name: InputTensor
        shape: [batch, in_features]
        description: 输入特征

      W:
        name: WeightMatrix
        shape: [out_features, in_features]
        description: 线性层权重

      b:
        name: BiasVector
        shape: [out_features]
        description: 偏置向量

      p:
        name: ComputePrecision
        description: 可选计算精度（如 float32）


  structure:

    architecture: customized_linear_family

    pipeline:

      - name: ParameterInitialization
        operation: trunc_normal_or_zero_init

      - name: OptionalPrecisionCast
        operation: cast_input_weight_bias_if_needed

      - name: LinearProjection
        operation: torch_linear

      - name: DtypeRestore
        operation: cast_back_to_input_dtype


  interface:

    parameters:

      in_features:
        type: int
        description: 输入维度

      out_features:
        type: int
        description: 输出维度

      bias:
        type: bool
        description: 是否使用偏置

      precision:
        type: torch.dtype
        default: null
        description: forward 计算时的强制精度

      initializer:
        type: str
        description: 权重初始化策略（default/relu/zeros）

      biasinit:
        type: float
        default: 0.0
        description: 仅在 ProtenixBiasInitLinear 中用于偏置常量初始化

    inputs:

      input:
        type: SequenceEmbedding
        shape: [*, in_features]
        dtype: float16|bfloat16|float32
        description: 任意前缀维度的输入张量

    outputs:

      output:
        type: SequenceEmbedding
        shape: [*, out_features]
        description: 线性变换结果


  types:

    SequenceEmbedding:
      shape: [*, feature_dim]
      description: 通用特征 embedding 表示


  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      class ProtenixLinear(nn.Linear):
          def __init__(self, in_features, out_features, bias=True, precision=None, initializer="default", **kwargs):
              self.use_bias = bias
              self.precision = precision
              self.initializer = initializer
              super().__init__(in_features=in_features, out_features=out_features, bias=bias, **kwargs)
              self._init_params()

          @torch.no_grad()
          def _init_params(self):
              if self.use_bias:
                  nn.init.zeros_(self.bias)
              if self.initializer == "default":
                  trunc_normal_init_(self.weight, scale=1.0)
              elif self.initializer == "relu":
                  trunc_normal_init_(self.weight, scale=2.0)
              elif self.initializer == "zeros":
                  nn.init.zeros_(self.weight)
              else:
                  raise ValueError(f"Invalid initializer: {self.initializer}.")

          def forward(self, input: torch.Tensor) -> torch.Tensor:
              if self.precision is not None:
                  input_dtype = input.dtype
                  with torch.cuda.amp.autocast(enabled=False):
                      bias = self.bias.to(dtype=self.precision) if self.bias is not None else None
                      return F.linear(input.to(dtype=self.precision), self.weight.to(dtype=self.precision), bias).to(dtype=input_dtype)
              return F.linear(input, self.weight, self.bias)

      class ProtenixLinearNoBias(ProtenixLinear):
          def __init__(self, in_features, out_features, **kwargs):
              super().__init__(in_features, out_features, bias=False, **kwargs)

      class ProtenixBiasInitLinear(ProtenixLinear):
          def __init__(self, in_features, out_features, bias=True, biasinit=0.0, **kwargs):
              super().__init__(in_features=in_features, out_features=out_features, bias=bias, **kwargs)
              nn.init.zeros_(tensor=self.weight)
              if bias:
                  nn.init.constant_(tensor=self.bias, val=biasinit)


  skills:

    build_protenix_linear:

      description: 构建带初始化和精度控制的 Protenix 线性层

      inputs:
        - in_features
        - out_features
        - bias
        - precision
        - initializer

      prompt_template: |

        构建 ProtenixLinear 模块。

        参数：
        in_features = {{in_features}}
        out_features = {{out_features}}
        bias = {{bias}}
        precision = {{precision}}
        initializer = {{initializer}}

        要求：
        支持 default/relu/zeros 初始化，并可选精度路径。


    diagnose_linear_init:

      description: 分析线性层初始化与数值稳定性问题

      checks:
        - invalid_initializer_key
        - precision_cast_overflow_underflow
        - bias_configuration_mismatch
        - degraded_training_from_zero_weight_init


  knowledge:

    usage_patterns:

      structure_modeling:

        pipeline:
          - ProtenixLayerNorm
          - ProtenixLinear
          - Activation

      precision_sensitive_training:

        pipeline:
          - autocast_off_for_linear
          - fp32_projection
          - cast_back


    hot_models:

      - model: AlphaFold3-like Architectures
        year: 2024
        role: 使用带定制初始化的线性层提升训练稳定性
        architecture: protein_structure_model

      - model: OpenFold Variants
        year: 2022
        role: 提供结构建模原语与初始化策略参考
        architecture: evoformer_style


    best_practices:

      - 在 mixed precision 训练中对关键投影层启用 precision 参数。
      - `initializer=relu` 适用于后续接 ReLU 的层，`default` 更通用。
      - 对偏置敏感任务可使用 ProtenixBiasInitLinear 明确设定初值。


    anti_patterns:

      - 在未知 initializer 字符串下继续训练而不报错。
      - 全网使用 zeros 初始化导致学习停滞。


    paper_references:

      - title: "Deep Residual Learning for Image Recognition"
        authors: He et al.
        year: 2015

      - title: "Attention Is All You Need"
        authors: Vaswani et al.
        year: 2017


  graph:

    is_a:
      - NeuralNetworkComponent
      - LinearLayer

    part_of:
      - ProtenixModules
      - ProteinModelBackbones

    depends_on:
      - torch.nn.Linear
      - trunc_normal_init
      - autocast

    variants:
      - ProtenixLinear
      - ProtenixLinearNoBias
      - ProtenixBiasInitLinear

    used_in_models:
      - AlphaFold3-like Models
      - Protenix Pipelines
      - OpenFold-based Architectures

    compatible_with:

      inputs:
        - SequenceEmbedding

      outputs:
        - SequenceEmbedding