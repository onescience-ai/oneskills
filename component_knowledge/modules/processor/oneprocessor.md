component:

  meta:
    name: OneProcessor
    alias: ProcessorFactory
    version: 1.0
    domain: deep_learning
    category: graph_neural_network
    subcategory: module_dispatcher
    author: OneScience
    license: Apache-2.0
    tags:
      - processor
      - registry
      - factory_pattern
      - graph_processing
      - wrapper


  concept:

    description: >
      OneProcessor 提供统一处理器调用接口，通过 style 从注册表中选择处理器实现。
      当前支持 BistrideGraphMessagePassing 与 GraphMessagePassing 两种后端，
      适合在同一训练框架中快速切换分层处理器与单层消息传递器。

    intuition: >
      它是图处理阶段的“统一入口”，让上层代码仅依赖接口而非具体算法实现。

    problem_it_solves:
      - 统一图处理器实例化和调用
      - 支持多种处理器风格切换
      - 对未知 style 提供明确报错
      - 降低处理器替换成本


  theory:

    formula:

      processor_dispatch:
        expression: |
          f_{proc} = Registry[style](**kwargs)
          y = f_{proc}(*args, **kwargs)

    variables:

      style:
        name: ProcessorStyle
        description: 处理器风格键

      Registry:
        name: ProcessorRegistry
        description: 处理器名称到类映射

      f_{proc}:
        name: ProcessorImpl
        description: 具体处理器实例


  structure:

    architecture: registry_based_wrapper

    pipeline:

      - name: StyleValidation
        operation: check_style

      - name: ProcessorInstantiation
        operation: create_processor_instance

      - name: ForwardDelegation
        operation: passthrough_forward


  interface:

    parameters:

      style:
        type: str
        description: 处理器实现名称

      kwargs:
        type: dict
        description: 构造处理器参数

    inputs:

      args:
        type: tuple
        description: 前向透传参数

      kwargs:
        type: dict
        description: 前向透传关键字参数

    outputs:

      output:
        type: TensorOrTuple
        description: 处理器输出


  types:

    TensorOrTuple:
      description: 由具体处理器定义的返回结构


  implementation:

    framework: pytorch

    code: |

      from .bistride_processor import BistrideGraphMessagePassing, GraphMessagePassing

      _PROCESSOR_REGISTRY = {
          "BistrideGraphMessagePassing": BistrideGraphMessagePassing,
          "GraphMessagePassing": GraphMessagePassing,
      }

      class OneProcessor(nn.Module):
          def __init__(self, style: str, **kwargs):
              super().__init__()
              if style not in _PROCESSOR_REGISTRY:
                  raise NotImplementedError(
                      f"Unknown processor style: '{style}'. Available: {list(_PROCESSOR_REGISTRY.keys())}"
                  )
              self.processor = _PROCESSOR_REGISTRY[style](**kwargs)

          def forward(self, *args, **kwargs):
              return self.processor(*args, **kwargs)


  skills:

    build_processor_dispatcher:

      description: 构建统一图处理器分发层

      inputs:
        - style
        - kwargs

      prompt_template: |

        构建 OneProcessor 模块。

        参数：
        style = {{style}}
        kwargs = {{kwargs}}

        要求：
        注册表中支持多个处理器并可前向透传。


    diagnose_processor_dispatcher:

      description: 分析处理器分发层中的配置和调用问题

      checks:
        - unknown_processor_style
        - missing_registry_entry
        - constructor_arg_mismatch
        - forward_contract_mismatch


  knowledge:

    usage_patterns:

      processor_switching:

        pipeline:
          - ReadConfigStyle
          - InstantiateOneProcessor
          - ExecuteSelectedProcessor

      experimentation_pipeline:

        pipeline:
          - CompareBistrideVsSingleLayer
          - KeepUnifiedTrainingLoop


    hot_models:

      - model: BistrideGraphMessagePassing
        year: 2023
        role: 分层图处理器实现
        architecture: graph_unet_style

      - model: GraphMessagePassing
        year: 2021
        role: 基础消息传递实现
        architecture: message_passing_gnn


    best_practices:

      - 通过配置管理 style，避免代码内硬编码。
      - 新增处理器时同步更新注册表与错误提示。
      - 为不同处理器设计统一输入输出契约。


    anti_patterns:

      - 分发层直接耦合训练逻辑。
      - 对未知 style 使用默认回退隐藏配置问题。


    paper_references:

      - title: "Design Patterns: Elements of Reusable Object-Oriented Software"
        authors: Gamma et al.
        year: 1994


  graph:

    is_a:
      - NeuralNetworkComponent
      - FactoryWrapper

    part_of:
      - ProcessorModuleSystem

    depends_on:
      - RegistryPattern
      - BistrideGraphMessagePassing
      - GraphMessagePassing

    variants:
      - OneNode
      - OnePooling
      - OneEdge

    used_in_models:
      - Hierarchical GNN Pipelines
      - Mesh Dynamics Models
      - Weather Graph Forecasting Systems

    compatible_with:

      inputs:
        - NodeEmbedding
        - MultiScaleEdgeIndex

      outputs:
        - TensorOrTuple