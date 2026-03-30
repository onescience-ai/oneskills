component:

  meta:
    name: OnePooling
    alias: PoolingFactory
    version: 1.0
    domain: deep_learning
    category: graph_neural_network
    subcategory: module_dispatcher
    author: OneScience
    license: Apache-2.0
    tags:
      - pooling
      - registry
      - factory_pattern
      - graph_pooling
      - wrapper


  concept:

    description: >
      OnePooling 是池化模块统一调用接口，通过 style 在注册表中分发具体池化器（当前为 RNNClusterPooling）。
      forward 透传参数，保持调用层与具体池化实现解耦。

    intuition: >
      它相当于“池化策略入口层”，上游只关心池化语义，不需要感知底层实现细节。

    problem_it_solves:
      - 统一池化模块实例化入口
      - 简化不同池化策略切换
      - 对未知 style 给出显式错误
      - 提高图层级建模代码复用度


  theory:

    formula:

      pooling_dispatch:
        expression: |
          f_{pool} = Registry[style](**kwargs)
          y = f_{pool}(*args, **kwargs)

    variables:

      style:
        name: PoolingStyle
        description: 池化风格键

      Registry:
        name: PoolingRegistry
        description: 风格到池化实现映射

      f_{pool}:
        name: PoolingModule
        description: 被实例化的池化模块


  structure:

    architecture: registry_based_wrapper

    pipeline:

      - name: StyleCheck
        operation: verify_style_in_registry

      - name: PoolerInstantiation
        operation: create_pooling_module

      - name: ForwardPassthrough
        operation: delegate_inputs_to_pooler


  interface:

    parameters:

      style:
        type: str
        description: 池化实现名称

      kwargs:
        type: dict
        description: 池化器构造参数

    inputs:

      args:
        type: tuple
        description: 前向透传参数

      kwargs:
        type: dict
        description: 前向透传关键字参数

    outputs:

      output:
        type: Tensor
        description: 池化结果张量


  types:

    Tensor:
      description: PyTorch 张量


  implementation:

    framework: pytorch

    code: |

      from .rnn_cluster_pooling import RNNClusterPooling

      _POOLING_REGISTRY = {
          "RNNClusterPooling": RNNClusterPooling,
      }

      class OnePooling(nn.Module):
          def __init__(self, style: str, **kwargs):
              super().__init__()
              if style not in _POOLING_REGISTRY:
                  raise NotImplementedError(
                      f"Unknown style: '{style}'. Available options are: {list(_POOLING_REGISTRY.keys())}"
                  )
              self.pooler = _POOLING_REGISTRY[style](**kwargs)

          def forward(self, *args, **kwargs):
              return self.pooler(*args, **kwargs)


  skills:

    build_pooling_dispatcher:

      description: 构建统一池化分发层

      inputs:
        - style
        - kwargs

      prompt_template: |

        构建 OnePooling 模块。

        参数：
        style = {{style}}
        kwargs = {{kwargs}}

        要求：
        通过注册表实例化池化层并前向透传。


    diagnose_pooling_dispatcher:

      description: 分析池化分发器中的注册和参数匹配问题

      checks:
        - unknown_pooling_style
        - missing_pooling_registration
        - forward_signature_incompatibility


  knowledge:

    usage_patterns:

      graph_hierarchy_pooling:

        pipeline:
          - SelectPoolingStyle
          - InstantiateOnePooling
          - PoolNodeFeatures

      modular_graph_pipeline:

        pipeline:
          - Encoder
          - OnePooling
          - CoarseGraphProcessor


    hot_models:

      - model: RNNClusterPooling
        year: 2023
        role: 当前注册的簇级池化实现
        architecture: sequence_based_cluster_pooling


    best_practices:

      - 保持 style 与注册表键一致。
      - 对不同池化器统一输出约定，减少上游适配代码。
      - 为新增池化器补充单测与基准。


    anti_patterns:

      - 在注册表外直接实例化池化器，破坏统一入口。
      - 对未知 style 进行静默回退。


    paper_references:

      - title: "Design Patterns: Elements of Reusable Object-Oriented Software"
        authors: Gamma et al.
        year: 1994


  graph:

    is_a:
      - NeuralNetworkComponent
      - FactoryWrapper

    part_of:
      - PoolingModuleSystem

    depends_on:
      - RegistryPattern
      - RNNClusterPooling

    variants:
      - OneProcessor
      - OneNode
      - OneEdge

    used_in_models:
      - Hierarchical GNN Pipelines
      - GraphCast-like Architectures
      - Mesh-to-Latent Systems

    compatible_with:

      inputs:
        - NodeEmbedding
        - ClusterIndex

      outputs:
        - ClusterEmbedding