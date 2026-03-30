component:

  meta:
    name: OneNode
    alias: NodeFactory
    version: 1.0
    domain: deep_learning
    category: graph_neural_network
    subcategory: module_dispatcher
    author: OneScience
    license: Apache-2.0
    tags:
      - node
      - registry
      - factory_pattern
      - gnn
      - wrapper


  concept:

    description: >
      OneNode 提供统一节点更新模块调用接口，通过 style 从注册表选择具体节点更新器（当前为 MeshNodeBlock）。
      forward 采用参数透传方式，适配不同节点更新实现签名。

    intuition: >
      它是节点更新层的“策略路由器”，让上层消息传递框架只依赖统一接口，
      不直接耦合具体实现。

    problem_it_solves:
      - 统一节点更新模块实例化入口
      - 简化节点更新策略切换
      - 对未知 style 提供可诊断错误
      - 提升 GNN 组件化复用能力


  theory:

    formula:

      registry_dispatch:
        expression: |
          f_{node} = Registry[style](**kwargs)
          y = f_{node}(*args, **kwargs)

    variables:

      style:
        name: NodeStyle
        description: 节点更新器风格名称

      Registry:
        name: NodeRegistry
        description: 风格与节点更新类映射

      f_{node}:
        name: NodeUpdater
        description: 具体节点更新实例


  structure:

    architecture: registry_based_wrapper

    pipeline:

      - name: StyleValidation
        operation: verify_style_key

      - name: UpdaterInstantiation
        operation: instantiate_node_updater

      - name: ForwardDelegation
        operation: passthrough_arguments


  interface:

    parameters:

      style:
        type: str
        description: 节点更新实现名称

      kwargs:
        type: dict
        description: 构造节点更新器参数

    inputs:

      args:
        type: tuple
        description: 前向透传参数（通常为 efeat, nfeat, graph）

      kwargs:
        type: dict
        description: 前向透传关键字参数

    outputs:

      output:
        type: TensorOrTuple
        description: 节点更新器输出


  types:

    TensorOrTuple:
      description: 具体节点更新器定义的输出结构


  implementation:

    framework: pytorch

    code: |

      from .mesh_node_block import MeshNodeBlock

      _NODE_REGISTRY = {
          "MeshNodeBlock": MeshNodeBlock,
      }

      class OneNode(nn.Module):
          def __init__(self, style: str, **kwargs):
              super().__init__()
              if style not in _NODE_REGISTRY:
                  raise NotImplementedError(
                      f"Unknown node style: '{style}'. Available: {list(_NODE_REGISTRY.keys())}"
                  )
              self.node_updater = _NODE_REGISTRY[style](**kwargs)

          def forward(self, *args, **kwargs):
              return self.node_updater(*args, **kwargs)


  skills:

    build_node_dispatcher:

      description: 构建统一节点更新分发器

      inputs:
        - style
        - kwargs

      prompt_template: |

        构建 OneNode 模块。

        参数：
        style = {{style}}
        kwargs = {{kwargs}}

        要求：
        使用注册表分发并在未知风格时报错。


    diagnose_node_dispatcher:

      description: 分析节点分发器中的注册和参数透传问题

      checks:
        - unknown_node_style
        - missing_registry_update
        - forward_signature_mismatch


  knowledge:

    usage_patterns:

      modular_node_update:

        pipeline:
          - SelectNodeStyle
          - InstantiateOneNode
          - ForwardToUpdater

      ablation_study:

        pipeline:
          - RegisterMultipleNodeBlocks
          - SwitchByStyle
          - CompareResults


    hot_models:

      - model: MeshNodeBlock
        year: 2023
        role: 当前默认节点更新实现
        architecture: residual_node_mlp


    best_practices:

      - style 命名与注册表键保持一致。
      - 新增节点更新器时同步更新错误提示列表。
      - 在分发层保持最小逻辑，仅做路由和透传。


    anti_patterns:

      - 在分发层加入业务逻辑导致职责不清。
      - 未知 style 时静默回退默认节点更新器。


    paper_references:

      - title: "Design Patterns: Elements of Reusable Object-Oriented Software"
        authors: Gamma et al.
        year: 1994


  graph:

    is_a:
      - NeuralNetworkComponent
      - FactoryWrapper

    part_of:
      - NodeModuleSystem

    depends_on:
      - RegistryPattern
      - MeshNodeBlock

    variants:
      - OneEdge
      - OneProcessor
      - OnePooling

    used_in_models:
      - GraphCast Pipelines
      - MeshGraph Workflows
      - Modular GNN Systems

    compatible_with:

      inputs:
        - EdgeEmbedding
        - NodeEmbedding
        - GraphStructure

      outputs:
        - TensorOrTuple