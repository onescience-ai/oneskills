component:

  meta:
    name: OneMSA
    alias: MSAFactory
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: module_dispatcher
    author: OneScience
    license: Apache-2.0
    tags:
      - msa
      - registry
      - factory_pattern
      - wrapper
      - protenix


  concept:

    description: >
      OneMSA 是 MSA 模块统一分发入口，通过 style 在注册表中选择具体 MSA 实现并透传参数。
      同时通过 __getattr__ 将未知属性代理到底层 msa 实例，便于保持接口兼容。
      当前源码中的注册表为空（实现导入被注释），因此默认会在 style 检查时报错。

    intuition: >
      该模块意图与其它 one* 封装一致：让调用端只依赖稳定入口，
      具体 MSA 实现由注册表管理，便于后续增删和实验切换。

    problem_it_solves:
      - 统一 MSA 组件实例化入口
      - 降低上层模型对具体 MSA 类的耦合
      - 提供属性代理以兼容底层实现接口
      - 在未知 style 场景提供显式错误


  theory:

    formula:

      registry_dispatch:
        expression: |
          f_{msa} = Registry[style](**kwargs)
          y = f_{msa}(*args, **kwargs)

      attribute_proxy:
        expression: |
          attr(OneMSA, name) = attr(msa_impl, name)\ \text{if local attr missing}

    variables:

      style:
        name: MSAStyle
        description: MSA 风格键

      Registry:
        name: MSARegistry
        description: 风格到实现类映射

      msa_impl:
        name: ConcreteMSAModule
        description: 被分发到的 MSA 实例


  structure:

    architecture: registry_proxy_wrapper

    pipeline:

      - name: StyleValidation
        operation: check_style_in_registry

      - name: ModuleInstantiation
        operation: create_selected_msa_module

      - name: ForwardDelegation
        operation: passthrough_forward

      - name: AttributeFallback
        operation: delegate_missing_attributes


  interface:

    parameters:

      style:
        type: str
        description: MSA 模块风格名称

      kwargs:
        type: dict
        description: 构造目标 MSA 模块参数

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
        description: 目标 MSA 模块返回结果


  types:

    TensorOrTuple:
      description: 由具体 MSA 实现定义


  implementation:

    framework: pytorch

    code: |

      import torch.nn as nn

      _MSA_REGISTRY = {
          # "ProtenixMSAModule": ProtenixMSAModule,
      }

      class OneMSA(nn.Module):
          def __init__(self, style: str, **kwargs):
              super().__init__()
              if style not in _MSA_REGISTRY:
                  raise NotImplementedError(f"Unknown style: {style}")
              self.msa = _MSA_REGISTRY[style](**kwargs)

          def __getattr__(self, name):
              try:
                  return super().__getattr__(name)
              except AttributeError:
                  return getattr(self.msa, name)

          def forward(self, *args, **kwargs):
              return self.msa(*args, **kwargs)


  skills:

    build_msa_dispatcher:

      description: 构建统一 MSA 分发器并支持属性代理

      inputs:
        - style
        - kwargs

      prompt_template: |

        构建 OneMSA 模块。

        参数：
        style = {{style}}
        kwargs = {{kwargs}}

        要求：
        使用注册表路由 MSA 实现，并在未知风格时报错。


    diagnose_msa_dispatcher:

      description: 分析 MSA 分发层的注册和调用问题

      checks:
        - empty_registry_configuration
        - unknown_msa_style
        - attribute_proxy_shadowing
        - missing_concrete_module_import


  knowledge:

    usage_patterns:

      modular_msa_entry:

        pipeline:
          - ParseStyle
          - InstantiateOneMSA
          - DelegateForward

      extension_workflow:

        pipeline:
          - ImplementConcreteMSAModule
          - RegisterIntoMSARegistry
          - UseStyleInConfig


    hot_models:

      - model: ProtenixMSAModule
        year: 2024
        role: 预期接入的 MSA 实现
        architecture: msa_pair_update_stack


    best_practices:

      - 保持注册表与导入语句同步，避免空注册表。
      - 将 style 配置放在统一配置层，减少硬编码。
      - 为 OneMSA 增加 style 可用性单测。


    anti_patterns:

      - 注册表为空仍在训练配置中启用 OneMSA。
      - 通过异常捕获静默回退默认实现，导致行为不透明。


    paper_references:

      - title: "Design Patterns: Elements of Reusable Object-Oriented Software"
        authors: Gamma et al.
        year: 1994


  graph:

    is_a:
      - NeuralNetworkComponent
      - FactoryWrapper

    part_of:
      - MSAModuleSystem

    depends_on:
      - RegistryPattern
      - ProtenixMSAModule

    variants:
      - OnePairformer
      - OneLinear
      - OneProcessor

    used_in_models:
      - Protenix Pipelines
      - AF3-style Prototype Systems
      - Modular Biomolecular Models

    compatible_with:

      inputs:
        - TensorOrTuple

      outputs:
        - TensorOrTuple