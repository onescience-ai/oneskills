component:

  meta:
    name: OneEquivariant
    alias: EquivariantFactory
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: module_dispatcher
    author: OneScience
    license: Apache-2.0
    tags:
      - equivariant
      - registry
      - factory_pattern
      - group_conv
      - wrapper


  concept:

    description: >
      OneEquivariant 是等变模块统一调用入口，基于 style 字符串从注册表实例化
      GroupEquivariantConv2d 或 GroupEquivariantConv3d，并在 forward 中透传参数到选中的层。

    intuition: >
      该模块将“二维或三维等变卷积选择”从模型主干逻辑中分离出来，
      使上层通过配置即可切换等变算子实现。

    problem_it_solves:
      - 统一等变层构造接口
      - 支持 2D 与 3D 等变卷积快速切换
      - 对非法 style 提供清晰错误提示
      - 降低调用方与具体等变实现的耦合


  theory:

    formula:

      registry_dispatch:
        expression: |
          f_{eq} = Registry[style](**kwargs)
          y = f_{eq}(*args, **kwargs)

    variables:

      style:
        name: EquivariantStyle
        description: 等变层风格名称

      Registry:
        name: EquivariantRegistry
        description: 风格到等变层类的映射

      f_{eq}:
        name: EquivariantLayer
        description: 被实例化的具体等变层


  structure:

    architecture: registry_based_wrapper

    pipeline:

      - name: StyleValidation
        operation: verify_style_exists

      - name: LayerInstantiation
        operation: instantiate_equivariant_layer

      - name: ForwardPassthrough
        operation: delegate_forward


  interface:

    parameters:

      style:
        type: str
        description: 等变层实现名称

      kwargs:
        type: dict
        description: 实例化目标等变层参数

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
        description: 目标等变层输出


  types:

    Tensor:
      description: PyTorch 张量


  implementation:

    framework: pytorch

    code: |

      import torch.nn as nn
      from .group_conv import GroupEquivariantConv2d, GroupEquivariantConv3d

      _EQUIVARIANT_REGISTRY = {
          "GroupEquivariantConv2d": GroupEquivariantConv2d,
          "GroupEquivariantConv3d": GroupEquivariantConv3d,
      }

      class OneEquivariant(nn.Module):
          def __init__(self, style: str, **kwargs):
              super().__init__()

              if style not in _EQUIVARIANT_REGISTRY:
                  raise NotImplementedError(
                      f"Unknown style: '{style}'. Available options are: {list(_EQUIVARIANT_REGISTRY.keys())}"
                  )

              self.equivariant_layer = _EQUIVARIANT_REGISTRY[style](**kwargs)

          def forward(self, *args, **kwargs):
              return self.equivariant_layer(*args, **kwargs)


  skills:

    build_equivariant_dispatcher:

      description: 构建统一等变层分发器

      inputs:
        - style
        - kwargs

      prompt_template: |

        构建 OneEquivariant 模块。

        参数：
        style = {{style}}
        kwargs = {{kwargs}}

        要求：
        使用注册表路由到 2D/3D 等变卷积实现，并在未知风格时抛错。


    diagnose_equivariant_dispatcher:

      description: 分析等变分发层中的注册和调用问题

      checks:
        - unknown_equivariant_style
        - missing_registry_update
        - forward_parameter_contract_mismatch


  knowledge:

    usage_patterns:

      symmetry_module_switching:

        pipeline:
          - ParseStyle
          - InstantiateOneEquivariant
          - ForwardToSelectedLayer

      multi_dimensional_equivariance:

        pipeline:
          - Select2DOr3DLayer
          - ConfigureGroupOptions
          - IntegrateInBackbone


    hot_models:

      - model: GroupEquivariantConv2d
        year: 2016
        role: 二维等变卷积实现
        architecture: group_equivariant_cnn

      - model: GroupEquivariantConv3d
        year: 2020
        role: 三维场景下的群等变扩展
        architecture: volumetric_equivariant_cnn


    best_practices:

      - 在配置层显式声明 style，避免默认值歧义。
      - 新增等变层时同步更新注册表和错误提示列表。
      - 保持 forward 透传逻辑最小化，避免破坏下游层接口。


    anti_patterns:

      - 通过字符串拼接动态导入层类，导致可维护性下降。
      - 对未知 style 静默降级到默认层，掩盖配置错误。


    paper_references:

      - title: "Group Equivariant Convolutional Networks"
        authors: Cohen and Welling
        year: 2016


  graph:

    is_a:
      - NeuralNetworkComponent
      - FactoryWrapper

    part_of:
      - EquivariantModuleSystem

    depends_on:
      - RegistryPattern
      - GroupEquivariantConv2d
      - GroupEquivariantConv3d

    variants:
      - OneAFNO
      - OneLinear
      - OneEdge

    used_in_models:
      - Symmetry-aware CNN Backbones
      - Equivariant Forecasting Models
      - Research Prototypes

    compatible_with:

      inputs:
        - GroupFeature2D
        - GroupFeature3D

      outputs:
        - GroupFeature2D
        - GroupFeature3D