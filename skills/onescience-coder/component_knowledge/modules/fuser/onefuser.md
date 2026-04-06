component:

  meta:
    name: OneFuser
    alias: Unified Fuser Interface
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: fuser_factory
    author: OneScience
    license: Apache-2.0
    tags:
      - fuser
      - factory_pattern
      - unified_interface
      - modular_design
      - feature_fusion

  concept:

    description: >
      OneFuser是统一的特征融合器接口模块，采用工厂模式设计，提供对不同融合器实现的统一访问。
      通过注册表机制管理多种融合器类型，包括Pangu系列、FengWu、FourCastNet、Xihe系列等，
      支持通过style参数动态选择具体的融合器实现。

    intuition: >
      就像通用混合器：一个设备可以处理不同类型的混合任务（气象数据融合、特征融合、时空融合等），
      只需选择对应的融合模式（style），设备就会自动调用相应的融合算法。OneFuser就是这样一个"通用融合器"。

    problem_it_solves:
      - 多种融合器实现的统一管理
      - 动态融合器选择和切换
      - 模块化设计中的接口统一
      - 特征融合算法的灵活配置

  theory:

    formula:

      factory_pattern:
        expression: |
          \text{fuser} = \text{OneFuser}(\text{style}, \text{**kwargs})
          \text{output} = \text{fuser}(\text{*args}, \text{**kwargs})

      registry_mechanism:
        expression: |
          \text{fuser} = \text{FUSER\_REGISTRY}[\text{style}](\text{**kwargs})
          \text{if } \text{style} \notin \text{FUSER\_REGISTRY}: \text{raise NotImplementedError}

    variables:

      style:
        name: FuserStyle
        description: 融合器类型标识符

      kwargs:
        name: FuserParameters
        description: 传递给具体融合器的参数

  structure:

    architecture: factory_pattern_interface

    pipeline:

      - name: StyleValidation
        operation: check_style_in_registry

      - name: FuserInstantiation
        operation: create_specific_fuser

      - name: ForwardDelegation
        operation: delegate_to_concrete_fuser

  interface:

    parameters:

      style:
        type: str
        description: 融合器类型，可选值见FUSER_REGISTRY

      kwargs:
        type: dict
        description: 传递给具体融合器的参数

    available_styles:

      PanguFuser:
        description: Pangu-Weather特征融合器

      PanguDistributedFuser:
        description: Pangu-Weather分布式特征融合器

      FengWuFuser:
        description: FengWu时空特征融合器

      FourCastNetFuser:
        description: FourCastNet AFNO融合器

      XiheLocalSIEFuser:
        description: Xihe局部SIE融合器

      XiheGlobalSIEFuser:
        description: Xihe全局SIE融合器

      XiheFuser:
        description: Xihe海洋模型融合器

    inputs:

      x:
        type: InputFeatures
        description: 传递给具体融合器的输入特征

    outputs:

      x:
        type: FusedFeatures
        description: 融合后的特征

  types:

    FuserRegistry:
      description: 融合器注册表，映射style到具体实现类

  implementation:

    framework: pytorch

    code: |

      import torch
      from torch import nn
      from .pangufuser import PanguFuser
      from .pangudistributedfuser import PanguDistributedFuser
      from .fengwufuser import FengWuFuser
      from .fourcastnetfuser import FourCastNetFuser
      from .xihelocalsiefuser import XiheLocalSIEFuser
      from .xiheglobalsiefuser import XiheGlobalSIEFuser
      from .xihefuse import XiheFuser

      _FUSER_REGISTRY = {
          "PanguFuser": PanguFuser,
          "PanguDistributedFuser": PanguDistributedFuser,
          "FengWuFuser": FengWuFuser,
          "FourCastNetFuser": FourCastNetFuser,
          "XiheLocalSIEFuser": XiheLocalSIEFuser,
          "XiheGlobalSIEFuser": XiheGlobalSIEFuser,
          "XiheFuser": XiheFuser,
      }

      class OneFuser(nn.Module):
          def __init__(self, style: str, **kwargs):
              super().__init__()
              
              if style not in _FUSER_REGISTRY:
                  raise NotImplementedError(f"Unknown style: {style}")
              
              self.Fuser = _FUSER_REGISTRY[style](**kwargs)
               
          def forward(self, x):
              return self.Fuser(x)

  skills:

    build_unified_fuser:

      description: 构建统一融合器接口

      inputs:
        - style
        - fuser_parameters

      prompt_template: |

        构建OneFuser统一接口，支持多种融合器类型。

        参数：
        style = {{style}}
        fuser_parameters = {{fuser_parameters}}

        要求：
        1. 支持动态融合器选择
        2. 统一的前向传播接口
        3. 错误处理和验证
        4. 参数透传机制

    manage_fuser_registry:

      description: 管理融合器注册表

      checks:
        - style_availability (检查style是否可用)
        - parameter_compatibility (参数兼容性)
        - interface_consistency (接口一致性)

  knowledge:

    usage_patterns:

      dynamic_fuser_selection:

        pipeline:
          - Config: 配置文件指定fuser类型
          - Factory: OneFuser工厂创建实例
          - Usage: 统一接口调用
          - Output: 融合后的特征

      modular_fusion:

        pipeline:
          - Registry: 融合器注册表
          - Interface: 统一访问接口
          - Implementation: 具体融合器实现
          - Integration: 系统集成

    hot_models:

      - model: Factory Pattern
        year: 1994
        role: 设计模式经典实现
        architecture: creational pattern

      - model: Feature Fusion
        year: 2010s
        role: 深度学习中的特征融合技术
        architecture: various fusion mechanisms

    best_practices:

      - 保持接口的一致性和简洁性
      - 提供清晰的错误信息
      - 支持参数透传
      - 维护完整的注册表文档

    anti_patterns:

      - 注册表不一致导致运行时错误
      - 接口设计过于复杂
      - 缺少适当的错误处理
      - 参数验证不充分

    paper_references:

      - title: "Design Patterns: Elements of Reusable Object-Oriented Software"
        authors: Gamma et al.
        year: 1994

      - title: "Feature Fusion for Deep Learning: A Survey"
        authors: Liu et al.
        year: 2020

  graph:

    is_a:
      - FactoryModule
      - UnifiedInterface
      - FeatureFusionWrapper

    part_of:
      - ModularArchitecture
      - FusionSystem
      - ComponentRegistry

    depends_on:
      - PanguFuser
      - PanguDistributedFuser
      - FengWuFuser
      - FourCastNetFuser
      - XiheLocalSIEFuser
      - XiheGlobalSIEFuser
      - XiheFuser

    variants:
      - OneEncoder (编码器统一接口)
      - OneDecoder (解码器统一接口)
      - OneMLP (MLP统一接口)

    used_in_models:
      - 模块化深度学习系统
      - 多模型集成框架
      - 动态融合选择系统

    compatible_with:

      inputs:
        - FuserConfigurations
        - ModelParameters
        - InputFeatures

      outputs:
        - FusedFeatures
        - IntegratedRepresentations
        - FusionResults
