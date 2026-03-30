component:

  meta:
    name: OneDecoder
    alias: Unified Decoder Interface
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: decoder_factory
    author: OneScience
    license: Apache-2.0
    tags:
      - decoder
      - factory_pattern
      - unified_interface
      - modular_design
      - decoder_registry

  concept:

    description: >
      OneDecoder是统一的解码器接口模块，采用工厂模式设计，提供对不同解码器实现的统一访问。
      通过注册表机制管理多种解码器类型，包括UNet系列、GraphViT、MeshGraph、FengWu等，
      支持通过style参数动态选择具体的解码器实现。

    intuition: >
      就像通用遥控器：一个设备可以控制不同品牌的电视（解码器），只需选择对应的品牌（style），
      遥控器就会自动调用该品牌的控制协议。OneDecoder就是这样一个"通用解码器遥控器"。

    problem_it_solves:
      - 多种解码器实现的统一管理
      - 动态解码器选择和切换
      - 模块化设计中的接口统一
      - 代码复用和维护性提升

  theory:

    formula:

      factory_pattern:
        expression: |
          \text{decoder} = \text{OneDecoder}(\text{style}, \text{**kwargs})
          \text{output} = \text{decoder}(\text{*args}, \text{**kwargs})

      registry_mechanism:
        expression: |
          \text{decoder} = \text{DECODER\_REGISTRY}[\text{style}](\text{**kwargs})
          \text{if } \text{style} \notin \text{DECODER\_REGISTRY}: \text{raise NotImplementedError}

    variables:

      style:
        name: DecoderStyle
        description: 解码器类型标识符

      kwargs:
        name: DecoderParameters
        description: 传递给具体解码器的参数

  structure:

    architecture: factory_pattern_interface

    pipeline:

      - name: StyleValidation
        operation: check_style_in_registry

      - name: DecoderInstantiation
        operation: create_specific_decoder

      - name: ForwardDelegation
        operation: delegate_to_concrete_decoder

      - name: AttributeDelegation
        operation: attribute_access_forwarding

  interface:

    parameters:

      style:
        type: str
        description: 解码器类型，可选值见DECODER_REGISTRY

      kwargs:
        type: dict
        description: 传递给具体解码器的参数

    available_styles:

      UNetDecoder1D:
        description: 一维U-Net解码器

      UNetDecoder2D:
        description: 二维U-Net解码器

      UNetDecoder3D:
        description: 三维U-Net解码器

      GraphViTDecoder:
        description: GraphViT解码器

      MeshGraphDecoder:
        description: MeshGraph解码器

      FengWuDecoder:
        description: FengWu气象解码器

    inputs:

      args:
        type: variadic
        description: 传递给具体解码器的位置参数

      kwargs:
        type: variadic
        description: 传递给具体解码器的关键字参数

    outputs:

      output:
        type: decoder_output
        description: 具体解码器的输出

  types:

      DecoderRegistry:
        description: 解码器注册表，映射style到具体实现类

  implementation:

    framework: pytorch

    code: |

      import torch
      from torch import nn
      from .unet_decoder import UNetDecoder1D, UNetDecoder2D, UNetDecoder3D
      from .graphvit_decoder import GraphViTDecoder
      from .mesh_graph_decoder import MeshGraphDecoder
      from .fengwudecoder import FengWuDecoder

      _DECODER_REGISTRY = {
          "UNetDecoder1D": UNetDecoder1D,
          "UNetDecoder2D": UNetDecoder2D,
          "UNetDecoder3D": UNetDecoder3D,
          "GraphViTDecoder": GraphViTDecoder,
          "MeshGraphDecoder": MeshGraphDecoder,
          "FengWuDecoder": FengWuDecoder,
      }

      class OneDecoder(nn.Module):
          def __init__(self, style: str, **kwargs):
              super().__init__()
              
              if style not in _DECODER_REGISTRY:
                  raise NotImplementedError(
                      f"Unknown style: '{style}'. Available options are: {list(_DECODER_REGISTRY.keys())}"
                  )
              
              self.decoder = _DECODER_REGISTRY[style](**kwargs)

          def forward(self, *args, **kwargs):
              return self.decoder(*args, **kwargs)
          
          def __getattr__(self, name):
              try:
                  return super().__getattr__(name)
              except AttributeError:
                  return getattr(self.decoder, name)

  skills:

    build_unified_decoder:

      description: 构建统一解码器接口

      inputs:
        - style
        - decoder_parameters

      prompt_template: |

        构建OneDecoder统一接口，支持多种解码器类型。

        参数：
        style = {{style}}
        decoder_parameters = {{decoder_parameters}}

        要求：
        1. 支持动态解码器选择
        2. 统一的前向传播接口
        3. 属性访问转发
        4. 错误处理和验证

    manage_decoder_registry:

      description: 管理解码器注册表

      checks:
        - style_availability (检查style是否可用)
        - parameter_compatibility (参数兼容性)
        - interface_consistency (接口一致性)

  knowledge:

    usage_patterns:

      dynamic_decoder_selection:

        pipeline:
          - Config: 配置文件指定decoder类型
          - Factory: OneDecoder工厂创建实例
          - Usage: 统一接口调用
          - Output: 具体decoder的输出

      modular_architecture:

        pipeline:
          - Registry: 解码器注册表
          - Interface: 统一访问接口
          - Implementation: 具体解码器实现
          - Integration: 系统集成

    hot_models:

      - model: Factory Pattern
        year: 1994
        role: 设计模式经典实现
        architecture: creational pattern

      - model: Registry Pattern
        year: 1990s
        role: 组件注册和管理
        architecture: behavioral pattern

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

  graph:

    is_a:
      - FactoryModule
      - UnifiedInterface
      - NeuralNetworkWrapper

    part_of:
      - ModularArchitecture
      - DecoderSystem
      - ComponentRegistry

    depends_on:
      - UNetDecoder1D/2D/3D
      - GraphViTDecoder
      - MeshGraphDecoder
      - FengWuDecoder

    variants:
      - OneEncoder (编码器统一接口)
      - OneFuser (融合器统一接口)
      - OneMLP (MLP统一接口)

    used_in_models:
      - 模块化深度学习系统
      - 多模型集成框架
      - 动态模型选择系统

    compatible_with:

      inputs:
        - DecoderConfigurations
        - ModelParameters
        - InputData

      outputs:
        - DecoderResults
        - ModelOutputs
        - Predictions
