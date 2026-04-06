component:

  meta:
    name: OneEmbedding
    alias: EmbeddingRegistry
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: embedding
    author: OneScience
    license: Apache-2.0
    tags:
      - embedding
      - registry
      - factory_pattern
      - routing_module

  concept:

    description: >
      OneEmbedding 是整个气象和流体模型库（OneScience）中各种复杂嵌入模块的中心注册表兼分配器。
      它使用了工厂模式（Factory Pattern）和类的动态代理机制，通过传入的 `style` 字符串参数，自动实例化对应的特定领域嵌入器（例如 PanguEmbedding, FuxiEmbedding 等），并将后续所有的模型输入或属性请求转发给这些底层的具体实现。

    intuition: >
      如果你去一家超级大餐厅点菜，可能中餐区有一个师傅，西餐区有一个师傅。
      OneEmbedding 就像是这家餐厅的服务员前台。你只需要告诉它你的菜系（`style`），它不仅会自动找对相应的厨师（构建具体的 Embedding 实例），且后续所有的送菜加料请求都会直接无缝对接给那位厨师处理，你甚至感觉不到它的存在。

    problem_it_solves:
      - 聚合各种离散且用法不同的气象模型组件代码入同一结构
      - 大幅度简化顶级配置文件的模型拓扑初始化，仅利用字典字符串解耦代码依赖
      - 实现各种 Embedding 调用的鸭子类型转换代理


  theory:

    formula:

      registry_routing:
        expression: |
          \text{Embedder}_{instance} = \text{REGISTRY}[style](**kwargs)
          \text{Output} = \text{Embedder}_{instance}(*args, **kwargs)

    variables:

      style:
        name: StyleIdentifier
        description: 用来指示初始化哪一种特定模型结构的字符串名称键值


  structure:

    architecture: dynamic_proxy_and_registry

    pipeline:

      - name: IdentifierLookup
        operation: dictionary_key_retrieval

      - name: ConcreteInstantiation
        operation: kwargs_parameters_passing

      - name: MethodDelegation
        operation: getattr_and_forwarding


  interface:

    parameters:

      style:
        type: str
        description: 模型类型的字串符注册名参数

      **kwargs:
        type: dict
        description: 透传给具体实例化类的任意参数

    inputs:

      *args:
        type: Any
        description: 输入的数据

    outputs:

      Output:
        type: Any
        description: 具体类投射处理后的输出特征张量


  types:

    RegistryDict:
      shape: [Hash Map]
      description: 包含模型类的可调用映射字典


  implementation:

    framework: pytorch

    code: |
      from torch import nn
      
      from .panguembedding2d import PanguEmbedding2D
      from .panguembedding3d import PanguEmbedding3D
      from .fourier_pos_embedding import FourierPosEmbedding
      from .fuxiembedding import FuxiEmbedding
      from .fourcastnetembedding import FourCastNetEmbedding
      from .xiheembedding import XiheEmbedding
      from .graphcast_embedder import GraphCastEncoderEmbedder, GraphCastDecoderEmbedder
      # from .protenixembedding import (
      #     ProtenixFourierEmbedding,
      #     ProtenixInputFeatureEmbedder,
      #     ProtenixTemplateEmbedder,
      # )
      
      _EMBEDDER_REGISTRY = {
          "PanguEmbedding2D": PanguEmbedding2D,
          "PanguEmbedding3D": PanguEmbedding3D,
          "FourierPosEmbedding": FourierPosEmbedding,
          "FuxiEmbedding": FuxiEmbedding,
          "FourCastNetEmbedding": FourCastNetEmbedding,
          "XiheEmbedding": XiheEmbedding,
          "GraphCastEncoderEmbedder": GraphCastEncoderEmbedder,
          "GraphCastDecoderEmbedder": GraphCastDecoderEmbedder,
          # "ProtenixFourierEmbedding": ProtenixFourierEmbedding,
          # "ProtenixInputFeatureEmbedder": ProtenixInputFeatureEmbedder,
          # "ProtenixTemplateEmbedder": ProtenixTemplateEmbedder,
      }
      
      class OneEmbedding(nn.Module):
      
          def __init__(self, style: str, **kwargs):
              super().__init__()
      
              if style not in _EMBEDDER_REGISTRY:
                  raise NotImplementedError(f"Unknown style: {style}")
      
              self.embedder = _EMBEDDER_REGISTRY[style](**kwargs)
          
          def __getattr__(self, name):
              try:
                  return super().__getattr__(name)
              except AttributeError:
                  return getattr(self.embedder, name)
      
          def forward(self, *args, **kwargs):
              return self.embedder(*args, **kwargs) 

  skills:

    build_oneembedding:

      description: 构建具有注册和代理功能的综合前端路由模块

      inputs:
        - registry_dict
        - style

      prompt_template: |

        请实现 OneEmbedding 模块。
        需包含对 `_EMBEDDER_REGISTRY` 字典的查找与构建。
        重写 `__getattr__` 实现对内部属性的穿透代理。


    diagnose_oneembedding:

      description: 分析 OneEmbedding 路由模块常见失效模式

      checks:
        - KeyError_for_unregistered_style
        - attribute_proxy_infinite_loops
        - mismatched_kwargs_propagation


  knowledge:

    usage_patterns:

      model_factory_init:

        pipeline:
          - Parse YAML style string
          - Init OneEmbedding
          - Seamlessly pass variables into Network 

    hot_models:

      - model: MMDetection/TIMM
        role: 也是采用了大量类似基于 Registry 装饰器的大型框架
        architecture: Registry

    best_practices:

      - 通过重写 `__getattr__`，主调用者可以直接获取到底层具体的内部参量（如 `embedder.patch_size`）而不需要改变原来的逻辑接口。
      - 被注释的掉的其它实验性嵌入模块依然可以在此地通过控制反转统一添加回代码主线。


    anti_patterns:

      - 不检查底层调用或在 `__getattr__` 里引入互相引用的操作极易导致 Python 的递归爆栈死锁。所以在 `except AttributeError` 之后再调 `getattr(self.embedder)` 是正规做法。

    paper_references:

      - title: "Design Patterns: Elements of Reusable Object-Oriented Software"
        authors: GoF
        year: 1994


  graph:

    is_a:
      - ProxyModule
      - FactoryPattern

    part_of:
      - OneScienceLibrary

    depends_on:
      - getattr

    variants:
      - DecoratorRegistry

    used_in_models:
      - OneScienceUnifiedModel

    compatible_with:

      inputs:
        - StrIdentifier

      outputs:
        - Any