component:

  meta:
    name: OneAttention
    alias: Attention Factory / Unified Dispatcher
    version: 1.0
    domain: deep_learning_and_ai4science
    category: neural_network
    subcategory: module_registry
    author: OneScience
    license: Apache-2.0
    tags:
      - factory_pattern
      - unified_interface
      - dynamic_dispatch
      - modular_design
      - attention_wrapper

  concept:

    description: >
      OneAttention 是一个用于统一管理和动态调用各种注意力机制的工厂模块（Factory Module）。
      它维护了一个全局的注意力注册表（Registry），包含了超过 20 种针对不同领域（自然语言、2D视觉、3D气象、非结构化物理网格、蛋白质大分子等）优化的注意力机制。
      通过传入特定的 style 字符串，它可以在运行时动态实例化并代理目标注意力层的计算。

    intuition: >
      在构建通用人工智能平台或 AI4Science 求解器时，我们往往需要根据不同的数据类型尝试不同的注意力算法。
      OneAttention 就像是一个“万能插座”或“中央调度员”。当你需要组装模型时，不用去仓库里翻找各种不同的零件并修改接口代码，
      只需告诉 OneAttention：“给我一个针对 3D 地球网格的注意力层（EarthAttention3D）”，它就会自动帮你把对应的零件装好并接通电源。

    problem_it_solves:
      - 解决在大型复杂算法库中，大量导入不同注意力类造成的代码强耦合问题。
      - 允许通过外部配置文件（如 YAML/JSON config）通过纯字符串的方式动态切换网络底层的核心注意力算子，而无需修改模型架构源码。
      - 极大地提升了 AI4Science 框架的可扩展性：未来添加新的注意力算法只需在注册表中增加一行，对上游完全透明。

  theory:

    formula:
      
      dynamic_dispatch:
        expression: Output = Registry[style](**kwargs).forward(*args, **kwargs)

    variables:

      style:
        name: AttentionStyleName
        description: 目标注意力机制的注册名称字符串，如 "Physics_Attention_Irregular_Mesh"。

      _ATTENTIONER_REGISTRY:
        name: GlobalRegistryDict
        description: 存储字符串到具体 PyTorch Module 类映射的全局哈希字典。

  structure:

    architecture: factory_and_delegate_pattern

    pipeline:

      - name: StyleLookup
        operation: dictionary_get (在初始化阶段，从 _ATTENTIONER_REGISTRY 查找对应的类)

      - name: InstanceInitialization
        operation: class_instantiation (利用透传的 **kwargs 实例化具体的注意力对象)

      - name: ForwardDelegation
        operation: method_forwarding (在 forward 阶段，将所有 *args 和 **kwargs 原封不动地代理给实例化的对象)

  interface:

    parameters:

      style:
        type: str
        description: 指定要实例化的注意力层名称，必须是已在 `_ATTENTIONER_REGISTRY` 注册的有效字符串。

      kwargs:
        type: dict
        description: 变长关键字参数，将被透传给具体目标注意力类的 __init__ 方法。

    inputs:

      args:
        type: tuple
        description: 变长位置参数，透传给目标 forward

      kwargs:
        type: dict
        description: 变长关键字参数，透传给目标 forward

    outputs:

      output:
        type: Tensor_or_Tuple
        description: 完全取决于由 `style` 动态实例化的那个具体注意力层返回什么格式的数据。

  types:

    AttentionStyleString:
      type: str
      description: 限定在 `_ATTENTIONER_REGISTRY.keys()` 范围内的枚举字符串。

  implementation:

    framework: pytorch

    code: |
      from torch import nn

      from .earthattention2d import EarthAttention2D
      from .earthattention3d import EarthAttention3D
      from .earthdistributedattention3d import EarthDistributedAttention3D
      from .physicsattention import Physics_Attention_Irregular_Mesh
      from .physicsattention import Physics_Attention_Irregular_Mesh_plus
      from .physicsattention import Physics_Attention_Structured_Mesh_1D
      from .physicsattention import Physics_Attention_Structured_Mesh_2D
      from .physicsattention import Physics_Attention_Structured_Mesh_3D
      from .factattention import FactAttention2D
      from .factattention import FactAttention3D
      from .flashattention import FlashAttention
      from .linearattention import LinearAttention
      from .linearattention import Vanilla_Linear_Attention
      from .multiheadattention import MultiHeadAttention
      from .selfattention import SelfAttention
      from .windowattention import WindowAttention
      from .nystrom_attention import NystromAttention
      from .xihefeaturegroupattention import FeatureGroupingAttention
      from .xihefeatureungroupattention import FeatureUngroupingAttention
      from .protenixattention import (
          ProtenixAttention,
          ProtenixAttentionPairBias,
          ProtenixAttentionPairBiasWithLocalAttn,
      )
      _ATTENTIONER_REGISTRY = {
          "EarthAttention2D": EarthAttention2D,
          "EarthAttention3D": EarthAttention3D,
          "EarthDistributedAttention3D": EarthDistributedAttention3D,
          "Physics_Attention_Irregular_Mesh": Physics_Attention_Irregular_Mesh,
          "Physics_Attention_Irregular_Mesh_plus": Physics_Attention_Irregular_Mesh_plus,
          "Physics_Attention_Structured_Mesh_1D": Physics_Attention_Structured_Mesh_1D,
          "Physics_Attention_Structured_Mesh_2D": Physics_Attention_Structured_Mesh_2D,
          "Physics_Attention_Structured_Mesh_3D": Physics_Attention_Structured_Mesh_3D,
          "FactAttention2D": FactAttention2D,
          "FactAttention3D": FactAttention3D,
          "FlashAttention": FlashAttention,
          "LinearAttention": LinearAttention,
          "Vanilla_Linear_Attention": Vanilla_Linear_Attention,
          "MultiHeadAttention": MultiHeadAttention,
          "SelfAttention": SelfAttention,
          "WindowAttention": WindowAttention,
          "NystromAttention": NystromAttention,
          "FeatureUngroupingAttention": FeatureUngroupingAttention,
          "FeatureGroupingAttention": FeatureGroupingAttention,
          "ProtenixAttention": ProtenixAttention,
          "ProtenixAttentionPairBias": ProtenixAttentionPairBias,
          "ProtenixAttentionPairBiasWithLocalAttn": ProtenixAttentionPairBiasWithLocalAttn,
      }


      class OneAttention(nn.Module):
          def __init__(self, style: str, **kwargs):
              super().__init__()

              if style not in _ATTENTIONER_REGISTRY:
                  raise NotImplementedError(f"Unknown style: {style}")

              self.attentioner = _ATTENTIONER_REGISTRY[style](**kwargs)

          def forward(self, *args, **kwargs):
              return self.attentioner(*args, **kwargs)
    |

  skills:

    build_unified_attention_wrapper:

      description: 构建一个可以动态代理任何底层注意力机制的模块外壳

      inputs:
        - style
        - kwargs

      prompt_template: 
        创建一个 OneAttention 层。
        目标代理算法的类型是 `style={{style}}`。
        请将以下参数传入透传给底层的 __init__：{{kwargs}}。

    diagnose_factory_registration:

      description: 排查由于组件未注册或透传参数不匹配导致的初始化失败

      checks:
        - not_implemented_error (传入的 `style` 字符串发生拼写错误或该算法尚未在 `_ATTENTIONER_REGISTRY` 中注册)
        - unexpected_keyword_argument (透传的 `kwargs` 包含了目标具体注意力类不支持的参数，导致其 `__init__` 函数报错)

  knowledge:

    usage_patterns:

      dynamic_model_building:
        pipeline:
          - ParseConfig (从 yaml 或 json 读取架构信息)
          - OneAttention(style=config.attn_style, **config.attn_kwargs)
          - FFN

    design_patterns:

      factory_and_strategy_pattern:
        structure:
          - 这是经典的面向对象设计模式结合。工厂模式（Factory）体现在通过字符串 `style` 统一创建对象；策略模式（Strategy）体现在算法的无缝替换，上游调用者（如 Transolver_block）不需要知道执行的是 MHA 还是 FlashAttention，只管调用 `forward`。

    hot_models:

      - model: OneScience Open Framework
        year: 2024
        role: 致力于大一统的 AI for Science 底层代码库
        architecture: Highly Modularized
        attention_type: Dynamic Delegated (OneAttention)

    model_usage_details:

      Transolver_Dynamic_Instantiator:
        style: "Physics_Attention_Irregular_Mesh" (受配置文件驱动改变)
        kwargs: {"dim": 256, "heads": 8, "slice_num": 32}

    best_practices:
      - 在开发新的自定义注意力层后，务必在顶部导入该类，并将其精确的字符串名称添加到 `_ATTENTIONER_REGISTRY` 字典中，否则 `OneAttention` 无法感知新模块的存在。
      - 所有的注意力子类最好实现一定程度上的接口统一（例如都具备 `dim` 和 `heads` 等通用参数），这能最大化 `OneAttention` 动态替换时的安全性和便利性。

    anti_patterns:
      - 在具体模型代码（如 Transformer Block）中直接 `import MultiHeadAttention` 并硬编码，这会破坏整个代码库的动态配置生态。
      - 传入未注册的 `style` 字符串并在捕获 `NotImplementedError` 时不做日志记录，导致排查困难。

    paper_references:

      - title: "Design Patterns: Elements of Reusable Object-Oriented Software"
        authors: Erich Gamma, Richard Helm, Ralph Johnson, John Vlissides (Gang of Four)
        year: 1994
        journal: Addison-Wesley (虽然不是深度学习论文，但这是此处所用工厂设计模式的开创性文献)

  graph:

    is_a:
      - NeuralNetworkComponent
      - ModuleFactory
      - ComponentWrapper

    part_of:
      - ModularAI4ScienceFramework

    depends_on:
      - _ATTENTIONER_REGISTRY
      - All Specific Attention Mechanisms

    compatible_with:
      inputs:
        - Dynamic (*args, **kwargs)
      outputs:
        - Dynamic