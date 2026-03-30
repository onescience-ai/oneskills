component:

  meta:
    name: OneEvolution
    alias: EvolutionFactory
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: module_dispatcher
    author: OneScience
    license: Apache-2.0
    tags:
      - evolution
      - registry_like_dispatch
      - wrapper
      - nowcastnet
      - factory_pattern


  concept:

    description: >
      OneEvolution 的目标是为演化模型提供统一实例化入口，当前仅路由到 nowcastnet 风格。
      文件中包含一个 nowcastnet 定义以及 OneEvolution 包装器雏形，
      其核心意图是按 style 选择具体演化模块并向外暴露一致接口。

    intuition: >
      该设计与其他 one* 封装模块一致，属于“按字符串风格选择后端实现”的工厂模式。
      当需要接入多个演化网络时，可通过扩展 style 分支实现快速切换。

    problem_it_solves:
      - 统一演化网络的调用入口
      - 通过 style 参数支持演化模型切换
      - 将上层训练代码与具体演化实现解耦
      - 为后续扩展更多 evolution backbone 预留接口


  theory:

    formula:

      style_dispatch:
        expression: |
          f_{evo} = Select(style)
          y = f_{evo}(x)

    variables:

      style:
        name: EvolutionStyle
        description: 演化模块风格名称

      f_{evo}:
        name: EvolutionModule
        description: 被选中的演化网络实例

      x:
        name: InputField
        shape: [batch, channels, H, W]
        description: 演化模型输入


  structure:

    architecture: dispatcher_wrapper

    pipeline:

      - name: StyleCheck
        operation: compare_style_with_supported_options

      - name: ModuleInstantiation
        operation: construct_nowcastnet_or_raise_error

      - name: ForwardDelegation
        operation: passthrough_to_selected_evolution_module


  interface:

    parameters:

      style:
        type: str
        description: 演化模型风格，默认意图为 nowcastnet

    inputs:

      x:
        type: FieldEmbedding
        shape: [batch, channels, H, W]
        dtype: float32
        description: 输入场张量

    outputs:

      output:
        type: TensorOrTuple
        description: 由具体演化模型定义的输出


  types:

    FieldEmbedding:
      shape: [batch, channels, H, W]
      description: 二维场数据 embedding

    TensorOrTuple:
      description: 具体演化模块输出结构


  implementation:

    framework: pytorch

    code: |

      import torch.nn.functional as F
      from .module import *

      class nowcastnet(nn.Module):
          def __init__(self, n_channels, n_classes, base_c=64, bilinear=True):
              super(nowcastnet, self).__init__()
              ...

          def forward(self, x):
              ...
              return x, v

      class OneEvolution(nn.module):
          def __init__(self, style=="nowcastnet"):
              if style == "nowcastnet":
                  self.Evolution = nowcastnet()
              else:
                  raise NotImplementedError


  skills:

    build_evolution_dispatcher:

      description: 构建演化网络统一调用封装

      inputs:
        - style

      prompt_template: |

        构建 OneEvolution 模块。

        参数：
        style = {{style}}

        要求：
        对支持的风格实例化对应演化模型，对未知风格抛错。


    diagnose_evolution_dispatcher:

      description: 分析演化分发层中的实现一致性与可用性问题

      checks:
        - invalid_class_inheritance_name
        - incorrect_default_argument_syntax
        - missing_super_init_call
        - absent_forward_method_in_wrapper


  knowledge:

    usage_patterns:

      evolution_module_switching:

        pipeline:
          - ChooseStyle
          - InstantiateOneEvolution
          - ForwardToBackend

      future_extension:

        pipeline:
          - AddNewEvolutionModel
          - ExtendStyleBranch
          - ValidateOutputContract


    hot_models:

      - model: nowcastnet
        year: 2022
        role: 当前 OneEvolution 目标后端
        architecture: dual_head_unet


    best_practices:

      - 使用注册表替代硬编码 style 分支以提升可扩展性。
      - 保证 OneEvolution 本身继承 nn.Module 且实现 forward。
      - 将 backend 构造参数透传，避免固定空参实例化。


    anti_patterns:

      - 使用 `nn.module`（错误大小写）导致类继承失败。
      - 在 `__init__(self, style==...)` 中误用比较表达式替代参数默认值。


    paper_references:

      - title: "Design Patterns: Elements of Reusable Object-Oriented Software"
        authors: Gamma et al.
        year: 1994


  graph:

    is_a:
      - NeuralNetworkComponent
      - FactoryWrapper

    part_of:
      - EvolutionModuleSystem

    depends_on:
      - nowcastnet
      - DispatcherPattern

    variants:
      - nowcastnet
      - oneevolution

    used_in_models:
      - Weather Evolution Pipelines
      - Nowcasting Prototypes
      - Spatiotemporal Forecasting Systems

    compatible_with:

      inputs:
        - FieldEmbedding

      outputs:
        - TensorOrTuple