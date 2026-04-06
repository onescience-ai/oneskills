component:

  meta:
    name: PanguTransformerLayers
    alias: PanguLayer
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: transformer_block
    author: OneScience
    license: Apache-2.0
    tags:
      - pangu_weather
      - mlp
      - drop_path
      - stochastic_depth
      - vision_transformer

  concept:

    description: >
      PanguLayer 包含盘古气象大模型（Pangu-Weather）中构建 3D Earth-Specific Transformer (3DEST) 架构的核心基础组件：
      DropPath（随机深度丢弃路径）和 Mlp（多层感知机）。DropPath 用于在残差连接的主路径上按样本随机丢弃特征，从而对深度网络进行正则化；
      Mlp 则作为 Transformer Block 中的前馈神经网络（Feed-Forward Network, FFN），在特征的通道维度上进行非线性变换。

    intuition: >
      在极深的网络（如气象大模型中堆叠数十层的 3D Transformer）中，梯度消失和过拟合是严重的问题。
      DropPath 就像是在训练时随机“短路”掉某些层，迫使网络不过分依赖任何单一层，使得整体模型像是一个浅层网络的隐式集成体（Ensemble）。
      Mlp 则负责在每次空间混合（Spatial Mixing，如注意力机制）之后，进行一次纯粹的通道级特征混合（Channel Mixing）。

    problem_it_solves:
      - (DropPath) 防止极深 Transformer 网络的过拟合，并增强其泛化能力
      - (DropPath) 在训练大模型时作为一种有效的正则化手段（Stochastic Depth）
      - (Mlp) 为自注意力层之后的特征提供通道维度的非线性映射能力

  theory:

    formula:

      drop_path_output:
        expression: y = \begin{cases} \frac{x}{1-p} & \text{with probability } 1-p \\ 0 & \text{with probability } p \end{cases} \quad \text{(Training)}

      mlp_output:
        expression: y = \text{Dropout}(W_2 \cdot \text{Dropout}(\text{GELU}(W_1 \cdot x + b_1)) + b_2)

    variables:

      p:
        name: DropProbability
        description: 丢弃路径的概率 (drop_prob)

      x:
        name: Input
        shape: dynamic
        description: 输入特征张量

      W_1:
        name: Weight1
        description: Mlp 中第一层的线性投影权重矩阵

      W_2:
        name: Weight2
        description: Mlp 中第二层的线性投影权重矩阵

  structure:

    architecture: transformer_components

    pipeline:

      - name: StochasticDepth (DropPath)
        operation: conditional_masking_and_scaling

      - name: FeedForward (Mlp)
        operation: linear_gelu_linear

  interface:

    parameters:

      drop_prob:
        type: float
        default: 0.0
        description: DropPath 的丢弃概率

      scale_by_keep:
        type: bool
        default: true
        description: 是否在训练时按保留概率缩放特征，以保持期望输出不变

      in_features:
        type: int
        description: Mlp 的输入特征维度

      hidden_features:
        type: int
        default: null
        description: Mlp 的隐藏层特征维度（默认为 in_features）

      out_features:
        type: int
        default: null
        description: Mlp 的输出特征维度（默认为 in_features）

      act_layer:
        type: nn.Module
        default: nn.GELU
        description: Mlp 隐藏层使用的非线性激活函数

      drop:
        type: float
        default: 0.0
        description: Mlp 内全连接层后的 Dropout 概率

    inputs:

      x:
        type: Tensor
        shape: dynamic
        dtype: float32
        description: 传递给模块的特征张量

    outputs:

      output:
        type: Tensor
        shape: dynamic
        description: 经过 DropPath 或 Mlp 处理后的输出张量

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn
      from ..func_utils.pangu_utils import drop_path

      class DropPath(nn.Module):
          """摘自 timm 仓库
          按样本丢弃路径（Drop paths / 随机深度 Stochastic Depth）当应用于残差块的主路径时）
          """
          def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
              super(DropPath, self).__init__()
              self.drop_prob = drop_prob
              self.scale_by_keep = scale_by_keep

          def forward(self, x):
              return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

          def extra_repr(self):
              return f"drop_prob={round(self.drop_prob,3):0.3f}"

      class Mlp(nn.Module):
          def __init__(
              self,
              in_features,
              hidden_features=None,
              out_features=None,
              act_layer=nn.GELU,
              drop=0.0,
          ):
              super().__init__()
              out_features = out_features or in_features
              hidden_features = hidden_features or in_features
              self.fc1 = nn.Linear(in_features, hidden_features)
              self.act = act_layer()
              self.fc2 = nn.Linear(hidden_features, out_features)
              self.drop = nn.Dropout(drop)

          def forward(self, x: torch.Tensor):
              x = self.fc1(x)
              x = self.act(x)
              x = self.drop(x)
              x = self.fc2(x)
              x = self.drop(x)
              return x

  skills:

    build_pangu_layer:

      description: 为 Pangu-Weather 等 Vision Transformer 模型构建基础网络模块

      inputs:
        - block_type (DropPath or Mlp)
        - params (对应模块的初始化参数)

      prompt_template: |
        构建一个 {{block_type}}。
        参数：{{params}}。
        如果是 DropPath，请明确丢弃概率；如果是 Mlp，请指定输入/隐藏层维度。

    diagnose_pangu_layer:

      description: 分析使用 DropPath 时的训练/推理行为差异

      checks:
        - scale_by_keep_behavior_during_eval
        - dimension_mismatch_in_mlp_layers

  knowledge:

    usage_patterns:

      residual_block_with_stochastic_depth:

        pipeline:
          - Input (x)
          - SubLayer (如 3D-Attention / Mlp)
          - DropPath
          - Residual Add (x + DropPath(SubLayer(x)))

    design_patterns:

      stochastic_depth:

        structure:
          - 在极深网络中，网络层数越多，梯度越容易消失或陷入局部最优。
          - 通过在残差块的分支上施加 DropPath，相当于在训练时动态缩减了网络的有效深度。
          - 推理时（eval），DropPath 成为无操作（Identity），利用全部深度的特征进行预测。

    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 华为提出的高精度全球AI气象预报模型
        architecture: 3D_earth_specific_transformer
        attention_type: 3D_Swin_Attention

      - model: Swin Transformer
        year: 2021
        role: 层次化视觉 Transformer 骨干网络
        architecture: hierarchical_transformer

    model_usage_details:

      Pangu-Weather:

        usage: 盘古气象大模型采用 3D Transformer 架构（引入时间维度或高度维度），这些基础组件（Mlp 和 DropPath）被大量堆叠在其 Encoder 和 Decoder 块中。DropPath 概率通常随网络深度线性增加。

    best_practices:

      - `DropPath` 只应应用于带有残差连接（Residual Connection）的路径上。如果在主线性的非残差路径上使用 DropPath，会导致该层在训练中彻底切断信息流，毁灭模型训练。
      - 当堆叠多个 Transformer Block 时，`drop_prob` 通常被设置为线性递增（Linear Decay），即网络最浅层的 block 丢弃率极低，最深层的 block 丢弃率最高。
      - `scale_by_keep` 通常保持为 `True`，以此保证训练和推理（评估）阶段期望特征分布的无缝对齐。

    anti_patterns:

      - 在评估（eval）模式下未能正确关闭 `DropPath` 的行为，导致推理结果随机波动（该模块内部的 `self.training` 标志位由 PyTorch 自动管理，需确保在验证时调用了 `model.eval()`）。

    paper_references:

      - title: "Deep Networks with Stochastic Depth"
        authors: Huang et al.
        year: 2016

      - title: "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast"
        authors: Bi et al.
        year: 2023

  graph:

    is_a:
      - NeuralNetworkComponent
      - RegularizationLayer
      - FeedForwardNetwork

    part_of:
      - PanguWeatherBlock
      - SwinTransformerBlock
      - VisionTransformer

    depends_on:
      - pangu_utils.drop_path
      - nn.Linear
      - nn.GELU

    variants:
      - nn.Dropout
      - SpatialDropout

    used_in_models:
      - Pangu-Weather
      - Swin Transformer
      - ConvNeXt

    compatible_with:

      inputs:
        - Tensor

      outputs:
        - Tensor