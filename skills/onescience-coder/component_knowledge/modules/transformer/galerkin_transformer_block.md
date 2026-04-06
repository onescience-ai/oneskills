component:

  meta:
    name: GalerkinTransformerBlock
    alias: GalerkinLinearBlock
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: linear_transformer
    author: OneScience
    license: Apache-2.0
    tags:
      - galerkin_transformer
      - linear_attention
      - pde_solver
      - earth_science
      - o_n_complexity


  concept:

    description: >
      Galerkin Transformer 编码器块是为高效处理物理场数据而设计的线性 Transformer 组件。
      它摒弃了传统的 Softmax 点积注意力机制，转而采用基于 Galerkin 投影的线性注意力（Linear Attention）。
      该架构不仅保留了 Transformer 捕获长距离依赖的能力，还使得计算开销随序列长度线性增长。

    intuition: >
      在标准 Attention 中，每个 Token 都要和其他所有 Token 计算相似度，这就像是一个每个人都在互相握手的房间，
      人数一多（比如高分辨率网格），握手次数就会呈平方级爆炸。
      Galerkin Attention 则像是在房间里设立了几个“特征基底”作为中转站，大家先把信息投影到这些基底上（Galerkin 投影），
      然后再从基底重构特征。这样就只需要计算每个人与基底的交互，将复杂度降维到了线性级别。

    problem_it_solves:
      - 突破标准 Transformer 处理高分辨率物理网格数据（如长序列 ERA5 数据集）时的 $O(N^2)$ 显存和算力瓶颈。
      - 使脱离超级计算机、在标准计算硬件上训练大规模物理场替代模型（Surrogate Models）成为可能。
      - 提供了神经算子（Neural Operator）架构下针对偏微分方程（PDE）求解的高效注意力范式。


  theory:

    formula:

      galerkin_attention_update:
        expression: $$x_{attn} = x + \text{GalerkinAttention}(\text{LN}_1(x), \text{LN}_{1a}(x))$$

      mlp_update:
        expression: $$x_{out} = x_{attn} + \text{MLP}(\text{LN}_2(x_{attn}))$$

      optional_projection:
        expression: $$\text{final\_out} = \text{Linear}(\text{LN}_3(x_{out}))$$

    variables:

      x:
        name: InputPhysicalField
        shape: [batch, num_points, hidden_dim]
        description: 展平后的物理场网格点特征序列

      GalerkinAttention:
        name: LinearGalerkinProjection
        description: 接收两组经过独立 LayerNorm 处理的输入（分别作为 Q/K 和 V 的基础），通过 Galerkin 积分形式完成特征聚合。


  structure:

    architecture: linear_complexity_transformer

    pipeline:

      - name: DualPreNorm
        operation: layer_norm (并行使用 ln_1 和 ln_1a)

      - name: GalerkinLinearAttention
        operation: one_attention (style="LinearAttention", attn_type="galerkin")

      - name: AttnResidual
        operation: add

      - name: MlpPreNorm
        operation: layer_norm

      - name: FeedForwardNetwork
        operation: one_mlp (StandardMLP)

      - name: MlpResidual
        operation: add

      - name: OptionalFinalNorm
        operation: layer_norm (仅当 last_layer=True 激活)

      - name: OptionalProjection
        operation: linear_projection (仅当 last_layer=True 激活)


  interface:

    parameters:

      num_heads:
        type: int
        description: 注意力头的数量

      hidden_dim:
        type: int
        description: 隐藏层特征维度 (Embedding Dimension)

      dropout:
        type: float
        description: 特征映射的 Dropout 概率

      act:
        type: str
        default: "gelu"
        description: MLP 层的激活函数类型

      mlp_ratio:
        type: int
        default: 4
        description: MLP 隐藏层维度相对于 hidden_dim 的扩展倍数

      last_layer:
        type: bool
        default: False
        description: 标记是否为网络的最末层，若开启则附加最终的线性投影映射

      out_dim:
        type: int
        default: 1
        description: 最终输出的物理场变量维度（如输出温度场则为1，输出风速 u/v 则为2）

    inputs:

      fx:
        type: PhysicalFieldSequence
        shape: "[batch, num_points, hidden_dim] 或 [batch, hidden_dim, H, W]"
        dtype: float32

    outputs:

      output:
        type: Tensor
        shape: "[batch, num_points, hidden_dim] 或 [batch, num_points, out_dim]"
        description: 更新后的物理场特征表示


  types:

    PhysicalFieldSequence:
      description: 离散化后的连续物理场数据，通常为极其漫长的序列（数十万级）


  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from typing import Optional
      from onescience.modules.mlp.onemlp import OneMlp
      from onescience.modules.attention.oneattention import OneAttention

      class Galerkin_Transformer_block(nn.Module):
          """Galerkin Transformer 编码器块"""
          def __init__(
              self, num_heads: int, hidden_dim: int, dropout: float,
              act="gelu", mlp_ratio=4, last_layer=False, out_dim=1,
          ):
              super().__init__()
              self.last_layer = last_layer
              
              # Norm layers
              self.ln_1 = nn.LayerNorm(hidden_dim)
              self.ln_1a = nn.LayerNorm(hidden_dim)
              
              # Attention
              self.Attn = OneAttention(
                  style="LinearAttention", dim=hidden_dim, heads=num_heads,
                  dim_head=hidden_dim // num_heads, dropout=dropout, attn_type="galerkin",
              )
              
              self.ln_2 = nn.LayerNorm(hidden_dim)
              self.mlp = OneMlp(
                  style="StandardMLP", input_dim=hidden_dim, output_dim=hidden_dim,
                  hidden_dims=[hidden_dim * mlp_ratio], activation=act, use_bias=True, 
              )
              
              # Output Projection 
              if self.last_layer:
                  self.ln_3 = nn.LayerNorm(hidden_dim)
                  self.mlp2 = nn.Linear(hidden_dim, out_dim)

          def forward(self, fx):
              # Attention Residual
              fx = self.Attn(self.ln_1(fx), self.ln_1a(fx)) + fx
              # MLP Residual
              fx = self.mlp(self.ln_2(fx)) + fx
              
              if self.last_layer:
                  return self.mlp2(self.ln_3(fx))
              else:
                  return fx


  skills:

    build_galerkin_block:

      description: 构建具有 $O(N)$ 复杂度的物理场 Transformer 编码器层

      inputs:
        - num_heads
        - hidden_dim
        - last_layer
        - out_dim

      prompt_template: |

        请实例化 Galerkin_Transformer_block。
        作为骨干网络堆叠时，保持 last_layer=False。
        当作为物理场预测的最后一层时，设置 last_layer=True 并根据预测目标（如流速、压强）设置 out_dim。


    diagnose_linear_attention:

      description: 排查 Linear Attention 的数值不稳定性

      checks:
        - galerkin_projection_divergence (投影矩阵特征值爆炸导致 NaN)
        - pre_norm_asymmetry (双 LayerNorm ln_1 和 ln_1a 梯度更新异常)



  knowledge:

    usage_patterns:

      neural_operator_surrogate:

        pipeline:
          - Grid Encoder (将物理坐标映射到高维)
          - [Galerkin_Transformer_block] x N (在隐空间求解 PDE)
          - Grid Decoder (投影回物理目标空间)


    hot_models:

      - model: Galerkin Transformer
        year: 2021
        role: 结合算子学习与线性注意力机制的通用 PDE 求解器
        architecture: Linear Attention + Pre-Norm

      - model: Fourier Neural Operator (FNO)
        year: 2020
        role: 同属神经算子领域的代表性工作，通过频域变换实现全局感受野


    best_practices:

      - 这里的 `Attn` 需要接收两个独立的归一化输入 `ln_1(fx)` 和 `ln_1a(fx)`。这是 Galerkin 注意力的独特设计，用于分离 Query/Key 生成空间与 Value 聚合空间的特征分布，切勿将两者的代码合并。
      - 由于复杂度仅为 $O(N)$，可以放心增加序列长度，甚至直接输入未进行 Patch 划分的原始高分辨率网格，而不用担心爆显存。


    anti_patterns:

      - 在序列长度 $N$ 较小（如 NLP 中常见的 512 或 1024）的任务中使用此模块。由于 Galerkin 投影的常数项开销，在短序列下它甚至可能慢于标准 $O(N^2)$ 点积注意力。


    paper_references:

      - title: "Galerkin Transformer"
        authors: Cao et al.
        year: 2021



  graph:

    is_a:
      - TransformerBlock
      - NeuralOperator

    part_of:
      - PDESolver
      - SurrogateModel

    depends_on:
      - OneAttention (LinearAttention)
      - OneMlp
      - LayerNorm
      - Linear

    variants:
      - FourierNeuralOperatorBlock
      - PerformerBlock

    used_in_models:
      - Galerkin Transformer

    compatible_with:

      inputs:
        - PhysicalFieldSequence

      outputs:
        - PhysicalFieldSequence
        - PhysicalFieldPrediction (if last_layer=True)