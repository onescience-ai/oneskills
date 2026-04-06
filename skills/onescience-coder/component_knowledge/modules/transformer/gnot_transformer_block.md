component:

  meta:
    name: GNOTTransformerBlock
    alias: GNOTBlock
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: neural_operator
    author: OneScience
    license: Apache-2.0
    tags:
      - gnot
      - neural_operator
      - mixture_of_experts
      - linear_attention
      - cross_attention


  concept:

    description: >
      GNOT (General Neural Operator Transformer) 编码器块。
      该模块为异构数据（如几何网格和物理场观测）的融合提供了一个高效的 Transformer 架构。
      它交替使用线性交叉注意力（融合几何与物理信息）和线性自注意力（几何特征内部交互），
      并引入了基于空间坐标路由的混合专家 (MoE) 前馈网络，以捕捉物理场在不同空间位置的非平稳映射关系。

    intuition: >
      处理复杂的物理模拟时，输入往往包含两部分：网格的几何形状和附着在网格上的物理观测值。
      这个模块首先让“几何”和“物理”开个会交换信息（Cross Attention）。
      随后，它会查看当前数据点在真实物理世界中的“坐标位置”。如果在流体边缘，它会唤醒“边缘处理专家”；
      如果在平稳中心，它会唤醒“平稳处理专家”（通过 GateNet 计算出的门控分数控制 MoE）。
      之后几何特征再进行一次内部复盘（Self Attention），并再次经过专家组的提炼。

    problem_it_solves:
      - 高效融合异构的输入数据（如输入特征维度不同或含义不同的几何和物理数据）
      - 解决传统网络在处理空间非平稳（Spatially Non-stationary）物理场时泛化能力不足的问题
      - 依靠线性注意力机制打破高分辨率网格数据计算的 $O(N^2)$ 显存瓶颈


  theory:

    formula:

      cross_attention_fusion:
        expression: $x_1 = x + \text{Dropout}(\text{CrossAttn}(\text{LN}_1(x), \text{LN}_2(y)))$

      spatial_moe_1:
        expression: $x_2 = x_1 + \text{LN}_3(\sum_{i=1}^{E} \text{Softmax}(\text{Gate}(pos))_i \cdot \text{MLP}^{(1)}_i(x_1))$

      self_attention_update:
        expression: $x_3 = x_2 + \text{Dropout}(\text{SelfAttn}(\text{LN}_4(x_2)))$

      spatial_moe_2:
        expression: $x_{out} = x_3 + \text{LN}_5(\sum_{i=1}^{E} \text{Softmax}(\text{Gate}(pos))_i \cdot \text{MLP}^{(2)}_i(x_3))$

    variables:

      x:
        name: GeometryFeatures
        shape: [batch, num_points, hidden_dim]
        description: 目标网格或几何结构的特征张量

      y:
        name: PhysicsFeatures
        shape: [batch, num_points, hidden_dim]
        description: 相关的物理场观测或条件特征张量

      pos:
        name: SpatialCoordinates
        shape: [batch, num_points, space_dim]
        description: 物理空间坐标（如 2D 的 [X, Y] 或 3D 的 [X, Y, Z]）

      Gate:
        name: GatingNetwork
        description: 根据坐标输出专家权重的多层感知机


  structure:

    architecture: spatial_moe_linear_transformer

    pipeline:

      - name: GateScoreComputation
        operation: standard_mlp + softmax (基于 pos 计算 MoE 权重)

      - name: HeterogeneousFusion
        operation: linear_cross_attention (x 作为 Query, y 作为 Key/Value)

      - name: SpatialMoE_Stage1
        operation: expert_mlps + weighted_sum

      - name: GeometrySelfInteraction
        operation: linear_self_attention

      - name: SpatialMoE_Stage2
        operation: expert_mlps + weighted_sum


  interface:

    parameters:

      num_heads:
        type: int
        description: 线性注意力头的数量

      hidden_dim:
        type: int
        description: 隐藏层特征维度

      dropout:
        type: float
        description: 注意力后的 Dropout 概率

      act:
        type: str
        default: "gelu"
        description: 门控网络和所有专家 MLP 的激活函数类型

      mlp_ratio:
        type: int
        default: 4
        description: MLP 隐藏层维度扩展倍数

      space_dim:
        type: int
        default: 2
        description: 空间坐标维度，必须与输入的 pos 维度匹配

      n_experts:
        type: int
        default: 3
        description: MoE 层中包含的独立专家 MLP 数量

    inputs:

      x:
        type: Tensor
        shape: [batch, num_points, hidden_dim]
        dtype: float32
        description: 待更新的几何或主干特征序列

      y:
        type: Tensor
        shape: [batch, num_points, hidden_dim]
        dtype: float32
        description: 用于提供外部信息的物理/辅助特征序列

      pos:
        type: Tensor
        shape: [batch, num_points, space_dim]
        dtype: float32
        description: 序列对应的物理坐标位置

    outputs:

      output:
        type: Tensor
        shape: [batch, num_points, hidden_dim]
        description: 融合了物理信息并经过空间自适应非线性映射后的特征


  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      import torch.nn.functional as F
      from onescience.modules.mlp.onemlp import OneMlp
      from onescience.modules.attention.oneattention import OneAttention

      class GNOTTransformerBlock(nn.Module):
          """GNOT Transformer 编码器块 (MoE 风格)"""
          def __init__(
              self, num_heads: int, hidden_dim: int, dropout: float,
              act="gelu", mlp_ratio=4, space_dim=2, n_experts=3,
          ):
              super().__init__()
              self.ln1, self.ln2, self.ln3, self.ln4, self.ln5 = [nn.LayerNorm(hidden_dim) for _ in range(5)]

              # Linear Attention 重构
              self.selfattn = OneAttention(style="LinearAttention", dim=hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout)
              self.crossattn = OneAttention(style="LinearAttention", dim=hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout)
              
              self.resid_drop1 = nn.Dropout(dropout)
              self.resid_drop2 = nn.Dropout(dropout)
              self.n_experts = n_experts
              
              # MoE MLP 专家组
              self.moe_mlp1 = nn.ModuleList([
                  OneMlp(style="StandardMLP", input_dim=hidden_dim, output_dim=hidden_dim, hidden_dims=[hidden_dim * mlp_ratio], activation=act, use_bias=True)
                  for _ in range(self.n_experts)
              ])
              self.moe_mlp2 = nn.ModuleList([
                  OneMlp(style="StandardMLP", input_dim=hidden_dim, output_dim=hidden_dim, hidden_dims=[hidden_dim * mlp_ratio], activation=act, use_bias=True)
                  for _ in range(self.n_experts)
              ])
              
              # 空间门控网络
              self.gatenet = OneMlp(style="StandardMLP", input_dim=space_dim, output_dim=self.n_experts, hidden_dims=[hidden_dim * mlp_ratio, hidden_dim * mlp_ratio], activation=act, use_bias=True)

          def forward(self, x, y, pos):
              # 1. 计算专家权重
              gate_score = F.softmax(self.gatenet(pos), dim=-1).unsqueeze(2)
              
              # 2. Cross Attention
              x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln2(y)))
              
              # 3. MoE Stage 1
              x_moe1 = torch.stack([self.moe_mlp1[i](x) for i in range(self.n_experts)], dim=-1)
              x_moe1 = (gate_score * x_moe1).sum(dim=-1, keepdim=False)
              x = x + self.ln3(x_moe1)
              
              # 4. Self Attention
              x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
              
              # 5. MoE Stage 2
              x_moe2 = torch.stack([self.moe_mlp2[i](x) for i in range(self.n_experts)], dim=-1)
              x_moe2 = (gate_score * x_moe2).sum(dim=-1, keepdim=False)
              x = x + self.ln5(x_moe2)
              
              return x


  skills:

    build_gnot_block:

      description: 构建以空间坐标为路由机制的 GNOT Transformer 块

      inputs:
        - hidden_dim
        - space_dim
        - n_experts

      prompt_template: |

        实例化 GNOTTransformerBlock。
        必须确认输入数据集的坐标维度（例如 2D 翼型边界是 2，3D 气象场是 3），并据此设置 space_dim 确保 GateNet 能够正确接收 pos 张量。


    diagnose_gnot:

      description: 分析异构特征融合和 MoE 训练的异常

      checks:
        - gatenet_collapse (门控网络对所有空间位置都输出相同的专家权重，导致 MoE 退化)
        - linear_attention_nan (由于数值稳定性问题导致 Cross Attention 输出包含 NaN)



  knowledge:

    usage_patterns:

      gnot_operator_framework:

        pipeline:
          - PointCloud / Grid Embedder (分别编码网格和观测物理量)
          - Multiple GNOTTransformerBlocks (基于坐标不断融合与推演)
          - Decoder (输出目标物理量)


    hot_models:

      - model: GNOT (General Neural Operator Transformer)
        year: 2023
        role: 能够处理不规则网格、异构输入和多尺度空间演变的通用物理算子
        architecture: MoE + Linear Attention


    best_practices:

      - 在训练阶段，应当监控 `gate_score` 的分布。如果发现 `gate_score` 在空间中没有任何区分度（所有专家权重趋于均等），说明坐标 `pos` 可能缺乏有效的归一化，或者 `gatenet` 的学习率过小。
      - `n_experts` 不宜设置过大（通常 3-8 即可），因为在不使用稀疏计算（如 Top-K）的这种 Soft MoE 实现中，所有专家的前向传播都会被完整计算，显存开销与专家数成正比。


    anti_patterns:

      - 将归一化前的绝对坐标直接作为 `pos` 输入。物理场的绝对坐标差异过大可能导致 `gatenet` 的梯度爆炸，应首先将 `pos` 归一化至 $[-1, 1]$ 范围内。


    paper_references:

      - title: "GNOT: A General Neural Operator Transformer for Operator Learning"
        authors: Hao et al.
        year: 2023



  graph:

    is_a:
      - TransformerBlock
      - NeuralOperator
      - MixtureOfExperts

    part_of:
      - GNOTModel
      - AIforScienceSurrogate

    depends_on:
      - OneAttention (LinearAttention)
      - OneMlp (GateNet & Experts)
      - LayerNorm
      - Softmax

    variants:
      - DeepONet
      - FNO

    used_in_models:
      - GNOT

    compatible_with:

      inputs:
        - PointCloudGeometry
        - PhysicalObservation

      outputs:
        - UpdatedGeometryFeatures