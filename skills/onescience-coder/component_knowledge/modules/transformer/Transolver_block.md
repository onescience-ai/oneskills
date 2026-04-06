component:

  meta:
    name: TransolverBlock
    alias: Transolver Encoder Block
    version: 1.0
    domain: ai4science
    category: neural_network
    subcategory: encoder_block
    author: OneScience
    license: Apache-2.0
    tags:
      - transolver
      - physics_attention
      - transformer
      - pde_solver
      - deep_learning

  concept:

    description: >
      Transolver 编码器块 (Transolver Encoder Block) 是 Transolver 模型的核心构建单元。
      它采用标准的 Transformer Encoder 架构，但集成了物理感知的注意力机制 (Physics-Attention)。
      它能够根据输入的几何类型（如结构化或非结构化网格），动态调用相应的注意力机制，并结合多层感知机 (MLP) 进行特征变换。

    intuition: >
      传统的 Transformer 在处理物理场数据时，由于网格点数量庞大且几何结构复杂，直接计算注意力会面临巨大的计算挑战。
      Transolver Block 像是一个“懂物理规律的观察者”，它通过 Physics-Attention 将离散的网格域自适应地划分为一系列形状灵活的可学习切片 (slices)。
      物理状态相似的网格点会被分配到同一个切片中，模型通过计算这些切片之间的注意力来捕捉复杂的物理相关性，从而实现线性复杂度的计算。

    problem_it_solves:
      - 解决传统自注意力机制在处理复杂几何形状的大规模物理网格时面临的计算与显存瓶颈
      - 提供统一的架构，无缝兼容结构化网格和非结构化网格数据
      - 提升偏微分方程 (PDE) 求解的精度与效率，特别适用于气象人工智能领域中复杂流体动力学的长期演化模拟以及大型工业气动力学仿真

  theory:

    formula:
      
      residual_attention:
        expression: X' = PhysicsAttention(LayerNorm(X)) + X
        
      residual_mlp:
        expression: X'' = StandardMLP(LayerNorm(X')) + X'
        
      optional_output_projection:
        expression: Output = Linear(LayerNorm(X'')) # 仅当 last_layer=True 时

    variables:

      X:
        name: InputFeatures
        shape: [B, N, C]
        description: 输入特征图，其中 B 为批次大小，N 为节点/网格点数量，C 为隐藏层维度

  structure:

    architecture: transformer_encoder_variant

    pipeline:

      - name: InputNorm
        operation: layer_normalization

      - name: PhysicsAttention
        operation: dynamic_attention_based_on_geotype

      - name: ResidualConnection1
        operation: add

      - name: MiddleNorm
        operation: layer_normalization

      - name: FeedForwardNetwork
        operation: standard_mlp

      - name: ResidualConnection2
        operation: add

      - name: OutputNormAndProjection
        operation: layer_norm_and_linear (可选，仅限最后一层)

  interface:

    parameters:

      num_heads:
        type: int
        description: 注意力头的数量

      hidden_dim:
        type: int
        description: 隐藏层的特征维度（输入和输出的主维度）

      dropout:
        type: float
        description: Dropout 概率，用于防止过拟合

      act:
        type: str
        default: "gelu"
        description: MLP层使用的激活函数类型

      mlp_ratio:
        type: int
        default: 4
        description: MLP 隐藏层维度相对于 hidden_dim 的放大倍率

      last_layer:
        type: bool
        default: false
        description: 标识是否为模型的最后一个 Block，若是则附加输出投影层

      out_dim:
        type: int
        default: 1
        description: 最终预测输出的特征维度（仅在 last_layer=True 时生效）

      slice_num:
        type: int
        default: 32
        description: 用于物理注意力机制的分片数量（针对特定非结构化几何类型）

      geotype:
        type: str
        default: "unstructured"
        description: 几何类型，决定调用的具体 Attention 变体（如 unstructured, structured_2D 等）

      shapelist:
        type: list
        default: null
        description: 网格形状列表，用于结构化网格的注意力跨度计算

    inputs:

      fx:
        type: NodeFeatures
        shape: [B, N, C]
        dtype: float32
        description: 节点或网格点的物理特征输入

    outputs:

      output:
        type: NodeOrPredictedFeatures
        shape: "[B, N, C] 或 [B, N, out_dim]"
        description: 如果 last_layer=False，返回同维度特征；如果为 True，返回投影后的预测值

  types:

    NodeFeatures:
      shape: [B, N, C]
      description: 包含空间坐标隐式信息的节点特征序列

  implementation:

    framework: pytorch

    code: |
      import torch
      import torch.nn as nn
      from onescience.modules.mlp.onemlp import OneMlp
      from onescience.modules.attention.oneattention import OneAttention
      _GEOTYPE_TO_ATTN_STYLE = {
          "unstructured": "Physics_Attention_Irregular_Mesh",
          "unstructured_plus": "Physics_Attention_Irregular_Mesh_plus",
          "structured_1D": "Physics_Attention_Structured_Mesh_1D",
          "structured_2D": "Physics_Attention_Structured_Mesh_2D",
          "structured_3D": "Physics_Attention_Structured_Mesh_3D",
      }
      class Transolver_block(nn.Module):
          """
          Transolver 编码器块 (Transolver Encoder Block)。

          这是 Transolver 模型的核心构建单元，采用标准的 Transformer Encoder 架构，但集成了物理感知的注意力机制（Physics Attention）。
          每个块包含两个主要子层：
          1.  **物理注意力层**: 根据几何类型（结构化或非结构化网格），通过 OneAttention 动态调用相应的注意力机制。
          2.  **前馈网络 (MLP)**: 通过 OneMlp 动态调用多层感知机进行特征变换和非线性映射。

          两个子层都采用了 Pre-LayerNorm 结构（即先归一化再进入层）和残差连接。

          Args:
              num_heads (int): 注意力头的数量。
              hidden_dim (int): 隐藏层的特征维度（输入和输出维度）。
              dropout (float): Dropout 概率，用于防止过拟合。
              act (str, optional): 激活函数类型，例如 'gelu', 'relu'。默认值: 'gelu'。
              mlp_ratio (float, optional): MLP 隐藏层维度相对于 hidden_dim 的倍率。默认值: 4。
              last_layer (bool, optional): 是否为最后一个 Block。如果是，会额外包含一个 LayerNorm 和线性投影层用于输出。默认值: False。
              out_dim (int, optional): 如果是 last_layer，则指定最终的输出维度。默认值: 1。
              slice_num (int, optional): 用于物理注意力机制的分片数量（针对特定几何类型）。默认值: 32。
              geotype (str, optional): 几何类型，决定使用的 Attention 变体。
                  支持: 'unstructured', 'unstructured_plus', 'structured_1D', 'structured_2D', 'structured_3D'。默认值: 'unstructured'。
              shapelist (list, optional): 网格形状列表，用于结构化网格的注意力计算。默认值: None。

          形状:
              输入 fx: (B, N, C)，其中 B 是批次大小，N 是节点/网格点数量，C 是 hidden_dim。
              输出:
                  - 如果 last_layer=False: (B, N, C)，形状与输入相同。
                  - 如果 last_layer=True: (B, N, out_dim)，经过投影后的输出。
          """

          def __init__(
              self,
              num_heads: int,
              hidden_dim: int,
              dropout: float,
              act="gelu",
              mlp_ratio=4,
              last_layer=False,
              out_dim=1,
              slice_num=32,
              geotype="unstructured",
              shapelist=None,
          ):
              super().__init__()
              self.last_layer = last_layer
              self.ln_1 = nn.LayerNorm(hidden_dim)

              # 检查 geotype 是否合法
              if geotype not in _GEOTYPE_TO_ATTN_STYLE:
                  raise ValueError(f"Unknown geotype: '{geotype}'. Available options are: {list(_GEOTYPE_TO_ATTN_STYLE.keys())}")
              
              attn_style = _GEOTYPE_TO_ATTN_STYLE[geotype]

              # 使用 OneAttention 重构注意力层调用
              self.Attn = OneAttention(
                  style=attn_style,
                  dim=hidden_dim, 
                  heads=num_heads,
                  dim_head=hidden_dim // num_heads,
                  dropout=dropout,
                  slice_num=slice_num,
                  shapelist=shapelist,
              )
              
              self.ln_2 = nn.LayerNorm(hidden_dim)

              # 2. OneMlp 重构 MLP 层调用
              self.mlp = OneMlp(
                  style="StandardMLP",        # 指定使用 StandardMLP
                  input_dim=hidden_dim,
                  hidden_dims=[int(hidden_dim * mlp_ratio)], 
                  output_dim=hidden_dim,
                  activation=act,
                  dropout_rate=dropout,  
                  use_bias=True
              )

              if self.last_layer:
                  self.ln_3 = nn.LayerNorm(hidden_dim)
                  self.mlp2 = nn.Linear(hidden_dim, out_dim)

          def forward(self, fx):
              fx = self.Attn(self.ln_1(fx)) + fx
              
              fx = self.mlp(self.ln_2(fx)) + fx

              if self.last_layer:
                  return self.mlp2(self.ln_3(fx))
              else:
                  return fx

  skills:

    build_transolver_block:

      description: 根据具体的物理网格形态构建相应的编码器块

      inputs:
        - num_heads
        - hidden_dim
        - geotype
        - last_layer

      prompt_template: |
        构建一个针对 {{geotype}} 网格的 Transolver Block。
        参数：
        hidden_dim = {{hidden_dim}}
        num_heads = {{num_heads}}
        如果是网络的最后一层，请将 last_layer 设为 True，并设定对应的 out_dim。

    diagnose_transolver:

      description: 分析物理网格建模时的常见错误

      checks:
        - wrong_geotype_for_data (数据是 2D 网格却使用了 unstructured 注意力)
        - slice_num_mismatch_with_nodes (分片数量与总节点数不匹配导致聚合失真或显存异常)

  knowledge:

    usage_patterns:

      ai4science_pde_solver:
        pipeline:
          - GeometricCoordinateEncoding
          - TransolverBlock (x N, 根据数据源选择特定的 geotype)
          - TransolverBlock (last_layer=True, out_dim=目标物理量维度)

    design_patterns:

      dynamic_physics_attention:
        structure:
          - 提取了几何先验与注意力机制的强耦合
          - 通过字典映射 `_GEOTYPE_TO_ATTN_STYLE` 实现策略模式 (Strategy Pattern)，解耦了注意力算法的具体实现

    hot_models:

      - model: Transolver
        year: 2024
        role: 高效通用物理场求解器
        architecture: Transformer-based
        attention_type: Physics-Attention

    model_usage_details:

      Transolver_Meteorological_Predictor:

        num_heads: 8
        hidden_dim: 256
        slice_num: 64
        geotype: unstructured

      Transolver_Aerodynamics_Simulation:

        num_heads: 16
        hidden_dim: 512
        slice_num: 128
        geotype: unstructured_plus

    best_practices:
      - 在处理极其不规则的散点物理数据时，必须确保 `slice_num` 设置合理，以在感受野（捕捉全局物理信息）和计算复杂度之间取得平衡。
      - 始终确保 `hidden_dim` 能够被 `num_heads` 整除，这是底层 `OneAttention` 内部计算 `dim_head` 所必需的。

    anti_patterns:
      - 传入了不支持的 `geotype` 字符串（如未在 `_GEOTYPE_TO_ATTN_STYLE` 中注册的类型），会导致初始化直接报错。
      - 在网络的中间隐层错误地设置了 `last_layer=True`，这将导致特征维度被提前截断到 `out_dim`，丢失关键物理特征。

    paper_references:

      - title: "Transolver: A Fast Transformer Solver for PDEs on General Geometries"
        authors: Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, Mingsheng Long
        year: 2024

  graph:

    is_a:
      - EncoderBlock
      - NeuralNetworkComponent

    part_of:
      - TransolverModel
      - AI4SciencePipeline

    depends_on:
      - OneAttention
      - OneMlp
      - LayerNorm

    compatible_with:
      inputs:
        - NodeFeatures
      outputs:
        - NodeFeatures
        - PredictedPhysicalFields