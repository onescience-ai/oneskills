component:

  meta:
    name: LinearAttentionMechanisms
    alias: Linear Attention (Vanilla & Generalized)
    version: 1.0
    domain: efficient_deep_learning_and_ai4science
    category: neural_network
    subcategory: attention_mechanism
    author: OneScience
    license: Apache-2.0
    tags:
      - linear_attention
      - o(n)_complexity
      - galerkin_attention
      - gnot
      - efficient_transformer

  concept:

    description: >
      线性注意力（Linear Attention）是一类将传统自注意力机制的空间和时间复杂度从序列长度的平方 O(N^2) 降低到线性 O(N) 的高效模块。
      该文件包含两个变体：朴素线性注意力（Vanilla Linear Attention）和支持多种归一化策略（如 l1, l2, galerkin）的通用线性注意力（Generalized Linear Attention）。
      通用版本不仅支持交叉注意力，还针对物理算子学习引入了残差链接设计。

    intuition: >
      在标准注意力中，模型必须先计算所有 Token 相互之间的关系（形成 N x N 的巨大注意力分数矩阵），然后再去提取信息。
      线性注意力利用了数学中矩阵乘法的结合律：先让 Key 和 Value 结合，提取出一个“全局上下文浓缩矩阵”（维度仅为 D x D），
      然后再让 Query 去向这个“浓缩矩阵”要信息。因为 D（特征维度）通常远小于 N（序列长度），这就像是从“所有人都互相开会”变成了“所有人把意见汇总给一个记录员，再由记录员分发”，从而极大地减少了计算量。

    problem_it_solves:
      - 彻底解决极长序列（如高分辨率网格、万字长文）在传统注意力计算下引发的 O(N^2) 显存溢出和计算灾难。
      - 为算子学习 (Operator Learning, 如 GNOT 模型) 提供了一种支持不同网格尺度泛化且计算高效的注意力架构。

  theory:

    formula:
      
      standard_attention:
        expression: Output = (Q * K^T) * V  # 复杂度 O(N^2)

      linear_attention:
        expression: Output = Q * (K^T * V) * D^{-1} # 复杂度 O(N * D^2)

    variables:

      D_inv:
        name: NormalizationFactor
        description: 归一化项。在不同的 attn_type 下有不同的计算方式（如基于序列长度 N、或者基于 softmax 概率和的求积归一化）。

      Context:
        name: GlobalContextMatrix
        shape: [B, H, D, D]
        description: 由 K^T 乘以 V 得到的紧凑全局上下文特征矩阵。

  structure:

    architecture: linear_complexity_attention

    pipeline:

      - name: QKVProjection
        operation: linear_projection

      - name: FeatureMasking
        operation: masked_fill (可选，屏蔽无效的 Padding Token)

      - name: NormalizationFactorCalculation
        operation: specific_norm_math (根据 l1, l2 或 galerkin 策略计算 D_inv)

      - name: ContextAggregation
        operation: matmul (计算 K^T @ V 得到全局上下文矩阵)

      - name: QueryApplication
        operation: matmul_and_scale (计算 Q @ Context 并应用 D_inv 缩放)

      - name: ResidualAndOutput
        operation: residual_add_and_linear (通用版包含跨层残差 + q，最后通过线性投影)

  interface:

    parameters:

      dim:
        type: int
        description: 输入特征图的总维度大小

      heads:
        type: int
        default: 8
        description: 多头注意力的头数

      dim_head:
        type: int
        default: 64
        description: 每个注意力头的维度特征数

      dropout:
        type: float
        default: 0.0
        description: 注意力输出的随机丢弃率

      scale:
        type: float
        default: null
        description: (仅 Vanilla) 缩放因子。若不填，默认采用 1 / 序列长度 (N)

      attn_type:
        type: str
        default: "l1"
        description: (仅 Generalized) 归一化类型，支持 'l1', 'l2', 'galerkin'

    inputs:

      x:
        type: Tensor
        shape: [B, T1, dim]
        dtype: float32
        description: Query 特征序列

      y:
        type: Tensor
        shape: [B, T2, dim]
        default: null
        description: Key/Value 特征序列，如果不提供，则默认执行自注意力 (y=x)

      mask:
        type: Tensor
        shape: "[B, T2] 或 [B, 1, T2, 1]"
        default: null
        description: 针对 Key/Value 的布尔掩码，用于屏蔽 Padding

    outputs:

      output:
        type: Tensor
        shape: [B, T1, dim]
        description: 提取特征后的序列张量

  types:

    LinearAttentionFeatures:
      shape: [B, N, dim]
      description: 经过 O(N) 注意力计算后的特征表达

  implementation:

    framework: pytorch, einops

    code: |
      import torch
      import torch.nn as nn
      import torch.nn.functional as F
      from einops import rearrange

      # ==========================================
      # 朴素线性注意力 (Simple / Vanilla)
      # ==========================================

      class Vanilla_Linear_Attention(nn.Module):
          """
          朴素线性注意力机制 (Simple Linear Attention)。

          该模块实现了一种计算复杂度为 O(N) 的高效注意力机制。
          通过利用矩阵乘法的结合律，即计算 Q(K^T V) 而非 (QK^T)V，它避免了显式计算 O(N^2) 的注意力矩阵。
          此版本是线性注意力的基础实现，特点是计算直接，通过除以序列长度或掩码有效长度进行归一化。
          它保留了特殊的 "先 Reshape 后 Linear" 的权重结构，以兼容特定的预训练权重。

          Args:
              dim (int): 输入特征维度。
              heads (int, optional): 注意力头数。默认值: 8。
              dim_head (int, optional): 每个注意力头的维度。默认值: 64。
              dropout (float, optional): Dropout 概率。默认值: 0.0。
              scale (float, optional): 自定义缩放因子。如果为 None，默认行为是除以序列长度 (1/N)。默认值: None。

          形状:
              输入 x: (B, N, C)，其中 C 必须等于 heads * dim_head。
              输入 mask (可选): (B, N) 或 (B, 1, 1, N) 的布尔掩码。True 表示有效 token，False 表示 padding。
              输出: (B, N, C)。

          Example:
              >>> attn = Vanilla_Linear_Attention(dim=128, heads=8, dim_head=16)
              >>> x = torch.randn(8, 100, 128)
              >>> # 创建掩码，假设后50个token是padding
              >>> mask = torch.ones(8, 100).bool()
              >>> mask[:, 50:] = False
              >>> out = attn(x, mask=mask)
              >>> out.shape
              torch.Size([8, 100, 128])
          """
          def __init__(self, dim, heads=8, dim_head=64, dropout=0., scale=None, **kwargs):
              super().__init__()
              inner_dim = dim_head * heads
              self.dim_head = dim_head
              self.heads = heads
              self.scale = scale
              self.dropout = nn.Dropout(dropout)
              
              self.to_q = nn.Linear(dim_head, dim_head, bias=False)
              self.to_k = nn.Linear(dim_head, dim_head, bias=False)
              self.to_v = nn.Linear(dim_head, dim_head, bias=False)
              
              self.to_out = nn.Sequential(
                  nn.Linear(inner_dim, dim),
                  nn.Dropout(dropout)
              )

          def forward(self, x, mask=None):
              B, N, C = x.shape
              # [B, N, C] -> [B, H, N, D]
              x = x.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() 
              
              q = self.to_q(x)
              k = self.to_k(x)
              v = self.to_v(x)

              if mask is not None:
                  if mask.ndim == 2:
                      mask = mask.view(B, 1, N, 1)
                  k = k.masked_fill(~mask, 0.)
                  v = v.masked_fill(~mask, 0.)
                  denorm = mask.sum(dim=-2, keepdim=True).float()
              else:
                  denorm = float(N)

              context = torch.matmul(k.transpose(-1, -2), v)
              
              if self.scale is not None:
                  context = context * self.scale
              else:
                  context = context / (denorm + 1e-8)

              context = self.dropout(context)
              res = torch.matmul(q, context) 
              res = rearrange(res, 'b h n d -> b n (h d)')
              return self.to_out(res)


      # ==========================================
      # 通用线性注意力 
      # ==========================================

      class LinearAttention(nn.Module):
          """
          通用线性注意力 (Generalized Linear Attention)。

          相比朴素版，它引入了更复杂的归一化机制（D^{-1}），通过特定的归一化项来处理 Query 和 Key 的加权和。
          它支持 'l1' (Softmax), 'l2', 'galerkin' 等多种归一化策略，在保持线性复杂度的同时提供了更好的数值稳定性和表达能力。
          此外，该模块支持交叉注意力（Cross Attention）和残差连接（GNOT 特有设计）。

          Args:
              dim (int): 输入特征维度。
              heads (int, optional): 注意力头数。默认值: 8。
              dim_head (int, optional): 每个注意力头的维度。默认值: 64。
              dropout (float, optional): Dropout 概率。默认值: 0.0。
              attn_type (str, optional): 归一化类型，支持 'l1', 'l2', 'galerkin'。默认值: 'l1'。

          形状:
              输入 x: (B, N, C)，Query 特征。
              输入 y (可选): (B, M, C)，Key/Value 特征。如果不提供，默认为自注意力 (y=x)。
              输入 mask (可选): (B, M) 或 (B, 1, M, 1) 的布尔掩码，用于屏蔽无效的 Key/Value。
              输出: (B, N, C)。

          Example:
              >>> l_attn = LinearAttention(dim=64, heads=8, dim_head=8, attn_type='l2')
              >>> x = torch.randn(4, 512, 64)
              >>> out = l_attn(x)
              >>> out.shape
              torch.Size([4, 512, 64])
          """
          def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_type='l1', **kwargs):
              super().__init__()
              self.n_head = heads
              self.dim_head = dim_head
              self.attn_type = attn_type
              
              self.key = nn.Linear(dim, dim)
              self.query = nn.Linear(dim, dim)
              self.value = nn.Linear(dim, dim)
              
              self.attn_drop = nn.Dropout(dropout)
              self.proj = nn.Linear(dim, dim)

          def forward(self, x, y=None, mask=None):
              """
              Args:
                  x: Query [B, N, C]
                  y: Key/Value [B, M, C] (Optional, default=x)
                  mask: [B, M] (针对 Key/Value 的 mask)
              """
              y = x if y is None else y
              B, T1, C = x.size()
              _, T2, _ = y.size()
              
              # 投影与分头
              q = self.query(x).view(B, T1, self.n_head, self.dim_head).transpose(1, 2) # [B, H, T1, D]
              k = self.key(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)   # [B, H, T2, D]
              v = self.value(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2) # [B, H, T2, D]
              if mask is not None:
                  if mask.ndim == 2:
                      mask = mask.view(B, 1, T2, 1) # [B, 1, M, 1]
                  
                  if self.attn_type in ['l1', 'galerkin']:
                      k = k.masked_fill(~mask, -1e9) 
                  elif self.attn_type == 'l2':
                      k = k.masked_fill(~mask, 0.)
                  
                  v = v.masked_fill(~mask, 0.)

              # 归一化处理
              if self.attn_type == 'l1':
                  q = q.softmax(dim=-1)
                  k = k.softmax(dim=-1)
                  
                  if mask is not None:
                      k = k.masked_fill(~mask, 0.)

                  k_cumsum = k.sum(dim=-2, keepdim=True) # [B, H, 1, D]
                  D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True) 
                  
              elif self.attn_type == "galerkin":
                  q = q.softmax(dim=-1)
                  k = k.softmax(dim=-1)
                  if mask is not None:
                      k = k.masked_fill(~mask, 0.)
                      valid_len = mask.sum(dim=-2, keepdim=True)
                      D_inv = 1. / (valid_len + 1e-8)
                  else:
                      D_inv = 1. / float(T2)
                      
              elif self.attn_type == "l2":
                  q = q / (q.norm(dim=-1, keepdim=True, p=1) + 1e-8)
                  k = k / (k.norm(dim=-1, keepdim=True, p=1) + 1e-8)
                  k_cumsum = k.sum(dim=-2, keepdim=True)
                  D_inv = 1. / ((q * k_cumsum).abs().sum(dim=-1, keepdim=True) + 1e-8)
                  
              else:
                  raise NotImplementedError

              # 核心线性 Attention 计算: Q (K^T V)
              context = k.transpose(-2, -1) @ v  # [B, H, D, D]
              
              y = self.attn_drop((q @ context) * D_inv + q)

              y = rearrange(y, 'b h n d -> b n (h d)')
              y = self.proj(y)
              return y

  skills:

    build_linear_attention:

      description: 为超长序列或异构网格算子构建线性复杂度的 Attention 层

      inputs:
        - dim
        - heads
        - attn_type

      prompt_template: |
        创建一个 LinearAttention 层。
        要求：特征维度 = {{dim}}，头数 = {{heads}}。
        如果你是在为 PDE 求解器或物理模型构建，请将 attn_type 设置为 'galerkin'。

    diagnose_normalization_instability:

      description: 分析线性注意力由于除法操作导致的 NaN (数值不稳定) 问题

      checks:
        - zero_division_in_D_inv (检查分母 `denorm` 或 `D_inv` 计算时是否遗漏了 `+ 1e-8` 的平滑项)
        - inf_values_after_masking (检查在使用 `galerkin` 时，如果全为 mask 是否导致有效长度 valid_len 为 0)

  knowledge:

    usage_patterns:

      gnot_operator_learning:
        pipeline:
          - HeterogeneousGridEmbedding
          - LinearAttention (Cross-Attention, y=Geometry)
          - LinearAttention (Self-Attention, y=None, attn_type='galerkin')
          - FFN

    design_patterns:

      associative_property_optimization:
        structure:
          - 打破标准 attention 中 Query 和 Key 必须计算成相似度分布的设定。
          - 优先将 Key 转置后与 Value 矩阵相乘，将其压缩成维度为 [D, D] 的固定大小上下文矩阵 `context`，无论序列长度 `N` 多大，计算瓶颈都被转移和削弱了。

    hot_models:

      - model: GNOT (General Neural Operator Transformer)
        year: 2023
        role: 顶级的通用物理偏微分方程求解算子模型
        architecture: Linear Cross-Attention / Galerkin Attention

      - model: Linear Transformer / Performer
        year: 2020
        role: 首批尝试将 Transformer 复杂度降为线性的基础大模型

    model_usage_details:

      GNOT_PDE_Solver:
        attn_type: "galerkin"
        dim: 256
        heads: 8
        residual_connection: True (q + q @ context)

    best_practices:
      - 在物理模型或处理连续几何场的问题时，优先使用 `attn_type="galerkin"`，它通过全局网格有效长度的均匀归一化，能更好地保持物理场的守恒性。
      - 如果你发现训练过程中出现 Loss 突变为 NaN 的情况，通常是因为 `q` 和 `k` 的内积非常小，导致 `D_inv` 的分母接近 0。

    anti_patterns:
      - 试图在极短的序列（如 NLP 中的短文本 N < 512）上使用线性注意力。由于额外的归一化步骤和矩阵转置开销，线性注意力在短序列上的实际运行速度可能比标准注意力还要慢。
      - 在使用带有 `l1`（Softmax）归一化的版本时，忘记使用 `masked_fill(~mask, -1e9)` 处理 Key，会导致模型吸收到无效的 padding 噪声。

    paper_references:

      - title: "GNOT: A General Neural Operator Transformer for Operator Learning"
        authors: Zhongkai Hao, Zhengyi Wang, Hang Su, Chengyang Ying, Yinpeng Dong, Songming Liu, Ze Cheng, Jian Song, Jun Zhu
        year: 2023
        journal: ICML

      - title: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
        authors: Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret
        year: 2020
        journal: ICML

  graph:

    is_a:
      - AttentionMechanism
      - EfficientTransformerComponent
      - NeuralOperator

    part_of:
      - LinearTransformer
      - GNOTModel

    depends_on:
      - torch.matmul
      - D_inv Normalization

    compatible_with:
      inputs:
        - LinearAttentionFeatures
      outputs:
        - LinearAttentionFeatures