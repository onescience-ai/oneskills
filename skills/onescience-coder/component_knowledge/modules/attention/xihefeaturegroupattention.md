component:

  meta:
    name: FeatureGroupingAttention
    alias: Global SIE Grouping Attention / Cross-Attention Pooling
    version: 1.0
    domain: meteorological_ai
    category: neural_network
    subcategory: attention_mechanism
    author: OneScience
    license: Apache-2.0
    tags:
      - cross_attention
      - perceiver_style_pooling
      - feature_grouping
      - global_context
      - xihe_model

  concept:

    description: >
      FeatureGroupingAttention 是针对高分辨率物理/气象模型设计的特征降维与聚合模块。
      作为全局空间信息提取（Global SIE）的第一步，它并非在网格节点之间计算自注意力，
      而是利用一组固定数量的“可学习组向量”（Learnable Group Vectors）作为 Query，
      与庞大的局部物理特征网格（作为 Key 和 Value）进行多头交叉注意力交互，
      从而将海量局部特征高效地压缩并聚合成高度浓缩的全局上下文表示。

    intuition: >
      想象你要从上百万个海洋或大气网格点中提取全球气候的宏观趋势。如果让所有点互相交流，网络会瞬间崩溃。
      这个模块的做法是“委派代表”：它在隐空间中初始化了少量（如 32 个）极具代表性的组向量。
      这些组向量带着“疑问”（Query），去全局的网格地图（Key/Value）上巡视和检索，把最关键的信息吸收到自己身上。
      这就把处理数十万节点的高难度任务，降维成了只处理这几十个代表的任务。

    problem_it_solves:
      - 解决在全球高分辨率 3D 物理网格（如 1/12° 海洋模拟）上直接计算全局自注意力所引发的 O(N^2) 显存与计算灾难。
      - 提供了一种端到端可导的软聚合（Soft Pooling）机制，比传统的卷积下采样或 Average Pooling 更具数据适应性和表示能力。
      - 妥善处理海陆掩码（Mask），确保在信息聚合时忽略陆地等无效区域。

  theory:

    formula:
      
      cross_attention_pooling:
        expression: G' = MultiHeadAttention(Query=G, Key=X, Value=X, Mask)

    variables:

      G:
        name: LearnableGroupVectors
        shape: [B, num_groups, dim]
        description: 在网络初始化时定义的一组可学习参数，在批次处理时通过 expand 扩展到 batch_size，充当信息收集器。

      X:
        name: LocalGridFeatures
        shape: [B, Pl * Lat * Lon, dim]
        description: 数量庞大、包含局部高频细节的三维网格特征展平序列。

  structure:

    architecture: perceiver_cross_attention_bottleneck

    pipeline:

      - name: InputNormalization
        operation: layer_norm (对输入的庞大特征序列 X 进行归一化)

      - name: GroupVectorExpansion
        operation: expand (将共享的 group_vectors 扩展以匹配当前 batch_size)

      - name: MaskPreprocessing
        operation: dimension_flattening (将输入的 2D/3D 海陆掩码展平为 [B, N] 的一维 Padding 掩码)

      - name: CrossAttentionInteraction
        operation: multihead_attention (组向量作为 Q，特征序列作为 K 和 V，在 Mask 的保护下聚合信息)

      - name: OutputProjection
        operation: linear_and_dropout (对更新后的紧凑组特征进行线性投影)

  interface:

    parameters:

      dim:
        type: int
        description: 输入网格特征和组向量的通道维度 C

      num_groups:
        type: int
        default: 32
        description: 紧凑的全局组向量数量 G。该数值越大，保留的全局细节越多，但后续计算量也越大。

      num_heads:
        type: int
        default: 12
        description: 交叉注意力的多头数量

      qkv_bias:
        type: bool
        default: true
        description: 是否在交叉注意力的 Q, K, V 投影上使用偏置

      attn_drop:
        type: float
        default: 0.0
        description: 注意力概率矩阵的 Dropout 比例

      proj_drop:
        type: float
        default: 0.0
        description: 输出组特征的 Dropout 比例

    inputs:

      obj:
        type: dict_or_object
        description: 统一的输入载体
        properties:
          x:
            shape: [B, N, C]
            description: 本地特征提取网络传来的全量局部网格特征
          mask:
            shape: "[B, N] 或 [B, H, W] 或 [B, 1, H, W]"
            description: 物理边界掩码，用于防止模型从无效网格（如陆地）中提取信息

    outputs:

      G_prime:
        type: Tensor
        shape: [B, num_groups, dim]
        description: 聚合了全局视野的高级组特征表示

  types:

    CompactGroupFeatures:
      shape: [B, num_groups, dim]
      description: 突破了空间分辨率限制的、高度压缩的全局隐空间特征

  implementation:

    framework: pytorch

    code: |
      import torch
      import torch.nn as nn
      from onescience.modules.func_utils import Mlp

      #GLOBAL 1
      class FeatureGroupingAttention(nn.Module):
          """
          全局 SIE 的第一步：特征分组模块，将局部特征聚合为少量的 group 表示。

          Args:
              dim (int): 输入通道数 C。
              num_groups (int): group vectors 数量 G，默认为 32。
              num_heads (int): 多头注意力的头数，默认为 12。
              qkv_bias (bool): 是否在 QKV 上添加可学习的偏置，默认为 True。
              attn_drop (float): 注意力权重的 dropout 比例，默认为 0.0。
              proj_drop (float): 输出投影的 dropout 比例，默认为 0.0。
              LN (nn.Module): 归一化层类型，默认为 nn.LayerNorm。
              drop_layer (nn.Module): dropout 层类型，默认为 nn.Dropout。

          形状:
              输入 obj: 包含以下属性的对象
                  - x (torch.Tensor): 输入张量，形状为 (B, N, C)，其中 N = Pl × Lat × Lon
                  - mask (torch.Tensor): 掩码张量，形状为 (B, N) 或 (B, 1, H, W) 或 (B, H, W)
              输出 G_prime (torch.Tensor): 聚合后的 group 特征，形状为 (B, G, C)

          Example:
              >>> grouping = FeatureGroupingAttention(
              ...     dim=192,
              ...     num_groups=32,
              ...     num_heads=12
              ... )
              >>> from types import SimpleNamespace
              >>> B, N, C = 2, 425984, 192  # N = 13*128*256
              >>> x = torch.randn(B, N, C)
              >>> mask = torch.ones(B, N, dtype=torch.bool)
              >>> obj = SimpleNamespace(x=x, mask=mask)
              >>> out = grouping(obj)
              >>> out.shape
              torch.Size([2, 32, 192])
          """

          def __init__(
              self,
              dim, 
              num_groups=32, 
              num_heads=12, 
              qkv_bias=True,
              attn_drop=0.0, 
              proj_drop=0.0,
              LN=nn.LayerNorm,
              drop_layer=nn.Dropout,
              ):
              super().__init__()
              self.dim = dim
              self.num_groups = num_groups  
              self.num_heads = num_heads  #自定义，分组多就可表示的更精细

              # 初始化 learnable group vectors (相当于 G_l)
              self.group_vectors = nn.Parameter(torch.randn(1, num_groups, dim))
              self.norm = LN(dim)
              # 多头注意力 (标准 vanilla Transformer Attention)
              self.attn = nn.MultiheadAttention(
                  embed_dim=dim, num_heads=num_heads, bias=qkv_bias, batch_first=True
              )
              self.attn_drop = drop_layer(attn_drop)
              self.proj = nn.Linear(dim, dim)
              self.proj_drop = drop_layer(proj_drop)
          # def forward(self, x,mask_tokens=None):
          def forward(self,obj,mask=None):
              """
              x: (B, N, C)  -> 来自 Local SIE 的特征
              """
              # x=obj.x
              # mask_tokens=obj.mask

              if isinstance(obj, dict):
                  # 字典方式访问
                  x=obj["x"]
                  mask_tokens=obj["mask"].clone().detach().float()
          
              # 判断是否为对象（非字典的其他类型）
              else:
                  # 对象方式访问        
                  x=obj.x
                  mask_tokens=obj.mask
                  obj={
                      "x":x,
                      "mask":mask_tokens,
                  }
                  
              B, N, C = x.shape
              x = self.norm(x)  # (B, N, C)
              
              #  expand group vectors (batch 内共享同一份 group 参数)
              G = self.group_vectors.expand(B, -1, -1)  # (B, G, C)
              # Multi-Head Cross-Attention
              if mask_tokens is None:
                  return None
              if mask_tokens.dim() == 4:              # (B,1,H,W)
                  mask_tokens = mask_tokens.squeeze(1)
              if mask_tokens.dim() == 3:              # (B,H,W)
                  mask_tokens = mask_tokens.reshape(B, -1)
              assert mask_tokens.shape == (B, N)
              key_padding_mask = None if mask_tokens is None else (mask_tokens == 0)
              G_prime, _ = self.attn(query=G, key=x, value=x,key_padding_mask=key_padding_mask)
              #  输出更新后的 group vectors
              G_prime = self.proj_drop(self.proj(G_prime))  # (B, G, C)

              return G_prime

  skills:

    build_feature_grouping_attention:

      description: 构建基于跨注意力机制的全局上下文信息降维与聚合模块

      inputs:
        - dim
        - num_groups
        - num_heads

      prompt_template: |
        创建一个 FeatureGroupingAttention 层。
        参数：
        输入与内部维度 = {{dim}}。
        设置聚合代表的组向量数量为 {{num_groups}}，多头头数为 {{num_heads}}。
        注意处理好对象或字典类型的复合输入。

    diagnose_mask_flattening_errors:

      description: 诊断物理边界掩码（Mask）在展平操作时维度与特征序列不匹配引发的崩溃

      checks:
        - mask_dimension_reduction (确保无论传入的 Mask 是 `[B, 1, H, W]` 还是 `[B, H, W]`，都能被正确 squeeze 和 reshape 为 `[B, N]`)
        - pytorch_mha_key_padding_mask_logic (原生 `nn.MultiheadAttention` 的 `key_padding_mask` 中，True 表示忽略/屏蔽，代码中通过 `mask_tokens == 0` 转换逻辑以确保陆地等无效区域被忽略)

  knowledge:

    usage_patterns:

      xihe_global_spatial_information_extraction:
        pipeline:
          - XihelocalTransformer (Local Spatial Extraction)
          - FeatureGroupingAttention (将局部 N 映射到全局 G)
          - GlobalTransformerBlock (在 G 个紧凑特征内部进行自注意力交互)
          - FeatureUngroupingAttention (将更新后的全局特征交织反投影回 N)

    design_patterns:

      learnable_queries_as_bottleneck:
        structure:
          - 这是一个经典的 "Attention Bottleneck" 设计。不再受限于输入数据的原始物理几何拓扑，利用独立初始化的张量 `group_vectors` 强行打破全连接图，不仅解决了显存问题，更强制模型将地球表面的稀疏信息蒸馏成最核心的动力学变量。

    hot_models:

      - model: XiHe
        year: 2024
        role: 高分辨率数据驱动的全球海洋涡旋分辨预报模型
        architecture: Local-Global Dual-Path Transformer
        attention_type: Feature Grouping Cross-Attention

      - model: Perceiver / Perceiver IO
        year: 2021
        role: 能够处理极其不规则、超高维度感官输入（视频、音频、点云）的通用架构
        architecture: Asymmetric Attention
        attention_type: Cross-Attention to Latent Array

    model_usage_details:

      XiHe_Global_SIE_Module:
        num_groups: 32
        dim: 192
        num_heads: 12

    best_practices:
      - 组特征数量 `num_groups` 一般远远小于节点数 $N$。对于含有 40 多万节点的三维物理场，32 到 128 通常是一个非常高效且能保持表现力的“信息瓶颈”大小。
      - `nn.MultiheadAttention` 设置了 `batch_first=True`，因此调用时无需像旧版 PyTorch 那样繁琐地在第一维放置序列长度。

    anti_patterns:
      - 错误地将目标（Target）局部特征作为 Query 输入，将少量的 `group_vectors` 作为 Key 和 Value。这会导致前向计算无法达到降维（从 $N$ 变为 $G$）的目的，而是毫无意义的冗余计算。

    paper_references:

      - title: "XiHe: A Data-Driven Model for Global Ocean Eddy-Resolving Forecasting"
        authors: Xiang Wang, Renzhi Wang, Junqiang Song, et al.
        year: 2024

      - title: "Perceiver: General Perception with Iterative Attention"
        authors: Andrew Jaegle, Felix Gimeno, Andrew Brock, Oriol Vinyals, Andrew Zisserman, Joao Carreira
        year: 2021
        journal: ICML

  graph:

    is_a:
      - AttentionMechanism
      - CrossAttention
      - SpatialInformationExtractor
      - PoolingLayer

    part_of:
      - XiHeModel
      - GlobalSIE

    depends_on:
      - nn.MultiheadAttention
      - LayerNorm
      - LearnableGroupVectors

    compatible_with:
      inputs:
        - LocalGridFeatures
      outputs:
        - CompactGroupFeatures