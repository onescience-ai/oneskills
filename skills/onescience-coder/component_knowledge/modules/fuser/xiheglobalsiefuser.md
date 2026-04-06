component:

  meta:
    name: XiheGlobalSIEFuser
    alias: Xihe Global SIE Fuser
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: global_attention
    author: OneScience
    license: Apache-2.0
    tags:
      - ocean_modeling
      - global_attention
      - grouped_attention
      - sea_ice
      - climate_model

  concept:

    description: >
      Xihe模型的全局特征融合模块，基于分组注意力机制实现全局特征交互。
      将特征分成多个组，在每个组内进行全局注意力计算，然后通过MLP进行组间信息传播，
      实现高效的全局特征融合。

    intuition: >
      就像全球气候会议：不同地区（分组）先内部讨论（组内注意力），然后代表发言交流（组间传播），
      最后形成全球共识。分组注意力既保证了全局视野，又控制了计算复杂度。

    problem_it_solves:
      - 全局海洋特征的高效交互
      - 大规模注意力的计算优化
      - 组间信息传播机制
      - 海洋系统的全局耦合建模

  theory:

    formula:

      grouped_attention:
        expression: |
          x_{grouped} = \text{Reshape}(x, [B, G, L/G, C])
          x_{attended} = \text{GroupAttention}(x_{grouped})
          x_{flattened} = \text{Reshape}(x_{attended}, [B, L, C])

      group_propagation:
        expression: |
          x_{group} = \text{Pool}(x_{flattened}, dim=L/G)
          x_{propagated} = \text{MLP}(x_{group})
          x_{broadcast} = \text{Broadcast}(x_{propagated}, L/G)

    variables:

      G:
        name: NumGroups
        description: 特征分组数量

      x_{grouped}:
        name: GroupedFeatures
        shape: [batch, num_groups, seq_len_per_group, channels]
        description: 分组后的特征

      x_{attended}:
        name: AttendedFeatures
        description: 组内注意力后的特征

  structure:

    architecture: grouped_global_attention

    pipeline:

      - name: InputNormalization
        operation: layer_normalization

      - name: GroupedAttention
        operation: multi_head_grouped_attention

      - name: GroupPropagation
        operation: mlp_on_group_vectors

      - name: ResidualConnection
        operation: addition_with_input

  interface:

    parameters:

      dim:
        type: int
        description: 输入通道数

      num_heads:
        type: int
        description: 注意力头数量

      num_groups:
        type: int
        description: 特征分组数量

      qkv_bias:
        type: bool
        description: 是否在QKV上添加偏置

      norm_layer:
        type: nn.Module
        description: 归一化层类型

    inputs:

      obj:
        type: OceanData
        description: 包含x和mask的对象

      x:
        type: Features
        shape: [batch, L, C]
        description: 输入特征

      mask:
        type: AttentionMask
        shape: [batch, L]
        description: 注意力掩码

      y:
        type: SkipFeatures
        shape: [batch, L, C]
        description: 用于残差连接的辅助特征

    outputs:

      x:
        type: GlobalFeatures
        shape: [batch, L, C]
        description: 全局融合后的特征

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules.attention.oneattention import OneAttention
      from onescience.modules.mlp.onemlp import OneMlp

      class XiheGlobalSIEFuser(nn.Module):
          def __init__(self, dim=192, num_heads=12, num_groups=32, **kwargs):
              super().__init__()
              self.dim = dim
              self.num_groups = num_groups
              
              # 分组注意力
              self.attention = OneAttention(
                  style="GroupedAttention", dim=dim, num_heads=num_heads,
                  num_groups=num_groups
              )
              
              # 组间传播MLP
              self.group_mlp = OneMlp(
                  style="XiheMLP", input_dim=dim, output_dim=dim,
                  num_groups=num_groups
              )
              
              self.norm = nn.LayerNorm(dim)

          def forward(self, obj):
              x, mask = obj.x, getattr(obj, 'mask', None)
              residual = x
              y = getattr(obj, 'y', residual)
              
              # 归一化
              x = self.norm(x)
              
              # 分组注意力
              x = self.attention(x, mask=mask)
              
              # 组间传播
              B, L, C = x.shape
              G = self.num_groups
              x_grouped = x.view(B, G, L // G, C).mean(dim=2)  # [B, G, C]
              x_propagated = self.group_mlp(x_grouped)  # [B, G, C]
              x_broadcast = x_propagated.unsqueeze(2).repeat(1, 1, L // G, 1)  # [B, G, L/G, C]
              x = x_broadcast.view(B, L, C)
              
              # 残差连接
              x = x + y
              return x

  skills:

    build_global_sie_fuser:

      description: 构建全局SIE融合器

  knowledge:

    hot_models:

      - model: Xihe
        year: 2023
        role: 海洋模型全局注意力模块
        architecture: grouped attention

    best_practices:

      - 组数应根据特征维度合理设置
      - 组头数要能被组数整除
      - 组间传播对全局信息很重要

    paper_references:

      - title: "Efficient Attention: Attention with Linear Complexities"
        authors: Shen et al.
        year: 2021

  graph:

    is_a:
      - GlobalAttentionModule
      - OceanModelingComponent
      - EfficientAttention

    used_in_models:
      - Xihe
      - Ocean climate models

    compatible_with:

      inputs:
        - OceanFeatures
        - ClimateData
        - AttentionMasks

      outputs:
        - GlobalOceanFeatures
        - ClimateRepresentations
