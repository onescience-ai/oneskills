component:

  meta:
    name: XiheMLP
    alias: Xihe Group Propagation MLP
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: group_mlp
    author: OneScience
    license: Apache-2.0
    tags:
      - ocean_modeling
      - group_propagation
      - mlp
      - sea_ice
      - climate_model

  concept:

    description: >
      全局SIE的第二步：组间传播模块，在group空间内进行信息交互与融合。
      将特征向量分组后，通过MLP在组维度上进行信息传播，实现不同组之间的特征交互，
      是Xihe模型中实现全局信息传播的关键组件。

    intuition: >
      就像联合国气候会议：不同国家/地区（组）的代表先内部讨论，然后通过发言（MLP传播）
      让其他组了解自己的情况，最终形成全球共识。组间传播确保了信息在不同组之间的有效流动。

    problem_it_solves:
      - 分组特征之间的信息传播
      - 全局信息的高效交互
      - 组间耦合建模
      - 海洋系统的全局连通性

  theory:

    formula:

      group_propagation:
        expression: |
          x_{norm} = \text{LayerNorm}(x_{group})
          x_{mixed} = \text{MLP}(x_{norm})
          x_{propagated} = x_{group} + x_{mixed}

      token_mixing:
        expression: |
          x_{output} = \text{TokenMixingMLP}(x_{input})
          \text{where mixing happens across group dimension}

    variables:

      G:
        name: NumGroups
        description: 特征分组数量

      x_{group}:
        name: GroupVectors
        shape: [batch, G, C]
        description: 分组特征向量

      x_{propagated}:
        name: PropagatedGroups
        shape: [batch, G, C]
        description: 传播后的组特征

  structure:

    architecture: group_propagation_mlp

    pipeline:

      - name: InputNormalization
        operation: layer_normalization

      - name: TokenMixingMLP
        operation: mlp_on_group_dimension

      - name: SecondNormalization
        operation: layer_normalization

      - name: ChannelMixingMLP
        operation: mlp_on_channel_dimension

      - name: ResidualConnection
        operation: addition_with_input

  interface:

    parameters:

      dim:
        type: int
        description: 输入通道数

      num_groups:
        type: int
        description: group vectors数量

      mlp_ratio:
        type: float
        description: MLP隐层扩展比例

      drop:
        type: float
        description: MLP的dropout比例

      act_layer:
        type: nn.Module
        description: 激活函数层类型

      LN:
        type: nn.Module
        description: 归一化层类型

    inputs:

      x:
        type: GroupVectors
        shape: [batch, G, C]
        description: 输入group vectors

    outputs:

      y:
        type: PropagatedVectors
        shape: [batch, G, C]
        description: 传播融合后的group vectors

  types:

    GroupVectors:
      shape: [batch, num_groups, channels]
      description: 分组特征向量

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      class XiheMLP(nn.Module):
          def __init__(self, dim=192, num_groups=32, mlp_ratio=4.0,
                       drop=0.0, act_layer=nn.GELU, LN=nn.LayerNorm):
              super().__init__()
              self.dim = dim
              self.num_groups = num_groups
              
              # 归一化层
              self.norm1 = LN(dim)
              self.norm2 = LN(dim)
              
              # Token-mixing MLP (在group维度上传播信息)
              hidden_dim = int(dim * mlp_ratio)
              self.token_mix = nn.Sequential(
                  nn.Linear(num_groups, hidden_dim),
                  act_layer(),
                  nn.Dropout(drop),
                  nn.Linear(hidden_dim, num_groups),
                  nn.Dropout(drop)
              )
              
              # Channel-mixing MLP
              self.channel_mix = nn.Sequential(
                  nn.Linear(dim, hidden_dim),
                  act_layer(),
                  nn.Dropout(drop),
                  nn.Linear(hidden_dim, dim),
                  nn.Dropout(drop)
              )

          def forward(self, x):
              # Token mixing: 在group维度上传播信息
              residual = x
              x = self.norm1(x)
              x = x.transpose(-1, -2)  # [B, C, G]
              x = self.token_mix(x)  # 在group维度上混合
              x = x.transpose(-1, -2)  # [B, G, C]
              x = x + residual
              
              # Channel mixing: 在channel维度上传播信息
              residual = x
              x = self.norm2(x)
              x = self.channel_mix(x)
              x = x + residual
              
              return x

  skills:

    build_group_mlp:

      description: 构建组传播MLP

  knowledge:

    hot_models:

      - model: Xihe
        year: 2023
        role: 海洋模型组传播模块
        architecture: group propagation

    best_practices:

      - 组数应根据特征维度合理设置
      - Token mixing和channel mixing都很重要
      - 残差连接有助于训练稳定性

    paper_references:

      - title: "gMLP: Learning MLPs with Group Convolutions"
        authors: Liu et al.
        year: 2021

  graph:

    is_a:
      - GroupPropagationModule
      - OceanModelingComponent
      - TokenMixingNetwork

    used_in_models:
      - Xihe
      - Group-based models

    compatible_with:

      inputs:
        - GroupVectors
        - ClusteredFeatures
        - PartitionedData

      outputs:
        - PropagatedVectors
        - MixedFeatures
        - GlobalRepresentations
