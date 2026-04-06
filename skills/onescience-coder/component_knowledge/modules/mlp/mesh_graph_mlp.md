component:

  meta:
    name: MeshGraphMLP
    alias: Mesh Graph Multi-Layer Perceptron
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: graph_mlp
    author: OneScience
    license: Apache-2.0
    tags:
      - graph_neural_network
      - mesh_processing
      - mlp
      - message_passing
      - graphcast

  concept:

    description: >
      专门用于图神经网络的多层感知机，支持MeshGraphEdgeMLP和MeshGraphMLP两种风格。
      MeshGraphEdgeMLP用于边特征更新，结合源节点、目标节点和边特征；MeshGraphMLP用于节点特征更新，
      支持拼接优化技巧以节省显存。

    intuition: >
      就像社交网络分析中的关系建模：边MLP分析两个人之间的关系强度，节点MLP更新每个人的状态。
      拼接技巧就像用求和替代显式拼接，既保持信息又节省内存。

    problem_it_solves:
      - 图神经网络中的边特征更新
      - 图神经网络中的节点特征更新
      - 大图处理的显存优化
      - 消息传递中的特征变换

  theory:

    formula:

      edge_mlp:
        expression: |
          e'_{ij} = \text{MLP}_{edge}(\text{Concat}(e_{ij}, h_i, h_j))
          \text{or} e'_{ij} = \text{MLP}_{edge}(e_{ij} + h_i + h_j) \quad \text{(sum trick)}

      node_mlp:
        expression: |
          h'_j = \text{MLP}_{node}(\text{Concat}(h_j, m_j))
          \text{where } m_j = \text{Aggregate}(e'_{ij})

      concat_trick:
        expression: |
          \text{Concat}(a, b) \approx \text{Linear}(a + b) \quad \text{(memory efficient)}

  structure:

    architecture: graph_aware_mlp

    pipeline:

      - name: FeaturePreparation
        operation: concatenation or summation

      - name: LinearTransformation
        operation: linear_projection

      - name: Activation
        operation: silu/relu activation

      - name: Normalization
        operation: layer_norm (optional)

      - name: Output
        operation: transformed_features

  interface:

    parameters:

      style:
        type: str
        description: MLP风格，可选"MeshGraphEdgeMLP"或"MeshGraphMLP"

      input_dim:
        type: int
        description: 输入特征维度

      output_dim:
        type: int
        description: 输出特征维度

      hidden_dim:
        type: int
        description: 隐藏层维度

      hidden_layers:
        type: int
        description: 隐藏层层数

      activation_fn:
        type: nn.Module
        description: 激活函数

      norm_type:
        type: str
        description: 归一化类型

    inputs:

      features:
        type: GraphFeatures
        description: 图特征（节点、边或聚合特征）

      node_features:
        type: NodeFeatures
        description: 节点特征（用于边MLP）

    outputs:

      output:
        type: TransformedFeatures
        description: 变换后的特征

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      class MeshGraphMLP(nn.Module):
          def __init__(self, style="MeshGraphMLP", input_dim=512, output_dim=512,
                       hidden_dim=512, hidden_layers=1, activation_fn=nn.SiLU(),
                       norm_type="LayerNorm", do_concat_trick=False):
              super().__init__()
              self.style = style
              self.do_concat_trick = do_concat_trick
              
              if style == "MeshGraphEdgeMLP":
                  # 边MLP：处理边特征 + 源节点特征 + 目标节点特征
                  if do_concat_trick:
                      self.input_proj = nn.Linear(input_dim, hidden_dim)
                  else:
                      self.input_proj = nn.Linear(input_dim * 3, hidden_dim)
              else:
                  # 节点MLP：处理节点特征 + 聚合消息
                  self.input_proj = nn.Linear(input_dim, hidden_dim)
              
              # 隐藏层
              self.hidden_layers = nn.ModuleList()
              for _ in range(hidden_layers - 1):
                  self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
              
              # 输出层
              self.output_proj = nn.Linear(hidden_dim, output_dim)
              
              # 归一化
              if norm_type == "LayerNorm":
                  self.norm = nn.LayerNorm(hidden_dim)
              else:
                  self.norm = nn.Identity()
              
              self.activation = activation_fn

          def forward(self, features, node_features=None, graph=None):
              if self.style == "MeshGraphEdgeMLP":
                  # 边MLP处理
                  if self.do_concat_trick:
                      x = features + node_features[0] + node_features[1]
                  else:
                      x = torch.cat([features, node_features[0], node_features[1]], dim=-1)
              else:
                  # 节点MLP处理
                  x = features
              
              # 前向传播
              x = self.input_proj(x)
              x = self.activation(x)
              x = self.norm(x)
              
              for layer in self.hidden_layers:
                  x = layer(x)
                  x = self.activation(x)
                  x = self.norm(x)
              
              x = self.output_proj(x)
              return x

  skills:

    build_graph_mlp:

      description: 构建图神经网络专用的MLP

  knowledge:

    hot_models:

      - model: GraphCast
        year: 2023
        role: DeepMind的气象预报图神经网络
        architecture: mesh-based GNN

    best_practices:

      - 使用concat_trick可以显著减少大图处理的显存占用
      - SiLU激活函数在GNN中表现良好
      - LayerNorm有助于稳定训练

    paper_references:

      - title: "GraphCast: Learning Skillful Medium-Range Global Weather Forecasting"
        authors: Lam et al.
        year: 2023

  graph:

    is_a:
      - GraphNeuralNetwork
      - MultiLayerPerceptron
      - FeatureTransformer

    used_in_models:
      - GraphCast
      - MeshGraphNet

    compatible_with:

      inputs:
        - EdgeFeatures
        - NodeFeatures
        - AggregatedMessages

      outputs:
        - UpdatedEdgeFeatures
        - UpdatedNodeFeatures
