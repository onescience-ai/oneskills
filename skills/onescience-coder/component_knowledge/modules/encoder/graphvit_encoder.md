component:

  meta:
    name: GraphViTEncoder
    alias: Graph Vision Transformer Encoder
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: graph_encoder
    author: OneScience
    license: Apache-2.0
    tags:
      - graph_neural_network
      - vision_transformer
      - node_encoding
      - edge_encoding
      - message_passing

  concept:

    description: >
      GraphViTEncoder是基于图神经网络（GNN）的编码器模块，负责将输入的物理状态（位置、速度、节点类型）编码为隐空间的节点特征（V）和边特征（E）。
      通过MLP对节点和边进行独立编码，然后通过多层GNN进行消息传递和特征更新，实现从原始物理状态到图结构化表示的转换。

    intuition: >
      就像社交网络分析中，先了解每个人的基本信息（节点编码），然后分析人与人之间的关系（边编码），
      最后通过信息传播了解整个网络的状态（GNN消息传递）。位置编码提供空间上下文，节点类型提供语义信息。

    problem_it_solves:
      - 物理系统到图结构的映射
      - 节点和边特征的联合编码
      - 图结构中的信息传播
      - 多层消息传递的特征学习

  theory:

    formula:

      node_edge_encoding:
        expression: |
          V_{base} = \text{MLP}_{node}(\text{Concat}(states, node\_type))
          E_{base} = \text{MLP}_{edge}(\text{Concat}(distance, norm))

      message_passing:
        expression: |
          V_{fused} = \text{Concat}(V, pos\_enc)
          \Delta V, \Delta E = \text{GNNLayer}(V_{fused}, E, edges)
          V = V + \Delta V
          E = E + \Delta E

  structure:

    architecture: graph_neural_encoder

    pipeline:

      - name: NodeFeaturePreparation
        operation: concatenation (states + node_type)

      - name: EdgeFeatureComputation
        operation: distance_calculation + normalization

      - name: IndependentEncoding
        operation: mlp_node + mlp_edge

      - name: MessagePassing
        operation: gnn_layers (with residual connections)

      - name: FeatureOutput
        operation: node_features + edge_features

  interface:

    parameters:

      nb_gn:
        type: int
        description: GNN层的堆叠数量（消息传递次数）

      state_size:
        type: int
        description: 输入物理状态的维度

      pos_length:
        type: int
        description: 位置编码的频带数量

    inputs:

      mesh_pos:
        type: NodePositions
        shape: [batch, N, D]

      edges:
        type: EdgeIndices
        shape: [batch, M, 2]

      states:
        type: PhysicalStates
        shape: [batch, N, S_in]

      node_type:
        type: NodeType
        shape: [batch, N, T_type]

      pos_enc:
        type: PositionalEncoding
        shape: [batch, N, P]

    outputs:

      V:
        type: NodeFeatures
        shape: [batch, N, 128]

      E:
        type: EdgeFeatures
        shape: [batch, M, 128]

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules.mlp import StandardMLP as MLP
      from onescience.modules.layer.gnn_layer import GNNLayer

      class GraphViTEncoder(nn.Module):
          def __init__(self, nb_gn=4, state_size=3, pos_length=7):
              super().__init__()
              _hidden_dim = 128
              
              self.encoder_node = MLP(
                  input_dim=9 + state_size, hidden_dims=[_hidden_dim],
                  output_dim=128, activation='relu', norm_layer=None
              )
              
              self.encoder_edge = MLP(
                  input_dim=3, hidden_dims=[_hidden_dim],
                  output_dim=128, activation='relu', norm_layer=None
              )

              node_size = 128 + pos_length * 8
              self.encoder_gn = nn.ModuleList([
                  GNNLayer(node_size=node_size, edge_size=128, 
                           output_size=128, layer_norm=True)
                  for _ in range(nb_gn)
              ])

          def forward(self, mesh_pos, edges, states, node_type, pos_enc):
              V = torch.cat([states, node_type], dim=-1)

              senders = torch.gather(mesh_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, 2))
              receivers = torch.gather(mesh_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, 2))
              distance = senders - receivers
              norm = torch.sqrt((distance**2).sum(-1, keepdims=True))
              E = torch.cat([distance, norm], dim=-1)

              V = self.encoder_node(V)
              E = self.encoder_edge(E) 

              for i in range(len(self.encoder_gn)):
                  inpt = torch.cat([V, pos_enc], dim=-1)
                  v, e = self.encoder_gn[i](inpt, E, edges)
                  V = V + v
                  E = E + e

              return V, E

  skills:

    build_graph_encoder:

      description: 构建基于GNN的图编码器

  knowledge:

    hot_models:

      - model: GraphViT
        year: 2023
        role: 结合图神经网络和Vision Transformer的架构
        architecture: GNN + Transformer hybrid

    best_practices:

      - 距离计算时考虑数值稳定性
      - 位置编码对空间图结构特别重要
      - GNN层数不宜过深，避免过度平滑

    paper_references:

      - title: "GraphViT: Introducing Graph Attention to Vision Transformer"
        authors: Chen et al.
        year: 2023

  graph:

    is_a:
      - GraphNeuralNetwork
      - NeuralNetworkEncoder
      - FeatureExtractor

    variants:
      - GraphEncoder (无位置编码)
      - TransformerEncoder (纯Transformer版本)

    used_in_models:
      - GraphViT
      - Particle Physics Models
