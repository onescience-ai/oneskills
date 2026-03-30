component:

  meta:
    name: GraphViTDecoder
    alias: Graph Vision Transformer Decoder
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: graph_decoder
    author: OneScience
    license: Apache-2.0
    tags:
      - graph_neural_network
      - vision_transformer
      - multi_scale
      - node_prediction
      - cluster_feature

  concept:

    description: >
      GraphViT解码器是基于GNN的解码/检索模块，与池化操作相反。
      它利用粗粒度的簇特征（W）和细粒度的节点特征（V）来恢复或预测节点的物理状态更新。
      通过GNN进行信息传播，将全局（簇）信息融合回局部（节点）图结构中，
      实现从粗粒度表示到细粒度预测的解码过程。

    intuition: >
      就像城市规划师先了解各个区域的整体特征（簇特征），然后结合每个地点的具体情况（节点特征），
      通过交通网络（图结构）传播信息，最终预测每个地点的发展变化。簇特征提供了全局上下文，
      节点特征保留了局部细节，GNN负责在图结构上进行信息融合。

    problem_it_solves:
      - 从粗粒度簇特征恢复细粒度节点状态
      - 图结构中的多尺度信息融合
      - 节点级别的物理状态预测
      - 图神经网络中的解码与检索任务
      - 簇级别到节点级别的特征传播

  theory:

    formula:

      cluster_expansion:
        expression: |
          W_{expanded} = \text{Expand}(W, \text{clusters})
          W_{nodes} = \text{Scatter}(W_{expanded}, \text{clusters})

      node_feature_fusion:
        expression: |
          H_{fused} = \text{Concat}(V, W_{nodes}, P)
          H_{updated} = \text{GNN}(H_{fused}, E, \text{edges})
          \Delta s = \text{MLP}(H_{updated})

    variables:

      W:
        name: ClusterFeatures
        shape: [batch, K, w_size]
        description: 粗粒度簇特征，K为簇数量

      V:
        name: NodeFeatures
        shape: [batch, N, 128]
        description: 细粒度节点特征，N为节点数量

      clusters:
        name: ClusterAssignment
        shape: [batch, K, C_max]
        description: 簇分配索引，表示节点到簇的映射关系

      P:
        name: PositionalEncoding
        shape: [batch, N, pos_length * 8]
        description: 节点位置编码

      E:
        name: EdgeFeatures
        shape: [batch, M, 128]
        description: 边特征，M为边数量

      \Delta s:
        name: StateUpdate
        shape: [batch, N, state_size]
        description: 预测的节点状态更新量

  structure:

    architecture: graph_neural_decoder

    pipeline:

      - name: ClusterExpansion
        operation: scatter_and_expand

      - name: FeatureFusion
        operation: concatenation (V + W_nodes + P)

      - name: GraphPropagation
        operation: gnn_layer

      - name: StatePrediction
        operation: mlp_output

  interface:

    parameters:

      w_size:
        type: int
        description: 输入簇特征的维度

      pos_length:
        type: int
        description: 位置编码长度

      state_size:
        type: int
        description: 最终输出的状态维度（例如预测的速度增量）

    inputs:

      W:
        type: ClusterFeatures
        shape: [batch, K, w_size]
        description: 簇特征

      V:
        type: NodeFeatures
        shape: [batch, N, 128]
        description: 节点特征

      clusters:
        type: ClusterAssignment
        shape: [batch, K, C_max]
        description: 簇分配索引

      positional_encoding:
        type: PositionalEncoding
        shape: [batch, N, P]
        description: 位置编码

      edges:
        type: EdgeIndex
        shape: [batch, M, 2]
        description: 细粒度图的边索引

      E:
        type: EdgeFeatures
        shape: [batch, M, 128]
        description: 细粒度图的边特征

    outputs:

      final_state:
        type: StateUpdate
        shape: [batch, N, state_size]
        description: 预测的节点状态更新量

  types:

    ClusterFeatures:
      shape: [batch, num_clusters, feature_dim]
      description: 簇级别的特征表示

    NodeFeatures:
      shape: [batch, num_nodes, feature_dim]
      description: 节点级别的特征表示

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules.mlp import StandardMLP as MLP
      from onescience.modules.layer.gnn_layer import GNNLayer

      class GraphViTDecoder(nn.Module):
          def __init__(self, w_size, pos_length, state_size):
              super().__init__()
              pos_size = pos_length * 8
              node_size = w_size + 128 + pos_size
              
              self.gnn = GNNLayer(node_size=node_size, output_size=128)
              self.final_mlp = MLP(
                  input_dim=128,
                  hidden_dims=[128, 128], 
                  output_dim=state_size,
                  activation='tanh',      
                  output_activation=None, 
                  norm_layer=None
              )

          def forward(self, W, V, clusters, positional_encoding, edges, E):
              B, N, _ = V.shape
              K = clusters.shape[1]
              C_dim = W.shape[-1]

              # 扩展簇特征到节点级别
              W_expanded = W.unsqueeze(-2).repeat(1, 1, clusters.shape[-1], 1).view(B, -1, C_dim)
              cluster_indices_flat = clusters.reshape(B, -1, 1).repeat(1, 1, C_dim)
              
              W_nodes = torch.zeros(B, max(N, cluster_indices_flat.shape[1]), C_dim, device=V.device)
              W_nodes = W_nodes.scatter(-2, cluster_indices_flat, W_expanded)
              W_nodes = W_nodes[:, :N] 

              # 融合特征
              nodes = torch.cat([V, W_nodes, positional_encoding], dim=-1)

              # GNN信息传播
              nodes, _ = self.gnn(nodes, E, edges)

              # 状态预测
              final_state = self.final_mlp(nodes)
              return final_state

  skills:

    build_graph_decoder:

      description: 构建基于GNN的多尺度解码器

      inputs:
        - w_size
        - pos_length
        - state_size

      prompt_template: |

        构建GraphViT解码器，实现从簇特征到节点状态的解码。

        参数：
        w_size = {{w_size}}
        pos_length = {{pos_length}}
        state_size = {{state_size}}

        要求：
        1. 支持簇特征到节点特征的扩展
        2. 融合位置编码信息
        3. 通过GNN进行图结构信息传播
        4. 输出节点级别的状态更新

    analyze_graph_decoding:

      description: 分析图解码过程中的信息流动

      checks:
        - cluster_expansion_correctness (簇扩展的正确性)
        - information_flow (信息在图中的流动)
        - position_encoding_effect (位置编码的影响)

  knowledge:

    usage_patterns:

      multi_scale_graph_prediction:

        pipeline:
          - Cluster Features: 全局/簇级特征
          - Expansion: 簇特征扩展到节点
          - GNN Propagation: 图结构信息传播
          - Node Prediction: 节点状态预测

      hierarchical_graph_learning:

        pipeline:
          - Coarse Level: 粗粒度特征学习
          - Fine Level: 细粒度特征恢复
          - Cross-level Fusion: 跨层级信息融合

    hot_models:

      - model: GraphViT
        year: 2023
        role: 结合图神经网络和Vision Transformer的架构
        architecture: GNN + Transformer hybrid

      - model: Graph U-Net
        year: 2023
        role: 图结构的U-Net架构
        architecture: graph pooling + unpooling

      - model: Graph Transformer
        year: 2023
        role: 基于Transformer的图神经网络
        architecture: transformer on graphs

    best_practices:

      - 簇特征扩展时要注意索引对齐，避免特征错位
      - 位置编码对空间图结构特别重要，能提升预测精度
      - GNN层数不宜过深，避免过度平滑问题
      - 使用tanh激活函数有助于状态更新的数值稳定性

    anti_patterns:

      - 簇分配不一致导致特征扩展错误
      - 忽略位置编码在空间图中的作用
      - GNN层数过深导致节点特征过度平滑
      - 边特征处理不当影响信息传播

    paper_references:

      - title: "GraphViT: Introducing Graph Attention to Vision Transformer"
        authors: Chen et al.
        year: 2023

      - title: "Graph U-Nets"
        authors: Gao and Ji
        year: 2019

      - title: "Masked Graph Prediction Networks"
        authors: Hu et al.
        year: 2023

  graph:

    is_a:
      - GraphNeuralNetwork
      - NeuralNetworkDecoder
      - MultiScaleProcessor

    part_of:
      - HierarchicalGraphModels
      - MultiScaleLearningSystems
      - GraphAutoencoders

    depends_on:
      - GNNLayer
      - StandardMLP
      - PositionalEncoding
      - ClusterAssignment

    variants:
      - GraphDecoder (无位置编码)
      - TransformerDecoder (纯Transformer版本)
      - ConvolutionalDecoder (卷积版本)

    used_in_models:
      - GraphViT
      - Hierarchical Graph Networks
      - Multi-scale Graph Models

    compatible_with:

      inputs:
        - ClusterFeatures
        - NodeFeatures
        - GraphStructure

      outputs:
        - NodeStateUpdate
        - NodePrediction
        - FeatureReconstruction
