component:

  meta:
    name: MeshGraphEncoder
    alias: Mesh Graph Encoder
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: graph_encoder
    author: OneScience
    license: Apache-2.0
    tags:
      - graph_neural_network
      - mesh_processing
      - bipartite_graph
      - graphcast
      - weather_forecasting

  concept:

    description: >
      MeshGraphEncoder是用于GraphCast或MeshGraphNet等模型中的编码器模块，作用于连接规则网格（Grid，代表输入物理域）和多尺度网格（Mesh，代表隐空间计算域）的二部图。
      负责将信息从输入网格（源节点）编码并传递到隐空间网格（目标节点）。

    intuition: >
      就像气象预报中的"网格插值"逆向过程：从密集的观测网格点（Grid节点）推断稀疏的计算网格点（Mesh节点）的特征。
      边特征更新建立网格点与计算点之间的关系模型，消息聚合收集所有网格点对某个计算点的影响。

    problem_it_solves:
      - 物理空间到隐空间的信息传递
      - 规则网格到不规则网格的特征映射
      - 二部图结构中的消息传递
      - 多尺度数据的编码与压缩

  theory:

    formula:

      edge_feature_update:
        expression: |
          e'_{ij} = \text{EdgeMLP}(e_{ij}, h_i, h_j)
          \text{where } i \in \text{Grid}, j \in \text{Mesh}

      message_aggregation:
        expression: |
          m_j = \sum_{i \in \mathcal{N}(j)} e'_{ij} \quad \text{or} \quad m_j = \frac{1}{|\mathcal{N}(j)|}\sum_{i \in \mathcal{N}(j)} e'_{ij}

      node_feature_update:
        expression: |
          h'_{grid} = h_{grid} + \text{NodeMLP}(h_{grid})
          h'_{mesh} = h_{mesh} + \text{NodeMLP}(\text{Concat}(h_{mesh}, m_j))

  structure:

    architecture: bipartite_graph_encoder

    pipeline:

      - name: EdgeFeatureUpdate
        operation: edge_mlp (grid + mesh + edge features)

      - name: MessageAggregation
        operation: graph_aggregation (sum or mean)

      - name: DualNodeUpdate
        operation: grid_node_mlp + mesh_node_mlp

  interface:

    parameters:

      aggregation:
        type: str
        description: 消息聚合方法，可选"sum"或"mean"

      input_dim_src_nodes:
        type: int
        description: 输入源节点（Grid）特征的维度

      input_dim_dst_nodes:
        type: int
        description: 输入目标节点（Mesh）特征的维度

      input_dim_edges:
        type: int
        description: 输入二部图边特征的维度

      output_dim_src_nodes:
        type: int
        description: 输出源节点（Grid）特征的维度

      output_dim_dst_nodes:
        type: int
        description: 输出目标节点（Mesh）特征的维度

      hidden_dim:
        type: int
        description: MLP隐藏层的神经元数量

    inputs:

      g2m_efeat:
        type: EdgeFeatures
        shape: [num_edges, input_dim_edges]

      grid_nfeat:
        type: GridNodeFeatures
        shape: [num_grid_nodes, input_dim_src_nodes]

      mesh_nfeat:
        type: MeshNodeFeatures
        shape: [num_mesh_nodes, input_dim_dst_nodes]

      graph:
        type: GraphStructure
        description: DGLGraph或CuGraphCSC对象

    outputs:

      grid_nfeat_out:
        type: UpdatedGridFeatures
        shape: [num_grid_nodes, output_dim_src_nodes]

      mesh_nfeat_out:
        type: UpdatedMeshFeatures
        shape: [num_mesh_nodes, output_dim_dst_nodes]

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules import OneMlp
      from onescience.modules.utils.gnnlayer_utils import aggregate_and_concat

      class MeshGraphEncoder(nn.Module):
          def __init__(self, aggregation="sum", input_dim_src_nodes=512, 
                       input_dim_dst_nodes=512, input_dim_edges=512, 
                       output_dim_src_nodes=512, output_dim_dst_nodes=512,
                       hidden_dim=512, **kwargs):
              super().__init__()
              self.aggregation = aggregation

              edge_mlp_style = "MeshGraphEdgeMLPSum" if do_concat_trick else "MeshGraphEdgeMLPConcat"
              self.edge_mlp = OneMlp(
                  style=edge_mlp_style, efeat_dim=input_dim_edges,
                  src_dim=input_dim_src_nodes, dst_dim=input_dim_dst_nodes,
                  output_dim=output_dim_edges, hidden_dim=hidden_dim
              )

              self.src_node_mlp = OneMlp(
                  style="MeshGraphMLP", input_dim=input_dim_src_nodes,
                  output_dim=output_dim_src_nodes, hidden_dim=hidden_dim
              )

              self.dst_node_mlp = OneMlp(
                  style="MeshGraphMLP", input_dim=input_dim_dst_nodes + output_dim_edges,
                  output_dim=output_dim_dst_nodes, hidden_dim=hidden_dim
              )

          def forward(self, g2m_efeat, grid_nfeat, mesh_nfeat, graph):
              efeat = self.edge_mlp(g2m_efeat, (grid_nfeat, mesh_nfeat), graph)
              cat_feat = aggregate_and_concat(efeat, mesh_nfeat, graph, self.aggregation)
              mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat)
              grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat)
              return grid_nfeat, mesh_nfeat

  skills:

    build_mesh_encoder:

      description: 构建用于Grid-Mesh映射的图编码器

  knowledge:

    hot_models:

      - model: GraphCast
        year: 2023
        role: DeepMind的气象预报图神经网络
        architecture: mesh-based GNN

    best_practices:

      - 根据图密度选择合适的聚合策略
      - 使用concat_trick可以显著减少大图处理的显存占用

    paper_references:

      - title: "GraphCast: Learning Skillful Medium-Range Global Weather Forecasting"
        authors: Lam et al.
        year: 2023

  graph:

    is_a:
      - GraphNeuralNetwork
      - NeuralNetworkEncoder
      - BipartiteGraphProcessor

    used_in_models:
      - GraphCast
      - MeshGraphNet

    compatible_with:

      inputs:
        - GridFeatures
        - MeshFeatures
        - BipartiteGraph

      outputs:
        - UpdatedGridFeatures
        - UpdatedMeshFeatures
