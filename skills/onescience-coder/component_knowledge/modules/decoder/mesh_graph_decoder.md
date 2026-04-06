component:

  meta:
    name: MeshGraphDecoder
    alias: Mesh Graph Decoder
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: graph_decoder
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
      MeshGraphDecoder是用于GraphCast或MeshGraphNet等模型中的解码器模块。
      它作用于连接多尺度网格（Mesh，代表隐空间）和规则栅格（Grid，代表输出物理域）的二部图（Bipartite Graph）。
      该模块负责将隐空间的演化特征传递回物理空间，通过边特征更新、消息聚合和节点特征更新三个步骤恢复出最终的预测结果。

    intuition: >
      就像天气预报中的"网格插值"过程：从稀疏的观测站点（Mesh节点）推断密集的网格点（Grid节点）的天气状况。
      边特征更新相当于建立站点与网格点之间的关系模型，消息聚合就像收集所有站点对某个网格点的影响，
      节点特征更新则综合这些影响给出该网格点的最终预测值。

    problem_it_solves:
      - 隐空间到物理空间的信息传递
      - 不规则网格到规则网格的特征映射
      - 二部图结构中的消息传递
      - 多尺度数据的融合与解码
      - 物理场预测中的空间插值

  theory:

    formula:

      edge_feature_update:
        expression: |
          e'_{ij} = \text{EdgeMLP}(e_{ij}, h_i, h_j)
          \text{where } i \in \text{Mesh}, j \in \text{Grid}

      message_aggregation:
        expression: |
          m_j = \sum_{i \in \mathcal{N}(j)} e'_{ij} \quad \text{or} \quad m_j = \frac{1}{|\mathcal{N}(j)|}\sum_{i \in \mathcal{N}(j)} e'_{ij}

      node_feature_update:
        expression: |
          h'_j = \text{NodeMLP}(\text{Concat}(h_j, m_j)) + h_j

    variables:

      e_{ij}:
        name: EdgeFeatures
        shape: [num_edges, edge_dim]
        description: Mesh到Grid边的特征

      h_i:
        name: MeshNodeFeatures
        shape: [num_mesh_nodes, mesh_dim]
        description: Mesh节点特征（源节点）

      h_j:
        name: GridNodeFeatures
        shape: [num_grid_nodes, grid_dim]
        description: Grid节点特征（目标节点）

      m_j:
        name: AggregatedMessage
        shape: [num_grid_nodes, message_dim]
        description: 聚合到Grid节点的消息

  structure:

    architecture: bipartite_graph_decoder

    pipeline:

      - name: EdgeFeatureUpdate
        operation: edge_mlp (mesh + grid + edge features)

      - name: MessageAggregation
        operation: graph_aggregation (sum or mean)

      - name: NodeFeatureUpdate
        operation: node_mlp + residual_connection

  interface:

    parameters:

      aggregation:
        type: str
        description: 消息聚合方法，可选"sum"或"mean"

      input_dim_src_nodes:
        type: int
        description: 输入源节点（Mesh）特征的维度

      input_dim_dst_nodes:
        type: int
        description: 输入目标节点（Grid）特征的维度

      input_dim_edges:
        type: int
        description: 输入边特征的维度

      output_dim_dst_nodes:
        type: int
        description: 输出目标节点（Grid）特征的维度

      output_dim_edges:
        type: int
        description: 输出边特征的维度

      hidden_dim:
        type: int
        description: MLP隐藏层的神经元数量

      hidden_layers:
        type: int
        description: 隐藏层的层数

      activation_fn:
        type: nn.Module
        description: 激活函数类型

      norm_type:
        type: str
        description: 归一化类型 ("LayerNorm" 或 "TELayerNorm")

      do_concat_trick:
        type: bool
        description: 是否使用"拼接技巧"优化显存

      recompute_activation:
        type: bool
        description: 是否启用激活重计算以节省显存

    inputs:

      m2g_efeat:
        type: EdgeFeatures
        shape: [num_edges, input_dim_edges]
        description: Mesh-to-Grid边特征

      grid_nfeat:
        type: GridNodeFeatures
        shape: [num_grid_nodes, input_dim_dst_nodes]
        description: Grid节点特征（目标节点）

      mesh_nfeat:
        type: MeshNodeFeatures
        shape: [num_mesh_nodes, input_dim_src_nodes]
        description: Mesh节点特征（源节点）

      graph:
        type: GraphStructure
        description: DGLGraph或CuGraphCSC对象，表示Mesh到Grid的二部图

    outputs:

      dst_feat:
        type: UpdatedGridFeatures
        shape: [num_grid_nodes, output_dim_dst_nodes]
        description: 更新后的Grid节点特征

  types:

    BipartiteGraph:
      description: 连接两种不同类型节点的二部图结构

    MessagePassing:
      description: 图神经网络中的消息传递机制

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules import OneMlp
      from onescience.modules.utils.gnnlayer_utils import aggregate_and_concat

      class MeshGraphDecoder(nn.Module):
          def __init__(self, aggregation="sum", input_dim_src_nodes=512, input_dim_dst_nodes=512,
                       input_dim_edges=512, output_dim_dst_nodes=512, output_dim_edges=512,
                       hidden_dim=512, hidden_layers=1, activation_fn=nn.SiLU(),
                       norm_type="LayerNorm", do_concat_trick=False, recompute_activation=False):
              super().__init__()
              self.aggregation = aggregation

              # 边MLP
              edge_mlp_style = "MeshGraphEdgeMLPSum" if do_concat_trick else "MeshGraphEdgeMLPConcat"
              self.edge_mlp = OneMlp(
                  style=edge_mlp_style,
                  efeat_dim=input_dim_edges,
                  src_dim=input_dim_src_nodes,
                  dst_dim=input_dim_dst_nodes,
                  output_dim=output_dim_edges,
                  hidden_dim=hidden_dim,
                  hidden_layers=hidden_layers,
                  activation_fn=activation_fn,
                  norm_type=norm_type,
                  recompute_activation=recompute_activation,
              )

              # 节点MLP
              self.node_mlp = OneMlp(
                  style="MeshGraphMLP",
                  input_dim=input_dim_dst_nodes + output_dim_edges,
                  output_dim=output_dim_dst_nodes,
                  hidden_dim=hidden_dim,
                  hidden_layers=hidden_layers,
                  activation_fn=activation_fn,
                  norm_type=norm_type,
                  recompute_activation=recompute_activation,
              )

          def forward(self, m2g_efeat, grid_nfeat, mesh_nfeat, graph):
              # 边特征更新
              efeat = self.edge_mlp(m2g_efeat, (mesh_nfeat, grid_nfeat), graph)
              
              # 消息聚合与拼接
              cat_feat = aggregate_and_concat(efeat, grid_nfeat, graph, self.aggregation)
              
              # 节点特征更新（带残差连接）
              dst_feat = self.node_mlp(cat_feat) + grid_nfeat
              
              return dst_feat

  skills:

    build_mesh_decoder:

      description: 构建用于Mesh-Grid映射的图解码器

      inputs:
        - aggregation
        - input_dims
        - output_dims
        - hidden_dim

      prompt_template: |

        构建MeshGraphDecoder，实现从隐空间Mesh到物理空间Grid的特征传递。

        参数：
        aggregation = {{aggregation}}
        input_dims = ({{input_dim_src_nodes}}, {{input_dim_dst_nodes}}, {{input_dim_edges}})
        output_dims = ({{output_dim_dst_nodes}}, {{output_dim_edges}})
        hidden_dim = {{hidden_dim}}

        要求：
        1. 支持二部图结构
        2. 实现边-节点消息传递
        3. 支持sum/mean聚合策略
        4. 包含残差连接

    optimize_graph_decoding:

      description: 优化图解码器的计算效率

      checks:
        - memory_efficiency (大图处理的显存使用)
        - aggregation_correctness (聚合操作的正确性)
        - gradient_flow (梯度在图中的流动)

  knowledge:

    usage_patterns:

      weather_forecasting_decoding:

        pipeline:
          - Mesh Evolution: 隐空间特征演化
          - Edge Update: 建立Mesh-Grid关系
          - Message Aggregation: 信息从Mesh传递到Grid
          - Grid Prediction: 物理空间预测

      multi_scale_mapping:

        pipeline:
          - Coarse Mesh: 低分辨率隐空间
          - Fine Grid: 高分辨率物理空间
          - Bipartite Mapping: 跨尺度映射
          - Feature Reconstruction: 特征重建

    hot_models:

      - model: GraphCast
        year: 2023
        role: DeepMind的气象预报图神经网络
        architecture: mesh-based GNN

      - model: MeshGraphNet
        year: 2023
        role: 基于网格的图神经网络
        architecture: mesh GNN

      - model: EarthFormer
        year: 2023
        role: 基于Transformer的地球科学模型
        architecture: transformer on graphs

    best_practices:

      - 根据图密度选择合适的聚合策略（sum适合稀疏图，mean适合稠密图）
      - 使用concat_trick可以显著减少大图处理的显存占用
      - 残差连接有助于保持梯度流动和训练稳定性
      - 激活重计算在处理超大图时可以节省显存但增加计算时间

    anti_patterns:

      - 忽略二部图的特殊结构，直接使用通用GNN
      - 聚合策略选择不当导致数值不稳定
      - 边特征和节点特征维度不匹配
      - 在大图处理时忽略显存优化

    paper_references:

      - title: "GraphCast: Learning Skillful Medium-Range Global Weather Forecasting"
        authors: Lam et al.
        year: 2023

      - title: "MeshGraphNets: Learning Mesh-Based Physics Simulators"
        authors: Pfaff et al.
        year: 2021

      - title: "Learning Mesh-Based Simulators with Graph Networks"
        authors: Sanchez-Gonzalez et al.
        year: 2020

  graph:

    is_a:
      - GraphNeuralNetwork
      - NeuralNetworkDecoder
      - BipartiteGraphProcessor

    part_of:
      - GraphCastArchitecture
      - MeshBasedModels
      - PhysicsInformedNeuralNetworks

    depends_on:
      - MeshGraphEdgeMLP
      - MeshGraphMLP
      - GraphAggregation
      - BipartiteGraph

    variants:
      - GeneralGraphDecoder (通用图解码器)
      - TransformerDecoder (基于Transformer的解码器)
      - ConvolutionalDecoder (卷积解码器)

    used_in_models:
      - GraphCast
      - MeshGraphNet
      - 其他基于网格的物理模型

    compatible_with:

      inputs:
        - MeshFeatures
        - GridFeatures
        - BipartiteGraph

      outputs:
        - UpdatedGridFeatures
        - PhysicalFieldPrediction
        - SpatialReconstruction
