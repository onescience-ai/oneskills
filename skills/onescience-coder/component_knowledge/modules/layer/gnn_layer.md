component:

  meta:
    name: GraphNeuralNetworkLayer
    alias: GNNLayer
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: graph_neural_network
    author: OneScience
    license: Apache-2.0
    tags:
      - graph_neural_network
      - message_passing
      - node_update
      - edge_update
      - scatter_sum

  concept:

    description: >
      GNNLayer 实现了一个标准的基于消息传递机制（Message Passing）的图神经网络层。
      它同时维护并更新节点（Node）特征和边（Edge）特征。在单次前向传播中，
      它首先结合相连节点的特征来更新边特征，然后将更新后的边特征聚合到对应的节点上，
      最后结合节点原有的特征更新节点自身的表示。

    intuition: >
      图结构数据可以用来表示物理系统中的粒子、网格点或分子结构。该层的更新过程模仿了物理世界中的相互作用：
      相连的两个实体（节点）根据它们当前的状态和相互关系产生新的相互作用（边更新）；
      随后，每个实体收集所有作用于它的外部影响（通过 scatter_sum 进行特征聚合），
      并据此改变自身的状态（节点更新）。

    problem_it_solves:
      - 处理非欧几里得空间（Non-Euclidean）的不规则数据结构（如图结构、网格数据）
      - 显式地建模实体之间的依赖和物理相互作用关系
      - 支持动态或任意大小的图输入，适应不同的节点和边数量

  theory:

    formula:

      edge_update:
        expression: $e_{ij}^{(t+1)} = \text{MLP}_{\text{edge}}(v_i^{(t)} \parallel v_j^{(t)} \parallel e_{ij}^{(t)})$

      node_aggregation:
        expression: $m_i^{(t+1)} = \sum_{j \in \mathcal{N}(i)} e_{ij}^{(t+1)}$

      node_update:
        expression: $v_i^{(t+1)} = \text{MLP}_{\text{node}}(v_i^{(t)} \parallel m_i^{(t+1)})$

    variables:

      v_i:
        name: SenderNodeFeature
        description: 发送节点的特征向量

      v_j:
        name: ReceiverNodeFeature
        description: 接收节点的特征向量

      e_{ij}:
        name: EdgeFeature
        description: 节点 i 到节点 j 之间的边特征

      m_i:
        name: AggregatedMessage
        description: 节点 i 收集到的所有相邻边的特征总和

      \parallel:
        name: Concatenation
        description: 特征拼接操作

  structure:

    architecture: message_passing_graph_network

    pipeline:

      - name: FeatureGathering
        operation: torch.gather (获取 sender 和 receiver 特征)

      - name: EdgeInputConcatenation
        operation: torch.cat (拼接 senders, receivers, edges)

      - name: EdgeUpdate
        operation: mlp_and_layer_norm

      - name: MessageAggregation
        operation: scatter_sum (按节点索引聚合边特征)

      - name: NodeInputConcatenation
        operation: torch.cat (拼接原始节点特征和聚合信息)

      - name: NodeUpdate
        operation: mlp_and_layer_norm

  interface:

    parameters:

      n_hidden:
        type: int
        default: 2
        description: 节点和边更新网络（MLP）中的隐藏层数量

      node_size:
        type: int
        default: 128
        description: 节点特征的维度

      edge_size:
        type: int
        default: 128
        description: 边特征的维度

      output_size:
        type: int
        default: null
        description: 输出节点特征的维度，如果为空则默认等于 node_size

      layer_norm:
        type: bool
        default: false
        description: 是否在节点和边 MLP 输出后应用 Layer Normalization

    inputs:

      V:
        type: Tensor
        shape: [..., num_nodes, node_size]
        dtype: float32
        description: 节点特征矩阵

      E:
        type: Tensor
        shape: [..., num_edges, edge_size]
        dtype: float32
        description: 边特征矩阵

      edges:
        type: Tensor
        shape: [..., num_edges, 2]
        dtype: int64
        description: 边索引矩阵，edges[..., 0] 为 senders，edges[..., 1] 为 receivers

    outputs:

      node_embeddings:
        type: Tensor
        shape: [..., num_nodes, output_size]
        description: 更新后的节点特征

      edge_embeddings:
        type: Tensor
        shape: [..., num_edges, edge_size]
        description: 更新后的边特征

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量，支持批处理 (Batch) 维度

  implementation:

    framework: pytorch

    code: |
      import torch
      import torch.nn as nn
      from torch_scatter import scatter_sum
      from onescience.modules.mlp import StandardMLP as MLP

      class GNNLayer(nn.Module):
          """
          图神经网络层 (Graph Neural Network Layer)
          基于消息传递机制 (Message Passing) 更新节点和边特征。
          """
          def __init__(self, n_hidden=2, node_size=128, edge_size=128, output_size=None, layer_norm=False):
              super(GNNLayer, self).__init__()
              output_size = output_size or node_size
              _hidden_dim = 128 
              edge_mlp_hidden_dims = [_hidden_dim] * n_hidden
              node_mlp_hidden_dims = [_hidden_dim] * n_hidden

              # 边更新网络
              self.f_edge = MLP(
                  input_dim=edge_size + node_size * 2, 
                  hidden_dims=edge_mlp_hidden_dims,
                  output_dim=edge_size,
                  activation='relu',
                  norm_layer=None, 
                  use_bias=True
              )
              self.edge_norm = nn.LayerNorm(edge_size) if layer_norm else nn.Identity()

              # 节点更新网络
              self.f_node = MLP(
                  input_dim=edge_size + node_size, 
                  hidden_dims=node_mlp_hidden_dims,
                  output_dim=output_size,
                  activation='relu',
                  norm_layer=None,
                  use_bias=True
              )
              self.node_norm = nn.LayerNorm(output_size) if layer_norm else nn.Identity()

          def forward(self, V, E, edges):
              # 收集特征
              senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
              receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, V.shape[-1]))

              # 更新边
              edge_inpt = torch.cat([senders, receivers, E], dim=-1)
              edge_embeddings = self.edge_norm(self.f_edge(edge_inpt))

              # 聚合
              col = edges[..., 0].unsqueeze(-1).repeat(1, 1, edge_embeddings.shape[-1])
              edge_sum = scatter_sum(edge_embeddings, col, dim=-2, dim_size=V.shape[-2])

              # 更新节点
              node_inpt = torch.cat([V, edge_sum], dim=-1)
              node_embeddings = self.node_norm(self.f_node(node_inpt))

              return node_embeddings, edge_embeddings

  skills:

    build_gnn_layer:

      description: 根据图节点和边的特征维度构建消息传递层

      inputs:
        - node_size
        - edge_size
        - n_hidden
        - layer_norm

      prompt_template: |
        构建一个 GNNLayer。
        参数：
        节点特征维度 = {{node_size}}
        边特征维度 = {{edge_size}}
        隐藏层数 = {{n_hidden}}
        启用LayerNorm = {{layer_norm}}

    diagnose_gnn:

      description: 分析使用图神经网络时常见的数据流与张量索引问题

      checks:
        - index_out_of_bounds_in_gather
        - mismatch_between_scatter_dim_size_and_node_count
        - graph_connectivity_causes_empty_scatter

  knowledge:

    usage_patterns:

      encode_process_decode:

        pipeline:
          - Node/Edge Encoder (将原始物理量映射到高维表示)
          - GNNLayer (重复 N 次进行 Message Passing)
          - GNNLayer
          - Node/Edge Decoder (映射回物理预测目标)

    design_patterns:

      explicit_edge_modeling:

        structure:
          - 传统的 GCN 往往只更新节点特征，将边视为静态权重
          - 这种架构显式地通过 MLP 更新边特征，并在下一层继续使用，极大地增强了对复杂交互物理法则（如流体粒子碰撞、弹簧引力等）的建模能力

    hot_models:

      - model: MeshGraphNet
        year: 2020
        role: 用于网格化偏微分方程仿真的通用框架
        architecture: graph_neural_network
        attention_type: None (依赖显式的 Message Passing 替代 Attention)

      - model: GraphCast
        year: 2022
        role: 高分辨率全球天气预报模型
        architecture: gnn_on_icosahedral_grid

    model_usage_details:

      MeshGraphNet / AI for Physics:

        usage: V 通常代表物理粒子/网格点的状态（如速度、压力），E 代表相对距离和位移向量。通过多层 GNN 模拟物理系统在一个时间步内的状态演化。

    best_practices:

      - `scatter_sum` 高度依赖正确的边索引传递，在构建 `edges` 张量时，务必确保其在目标设备（CPU/GPU）上是连续且合法的 `int64` 数据。
      - 当层数较深时，强烈建议将 `layer_norm` 设为 `True`，可以有效防止图特征在多次聚合后数值爆炸或过平滑（Over-smoothing）。
      - 在物理系统建模中，通常推荐为双向连接的边建立有向边表示，这可以赋予模型不对称的相互作用学习能力。

    anti_patterns:

      - `V` 的 `dim_size` 在 `scatter_sum` 中未明确指定或传递错误。如果某些节点没有任何边相连，缺少 `dim_size` 会导致输出张量被截断，引发下游形状不匹配。
      - 未能保证 `edges` 索引的范围落在 `[0, num_nodes - 1]` 内，触发 `torch.gather` 的越界显存访问错误。

    paper_references:

      - title: "Learning to Simulate Complex Physics with Graph Networks"
        authors: Sanchez-Gonzalez et al.
        year: 2020

      - title: "Relational inductive biases, deep learning, and graph networks"
        authors: Battaglia et al.
        year: 2018

  graph:

    is_a:
      - NeuralNetworkComponent
      - GraphNeuralNetwork
      - MessagePassingLayer

    part_of:
      - GraphNetProcessor
      - GraphNeuralSimulator

    depends_on:
      - MLP
      - torch_scatter.scatter_sum
      - torch.gather
      - LayerNorm

    variants:
      - Graph Convolutional Network (GCN)
      - Graph Attention Network (GAT)
      - GraphSAGE

    used_in_models:
      - MeshGraphNet
      - GraphCast
      - GNS (Graph Network Simulator)

    compatible_with:

      inputs:
        - Tensor (Nodes)
        - Tensor (Edges)
        - Tensor (Indices)

      outputs:
        - Tensor (Nodes)
        - Tensor (Edges)