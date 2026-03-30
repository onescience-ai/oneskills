component:

  meta:
    name: MeshNodeBlock
    alias: NodeBlock
    version: 1.0
    domain: deep_learning
    category: graph_neural_network
    subcategory: node_update
    author: OneScience
    license: Apache-2.0
    tags:
      - graphcast
      - meshgraphnet
      - node_update
      - aggregation
      - residual_connection


  concept:

    description: >
      MeshNodeBlock 是图神经网络中的节点更新模块。
      它先按图拓扑聚合边消息并与节点特征拼接，再通过 MLP 变换后与原节点特征做残差相加，
      最终返回原边特征与更新后的节点特征。

    intuition: >
      节点更新可视为“先收消息再自我更新”：
      邻边信息反映局部上下文，残差连接确保原语义不被过度覆盖。

    problem_it_solves:
      - 在网格图中融合边消息更新节点表示
      - 支持 sum/mean 聚合策略切换
      - 提供残差路径增强训练稳定性
      - 兼容 DGLGraph 与 CuGraphCSC


  theory:

    formula:

      message_aggregation:
        expression: |
          m_i = Agg_{e\in\mathcal{N}(i)}(e)

      node_update:
        expression: |
          h_i' = MLP([h_i, m_i]) + h_i

    variables:

      h_i:
        name: NodeFeature
        shape: [N, C_node]
        description: 节点特征

      e:
        name: EdgeFeature
        shape: [E, C_edge]
        description: 边特征

      Agg:
        name: AggregationOperator
        description: sum 或 mean 聚合算子


  structure:

    architecture: residual_node_processor

    pipeline:

      - name: EdgeToNodeAggregation
        operation: aggregate_and_concat

      - name: NodeMLP
        operation: mesh_graph_mlp

      - name: ResidualAdd
        operation: add_original_node_feature

      - name: ReturnTuple
        operation: return_edge_passthrough_and_updated_node


  interface:

    parameters:

      aggregation:
        type: str
        description: 聚合方式（sum/mean）

      input_dim_nodes:
        type: int
        description: 输入节点维度

      input_dim_edges:
        type: int
        description: 输入边维度

      output_dim:
        type: int
        description: 输出节点维度

      hidden_dim:
        type: int
        description: MLP 隐层维度

      hidden_layers:
        type: int
        description: MLP 隐层层数

      activation_fn:
        type: nn.Module
        description: 激活函数

      norm_type:
        type: str
        description: 归一化类型

      recompute_activation:
        type: bool
        description: 是否启用激活重计算

    inputs:

      efeat:
        type: EdgeEmbedding
        shape: [E, C_edge]
        dtype: float32
        description: 边特征

      nfeat:
        type: NodeEmbedding
        shape: [N, C_node]
        dtype: float32
        description: 节点特征

      graph:
        type: GraphStructure
        description: 图拓扑对象

    outputs:

      edge_passthrough:
        type: EdgeEmbedding
        shape: [E, C_edge]
        description: 原样返回的边特征

      node_output:
        type: NodeEmbedding
        shape: [N, C_node]
        description: 更新后的节点特征


  types:

    EdgeEmbedding:
      shape: [E, C_edge]
      description: 边 embedding

    NodeEmbedding:
      shape: [N, C_node]
      description: 节点 embedding

    GraphStructure:
      description: 图连接结构


  implementation:

    framework: pytorch

    code: |

      class MeshNodeBlock(nn.Module):
          def __init__(..., aggregation="sum", ...):
              self.node_mlp = OneMlp(style="MeshGraphMLP", input_dim=input_dim_nodes + input_dim_edges, ...)

          def forward(self, efeat, nfeat, graph):
              cat_feat = aggregate_and_concat(efeat, nfeat, graph, self.aggregation)
              nfeat_new = self.node_mlp(cat_feat) + nfeat
              return efeat, nfeat_new


  skills:

    build_node_block:

      description: 构建图节点更新残差模块

      inputs:
        - aggregation
        - input_dim_nodes
        - input_dim_edges
        - output_dim
        - hidden_dim

      prompt_template: |

        构建 MeshNodeBlock 模块。

        参数：
        aggregation = {{aggregation}}
        input_dim_nodes = {{input_dim_nodes}}
        input_dim_edges = {{input_dim_edges}}
        output_dim = {{output_dim}}
        hidden_dim = {{hidden_dim}}

        要求：
        聚合边消息后更新节点并使用残差连接。


    diagnose_node_block:

      description: 分析节点更新模块中的聚合和维度问题

      checks:
        - invalid_aggregation_mode
        - dimension_mismatch_in_concat
        - residual_shape_mismatch
        - graph_backend_incompatibility


  knowledge:

    usage_patterns:

      graphcast_node_update:

        pipeline:
          - EdgeAggregation
          - NodeMLP
          - ResidualNodeUpdate

      message_passing_stage:

        pipeline:
          - EdgeUpdate
          - NodeUpdate
          - Readout


    hot_models:

      - model: GraphCast
        year: 2023
        role: 网格图节点更新模块
        architecture: mesh_graph_network

      - model: MeshGraphNet
        year: 2021
        role: 通用网格图消息传递框架
        architecture: gnn


    best_practices:

      - 保证 output_dim 与 input_dim_nodes 一致以支持残差。
      - 在噪声较大场景优先使用 mean 聚合稳定训练。
      - 对大图训练配合激活重计算控制显存。


    anti_patterns:

      - 忽略聚合输出维度检查直接送入 MLP。
      - 节点更新后丢失残差路径导致梯度退化。


    paper_references:

      - title: "Learning Mesh-Based Simulation with Graph Networks"
        authors: Pfaff et al.
        year: 2021


  graph:

    is_a:
      - NeuralNetworkComponent
      - NodeUpdateModule

    part_of:
      - GraphNeuralNetworks
      - MeshGraphModels

    depends_on:
      - AggregateAndConcat
      - MeshGraphMLP
      - ResidualConnection

    variants:
      - MeshNodeBlock

    used_in_models:
      - GraphCast
      - MeshGraphNet
      - Message Passing GNNs

    compatible_with:

      inputs:
        - EdgeEmbedding
        - NodeEmbedding
        - GraphStructure

      outputs:
        - EdgeEmbedding
        - NodeEmbedding