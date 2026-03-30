component:

  meta:
    name: RNNClusterPooling
    alias: ClusterGRUPooling
    version: 1.0
    domain: deep_learning
    category: graph_neural_network
    subcategory: hierarchical_pooling
    author: OneScience
    license: Apache-2.0
    tags:
      - rnn
      - gru
      - cluster_pooling
      - graph_hierarchy
      - positional_encoding


  concept:

    description: >
      RNNClusterPooling 用于把细粒度节点特征聚合为簇级表示。
      模块按簇索引从节点与位置编码中 gather 特征，将每个簇视作序列输入 GRU，
      再取每个簇最后有效节点隐状态并经 MLP 投影，得到簇表示 W。

    intuition: >
      相比简单 mean/sum pooling，序列模型可以在簇内编码顺序化上下文，
      对簇内部结构差异具备更强表达能力。

    problem_it_solves:
      - 将节点级信息压缩为簇级潜在表示
      - 融合位置编码增强簇内结构感知
      - 通过 cluster_mask 处理不同簇大小
      - 兼容层级图从细粒度到粗粒度映射


  theory:

    formula:

      cluster_gather:
        expression: |
          V_k = Gather(V, clusters_k),\ P_k = Gather(P, clusters_k)

      rnn_pool:
        expression: |
          H_k = GRU([V_k, P_k])
          w_k = H_k[t_{last}(mask_k)]

      projection:
        expression: |
          W_k = MLP(w_k)

    variables:

      V:
        name: NodeFeature
        shape: [B, N, C_v]
        description: 节点特征

      P:
        name: PositionalEncoding
        shape: [B, N, C_p]
        description: 节点位置编码

      clusters:
        name: ClusterIndex
        shape: [B, K, C_max]
        description: 每个簇的节点索引

      W:
        name: ClusterEmbedding
        shape: [B, K, w_size]
        description: 输出簇特征


  structure:

    architecture: rnn_based_cluster_pooling

    pipeline:

      - name: GatherClusterFeatures
        operation: gather_node_and_position_by_cluster_index

      - name: SequenceEncoding
        operation: gru_over_cluster_sequence

      - name: ValidStateSelection
        operation: pick_last_valid_state_via_mask

      - name: OutputProjection
        operation: mlp_projection_to_cluster_embedding


  interface:

    parameters:

      w_size:
        type: int
        description: 输出簇特征维度

      pos_length:
        type: int
        description: 位置编码长度参数

    inputs:

      V:
        type: NodeEmbedding
        shape: [B, N, 128]
        dtype: float32
        description: 节点特征

      clusters:
        type: ClusterIndex
        shape: [B, K, C_max]
        description: 簇索引

      positional_encoding:
        type: PositionEmbedding
        shape: [B, N, pos_length*8]
        description: 位置编码

      cluster_mask:
        type: Mask
        shape: [B, K, C_max]
        description: 有效节点掩码

    outputs:

      W:
        type: ClusterEmbedding
        shape: [B, K, w_size]
        description: 簇级特征


  types:

    NodeEmbedding:
      shape: [B, N, C]
      description: 节点 embedding

    PositionEmbedding:
      shape: [B, N, C_p]
      description: 位置编码

    ClusterIndex:
      shape: [B, K, C_max]
      description: 簇索引张量

    ClusterEmbedding:
      shape: [B, K, C]
      description: 簇 embedding

    Mask:
      shape: [B, K, C_max]
      description: 有效位掩码


  implementation:

    framework: pytorch

    code: |

      class RNNClusterPooling(nn.Module):
          def __init__(self, w_size, pos_length):
              input_size = 128 + pos_length * 8
              self.rnn_pooling = nn.GRU(input_size=input_size, hidden_size=w_size, batch_first=True)
              self.linear_rnn = OneMlp(style="StandardMLP", input_dim=w_size, output_dim=w_size, hidden_dims=[w_size])

          def forward(self, V, clusters, positional_encoding, cluster_mask):
              # gather -> GRU -> 根据mask取最后有效状态 -> MLP
              ...
              return W


  skills:

    build_rnn_cluster_pooling:

      description: 构建基于 GRU 的簇级池化模块

      inputs:
        - w_size
        - pos_length

      prompt_template: |

        构建 RNNClusterPooling 模块。

        参数：
        w_size = {{w_size}}
        pos_length = {{pos_length}}

        要求：
        按簇 gather 特征并通过 GRU 编码后输出簇表示。


    diagnose_cluster_pooling:

      description: 分析簇级池化中的索引与掩码问题

      checks:
        - invalid_cluster_indices
        - mask_last_index_underflow
        - positional_feature_dim_mismatch
        - inefficient_gather_memory_usage


  knowledge:

    usage_patterns:

      mesh_to_latent_pooling:

        pipeline:
          - GatherByCluster
          - GRUEncodeCluster
          - ProduceLatentNodes

      hierarchical_gnn:

        pipeline:
          - NodeEncoder
          - ClusterPooling
          - CoarseGraphProcessing


    hot_models:

      - model: GraphCast-like Hierarchical Models
        year: 2023
        role: 细网格到粗图特征压缩
        architecture: hierarchical_gnn


    best_practices:

      - 训练前检查 cluster 索引范围，避免 gather 越界。
      - 使用 cluster_mask 处理空簇和变长簇。
      - 将 GRU 输出再做 MLP 投影，提高表达灵活性。


    anti_patterns:

      - 忽略 mask 直接取最后时刻，导致空簇错误。
      - 假设固定簇大小而不支持变长输入。


    paper_references:

      - title: "Graph U-Nets"
        authors: Gao and Ji
        year: 2019


  graph:

    is_a:
      - NeuralNetworkComponent
      - HierarchicalPoolingModule

    part_of:
      - PoolingModuleSystem
      - HierarchicalGNNPipelines

    depends_on:
      - GRU
      - Gather
      - ClusterMask
      - MLP

    variants:
      - RNNClusterPooling

    used_in_models:
      - Hierarchical Graph Networks
      - Mesh-to-Latent Pipelines
      - Weather Graph Models

    compatible_with:

      inputs:
        - NodeEmbedding
        - PositionEmbedding
        - ClusterIndex
        - Mask

      outputs:
        - ClusterEmbedding