component:

  meta:
    name: BistrideGraphMessagePassing
    alias: BiStrideProcessor
    version: 1.0
    domain: deep_learning
    category: graph_neural_network
    subcategory: hierarchical_processor
    author: OneScience
    license: Apache-2.0
    tags:
      - bistride
      - graph_unet
      - message_passing
      - pooling_unpooling
      - weighted_edge_conv


  concept:

    description: >
      BistrideGraphMessagePassing 实现分层图处理器，结构类似图 U-Net。
      模块由 Down pass、Bottom pass、Up pass 组成：
      下采样阶段在多尺度图上执行消息传递与加权边卷积；
      底层执行瓶颈消息传递；
      上采样阶段通过 Unpool 和跳连恢复细粒度节点特征。
      文件还包含 GraphMessagePassing、WeightedEdgeConv、Unpool 与 scatter/degree 工具函数。

    intuition: >
      该处理器把“局部交互”和“跨尺度传播”结合：
      低分辨率图增强全局感受野，高分辨率路径保留细节，
      最终通过跳连融合两者优势。

    problem_it_solves:
      - 在多尺度图上执行高效消息传递
      - 支持池化/反池化的特征跨层传输
      - 通过加权边卷积稳定层级信息流
      - 提供 2D/3D batch 与非 batch 输入兼容


  theory:

    formula:

      gmp_update:
        expression: |
          e_{ij} = MLP_e([x_i, x_j, p_i-p_j, ||p_i-p_j||])
          m_j = \sum_{i\in\mathcal{N}(j)} e_{ij}
          x_j' = MLP_n([x_j, m_j]) + x_j

      weighted_edge_conv:
        expression: |
          y_j = \sum_{i\in\mathcal{N}(j)} w_{ij} x_i

      unpool:
        expression: |
          x_{fine}[idx] = x_{coarse}

    variables:

      x:
        name: NodeFeature
        shape: [N, C] or [B, N, C]
        description: 节点特征

      g:
        name: EdgeIndex
        shape: [2, E]
        description: COO 边索引

      p:
        name: NodePosition
        shape: [N, pos_dim] or [B, N, pos_dim]
        description: 节点坐标

      w_{ij}:
        name: EdgeWeight
        description: 边权重


  structure:

    architecture: hierarchical_graph_unet_processor

    pipeline:

      - name: DownPass
        operation: gmp_then_pool_with_weighted_edge_conv

      - name: BottomPass
        operation: gmp_on_coarsest_graph

      - name: UpPass
        operation: unpool_then_weighted_reverse_conv_then_gmp

      - name: SkipFusion
        operation: add_downstream_skip_features


  interface:

    parameters:

      unet_depth:
        type: int
        description: U-Net 深度

      latent_dim:
        type: int
        description: 特征维度

      hidden_layer:
        type: int
        description: MLP 隐层层数

      pos_dim:
        type: int
        description: 位置维度

    inputs:

      h:
        type: NodeEmbedding
        shape: [N, C] or [B, N, C]
        dtype: float32
        description: 输入节点特征

      m_ids:
        type: MultiScaleIndex
        description: 多尺度下采样索引列表

      m_gs:
        type: MultiScaleEdgeIndex
        description: 多尺度图边索引列表

      pos:
        type: PositionEmbedding
        shape: [N, pos_dim] or [B, N, pos_dim]
        description: 节点坐标

    outputs:

      h_out:
        type: NodeEmbedding
        description: 更新后的节点特征


  types:

    NodeEmbedding:
      shape: [N, C] or [B, N, C]
      description: 节点 embedding

    PositionEmbedding:
      shape: [N, pos_dim] or [B, N, pos_dim]
      description: 位置 embedding

    MultiScaleIndex:
      description: 各层下采样索引集合

    MultiScaleEdgeIndex:
      description: 各层图结构集合


  implementation:

    framework: pytorch

    code: |

      class BistrideGraphMessagePassing(nn.Module):
          # Down pass -> Bottom pass -> Up pass with skip connections
          ...

      class GraphMessagePassing(nn.Module):
          # 构造边特征并 scatter_sum 聚合到节点，再残差更新
          ...

      class WeightedEdgeConv(nn.Module):
          # 根据边权重做加权聚合，并提供 cal_ew 计算权重
          ...

      class Unpool(nn.Module):
          # 将 coarse 特征按 idx 写回 fine 图
          ...


  skills:

    build_bistride_processor:

      description: 构建多尺度图 U-Net 风格处理器

      inputs:
        - unet_depth
        - latent_dim
        - hidden_layer
        - pos_dim

      prompt_template: |

        构建 BistrideGraphMessagePassing 模块。

        参数：
        unet_depth = {{unet_depth}}
        latent_dim = {{latent_dim}}
        hidden_layer = {{hidden_layer}}
        pos_dim = {{pos_dim}}

        要求：
        实现 down/bottom/up 三阶段并保留 skip 连接。


    diagnose_bistride_processor:

      description: 分析分层图处理器中的索引与聚合问题

      checks:
        - incorrect_multiscale_indices
        - scatter_dim_mismatch
        - unpool_target_size_error
        - unstable_edge_weight_normalization


  knowledge:

    usage_patterns:

      hierarchical_message_passing:

        pipeline:
          - DownsampleGraph
          - ProcessCoarseGraph
          - UpsampleAndFuse

      graph_unet_style:

        pipeline:
          - EncoderGMP
          - BottleneckGMP
          - DecoderGMP


    hot_models:

      - model: Graph U-Net Family
        year: 2019
        role: 分层图编码解码框架参考
        architecture: hierarchical_gnn

      - model: MeshGraph-based Dynamics Models
        year: 2021
        role: 网格图多尺度消息传递
        architecture: mesh_gnn


    best_practices:

      - 严格校验 m_ids 与 m_gs 的层级对应关系。
      - 对加权边卷积中的度归一化添加数值稳定 eps。
      - 上采样后立即做 skip 融合以恢复高分辨率细节。


    anti_patterns:

      - 在不同层级混用边索引，导致拓扑错配。
      - 忽略 batch/非 batch 两种输入分支维度处理。


    paper_references:

      - title: "Graph U-Nets"
        authors: Gao and Ji
        year: 2019

      - title: "Learning Mesh-Based Simulation with Graph Networks"
        authors: Pfaff et al.
        year: 2021


  graph:

    is_a:
      - NeuralNetworkComponent
      - HierarchicalGraphProcessor

    part_of:
      - ProcessorModuleSystem
      - MultiscaleGNNPipelines

    depends_on:
      - GraphMessagePassing
      - WeightedEdgeConv
      - Unpool
      - ScatterSum

    variants:
      - BistrideGraphMessagePassing
      - GraphMessagePassing

    used_in_models:
      - Hierarchical Mesh GNNs
      - Graph Dynamics Models
      - Weather Graph Processors

    compatible_with:

      inputs:
        - NodeEmbedding
        - PositionEmbedding
        - MultiScaleIndex
        - MultiScaleEdgeIndex

      outputs:
        - NodeEmbedding