component:

  meta:
    name: FourierPosEmbedding
    alias: FourierPositionEncoder
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: embedding
    author: OneScience
    license: Apache-2.0
    tags:
      - embedding
      - fourier_features
      - positional_encoding
      - nerf_style
      - graph_neural_network
      - clustering

  concept:

    description: >
      傅里叶特征位置编码器（FourierPosEmbedding）负责生成基于连续坐标的高频傅里叶位置编码。
      它不仅计算节点的绝对位置编码，同时根据节点所属的簇（Cluster），计算节点相对于簇中心的相对位置距离，并将这两者的傅里叶特征拼接，为下游模型提供丰富的全局坐标与局部相对几何依赖信息。

    intuition: >
      在处理连续空间的坐标（例如 3D Mesh 或纯物理空间坐标）时，普通的线性映射由于偏向低频信号，难以捕捉高频剧变的细节（Spectral Bias）。
      利用不同频率带 (2^k) 强行将低维坐标映射到高维空间的三角函数值（类似于 NeRF 的位置编码），相当于给坐标装上了显微镜，不仅放大低频全局轮廓，更突出了细微尺度的局部差异。由于结合了簇结构，编码还自然地融入了局部相对视角。

    problem_it_solves:
      - 缓解多层感知机 (MLP) 固有的坐标输入“谱偏差”（Spectral Bias），捕获高频细节
      - 强化模型对图形/网格结构数据的图局部性（相对簇中心的局部坐标系）感知能力
      - 聚合生成多尺度维度的坐标特征信息以替代简单的线性坐标输入


  theory:

    formula:

      fourier_feature_mapping:
        expression: |
          \gamma(p)_k = [\cos(2^{k+start} \cdot \pi \cdot p), \sin(2^{k+start} \cdot \pi \cdot p)]
      
      relative_distance:
        expression: |
          \Delta p_{i} = C_{k} - p_{i} \quad \text{for } p_i \in \text{Cluster}_k

    variables:

      p:
        name: PositionalCoordinate
        description: 原始输入的物理空间坐标值（如 x, y, z 等）

      k:
        name: FrequencyScaleIndex
        description: 控制特征层级的频率幂次索引值

      start:
        name: PositionStart
        description: 最小频率缩放参数的偏移量，决定捕获信号的最低频率


  structure:

    architecture: positional_encoding_components

    pipeline:

      - name: ClusterCenterCalculation
        operation: gather_coordinates_and_average

      - name: RelativeDistanceComputation
        operation: center_minus_node_coordinates

      - name: FourierFeatureEmbedding
        operation: map_to_sin_cos_space

      - name: FeatureAggregation
        operation: scatter_and_concatenate (绝对特征与相对特征拼接)


  interface:

    parameters:

      pos_start:
        type: int
        description: 频率指数的起始偏移量（控制最低频率范围）

      pos_length:
        type: int
        description: 生成特征的频率带的数量（控制最高频以及产生的特征截面大小）

    inputs:

      mesh_pos:
        type: Tensor
        shape: [B, N, D]
        dtype: float32
        description: 输入拓扑或网格结构的节点坐标表示

      clusters:
        type: Tensor
        shape: [B, K, C_max]
        dtype: int64
        description: 各簇分配的节点索引（用于 Scatter/Gather）

      cluster_mask:
        type: Tensor
        shape: [B, K, C_max]
        description: 簇掩码，去除补齐产生的无效分配节点

    outputs:

      nodes_embedding:
        type: Tensor
        shape: [B, N, P_total]
        description: 拼接了绝对位置和针对簇的相对位置编码后的整体节点特征

      cluster_embedding:
        type: Tensor
        shape: [B, K, P_embed]
        description: 生成的簇中心绝对位置编码特征


  types:

    NodeCoordinates:
      shape: [B, N, D]
      description: 批次中包含 N 个节点的连续物理空间坐标


  implementation:

    framework: pytorch

    code: |
      import torch
      import torch.nn as nn
      
      class FourierPosEmbedding(nn.Module):
          """
              傅里叶特征位置编码器。
      
              该模块生成基于坐标的傅里叶位置编码。它不仅计算节点的绝对位置编码，还计算节点相对于其所属簇中心的相对位置编码。
              编码公式类似于 NeRF 中的位置编码：gamma(p) = [cos(2^k * pi * p), sin(2^k * pi * p)]。
      
              Args:
                  pos_start (int): 频率指数的起始值（控制最低频率）。
                  pos_length (int): 频率带的数量（控制频带宽度和输出维度）。
      
              形状:
                  输入 mesh_pos: (B, N, D)，节点坐标。
                  输入 clusters: (B, K, C_max)，簇分配索引。
                  输入 cluster_mask: (B, K, C_max)，簇掩码。
                  输出 nodes_embedding: (B, N, P_total)，包含绝对位置和相对位置的节点编码。
                  输出 cluster_embedding: (B, K, P_embed)，簇中心的绝对位置编码。
      
              Example:
                  >>> pos_enc = FourierPosEmbedding(pos_start=-3, pos_length=8)
                  >>> # node_emb, cluster_emb = pos_enc(pos, clusters, mask)
          """
          def __init__(self, pos_start, pos_length):
              super(FourierPosEmbedding, self).__init__()
              self.pos_length = pos_length
              self.pos_start = pos_start
      
          def forward(self, mesh_pos, clusters, cluster_mask):
              """
              参数:
                  mesh_pos: 节点坐标 [B, N, D]
                  clusters: 簇分配索引 [B, K, C_max]
                  cluster_mask: 簇掩码
              返回:
                  nodes_embedding: 节点位置特征 (绝对+相对) [B, N, P_total]
                  cluster_embedding: 簇中心位置特征 [B, K, P_embed]
              """
              B, N, _ = mesh_pos.shape
              _, K, C_max = clusters.shape
      
              # 收集簇内节点的坐标
              meshpos_gather_idx = clusters.reshape(B, -1, 1).repeat(1, 1, mesh_pos.shape[-1])
              meshpos_by_cluster = torch.gather(mesh_pos, -2, meshpos_gather_idx)
              meshpos_by_cluster = meshpos_by_cluster.reshape(*clusters.shape, -1)
      
              # 计算簇中心 
              clusters_centers = meshpos_by_cluster.sum(dim=-2)
              clusters_centers = clusters_centers / (
                  cluster_mask.sum(-1, keepdim=True) + 1e-8
              )
      
              # 计算相对距离
              distances_to_cluster = clusters_centers.unsqueeze(-2) - meshpos_by_cluster
              
              # 编码相对位置
              pos_embeddings = self.embed(distances_to_cluster)
              S = pos_embeddings.shape[-1]
              
              # 将相对位置编码 Scatter 回节点顺序
              pos_embeddings_flat = pos_embeddings.reshape(B, -1, S)
              cluster_indices_flat = clusters.reshape(B, -1, 1).repeat(1, 1, S)
              
              relative_positions = torch.zeros(B, max(N, cluster_indices_flat.shape[1]), S, device=mesh_pos.device)
              relative_positions = relative_positions.scatter(
                  -2,
                  cluster_indices_flat,
                  pos_embeddings_flat,
              )
              relative_positions = relative_positions[:, :N]
      
              # 编码绝对位置并拼接
              nodes_embedding = torch.cat([self.embed(mesh_pos), relative_positions], dim=-1)
      
              return nodes_embedding, self.embed(clusters_centers)
      
          def embed(self, pos):
              """
              傅里叶特征映射函数
              Formula: [cos(2^k * pi * p), sin(2^k * pi * p)]
              """
              original_shape = pos.shape
              # 展平以便处理任意维度的输入
              pos = pos.reshape(-1, original_shape[-1])
              
              index = torch.arange(
                  self.pos_start, self.pos_start + self.pos_length, device=pos.device
              )
              index = index.float()
              freq = 2**index * torch.pi # [Length]
              
              # 计算频率项: pos [M, D] * freq [L] -> [M, D, L]
              args = freq.view(1, 1, -1) * pos.unsqueeze(-1)
              
              # 计算 sin 和 cos
              cos_feat = torch.cos(args)
              sin_feat = torch.sin(args)
              
              # 拼接: [M, D, 2*L]
              embedding = torch.cat([cos_feat, sin_feat], dim=-1)
              
              # 恢复原始形状，最后维度变为 D * 2 * Length
              embedding = embedding.view(*original_shape[:-1], -1)
              return embedding

  skills:

    build_fourier_pos_embedding:

      description: 构建兼顾高频细节和相对几何关系的傅里叶位置编码

      inputs:
        - pos_start
        - pos_length

      prompt_template: |

        编写一个基于坐标的傅里叶位置编码组件。
        要求：
        1. 初始化包含频带偏移 (pos_start) 与层级数量 (pos_length)。
        2. 基于节点相对簇中心的连线矢量计算相对傅里叶位姿特征，并通过 scatter/gather 进行形状对齐。
        3. 输入连续坐标返回 cos/sin 频率特征。




    diagnose_fourierposembedding:

      description: 分析 FourierPosEmbedding 模块在使用中的常见失效模式

      checks:
        - padding_or_resolution_errors
        - bad_device_alignment
        - output_shape_collapse
  knowledge:

    usage_patterns:

      graph_learning_pipeline:

        pipeline:
          - Extract Mesh Coordinates
          - Cluster Assignments (Graph Partitioning)
          - FourierPosEmbedding (获得高频几何节点&簇特征)

    hot_models:

      - model: NeRF
        year: 2020
        role: 普及了高频正弦/余弦坐标编码
        architecture: MLP + Positional Encoding

    best_practices:

      - `pos_length` 与目标的坐标空间尺度紧密相关，应保证能覆盖网格的最小网格间距分辨率。
      - 在图计算中，使用相对中心位姿能够帮助网络学习平移不变性（Translation Invariant）表征。


    anti_patterns:

      - 数据未经过归一化便传入基于 $2^k$ 指数爆炸的高频三角函数之中，极易引起数值不稳定与条纹噪声。

    paper_references:

      - title: "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
        authors: Tancik et al.
        year: 2020


  graph:

    is_a:
      - CoordinateEmbedding
      - PositionalEncoding

    part_of:
      - MeshGraphNets
      - NeuralFields

    depends_on:
      - TrigonometricFunctions
      - GatherScatterOperations

    variants:
      - SphericalHarmonicsEncoding

    used_in_models:
      - Keisuke-NeRF
      - GraphCast

    compatible_with:

      inputs:
        - ContinuousCoordinates

      outputs:
        - NodeFeatureTensor