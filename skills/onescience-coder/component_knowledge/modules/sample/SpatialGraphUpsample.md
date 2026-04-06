component:

  meta:
    name: SpatialGraphUpsample
    alias: GeometricNearestNeighborInterpolator
    version: 1.0
    domain: physics_informed_machine_learning
    category: neural_network
    subcategory: graph_upsampling
    author: OneScience
    tags:
      - graph_neural_network
      - point_cloud
      - nearest_neighbor_interpolation
      - unpooling

  concept:

    description: >
      SpatialGraphUpsample 构成了空间图卷积网络系统内的逆过程生成算子，它的目的在于将经过高层池化深度学习后提取到的，蕴含有庞大上下文物理信息的、
      数量极其稀少的低分辨率图谱的顶点特征值，重新映射、“溅射”或扩散回具有原始宏伟规模且稠密的高分辨率物理节点空间网络。
      其底层的理论逻辑高度简化而直接，它不涉及反卷积扩展，而是计算两个解构空间点的欧几里得距离，执行无权重的直接“最近邻（Nearest Neighbor）”状态复制传承。

    intuition: >
      从宏观的天气图向微观站点映射的时候。一旦深层网络算出来某一大块区域的核心气旋特征量（极低分辨率点的隐藏态），那散布在这个核心代表点附近的所有原始实际测绘高密气象站，
      理应无条件继承属于他们所处这一大块区域内的宏观背景值基底状态特征，从而将其与自己本地保留着的高频噪声合并去推演自身的终点物理预估。

    problem_it_solves:
      - 打通高维抽象无结构空间表征与稠密低维坐标的输出鸿沟。
      - 纯坐标基准处理手段解决在降噪图架构中的 Unpool 特征扩散重铺难题。
      - 使计算量降低到通过 Kd-Tree 或者单纯欧几里得坐标近似计算的最简矩阵代价之中。

  theory:

    formula:
      nearest_neighbor_search:
        expression: |
          j^*_i = \arg\min_{j \in M} \| \text{pos\_up}_i - \text{pos\_down}_j \|_2
          X_{\text{up}}[i] = X_{\text{down}}[j^*_i] \quad \forall i \in \{1, 2, \dots, N\}

    variables:
      \text{pos\_up}:
        name: HighResCoordinates
        description: 期待还原至原本大数量等级位置坐标的三维地球地理向量映射点云 (N > M)
      
      \text{pos\_down}:
        name: LowResCoordinates
        description: 从深层模型端反馈出来的寥寥无几但极具抽象特征的少部分宏观站点的三维空间落点 (M < N)

  structure:

    architecture: graph_nearest_interpolation

    pipeline:
      - name: AdjacencyDistanceSearch
        operation: `nng.nearest(target_coords, source_coords)`
      - name: DirectFeatureAssignment
        operation: Broadcast or index slice tensor representations

  interface:

    parameters:
      none:
        description: 本模块作为纯粹插值逻辑门和运算组件类不附带和产生任何需要优化的学习权重系数，因此是完全的无状态网络门（Stateless Layer）。

    inputs:
      x_down:
        type: Tensor
        shape: [M, C]
        description: 低分辨率状态下的图提取特征向量集
        
      pos_down:
        type: Tensor
        shape: [M, D]
        description: 低级稀疏抽象层点的拓朴地理位置张量。

      pos_up:
        type: Tensor
        shape: [N, D]
        description: 原本的高分辨率待补足插点地理占位位置分布图。

    outputs:
      result:
        type: Tensor
        shape: [N, C]
        description: N个顶点被分别从各自所最近的中心簇(cluster)所取色填涂填充完本以后的致密物理状态系。

  implementation:

    framework: pytorch_geometric

    code: |
      import torch
      import torch.nn as nn
      import torch_geometric.nn as nng
      import random
      
      class SpatialGraphUpsample(nn.Module):
      
          """
          空间图上采样模块 (Spatial Graph Upsampling)。
      
          该模块对应于 CNN 中的 Unpooling 或 Upsample。
          它利用最近邻插值 (Nearest Neighbor Interpolation) 将低分辨率图的特征映射回高分辨率图的结构中。
          对于高分辨率图中的每个节点，寻找其在低分辨率图中空间距离最近的节点，并复制其特征。
      
          Args:
              无参数 (无状态模块)。
      
          形状:
              输入 x_down: (M, C)，低分辨率特征。
              输入 pos_down: (M, D)，低分辨率坐标。
              输入 pos_up: (N, D)，高分辨率坐标 (目标位置)。
              输出 x_up: (N, C)，上采样后的特征。
      
          Example:
              >>> upsample = SpatialGraphUpsample()
              >>> x_down = torch.randn(50, 32)   # 50 个点
              >>> pos_down = torch.randn(50, 2)
              >>> pos_up = torch.randn(100, 2)   # 恢复到 100 个点
              >>> x_up = upsample(x_down, pos_down, pos_up)
              >>> print(x_up.shape)
              torch.Size([100, 32])
          """
          def __init__(self):
              super().__init__()
      
          def forward(self, x_down, pos_down, pos_up):
              # cluster[i] 是 pos_up[i] 在 pos_down 中最近邻的索引
              cluster = nng.nearest(pos_up, pos_down)
              x_up = x_down[cluster]
              return x_up

  skills:

    build_spatialgraphupsample:

      description: 构建支持物理约束的三维/二维降维或插值组件，整合上下文环境与特征通道

      inputs:
        - input_resolution
        - output_resolution
        - in_dim
        - out_dim

      prompt_template: |

        构建名为 SpatialGraphUpsample 的气象特征采样模块。

        参数：
        输入分辨率 = {{{input_resolution}}}
        输出分辨率 = {{{output_resolution}}}
        特征维度 = {{{in_dim}}}
        预期输出维度 = {{{out_dim}}}

        要求：
        必须严格遵守该网络结构的物理特征（例如三维气压层分离、拓扑点云近邻不损失等）恒定映射规律。


    diagnose_tensor_shape:

      description: 调试在不同尺度缩放或维数跳转（如 .permute 或 .reshape 等多级重组）时出现的维度不匹配错误

      checks:
        - shape_mismatch_at_boundaries
        - incorrect_permute_strides
        - loss_of_physical_meaning (例如跨气压层或跨时间维的污染混合)


  knowledge:

    usage_patterns:

      spatial_scaling_framework:
        description: 控制多尺度空间特征提取的标准管线
        pipeline:
          - Extract_Macro_Features (利用DownSample提取低频气候抽象)
          - Message_Passing (中心GNN或Transformer处理)
          - Interpolate_Back (利用UpSample恢复至高频物理网格)
          
      multiscale_processor:
        pipeline:
          - Encoder_Block
          - DownSample_Layer
          - Decoder_Block
          - UpSample_Layer


    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 提供基于地球表面的 3D Patch Merging 的空间降维/升维范式
        architecture: 3D Earth-Specific Transformer

      - model: GraphCast
        year: 2023
        role: 提供基于 Mesh 的不规则点云抽象以及边重构拓扑网络
        architecture: Hierarchical Graph Neural Network

      - model: FuXi
        year: 2023
        role: 提供基于经典可变形卷积和群归一化的级联采样结构
        architecture: Cascaded Swin-Transformer + CNN 


    best_practices:

      - 结合 Skip-Connection 获取原有高分层属性使用单纯最短距离 nearest 实现赋值扩散。
      - 必须保证上采样输出与同层级下采样输出维度与分辨率完全匹配以备特征融合。
      - 所有的池化步幅和卷积核需要匹配地球环面展开特性（极点、赤道），处理由于180°与360°分辨率的不整除造成的尺寸零头。


    anti_patterns:

      - 在大规模 Mesh 上使用自己编写的距离矩阵导致 OOM，应直接利用 torch_geometric 的 C++ 内构 nearest 计算。
      - 不考虑任何物理约束的简单 AdaptiveAvgPool2d 会让复杂地理特征严重失真。
      - 忽略对多层级气压、高度、地表层的数据异构型差异盲目共用池化算子。


    paper_references:

      - title: "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast"
        authors: Bi et al.
        year: 2023

      - title: "GraphCast: Learning skillful medium-range global weather forecasting"
        authors: Lam et al.
        year: 2023

      - title: "FuXi: A cascade machine learning forecasting system for 15-day global weather forecast"
        authors: Chen et al.
        year: 2023

  graph:
    is_a:
      - InterpolationLayer
      - GeometricUpsampler
    part_of:
      - MeshGraphNet
      - GraphCastModel
    depends_on:
      - torch_geometric.nn.nearest
    compatible_with:
      - SpatialGraphDownsample