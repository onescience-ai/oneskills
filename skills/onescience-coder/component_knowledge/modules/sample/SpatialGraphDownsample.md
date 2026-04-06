component:

  meta:
    name: SpatialGraphDownsample
    alias: MeshPoolingDownsampler
    version: 1.0
    domain: physics_informed_machine_learning
    category: neural_network
    subcategory: graph_pooling
    author: OneScience
    tags:
      - graph_neural_network
      - point_cloud
      - spatial_pooling
      - mesh_graph_net

  concept:

    description: >
      SpatialGraphDownsample 是专为非结构化几何拓扑空间（例如气候模型中的非均匀测站和多面体表面网格 Point_Cloud）设计的池化和提取工具。
      由于气象传感器或者 CFD 求解器的输出多为不均匀分布的空间点，无法使用标准卷积和 Transformer 的 $2 \times 2$ Token 合并来压维。
      本模块负责：(1) 使用打分机制（TopK）或统计手段随机去除部分图节点，减少总体节点数比例。(2) 围绕遗留下来的“骨干中心点”，结合设定的空间通信距离（$r$）动态重新拉建新的 Radius Graph 边拓扑网络网，从而使得下层网络能够在更加抽象并且更宏大的物理邻域上学习到低频率主导传播特征。

    intuition: >
      想象全国的雨量测量站有密有疏（东部极密，西部极少）。通过池化系统，我们不将其强制配准为不准确的高低像素框网格，
      而是保留重要地貌节点，砍掉重合密集的冗余计算点。因为留下的站点间距拉大了，原有的边已断，我们需要立刻用“搜寻方圆 $R$ 公里的邻居”这把新尺子把剩下的少数精英站点重新搭上线，建立出中国气候的宏观传输网络。

    problem_it_solves:
      - 将极其复杂的任意曲面或流形成块降维。
      - 维持无标度结构数据的排列与平移不变形特性（Permutation Invariance）。
      - 在图降维后重建新的连通性拓扑支撑以供下游 GNN 执行信息传递消息层。

  theory:

    formula:

      top_k_scoring:
        expression: |
          \text{score} = \frac{x \cdot W_{proj}}{\|W_{proj}\|}
          \text{idx} = \text{Top}_{K}(\text{score}, k=N \times ratio) 
          x_{out} = x[\text{idx}] \cdot \sigma(\text{score}[\text{idx}])

      random_survival:
        expression: |
          \text{idx} = \text{RandPerm}(N)[0 : N \times ratio]
          x_{out} = x[\text{idx}]

      radius_graph_reconnect:
        expression: |
          E_{\text{new}} = \{ (i,j) \mid \| \text{pos}[i] - \text{pos}[j] \|_2 \le r \}

    variables:
      r:
        name: RadiusThreshold
        description: 欧几里德距离限界，当存活下来的点在三维世界的距离小于此变量时重新建边。

  structure:

    architecture: graph_spatial_pooling

    pipeline:
      - name: StrategyConditioning
        operation: "If 'topk' -> GraphPooling Score calculation. If 'random' -> Uniform sampling."
      - name: NodeSlicing
        operation: "extract `x_pooled` and `pos_pooled` via index array `perm`"
      - name: TopologyReconstruction
        operation: "torch_geometric.nn.radius_graph(...) -> Create entirely new edge_index based on geometric proximity."

  interface:

    parameters:

      in_channels:
        type: int
        description: 输入节点特征维数（为使用投影权重的 TopK 计算所依赖）。

      ratio:
        type: float
        default: 0.5
        description: 在图规模缩减之后应当保留下来的顶点的存活比率。

      r:
        type: float
        default: 0.1
        description: 池化抛弃多余点之后，由于距离增加，重新连结网格点所用的搜寻半径距离极限。

      max_num_neighbors:
        type: int
        default: 64
        description: 控制单新点的出入度连接总量，预防图爆炸（过于稠密化导致显存 OOM 崩溃）。

      pool_method:
        type: str
        default: 'random'
        description: 抽点战略类型选择分支: 'random' 或者 'topk'。

    inputs:
      x:
        type: Tensor
        shape: [N, C]
        description: 当下图谱上的全部点信道表征。
        
      pos:
        type: Tensor
        shape: [N, D]
        description: N个图中空间存在的 D 维地理定位坐标点云。

    outputs:
      result:
        type: Tuple[Tensor, Tensor, Tensor, Tensor]
        shape: "(x_pooled, pos_pooled, edge_index_pooled, perm)"
        description: 返回高度压缩的属性点与新建立出边的连接组合，以及抽取追踪序列。

  implementation:

    framework: pytorch_geometric

    code: |
      import torch
      import torch.nn as nn
      import torch_geometric.nn as nng
      import random
      
      class SpatialGraphDownsample(nn.Module):
          """
          空间图下采样模块 (Spatial Graph Downsampling)。
      
          该模块执行两个核心操作：
          1. **节点选择 (Pooling)**: 减少节点数量。支持 'random' (随机采样) 或 'topk' (基于投影分数的 TopKPooling)。
          2. **图拓扑重构 (Topology Reconstruction)**: 基于采样后节点的空间位置，利用半径图 (Radius Graph) 算法重新构建邻接关系。
          
          这模拟了 CNN 中的 Pooling 操作，但在非结构化几何数据（如点云、网格）上进行。
      
          Args:
              in_channels (int): 输入特征维度（仅当 pool_method='topk' 时需要，用于计算投影分数）。
              ratio (float, optional): 池化比率 (保留节点的比例)。默认值: 0.5。
              r (float, optional): 半径图构建时的半径阈值。默认值: 0.1。
              max_num_neighbors (int, optional): 每个节点的最大邻居数。默认值: 64。
              pool_method (str, optional): 池化方法，支持 'random' 或 'topk'。默认值: 'random'。
      
          形状:
              输入 x: (N, C)，节点特征。
              输入 pos: (N, D)，节点坐标 (通常 D=2 或 3)。
              输入 edge_index (可选): (2, E)，当使用 'topk' 时需要原始边信息。
              输出 x_pooled: (M, C)，下采样后的特征，其中 M = N * ratio。
              输出 pos_pooled: (M, D)，下采样后的坐标。
              输出 edge_index_pooled: (2, E_new)，重构后的邻接矩阵。
              输出 perm: (M,)，被选中节点的索引。
      
          Example:
              >>> # 假设有 100 个节点，特征维度 32，2D 坐标
              >>> downsample = SpatialGraphDownsample(in_channels=32, ratio=0.5, r=0.2, pool_method='random')
              >>> x = torch.randn(100, 32)
              >>> pos = torch.randn(100, 2)
              >>> x_pool, pos_pool, edge_index_pool, perm = downsample(x, pos)
              >>> print(x_pool.shape)
              torch.Size([50, 32])
          """
          def __init__(self, in_channels, ratio=0.5, r=0.1, max_num_neighbors=64, pool_method='random'):
              super().__init__()
              self.ratio = ratio
              self.r = r
              self.max_num_neighbors = max_num_neighbors
              self.pool_method = pool_method
      
              if self.pool_method == 'topk':
                  self.scorer = nng.TopKPooling(in_channels, ratio=ratio, nonlinearity=torch.sigmoid)
              else:
                  self.scorer = None
      
          def forward(self, x, pos, edge_index=None):
              num_nodes = x.size(0)
      
              if self.scorer is not None:
                  # TopK Pooling: 需要 edge_index 来传播分数
                  if edge_index is None:
                      raise ValueError("edge_index is required for TopK pooling")
                  # x, edge_index, edge_attr, batch, perm, score
                  x_pooled, _, _, _, perm, _ = self.scorer(x, edge_index)
              else:
                  # Random Pooling
                  k = int((self.ratio * float(num_nodes)))
                  # 保持 device 一致
                  perm = torch.randperm(num_nodes, device=x.device)[:k]
                  x_pooled = x[perm]
      
              pos_pooled = pos[perm]
      
              # 基于新的空间位置重构图结构
              edge_index_pooled = nng.radius_graph(
                  x=pos_pooled,
                  r=self.r,
                  loop=True,
                  max_num_neighbors=self.max_num_neighbors
              )
      
              return x_pooled, pos_pooled, edge_index_pooled, perm

  skills:

    build_spatialgraphdownsample:

      description: 构建支持物理约束的三维/二维降维或插值组件，整合上下文环境与特征通道

      inputs:
        - input_resolution
        - output_resolution
        - in_dim
        - out_dim

      prompt_template: |

        构建名为 SpatialGraphDownsample 的气象特征采样模块。

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

      - 池化比例 ratio 不宜过低，伴随重新建图距离 r 必须被适量调节放大，以免产生图孤岛。
      - random 抽点在大型气象场景中往往优于 topk 算法。
      - 必须保证上采样输出与同层级下采样输出维度与分辨率完全匹配以备特征融合。
      - 所有的池化步幅和卷积核需要匹配地球环面展开特性（极点、赤道），处理由于180°与360°分辨率的不整除造成的尺寸零头。


    anti_patterns:

      - 脱离 GPU device 在 CPU 上频繁倒腾计算导致巨大 IO 延迟。
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
      - GraphPoolingLayer
      - GeometricDownsample
    part_of:
      - GraphCastModel
      - MeshGraphNet
    depends_on:
      - torch_geometric.nn.radius_graph
      - torch_geometric.nn.TopKPooling
    compatible_with:
      - SpatialGraphUpsample