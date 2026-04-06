component:

  meta:
    name: GraphCastEncoderEmbedder
    alias: GraphCastEmbedder
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: embedding
    author: OneScience
    license: Apache-2.0
    tags:
      - embedding
      - graph_neural_network
      - graphcast
      - earth_system_modeling
      - msg_pass

  concept:

    description: >
      GraphCast 编码器嵌入层系统（GraphCast Encoder Embedder）负责将气象/图预测模型所需的四种初始特征映射进入高维隐空间（Latent Space）。
      在 GraphCast 等基于多尺度网格（Multi-mesh）的消息传递算法中，图包含了规则的经纬度网格节点 (Grid)、二十面体演化而来的多尺度内部网格节点 (Mesh)、以及互联的边 (Grid2Mesh, Mesh2Mesh)，此模块为它们分别创建了独立的神经映射通道。

    intuition: >
      GraphCast 就像一个物流分拣中心，有4种不同来源或去向的货物（4种不同维度的节点和边特征）。
      普通的 Embedding 可能只有一个进口，而这个 Embedder 拥有 4 条独立但结构一样的专门流水线（MLP），它们把原始尺寸各异、意义不同的特征全部打包、统一重塑成同样规格（如 512维）的“标准集装箱”，方便后续图神经网络（GNN）无缝传递和融合计算。

    problem_it_solves:
      - 统一复杂多部件图网络结构（规则网格与基于二十面体的异构网格）中各种实体的特征尺寸
      - 对异构节点（经纬点、Mesh点）和多向连接边进行解耦但平行的初始特征提取


  theory:

    formula:

      mesh_graph_mlp_mapping:
        expression: |
          Grid_{emb} = \text{MLP}_{grid}(Grid_{feat})
          Mesh_{emb} = \text{MLP}_{mesh}(Mesh_{feat})
          G2M_{emb} = \text{MLP}_{g2m}(G2M_{feat})
          M2M_{emb} = \text{MLP}_{m2m}(M2M_{feat})

    variables:

      Grid_{feat}:
        name: GridNodeFeatures
        description: 原始高分辨率经纬度网格的节点特征（如各种气象态）

      Mesh_{feat}:
        name: MeshNodeFeatures
        description: 多尺度隐空间网格的节点特征（如相对三维坐标的先验）


  structure:

    architecture: parallel_mlp_embedder

    pipeline:

      - name: GridEdgeEmbedding
        operation: dense_mlp_with_silu

      - name: MeshEdgeEmbedding
        operation: dense_mlp_with_silu

      - name: G2MEdgeEmbedding
        operation: dense_mlp_with_silu

      - name: MeshInternalEdgeEmbedding
        operation: dense_mlp_with_silu


  interface:

    parameters:

      input_dim_grid_nodes:
        type: int
        description: 输入规则网格特征的通道维度（例如包含历史步骤特征合集的大维数 474）

      input_dim_mesh_nodes:
        type: int
        description: 输入隐多尺度网格特征的维度，常为空间笛卡尔坐标的 3 维

      input_dim_edges:
        type: int
        description: 各种边特征的初始维数，如 4 (距离、方向等)

      output_dim:
        type: int
        description: 所有独立 MLP 统一投射的输出隐维度（典型为 512）

      hidden_dim:
        type: int
        description: MLP 中间隐藏层的神经元数量（典型为 512）

    inputs:

      grid_nfeat:
        type: Tensor
        shape: [N_{grid}, input_dim_{grid\_nodes}]

      mesh_nfeat:
        type: Tensor
        shape: [N_{mesh}, input_dim_{mesh\_nodes}]

      g2m_efeat:
        type: Tensor
        shape: [N_{g2m}, input_dim_{edges}]

      mesh_efeat:
        type: Tensor
        shape: [N_{mesh\_edges}, input_dim_{edges}]

    outputs:

      grid_nfeat_emb:
        type: Tensor
        shape: [N_{grid}, output_dim]

      mesh_nfeat_emb:
        type: Tensor
        shape: [N_{mesh}, output_dim]

      g2m_efeat_emb:
        type: Tensor
        shape: [N_{g2m}, output_dim]

      mesh_efeat_emb:
        type: Tensor
        shape: [N_{mesh\_edges}, output_dim]


  types:

    GraphFeaturesTuple:
      shape: Tuple of 4 Tensors
      description: 包含对应图组件特征的高维向量集合


  implementation:

    framework: pytorch

    code: |
      from typing import Tuple
      import torch.nn as nn
      from torch import Tensor
      
      from onescience.modules.mlp.mesh_graph_mlp import MeshGraphMLP
      
      class GraphCastEncoderEmbedder(nn.Module):
          """
          GraphCast 编码器嵌入层 (GraphCast Encoder Embedder)。
      
          该模块负责将 GraphCast 模型所需的四类输入特征映射到高维隐空间：
          1. 网格节点特征 (Grid Node Features)
          2. 多尺度网格节点特征 (Multi-mesh Node Features)
          3. 网格到多尺度网格的边特征 (Grid2Mesh Edge Features)
          4. 多尺度网格内部边特征 (Multi-mesh Edge Features)
      
          每个特征都通过一个独立的 MLP 进行嵌入。
      
          Args:
              input_dim_grid_nodes (int, optional): 网格节点特征的输入维度。默认值: 474。
              input_dim_mesh_nodes (int, optional): 多尺度网格节点特征的输入维度。默认值: 3。
              input_dim_edges (int, optional): 边特征的输入维度。默认值: 4。
              output_dim (int, optional): 嵌入后的特征维度 (Latent Dim)。默认值: 512。
              hidden_dim (int, optional): MLP 隐藏层神经元数量。默认值: 512。
              hidden_layers (int, optional): MLP 隐藏层层数。默认值: 1。
              activation_fn (nn.Module, optional): 激活函数类型。默认值: nn.SiLU()。
              norm_type (str, optional): 归一化类型。默认值: "LayerNorm"。
              recompute_activation (bool, optional): 是否在反向传播中重计算激活以节省显存。默认值: False。
      
          形状:
              输入 grid_nfeat: (N_grid, input_dim_grid_nodes)
              输入 mesh_nfeat: (N_mesh, input_dim_mesh_nodes)
              输入 g2m_efeat: (N_g2m, input_dim_edges)
              输入 mesh_efeat: (N_mesh_edges, input_dim_edges)
              输出: 返回一个包含四个张量的元组，形状分别为:
                  - (N_grid, output_dim)
                  - (N_mesh, output_dim)
                  - (N_g2m, output_dim)
                  - (N_mesh_edges, output_dim)
      
          Example:
              >>> embedder = GraphCastEncoderEmbedder(
              ...     input_dim_grid_nodes=474,
              ...     input_dim_mesh_nodes=3,
              ...     input_dim_edges=4,
              ...     output_dim=512
              ... )
              >>> grid_n = torch.randn(100, 474)
              >>> mesh_n = torch.randn(50, 3)
              >>> g2m_e = torch.randn(200, 4)
              >>> mesh_e = torch.randn(300, 4)
              >>> out_grid, out_mesh, out_g2m, out_mesh_e = embedder(grid_n, mesh_n, g2m_e, mesh_e)
              >>> print(out_grid.shape)
              torch.Size([100, 512])
          """
      
          def __init__(
              self,
              input_dim_grid_nodes: int = 474,
              input_dim_mesh_nodes: int = 3,
              input_dim_edges: int = 4,
              output_dim: int = 512,
              hidden_dim: int = 512,
              hidden_layers: int = 1,
              activation_fn: nn.Module = nn.SiLU(),
              norm_type: str = "LayerNorm",
              recompute_activation: bool = False,
          ):
              super().__init__()
      
              # MLP for grid node embedding
              self.grid_node_mlp = MeshGraphMLP(
                  input_dim=input_dim_grid_nodes,
                  output_dim=output_dim,
                  hidden_dim=hidden_dim,
                  hidden_layers=hidden_layers,
                  activation_fn=activation_fn,
                  norm_type=norm_type,
                  recompute_activation=recompute_activation,
              )
      
              # MLP for mesh node embedding
              self.mesh_node_mlp = MeshGraphMLP(
                  input_dim=input_dim_mesh_nodes,
                  output_dim=output_dim,
                  hidden_dim=hidden_dim,
                  hidden_layers=hidden_layers,
                  activation_fn=activation_fn,
                  norm_type=norm_type,
                  recompute_activation=recompute_activation,
              )
      
              # MLP for mesh edge embedding
              self.mesh_edge_mlp = MeshGraphMLP(
                  input_dim=input_dim_edges,
                  output_dim=output_dim,
                  hidden_dim=hidden_dim,
                  hidden_layers=hidden_layers,
                  activation_fn=activation_fn,
                  norm_type=norm_type,
                  recompute_activation=recompute_activation,
              )
      
              # MLP for grid2mesh edge embedding
              self.grid2mesh_edge_mlp = MeshGraphMLP(
                  input_dim=input_dim_edges,
                  output_dim=output_dim,
                  hidden_dim=hidden_dim,
                  hidden_layers=hidden_layers,
                  activation_fn=activation_fn,
                  norm_type=norm_type,
                  recompute_activation=recompute_activation,
              )
      
          def forward(
              self,
              grid_nfeat: Tensor,
              mesh_nfeat: Tensor,
              g2m_efeat: Tensor,
              mesh_efeat: Tensor,
          ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
              # Input node feature embedding
              grid_nfeat = self.grid_node_mlp(grid_nfeat)
              mesh_nfeat = self.mesh_node_mlp(mesh_nfeat)
              # Input edge feature embedding
              g2m_efeat = self.grid2mesh_edge_mlp(g2m_efeat)
              mesh_efeat = self.mesh_edge_mlp(mesh_efeat)
              return grid_nfeat, mesh_nfeat, g2m_efeat, mesh_efeat
      
      
      class GraphCastDecoderEmbedder(nn.Module):
          """
          GraphCast 解码器嵌入层 (GraphCast Decoder Embedder)。
      
          
      
          该模块用于将多尺度网格回到原始网格 (Mesh2Grid) 的边特征进行嵌入。
          这是 GraphCast 解码过程的第一步，用于处理从 latent mesh 传回 grid 的信息。
      
          Args:
              input_dim_edges (int, optional): 输入边特征的维度。默认值: 4。
              output_dim (int, optional): 嵌入后的特征维度。默认值: 512。
              hidden_dim (int, optional): MLP 隐藏层神经元数量。默认值: 512。
              hidden_layers (int, optional): MLP 隐藏层层数。默认值: 1。
              activation_fn (nn.Module, optional): 激活函数类型。默认值: nn.SiLU()。
              norm_type (str, optional): 归一化类型 ["TELayerNorm", "LayerNorm"]。默认值: "LayerNorm"。
              recompute_activation (bool, optional): 是否重计算激活。默认值: False。
      
          形状:
              输入 m2g_efeat: (N_m2g, input_dim_edges)，Mesh2Grid 的边特征。
              输出: (N_m2g, output_dim)，嵌入后的边特征。
      
          Example:
              >>> embedder = GraphCastDecoderEmbedder(input_dim_edges=4, output_dim=512)
              >>> m2g_edge = torch.randn(150, 4)
              >>> out = embedder(m2g_edge)
              >>> print(out.shape)
              torch.Size([150, 512])
          """
      
          def __init__(
              self,
              input_dim_edges: int = 4,
              output_dim: int = 512,
              hidden_dim: int = 512,
              hidden_layers: int = 1,
              activation_fn: nn.Module = nn.SiLU(),
              norm_type: str = "LayerNorm",
              recompute_activation: bool = False,
          ):
              super().__init__()
      
              # MLP for mesh2grid edge embedding
              self.mesh2grid_edge_mlp = MeshGraphMLP(
                  input_dim=input_dim_edges,
                  output_dim=output_dim,
                  hidden_dim=hidden_dim,
                  hidden_layers=hidden_layers,
                  activation_fn=activation_fn,
                  norm_type=norm_type,
                  recompute_activation=recompute_activation,
              )
      
          def forward(
              self,
              m2g_efeat: Tensor,
          ) -> Tensor:
              m2g_efeat = self.mesh2grid_edge_mlp(m2g_efeat)
              return m2g_efeat

  skills:

    build_graphcast_embedder:

      description: 构建具有多输入通道的图拓扑解耦 MLP 组合提取器

      inputs:
        - feature_dims
        - output_dim
        - mlp_structure

      prompt_template: |

        请实现 GraphCastEncoderEmbedder。
        需要为网格节点、网格与二十面体边等四种实体提供平行的多级感知层映射结构。
        隐藏激活使用 SiLU。




    diagnose_graphcastencoderembedder:

      description: 分析 GraphCastEncoderEmbedder 模块在使用中的常见失效模式

      checks:
        - padding_or_resolution_errors
        - bad_device_alignment
        - output_shape_collapse
  knowledge:

    usage_patterns:

      graph_message_passing:

        pipeline:
          - Extract 4 Graph Features
          - GraphCastEncoderEmbedder (Mapping to 512 dims)
          - G2M Message Passing
          - Mesh-to-Mesh Message Passing

    hot_models:

      - model: GraphCast
        year: 2023
        role: 顶尖性能的数据驱动气象全图模型，主推多尺度信息的高效传递
        architecture: GNN with Icosahedron Multimesh

    best_practices:

      - 对于极端大规模的图模型训练（节点过万），可采用 `recompute_activation` 机制使用梯度重激活以节省显存。
      - 图模型对边和点的数据非常敏感，分离 MLP 对于不同实体类型进行针对性表达，能够保证节点表征不产生混退或平滑化（Oversmoothing）。


    anti_patterns:

      - 由于有 4 种类型输入特征，容易在 Forward 输入或者解包输出时发生张量位置对齐错乱从而导致训练静默发散。需严格按照 Type Hint 约束和命名传参。

    paper_references:

      - title: "Learning skillful medium-range global weather forecasting"
        authors: Lam et al. (Science)
        year: 2023

  graph:

    is_a:
      - GraphEmbedding
      - FeatureEncoder

    part_of:
      - GraphCast
      - MessagePassingGraphNets

    depends_on:
      - MeshGraphMLP
      - SiLU

    variants:
      - GraphCastDecoderEmbedder (解码网格时的专用映射层)

    used_in_models:
      - GraphCast

    compatible_with:

      inputs:
        - HeterogeneousGraphFeatures

      outputs:
        - LatentGraphFeatures