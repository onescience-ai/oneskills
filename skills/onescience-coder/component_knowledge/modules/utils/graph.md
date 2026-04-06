component:

  meta:
    name: CuGraphCSCWrapper
    alias: graph
    version: 1.0
    domain: ai_for_science
    category: data_structure
    subcategory: graph_neural_network
    author: OneScience
    license: Apache-2.0
    tags:
      - graph_structure
      - csc_format
      - dgl
      - cugraph_ops
      - distributed_graph
      - wrapper

  concept:

    description: >
      该模块实现了一个 `CuGraphCSC` 类，它是一个通用的图数据结构对象，封装了典型的压缩稀疏列（CSC）表示的核心字段（如 offsets 和 indices）。
      它的主要目的是为了方便调用优化的 NVIDIA `cugraph-ops` 底层计算例程，同时也是分布式环境下切分图的便捷包装器。
      它支持从标准的 DGL 图转换而来，并且能够在必要时（如当底层 C++ 算子不可用时）回退转换为 DGL 兼容的数据结构。

    intuition: >
      在图神经网络中，由于图的极度不规则性，直接传递源节点和目标节点的坐标是非常低效的。
      CSC（压缩稀疏列）就像是一本“高效字典”，`offsets` 是目录，告诉你每个目标节点相关的边在哪一页起止；`indices` 则是具体的正文，列出了所有的源节点。
      `CuGraphCSC` 就是这本字典的“精装版保护套”，它负责在内部打点好内存格式、分布式切片以及设备转移，使得上层的 GNN 算子可以直接拿去用，而不需要关心底层的指针和并行同步细节。

    problem_it_solves:
      - 简化和统一高性能图计算后端（CuGraph-Ops）对静态图和二分图（Bipartite Graph）的输入要求
      - 隐藏分布式图计算（Multi-GPU）中复杂的 Global ID 与 Local ID 特征拉取与同步逻辑
      - 提供在不同的图框架（DGL 与 PyTorch Tensors）之间低开销、按需转换的机制

  theory:

    formula:

      csc_indexing:
        expression: \text{neighbors}(j) = \text{indices}[\text{offsets}[j] : \text{offsets}[j+1]]

    variables:

      offsets:
        name: ColumnOffsets
        shape: [num_dst_nodes + 1]
        description: 目标节点的列偏移数组，记录每个节点入边在 indices 数组中的起始和终止位置

      indices:
        name: RowIndices
        shape: [num_edges]
        description: 行索引数组，记录每条边对应的源节点 ID

      ef_indices:
        name: EdgeFeatureIndices
        description: 可选的边特征索引，用于将 CSC 的边顺序映射回最初的 COO（坐标）格式顺序

  structure:

    architecture: graph_data_wrapper

    pipeline:

      - name: InitializationAndPartitionCheck
        operation: check_distributed_group_and_partition_if_needed

      - name: FeatureRouting
        operation: provide_methods_to_fetch_local_or_global_features

      - name: BackendConversion
        operation: convert_to_cugraph_or_dgl_on_the_fly

  interface:

    parameters:

      num_src_nodes:
        type: int
        description: 图中源节点的总数

      num_dst_nodes:
        type: int
        description: 图中目标节点的总数

      partition_size:
        type: int
        default: -1
        description: 分布式图的切分数量。如果 <= 1，则为单卡普通图

      cache_graph:
        type: bool
        default: true
        description: 是否缓存生成的 cugraph-ops 结构以加速重复调用

    inputs:

      offsets:
        type: Tensor
        dtype: int32 or int64
        description: CSC 偏移张量

      indices:
        type: Tensor
        dtype: int32 or int64
        description: CSC 索引张量

    outputs:

      cugraph_object:
        type: CustomClass
        description: 包装好的 CuGraphCSC 实例

  types:

    GraphPartition:
      shape: dataclass
      description: 预先计算好的分布式图切分与映射状态

  implementation:

    framework: pytorch, dgl

    code: |
      from typing import Any, List, Optional

      import dgl
      import torch
      from dgl import DGLGraph
      from torch import Tensor

      try:
          from typing import Self
      except ImportError:
          from typing_extensions import Self

      from onescience.distributed import DistributedManager
      from onescience.modules.utils.distributed_graph import (
          DistributedGraph,
          GraphPartition,
          partition_graph_by_coordinate_bbox,
      )

      try:
          from pylibcugraphops.pytorch import BipartiteCSC, StaticCSC
          USE_CUGRAPHOPS = True
      except ImportError:
          StaticCSC = None
          BipartiteCSC = None
          USE_CUGRAPHOPS = False

      class CuGraphCSC:
          def __init__(
              self,
              offsets: Tensor,
              indices: Tensor,
              num_src_nodes: int,
              num_dst_nodes: int,
              ef_indices: Optional[Tensor] = None,
              reverse_graph_bwd: bool = True,
              cache_graph: bool = True,
              partition_size: Optional[int] = -1,
              partition_group_name: Optional[str] = None,
              graph_partition: Optional[GraphPartition] = None,
          ) -> None:
              self.offsets = offsets
              self.indices = indices
              self.num_src_nodes = num_src_nodes
              self.num_dst_nodes = num_dst_nodes
              self.ef_indices = ef_indices
              self.reverse_graph_bwd = reverse_graph_bwd
              self.cache_graph = cache_graph

              # cugraph-ops structures
              self.bipartite_csc = None
              self.static_csc = None
              # dgl graph
              self.dgl_graph = None

              self.is_distributed = False
              self.dist_csc = None

              if partition_size <= 1:
                  self.is_distributed = False
                  return

              if self.ef_indices is not None:
                  raise AssertionError("DistributedGraph does not support mapping CSC-indices to COO-indices.")

              self.dist_graph = DistributedGraph(
                  self.offsets,
                  self.indices,
                  partition_size,
                  partition_group_name,
                  graph_partition=graph_partition,
              )

              # overwrite graph information with local graph after distribution
              self.offsets = self.dist_graph.graph_partition.local_offsets
              self.indices = self.dist_graph.graph_partition.local_indices
              self.num_src_nodes = self.dist_graph.graph_partition.num_local_src_nodes
              self.num_dst_nodes = self.dist_graph.graph_partition.num_local_dst_nodes
              self.is_distributed = True

          @staticmethod
          def from_dgl(
              graph: DGLGraph,
              partition_size: int = 1,
              partition_group_name: Optional[str] = None,
              partition_by_bbox: bool = False,
              src_coordinates: Optional[torch.Tensor] = None,
              dst_coordinates: Optional[torch.Tensor] = None,
              coordinate_separators_min: Optional[List[List[Optional[float]]]] = None,
              coordinate_separators_max: Optional[List[List[Optional[float]]]] = None,
          ):
              if hasattr(graph, "adj_tensors"):
                  offsets, indices, edge_perm = graph.adj_tensors("csc")
              elif hasattr(graph, "adj_sparse"):
                  offsets, indices, edge_perm = graph.adj_sparse("csc")
              else:
                  raise ValueError("Passed graph object doesn't support conversion to CSC.")

              n_src_nodes, n_dst_nodes = (graph.num_src_nodes(), graph.num_dst_nodes())
              graph_partition = None

              if partition_by_bbox and partition_size > 1:
                  dist_manager = DistributedManager()
                  partition_rank = dist_manager.group_rank(name=partition_group_name)

                  graph_partition = partition_graph_by_coordinate_bbox(
                      offsets.to(dtype=torch.int64),
                      indices.to(dtype=torch.int64),
                      src_coordinates=src_coordinates,
                      dst_coordinates=dst_coordinates,
                      coordinate_separators_min=coordinate_separators_min,
                      coordinate_separators_max=coordinate_separators_max,
                      partition_size=partition_size,
                      partition_rank=partition_rank,
                      device=dist_manager.device,
                  )

              graph_csc = CuGraphCSC(
                  offsets.to(dtype=torch.int64),
                  indices.to(dtype=torch.int64),
                  n_src_nodes,
                  n_dst_nodes,
                  partition_size=partition_size,
                  partition_group_name=partition_group_name,
                  graph_partition=graph_partition,
              )
              return graph_csc, edge_perm

          def get_src_node_features_in_partition(self, global_src_feat: torch.Tensor, scatter_features: bool = False, src_rank: int = 0) -> torch.Tensor:
              if self.is_distributed:
                  return self.dist_graph.get_src_node_features_in_partition(global_src_feat, scatter_features=scatter_features, src_rank=src_rank)
              return global_src_feat

          def get_src_node_features_in_local_graph(self, local_src_feat: torch.Tensor) -> torch.Tensor:
              if self.is_distributed:
                  return self.dist_graph.get_src_node_features_in_local_graph(local_src_feat)
              return local_src_feat

          def get_dst_node_features_in_partition(self, global_dst_feat: torch.Tensor, scatter_features: bool = False, src_rank: int = 0) -> torch.Tensor:
              if self.is_distributed:
                  return self.dist_graph.get_dst_node_features_in_partition(global_dst_feat, scatter_features=scatter_features, src_rank=src_rank)
              return global_dst_feat

          def get_edge_features_in_partition(self, global_efeat: torch.Tensor, scatter_features: bool = False, src_rank: int = 0) -> torch.Tensor:
              if self.is_distributed:
                  return self.dist_graph.get_edge_features_in_partition(global_efeat, scatter_features=scatter_features, src_rank=src_rank)
              return global_efeat

          def get_global_src_node_features(self, local_nfeat: torch.Tensor, get_on_all_ranks: bool = True, dst_rank: int = 0) -> torch.Tensor:
              if self.is_distributed:
                  return self.dist_graph.get_global_src_node_features(local_nfeat, get_on_all_ranks, dst_rank=dst_rank)
              return local_nfeat

          def get_global_dst_node_features(self, local_nfeat: torch.Tensor, get_on_all_ranks: bool = True, dst_rank: int = 0) -> torch.Tensor:
              if self.is_distributed:
                  return self.dist_graph.get_global_dst_node_features(local_nfeat, get_on_all_ranks, dst_rank=dst_rank)
              return local_nfeat

          def get_global_edge_features(self, local_efeat: torch.Tensor, get_on_all_ranks: bool = True, dst_rank: int = 0) -> torch.Tensor:
              if self.is_distributed:
                  return self.dist_graph.get_global_edge_features(local_efeat, get_on_all_ranks, dst_rank=dst_rank)
              return local_efeat

          def to(self, *args: Any, **kwargs: Any) -> Self:
              device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
              if dtype not in (None, torch.int32, torch.int64):
                  raise TypeError(f"Invalid dtype, expected torch.int32 or torch.int64, got {dtype}.")
              self.offsets = self.offsets.to(device=device, dtype=dtype)
              self.indices = self.indices.to(device=device, dtype=dtype)
              if self.ef_indices is not None:
                  self.ef_indices = self.ef_indices.to(device=device, dtype=dtype)
              return self

          def to_bipartite_csc(self, dtype=None) -> BipartiteCSC:
              if not (USE_CUGRAPHOPS):
                  raise RuntimeError("Conversion failed, expected cugraph-ops to be installed.")
              if not self.offsets.is_cuda:
                  raise RuntimeError("Expected the graph structures to reside on GPU.")

              if self.bipartite_csc is None or not self.cache_graph:
                  graph_offsets = self.offsets
                  graph_indices = self.indices
                  graph_ef_indices = self.ef_indices

                  if dtype is not None:
                      graph_offsets = self.offsets.to(dtype=dtype)
                      graph_indices = self.indices.to(dtype=dtype)
                      if self.ef_indices is not None:
                          graph_ef_indices = self.ef_indices.to(dtype=dtype)

                  graph = BipartiteCSC(
                      graph_offsets,
                      graph_indices,
                      self.num_src_nodes,
                      graph_ef_indices,
                      reverse_graph_bwd=self.reverse_graph_bwd,
                  )
                  self.bipartite_csc = graph
              return self.bipartite_csc

          def to_static_csc(self, dtype=None) -> StaticCSC:
              if not (USE_CUGRAPHOPS):
                  raise RuntimeError("Conversion failed, expected cugraph-ops to be installed.")
              if not self.offsets.is_cuda:
                  raise RuntimeError("Expected the graph structures to reside on GPU.")

              if self.static_csc is None or not self.cache_graph:
                  graph_offsets = self.offsets
                  graph_indices = self.indices
                  graph_ef_indices = self.ef_indices

                  if dtype is not None:
                      graph_offsets = self.offsets.to(dtype=dtype)
                      graph_indices = self.indices.to(dtype=dtype)
                      if self.ef_indices is not None:
                          graph_ef_indices = self.ef_indices.to(dtype=dtype)

                  graph = StaticCSC(
                      graph_offsets,
                      graph_indices,
                      graph_ef_indices,
                  )
                  self.static_csc = graph
              return self.static_csc

          def to_dgl_graph(self) -> DGLGraph:
              if self.dgl_graph is None or not self.cache_graph:
                  if self.ef_indices is not None:
                      raise AssertionError("ef_indices is not supported.")
                  graph_offsets = self.offsets
                  dst_degree = graph_offsets[1:] - graph_offsets[:-1]
                  src_indices = self.indices
                  dst_indices = torch.arange(
                      0,
                      graph_offsets.size(0) - 1,
                      dtype=graph_offsets.dtype,
                      device=graph_offsets.device,
                  )
                  dst_indices = torch.repeat_interleave(dst_indices, dst_degree, dim=0)

                  self.dgl_graph = dgl.heterograph(
                      {("src", "src2dst", "dst"): ("coo", (src_indices, dst_indices))},
                      idtype=torch.int32,
                  )
              return self.dgl_graph

  skills:

    build_cugraph_csc:

      description: 根据原始的张量信息构建高度优化的 CSC 包装器对象

      inputs:
        - offsets
        - indices
        - num_src_nodes
        - num_dst_nodes
        - partition_size

      prompt_template: |
        如果你有原始的 offsets 和 indices，使用它们直接实例化 CuGraphCSC；
        如果你手头是一个现成的 DGLGraph，请调用 CuGraphCSC.from_dgl(graph, ...) 快速完成格式转换与打包。

    diagnose_graph_conversion_issues:

      description: 分析在转换不同图后端（DGL与CuGraphOps）时的数据类型和设备报错

      checks:
        - verify_cugraphops_installation_if_bipartite_or_static_fails
        - ensure_offsets_and_indices_are_on_gpu_before_cugraph_conversion
        - idxt_overflow_when_using_int32_on_large_graphs

  knowledge:

    usage_patterns:

      graph_operator_routing:

        pipeline:
          - Initialize CuGraphCSC (from scratch or DGL)
          - If Distributed: GNN Layer requests cross-GPU features via graph.get_src_node_features_in_local_graph()
          - Compute Message Passing (calls graph.to_bipartite_csc() or graph.to_dgl_graph() on the fly)
          - Return Updated Node Features

    design_patterns:

      lazy_initialization_and_caching:

        structure:
          - `to_bipartite_csc`, `to_static_csc`, 和 `to_dgl_graph` 使用了惰性初始化（Lazy Initialization）。
          - 只有在第一次被调用时才会执行真实的张量重构（如计算 `dst_degree`），生成后将实例保存在 `self.dgl_graph` 等内部变量中，后续调用直接返回缓存引用，极大地避免了在每一层 GNN 前向传播时重复计算导致的严重开销。

    hot_models:

      - model: Deep Graph Library (DGL) Ecosystem
        year: 2019+
        role: 主流的高效图神经网络框架
        architecture: graph_data_structures

      - model: MeshGraphNet (AI Fluid Simulators)
        year: 2020+
        role: 高分辨率流体力学仿真
        architecture: highly_optimized_message_passing

    model_usage_details:

      CuGraph Integration:

        usage: 在图的边数（如大于一亿）或特征维度极大时，原生的 PyTorch/DGL 散点更新（Scatter）操作性能不佳。NVIDIA CuGraph 提供了底层的 CUDA 融合算子。此类 `CuGraphCSC` 作为中间层，将用户的 DGL 图转换成了 C++ 能够直接识别的高效内存布局。

    best_practices:

      - 当底层的 CUDA 核发生类型溢出报错（IdxT Type Overflow）时，可以在调用 `to_bipartite_csc(dtype=torch.int64)` 时强制传入 `int64` 数据类型，临时解决某些 CuGraph 版本只能使用 64 位整型寻址大型图指针的问题。
      - 尽量不要传入 `ef_indices`，因为在 `to_dgl_graph` 以及分布式路由等多个核心方法中明确抛出了 `AssertionError("ef_indices is not supported.")`。
      - 如果当前图的拓扑是动态的（即每一层的连接关系都在变化，如动态图或不断增加新边的模型），务必在实例化时将 `cache_graph` 设置为 `False`，否则模型将错误地复用过期缓存的拓扑结构。

    anti_patterns:

      - 尝试在 CPU 上直接调用 `to_bipartite_csc()` 或 `to_static_csc()`。这会立即抛出 `RuntimeError("Expected the graph structures to reside on GPU.")`。`cugraph-ops` 是严格的 GPU 专属库。

    paper_references:

      - title: "Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks"
        authors: Wang et al.
        year: 2019

      - title: "CuGraph: A GPU-Accelerated Graph Analytics Library" (NVIDIA documentation and related engineering papers)
        authors: NVIDIA
        year: 2019+

  graph:

    is_a:
      - GraphDataContainer
      - BackendAdapter
      - DistributedManager

    part_of:
      - GNNLayerUtilities
      - DistributedGNNFramework

    depends_on:
      - pylibcugraphops
      - dgl.DGLGraph
      - DistributedGraph

    variants:
      - PyTorch Geometric (PyG) Data Object (类似生态位)

    used_in_models:
      - 任何需要底层 CUDA 算子加速的大规模图神经网络

    compatible_with:

      inputs:
        - Tensor (Offsets, Indices)
        - DGLGraph

      outputs:
        - BipartiteCSC
        - StaticCSC
        - DGLGraph