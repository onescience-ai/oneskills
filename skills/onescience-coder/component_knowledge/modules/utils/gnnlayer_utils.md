component:

  meta:
    name: GNNLayerUtilities
    alias: gnnlayer_utils
    version: 1.0
    domain: ai_for_science
    category: utility
    subcategory: graph_neural_network
    author: OneScience
    license: Apache-2.0
    tags:
      - graph_neural_network
      - message_passing
      - dgl
      - cugraph
      - feature_aggregation
      - distributed_training

  concept:

    description: >
      提供了图神经网络（GNN）层所需的核心底层算子和辅助函数。主要用于在图的节点（Node）和边（Edge）之间进行特征的拼接（Concatenation）、求和（Summation）以及聚合（Aggregation，如 sum 或 mean）。
      该模块通过集成标准 DGL 原语与高性能的 CuGraph-Ops 后端，并原生支持分布式图计算，显著提升了大规模图模型在特征交互阶段的计算效率与显存管理能力。

    intuition: >
      在图神经网络的消息传递（Message Passing）过程中，中心节点需要收集邻居节点和连接它们边上的信息。
      这个模块就像是一个高效的“快递分拣中心”，它可以把相邻节点和边的包裹打包在一起（concat_efeat），或者把它们的值加在一起（sum_efeat），
      然后把所有寄给同一个目的地的包裹合并成一个大包裹（aggregate_and_concat）。
      此外，它还会智能地判断当前设备是否支持 NVIDIA GPU 加速算子（CuGraphOps），如果支持，它会抄“高速公路”，否则平滑回退到标准路线（DGL）。

    problem_it_solves:
      - 统一并加速 GNN 中的 Edge-to-Edge (e2e) 和 Edge-to-Node (e2n) 特征更新操作
      - 透明地兼容标准 DGL 后端与极速但接口复杂的 NVIDIA CuGraph-Ops 后端
      - 在分布式图训练中，自动处理跨 GPU 节点的特征拉取（Fetch）与聚合
      - 提供可切换的梯度检查点（Gradient Checkpointing）功能，缓解深层 GNN 的显存压力

  theory:

    formula:

      edge_feature_concat:
        expression: e_{ij}^{(new)} = e_{ij} \parallel v_i \parallel v_j

      edge_feature_sum:
        expression: e_{ij}^{(new)} = e_{ij} + v_i + v_j

      node_aggregation_concat:
        expression: v_j^{(new)} = v_j \parallel \bigoplus_{i \in \mathcal{N}(j)} e_{ij}

    variables:

      e_{ij}:
        name: EdgeFeature
        description: 节点 i 到节点 j 之间的边特征

      v_i:
        name: SourceNodeFeature
        description: 边的起点（源节点）特征

      v_j:
        name: DestinationNodeFeature
        description: 边的终点（目标节点）特征

      \parallel:
        name: Concatenation
        description: 特征拼接操作

      \bigoplus:
        name: AggregationOperator
        description: 图节点邻域的聚合算子（代码中支持 sum 和 mean）

  structure:

    architecture: message_passing_utilities

    pipeline:

      - name: CheckpointRouting
        operation: set_checkpoint_fn

      - name: EdgeFeatureUpdate
        operation: concat_efeat / sum_efeat

      - name: NodeFeatureAggregation
        operation: aggregate_and_concat

  interface:

    parameters:

      do_checkpointing:
        type: bool
        description: 是否启用梯度检查点以节省显存

      aggregation:
        type: str
        description: 聚合方式，支持 'sum' 或 'mean'

    inputs:

      efeat:
        type: Tensor
        description: 边特征张量

      nfeat:
        type: Union[Tensor, Tuple[Tensor]]
        description: 节点特征张量（可以是单一的全局特征，或是区分源节点与目标节点的元组）

      graph:
        type: Union[DGLGraph, CuGraphCSC]
        description: 图拓扑结构对象，包含了图的连接关系和分布式状态信息

    outputs:

      output:
        type: Tensor
        description: 更新后的边特征，或完成聚合与拼接后的目标节点特征

  types:

    CuGraphCSC:
      shape: custom_class
      description: 针对 CuGraph 优化并支持分布式图拓扑的自定义图结构（基于压缩稀疏列格式）

  implementation:

    framework: pytorch, dgl, cugraphops

    code: |
      from typing import Any, Callable, Dict, Tuple, Union

      import dgl.function as fn
      import torch
      from dgl import DGLGraph
      from torch import Tensor
      from torch.utils.checkpoint import checkpoint

      from onescience.modules.utils.graph import CuGraphCSC

      try:
          from pylibcugraphops.pytorch.operators import (
              agg_concat_e2n,
              update_efeat_bipartite_e2e,
              update_efeat_static_e2e,
          )

          USE_CUGRAPHOPS = True

      except ImportError:
          update_efeat_bipartite_e2e = None
          update_efeat_static_e2e = None
          agg_concat_e2n = None
          USE_CUGRAPHOPS = False


      def checkpoint_identity(layer: Callable, *args: Any, **kwargs: Any) -> Any:
          """Applies the identity function for checkpointing.

          This function serves as an identity function for use with model layers
          when checkpointing is not enabled. It simply forwards the input arguments
          to the specified layer and returns its output.

          Parameters
          ----------
          layer : Callable
              The model layer or function to apply to the input arguments.
          *args
              Positional arguments to be passed to the layer.
          **kwargs
              Keyword arguments to be passed to the layer.

          Returns
          -------
          Any
              The output of the specified layer after processing the input arguments.
          """
          return layer(*args)


      def set_checkpoint_fn(do_checkpointing: bool) -> Callable:
          """Sets checkpoint function.

          This function returns the appropriate checkpoint function based on the
          provided `do_checkpointing` flag. If `do_checkpointing` is True, the
          function returns the checkpoint function from PyTorch's
          `torch.utils.checkpoint`. Otherwise, it returns an identity function
          that simply passes the inputs through the given layer.

          Parameters
          ----------
          do_checkpointing : bool
              Whether to use checkpointing for gradient computation. Checkpointing
              can reduce memory usage during backpropagation at the cost of
              increased computation time.

          Returns
          -------
          Callable
              The selected checkpoint function to use for gradient computation.
          """
          if do_checkpointing:
              return checkpoint
          else:
              return checkpoint_identity


      def concat_message_function(edges: Tensor) -> Dict[str, Tensor]:
          """Concatenates source node, destination node, and edge features.

          Parameters
          ----------
          edges : Tensor
              Edges.

          Returns
          -------
          Dict[Tensor]
              Concatenated source node, destination node, and edge features.
          """
          # concats src node , dst node, and edge features
          cat_feat = torch.cat((edges.data["x"], edges.src["x"], edges.dst["x"]), dim=1)
          return {"cat_feat": cat_feat}


      @torch.jit.ignore()
      def concat_efeat_dgl(
          efeat: Tensor,
          nfeat: Union[Tensor, Tuple[torch.Tensor, torch.Tensor]],
          graph: DGLGraph,
      ) -> Tensor:
          """Concatenates edge features with source and destination node features.
          Use for homogeneous graphs.

          Parameters
          ----------
          efeat : Tensor
              Edge features.
          nfeat : Tensor | Tuple[Tensor, Tensor]
              Node features.
          graph : DGLGraph
              Graph.

          Returns
          -------
          Tensor
              Concatenated edge features with source and destination node features.
          """
          if isinstance(nfeat, Tuple):
              src_feat, dst_feat = nfeat
              with graph.local_scope():
                  graph.srcdata["x"] = src_feat
                  graph.dstdata["x"] = dst_feat
                  graph.edata["x"] = efeat
                  graph.apply_edges(concat_message_function)
                  return graph.edata["cat_feat"]

          with graph.local_scope():
              graph.ndata["x"] = nfeat
              graph.edata["x"] = efeat
              graph.apply_edges(concat_message_function)
              return graph.edata["cat_feat"]


      def concat_efeat(
          efeat: Tensor,
          nfeat: Union[Tensor, Tuple[Tensor]],
          graph: Union[DGLGraph, CuGraphCSC],
      ) -> Tensor:
          """Concatenates edge features with source and destination node features.
          Use for homogeneous graphs.

          Parameters
          ----------
          efeat : Tensor
              Edge features.
          nfeat : Tensor | Tuple[Tensor]
              Node features.
          graph : DGLGraph | CuGraphCSC
              Graph.

          Returns
          -------
          Tensor
              Concatenated edge features with source and destination node features.
          """
          if isinstance(nfeat, Tensor):
              if isinstance(graph, CuGraphCSC):
                  if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                      src_feat, dst_feat = nfeat, nfeat
                      if graph.is_distributed:
                          src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                      efeat = concat_efeat_dgl(
                          efeat, (src_feat, dst_feat), graph.to_dgl_graph()
                      )

                  else:
                      if graph.is_distributed:
                          src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                          # torch.int64 to avoid indexing overflows due tu current behavior of cugraph-ops
                          bipartite_graph = graph.to_bipartite_csc(dtype=torch.int64)
                          dst_feat = nfeat
                          efeat = update_efeat_bipartite_e2e(
                              efeat, src_feat, dst_feat, bipartite_graph, "concat"
                          )

                      else:
                          static_graph = graph.to_static_csc()
                          efeat = update_efeat_static_e2e(
                              efeat,
                              nfeat,
                              static_graph,
                              mode="concat",
                              use_source_emb=True,
                              use_target_emb=True,
                          )

              else:
                  efeat = concat_efeat_dgl(efeat, nfeat, graph)

          else:
              src_feat, dst_feat = nfeat
              # update edge features through concatenating edge and node features
              if isinstance(graph, CuGraphCSC):
                  if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                      if graph.is_distributed:
                          src_feat = graph.get_src_node_features_in_local_graph(src_feat)
                      efeat = concat_efeat_dgl(
                          efeat, (src_feat, dst_feat), graph.to_dgl_graph()
                      )

                  else:
                      if graph.is_distributed:
                          src_feat = graph.get_src_node_features_in_local_graph(src_feat)
                      # torch.int64 to avoid indexing overflows due tu current behavior of cugraph-ops
                      bipartite_graph = graph.to_bipartite_csc(dtype=torch.int64)
                      efeat = update_efeat_bipartite_e2e(
                          efeat, src_feat, dst_feat, bipartite_graph, "concat"
                      )
              else:
                  efeat = concat_efeat_dgl(efeat, (src_feat, dst_feat), graph)

          return efeat


      @torch.jit.script
      def sum_efeat_dgl(
          efeat: Tensor, src_feat: Tensor, dst_feat: Tensor, src_idx: Tensor, dst_idx: Tensor
      ) -> Tensor:
          """Sums edge features with source and destination node features.

          Parameters
          ----------
          efeat : Tensor
              Edge features.
          src_feat : Tensor
              Source node features.
          dst_feat : Tensor
              Destination node features.
          src_idx : Tensor
              Source node indices.
          dst_idx : Tensor
              Destination node indices.

          Returns
          -------
          Tensor
              Sum of edge features with source and destination node features.
          """

          return efeat + src_feat[src_idx] + dst_feat[dst_idx]


      def sum_efeat(
          efeat: Tensor,
          nfeat: Union[Tensor, Tuple[Tensor]],
          graph: Union[DGLGraph, CuGraphCSC],
      ):
          """Sums edge features with source and destination node features.

          Parameters
          ----------
          efeat : Tensor
              Edge features.
          nfeat : Tensor | Tuple[Tensor]
              Node features (static setting) or tuple of node features of
              source and destination nodes (bipartite setting).
          graph : DGLGraph | CuGraphCSC
              The underlying graph.

          Returns
          -------
          Tensor
              Sum of edge features with source and destination node features.
          """
          if isinstance(nfeat, Tensor):
              if isinstance(graph, CuGraphCSC):
                  if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                      src_feat, dst_feat = nfeat, nfeat
                      if graph.is_distributed:
                          src_feat = graph.get_src_node_features_in_local_graph(src_feat)

                      src, dst = (item.long() for item in graph.to_dgl_graph().edges())
                      sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)

                  else:
                      if graph.is_distributed:
                          src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                          dst_feat = nfeat
                          bipartite_graph = graph.to_bipartite_csc()
                          sum_efeat = update_efeat_bipartite_e2e(
                              efeat, src_feat, dst_feat, bipartite_graph, mode="sum"
                          )

                      else:
                          static_graph = graph.to_static_csc()
                          sum_efeat = update_efeat_bipartite_e2e(
                              efeat, nfeat, static_graph, mode="sum"
                          )

              else:
                  src_feat, dst_feat = nfeat, nfeat
                  src, dst = (item.long() for item in graph.edges())
                  sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)

          else:
              src_feat, dst_feat = nfeat
              if isinstance(graph, CuGraphCSC):
                  if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                      if graph.is_distributed:
                          src_feat = graph.get_src_node_features_in_local_graph(src_feat)

                      src, dst = (item.long() for item in graph.to_dgl_graph().edges())
                      sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)

                  else:
                      if graph.is_distributed:
                          src_feat = graph.get_src_node_features_in_local_graph(src_feat)

                      bipartite_graph = graph.to_bipartite_csc()
                      sum_efeat = update_efeat_bipartite_e2e(
                          efeat, src_feat, dst_feat, bipartite_graph, mode="sum"
                      )
              else:
                  src, dst = (item.long() for item in graph.edges())
                  sum_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)

          return sum_efeat


      @torch.jit.ignore()
      def agg_concat_dgl(
          efeat: Tensor, dst_nfeat: Tensor, graph: DGLGraph, aggregation: str
      ) -> Tensor:
          """Aggregates edge features and concatenates result with destination node features.

          Parameters
          ----------
          efeat : Tensor
              Edge features.
          nfeat : Tensor
              Node features (destination nodes).
          graph : DGLGraph
              Graph.
          aggregation : str
              Aggregation method (sum or mean).

          Returns
          -------
          Tensor
              Aggregated edge features concatenated with destination node features.

          Raises
          ------
          RuntimeError
              If aggregation method is not sum or mean.
          """
          with graph.local_scope():
              # populate features on graph edges
              graph.edata["x"] = efeat

              # aggregate edge features
              if aggregation == "sum":
                  # print(graph.edata["x"].dtype) 
                  graph.update_all(fn.copy_e("x", "m"), fn.sum("m", "h_dest"))
              elif aggregation == "mean":
                  graph.update_all(fn.copy_e("x", "m"), fn.mean("m", "h_dest"))
              else:
                  raise RuntimeError("Not a valid aggregation!")

              # concat dst-node & edge features
              cat_feat = torch.cat((graph.dstdata["h_dest"], dst_nfeat), -1)
              
              return cat_feat


      def aggregate_and_concat(
          efeat: Tensor,
          nfeat: Tensor,
          graph: Union[DGLGraph, CuGraphCSC],
          aggregation: str,
      ):
          """
          Aggregates edge features and concatenates result with destination node features.

          Parameters
          ----------
          efeat : Tensor
              Edge features.
          nfeat : Tensor
              Node features (destination nodes).
          graph : DGLGraph
              Graph.
          aggregation : str
              Aggregation method (sum or mean).

          Returns
          -------
          Tensor
              Aggregated edge features concatenated with destination node features.

          Raises
          ------
          RuntimeError
              If aggregation method is not sum or mean.
          """

          if isinstance(graph, CuGraphCSC):
              # in this case, we don't have to distinguish a distributed setting
              # or the defalt setting as both efeat and nfeat are already
              # gurantueed to be on the same rank on both cases due to our
              # partitioning scheme

              if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                  cat_feat = agg_concat_dgl(efeat, nfeat, graph.to_dgl_graph(), aggregation)

              else:
                  static_graph = graph.to_static_csc()
                  cat_feat = agg_concat_e2n(nfeat, efeat, static_graph, aggregation)
          else:
              cat_feat = agg_concat_dgl(efeat, nfeat, graph, aggregation)

          return cat_feat

  skills:

    execute_efficient_message_passing:

      description: 调用该模块执行具备分布式与算子加速能力的消息传递更新

      inputs:
        - edge_features
        - node_features
        - graph_topology

      prompt_template: |
        调用 aggregate_and_concat 或 concat_efeat 函数完成底层更新。
        传递张量与图结构对象，内部逻辑会自动判定是使用 CuGraphOps 还是 DGL 以及是否需要从集群其他 GPU 获取邻居节点。

    diagnose_cugraph_compatibility:

      description: 分析由图底层算子回退或导入失败引起的问题

      checks:
        - verify_pylibcugraphops_installation_if_speed_drops
        - ensure_distributed_graph_methods_are_accessible_on_cugraphcsc

  knowledge:

    usage_patterns:

      gnn_layer_forward_pass:

        pipeline:
          - concat_efeat (将源节点、目标节点特征拼接到边上)
          - Edge_MLP (针对拼接特征计算得到新的边特征)
          - aggregate_and_concat (聚合相邻新边特征至目标节点并与原节点特征拼接)
          - Node_MLP (更新目标节点特征)

    design_patterns:

      backend_fallback_strategy:

        structure:
          - 模块在顶部采用 `try...except ImportError` 尝试导入 `pylibcugraphops`。
          - 在各个接口函数（如 `concat_efeat`）内部，先判断图结构是否为 `CuGraphCSC` 类型，同时检查全局标志 `USE_CUGRAPHOPS`。
          - 如果高性能加速库不存在或无法使用，代码将无缝回退（Fallback）至纯 DGL 的消息传递实现（如 `update_all(fn.copy_e, fn.sum)`），保证了工程的极高鲁棒性。

    hot_models:

      - model: MeshGraphNet
        year: 2020
        role: 物理模拟与偏微分方程求解大模型
        architecture: message_passing_gnn
        attention_type: None

      - model: GraphCast
        year: 2022
        role: DeepMind 基于图神经网络的全球气象预测模型
        architecture: multi_mesh_gnn
        attention_type: None

    model_usage_details:

      AI for Science Simulators:

        usage: 物理和气象 GNN 模型通常拥有数百万个网格点，使用标准的 PyTorch DGL 操作会在此处成为严重的计算瓶颈。引入 NVIDIA `pylibcugraphops` 能够实现显著的显存优化和端到端提速。

    best_practices:

      - 在分布式设置（`graph.is_distributed == True`）下，中心节点所依赖的一些邻居源节点可能在其他 GPU 上。这套代码内部自动调用了 `graph.get_src_node_features_in_local_graph` 进行底层网络通信拉取特征，使用者可以完全透明地像在单卡上一样调用拼接操作。
      - 聚合函数目前仅支持 `"sum"` 或 `"mean"`，如果在传参时输入拼写错误，将抛出 `RuntimeError`。
      - 对于极大深度的 GNN 模型，可向层模块传递 `do_checkpointing=True`，让 `set_checkpoint_fn` 返回 PyTorch 的梯度检查点函数以节约显存。

    anti_patterns:

      - 在自行扩展分布式操作时，忽略全局图和局部图的差异，直接使用源节点索引 `src_idx` 切片访问张量 `src_feat[src_idx]`。因为远端节点的 ID 并没有本地映射特征，这将直接导致严重的越界报错（IndexError）。本模块中的逻辑已规避此问题。

    paper_references:

      - title: "Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks"
        authors: Wang et al.
        year: 2019

      - title: "Learning to Simulate Complex Physics with Graph Networks"
        authors: Sanchez-Gonzalez et al.
        year: 2020

  graph:

    is_a:
      - HelperFunctions
      - MessagePassingUtility
      - GraphOperatorBackend

    part_of:
      - GraphNeuralNetworkLayer
      - DistributedGraphManager

    depends_on:
      - dgl.function
      - pylibcugraphops (Optional)
      - torch.utils.checkpoint

    variants:
      - PyG Message Passing (PyTorch Geometric 对应的消息传递原语)

    used_in_models:
      - 基于网格的物理场替代模型
      - 高分辨率气象预报图模型

    compatible_with:

      inputs:
        - Tensor
        - DGLGraph
        - CuGraphCSC

      outputs:
        - Tensor