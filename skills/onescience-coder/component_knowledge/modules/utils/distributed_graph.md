component:

  meta:
    name: DistributedGraphManager
    alias: distributed_graph
    version: 1.0
    domain: ai_for_science
    category: distributed_training
    subcategory: graph_neural_network
    author: OneScience
    license: Apache-2.0
    tags:
      - distributed_training
      - graph_partitioning
      - message_passing
      - csc_format
      - multi_gpu

  concept:

    description: >
      该模块为分布式图神经网络提供了一套完整的图切分（Graph Partitioning）和通信管理工具。
      它接收全局的 CSC（压缩稀疏列）格式图结构，通过指定的策略（如节点均分或基于空间坐标的边界框切分），将目标节点及其入度边分配到不同的 GPU 进程（Ranks）上。
      它维护了全局 ID 到局部 ID 的双向映射，并封装了底层的通信原语，使得上层的 GNN 层可以执行跨设备的分布式消息传递。

    intuition: >
      在处理极其庞大的物理网格或分子图时，单张 GPU 的显存无法容纳整个图。
      本模块将大图切分成多个子图分配给不同 GPU。在进行 GNN 消息传递前，模块会自动识别出哪些节点的特征需要跨设备获取，通过高效的通信算子（如 all_to_all）完成特征的发送与接收拼装，计算完成后再负责将更新后的特征同步回全局状态。

    problem_it_solves:
      - 打破大尺度物理模拟或大规模图中 GNN 训练的单卡显存限制
      - 解决图被切分后跨设备边（Cross-partition Edges）的特征同步与聚合问题
      - 自动管理 Global ID 与 Local ID 之间的映射与转换，屏蔽底层分布式通信细节

  theory:

    formula:

      distributed_message_passing_gather:
        expression: h_{src}^{local} = \text{IndexedAllToAll}(h_{src}^{partitioned}, \text{scatter\_indices})

    variables:

      h_{src}^{partitioned}:
        name: PartitionedSourceFeatures
        description: 分布在当前 GPU 上的源节点特征片段

      \text{scatter\_indices}:
        name: ScatterIndices
        description: 通信路由表，指示当前 GPU 需要将哪些节点特征发送给其他 GPU

      h_{src}^{local}:
        name: LocalSourceFeatures
        description: 跨卡通信拼装后，满足当前 GPU 本地子图消息传递所需的所有源节点特征

  structure:

    architecture: distributed_graph_manager

    pipeline:

      - name: PartitionStrategy
        operation: assign_node_ids_to_ranks

      - name: BufferInitialization
        operation: build_global_to_local_mappings

      - name: FeatureExchange
        operation: all_to_all_communication_during_forward_pass

  interface:

    parameters:

      partition_size:
        type: int
        description: 图被切分的总份数

      partition_rank:
        type: int
        description: 当前所在的切分进程 Rank

      device:
        type: torch.device
        description: 当前 Rank 绑定的计算设备

    inputs:

      global_offsets:
        type: Tensor
        shape: "[num_global_dst_nodes + 1]"
        description: 全局图的 CSC 列偏移量（建议在 CPU）

      global_indices:
        type: Tensor
        shape: "[num_global_edges]"
        description: 全局图的 CSC 行索引（源节点 ID，建议在 CPU）

    outputs:

      graph_partition:
        type: GraphPartition
        description: 包含本地 offsets、indices 以及 ID 映射核心数据结构

  types:

    GraphPartition:
      shape: dataclass
      description: 存储切分后局部子图拓扑和通信路由表的容器

  implementation:

    framework: pytorch

    code: |
      from dataclasses import dataclass
      from typing import List, Optional

      import torch
      import torch.distributed as dist

      from onescience.distributed import (
          DistributedManager,
          all_gather_v,
          gather_v,
          indexed_all_to_all_v,
          scatter_v,
      )


      @dataclass
      class GraphPartition:
          partition_size: int
          partition_rank: int
          device: torch.device

          local_offsets: Optional[torch.Tensor] = None
          local_indices: Optional[torch.Tensor] = None
          num_local_src_nodes: int = -1
          num_local_dst_nodes: int = -1
          num_local_indices: int = -1
          map_partitioned_src_ids_to_global: Optional[torch.Tensor] = None
          map_concatenated_local_src_ids_to_global: Optional[torch.Tensor] = None
          map_partitioned_dst_ids_to_global: Optional[torch.Tensor] = None
          map_concatenated_local_dst_ids_to_global: Optional[torch.Tensor] = None
          map_partitioned_edge_ids_to_global: Optional[torch.Tensor] = None
          map_concatenated_local_edge_ids_to_global: Optional[torch.Tensor] = None
          map_global_src_ids_to_concatenated_local: Optional[torch.Tensor] = None
          map_global_dst_ids_to_concatenated_local: Optional[torch.Tensor] = None
          map_global_edge_ids_to_concatenated_local: Optional[torch.Tensor] = None

          sizes: Optional[List[List[int]]] = None
          scatter_indices: Optional[List[torch.Tensor]] = None
          num_src_nodes_in_each_partition: Optional[List[int]] = None
          num_dst_nodes_in_each_partition: Optional[List[int]] = None
          num_indices_in_each_partition: Optional[List[int]] = None

          def __post_init__(self):
              if self.partition_size <= 0:
                  raise ValueError(f"Expected partition_size > 0, got {self.partition_size}")
              if not (0 <= self.partition_rank < self.partition_size):
                  raise ValueError(
                      f"Expected 0 <= partition_rank < {self.partition_size}, got {self.partiton_rank}"
                  )

              if self.sizes is None:
                  self.sizes = [
                      [None for _ in range(self.partition_size)]
                      for _ in range(self.partition_size)
                  ]

              if self.scatter_indices is None:
                  self.scatter_indices = [None] * self.partition_size
              if self.num_src_nodes_in_each_partition is None:
                  self.num_src_nodes_in_each_partition = [None] * self.partition_size
              if self.num_dst_nodes_in_each_partition is None:
                  self.num_dst_nodes_in_each_partition = [None] * self.partition_size
              if self.num_indices_in_each_partition is None:
                  self.num_indices_in_each_partition = [None] * self.partition_size

          def to(self, *args, **kwargs):
              for attr in dir(self):
                  attr_val = getattr(self, attr)
                  if isinstance(attr_val, torch.Tensor):
                      setattr(self, attr, attr_val.to(*args, **kwargs))

              self.scatter_indices = [idx.to(*args, **kwargs) for idx in self.scatter_indices]

              return self


      def partition_graph_with_id_mapping(
          global_offsets: torch.Tensor,
          global_indices: torch.Tensor,
          mapping_src_ids_to_ranks: torch.Tensor,
          mapping_dst_ids_to_ranks: torch.Tensor,
          partition_size: int,
          partition_rank: int,
          device: torch.device,
      ) -> GraphPartition:

          graph_partition = GraphPartition(
              partition_size=partition_size, partition_rank=partition_rank, device=device
          )

          dst_nodes_in_each_partition = [None] * partition_size
          src_nodes_in_each_partition = [None] * partition_size
          num_dst_nodes_in_each_partition = [None] * partition_size
          num_src_nodes_in_each_partition = [None] * partition_size

          dtype = global_indices.dtype
          input_device = global_indices.device

          graph_partition.map_concatenated_local_src_ids_to_global = torch.empty_like(
              mapping_src_ids_to_ranks
          )
          graph_partition.map_concatenated_local_dst_ids_to_global = torch.empty_like(
              mapping_dst_ids_to_ranks
          )
          graph_partition.map_concatenated_local_edge_ids_to_global = torch.empty_like(
              global_indices
          )
          graph_partition.map_global_src_ids_to_concatenated_local = torch.empty_like(
              mapping_src_ids_to_ranks
          )
          graph_partition.map_global_dst_ids_to_concatenated_local = torch.empty_like(
              mapping_dst_ids_to_ranks
          )
          graph_partition.map_global_edge_ids_to_concatenated_local = torch.empty_like(
              global_indices
          )
          _map_global_src_ids_to_local = torch.empty_like(mapping_src_ids_to_ranks)

          _src_id_offset = 0
          _dst_id_offset = 0
          _edge_id_offset = 0

          for rank in range(partition_size):
              dst_nodes_in_each_partition[rank] = torch.nonzero(
                  mapping_dst_ids_to_ranks == rank
              ).view(-1)
              src_nodes_in_each_partition[rank] = torch.nonzero(
                  mapping_src_ids_to_ranks == rank
              ).view(-1)
              num_nodes = dst_nodes_in_each_partition[rank].numel()
              if num_nodes == 0:
                  raise RuntimeError(
                      f"Aborting partitioning, rank {rank} has 0 destination nodes to work on."
                  )
              num_dst_nodes_in_each_partition[rank] = num_nodes

              num_nodes = src_nodes_in_each_partition[rank].numel()
              num_src_nodes_in_each_partition[rank] = num_nodes
              if num_nodes == 0:
                  raise RuntimeError(
                      f"Aborting partitioning, rank {rank} has 0 source nodes to work on."
                  )

              ids = src_nodes_in_each_partition[rank]
              mapped_ids = torch.arange(
                  start=_src_id_offset,
                  end=_src_id_offset + ids.numel(),
                  dtype=dtype,
                  device=input_device,
              )
              _map_global_src_ids_to_local[ids] = mapped_ids - _src_id_offset
              graph_partition.map_global_src_ids_to_concatenated_local[ids] = mapped_ids
              graph_partition.map_concatenated_local_src_ids_to_global[mapped_ids] = ids
              _src_id_offset += ids.numel()

              ids = dst_nodes_in_each_partition[rank]
              mapped_ids = torch.arange(
                  start=_dst_id_offset,
                  end=_dst_id_offset + ids.numel(),
                  dtype=dtype,
                  device=input_device,
              )
              graph_partition.map_global_dst_ids_to_concatenated_local[ids] = mapped_ids
              graph_partition.map_concatenated_local_dst_ids_to_global[mapped_ids] = ids
              _dst_id_offset += ids.numel()

          graph_partition.num_src_nodes_in_each_partition = num_src_nodes_in_each_partition
          graph_partition.num_dst_nodes_in_each_partition = num_dst_nodes_in_each_partition

          for rank in range(partition_size):
              offset_start = global_offsets[dst_nodes_in_each_partition[rank]].view(-1)
              offset_end = global_offsets[dst_nodes_in_each_partition[rank] + 1].view(-1)
              degree = offset_end - offset_start
              local_offsets = degree.view(-1).cumsum(dim=0)
              local_offsets = torch.cat(
                  [
                      torch.Tensor([0]).to(
                          dtype=dtype,
                          device=input_device,
                      ),
                      local_offsets,
                  ]
              )

              partitioned_edge_ids = torch.cat(
                  [
                      torch.arange(
                          start=offset_start[i],
                          end=offset_end[i],
                          dtype=dtype,
                          device=input_device,
                      )
                      for i in range(len(offset_start))
                  ]
              )

              ids = partitioned_edge_ids
              mapped_ids = torch.arange(
                  _edge_id_offset,
                  _edge_id_offset + ids.numel(),
                  device=ids.device,
                  dtype=ids.dtype,
              )
              graph_partition.map_global_edge_ids_to_concatenated_local[ids] = mapped_ids
              graph_partition.map_concatenated_local_edge_ids_to_global[mapped_ids] = ids
              _edge_id_offset += ids.numel()

              partitioned_src_ids = torch.cat(
                  [
                      global_indices[offset_start[i] : offset_end[i]].clone()
                      for i in range(len(offset_start))
                  ]
              )

              global_src_ids_on_rank, inverse_mapping = partitioned_src_ids.unique(
                  sorted=True, return_inverse=True
              )
              remote_local_src_ids_on_rank = _map_global_src_ids_to_local[
                  global_src_ids_on_rank
              ]

              _map_global_src_ids_to_local_graph = torch.zeros_like(mapping_src_ids_to_ranks)
              _num_local_indices = 0
              for rank_offset in range(partition_size):
                  mask = mapping_src_ids_to_ranks[global_src_ids_on_rank] == rank_offset
                  if partition_rank == rank_offset:
                      graph_partition.scatter_indices[rank] = (
                          remote_local_src_ids_on_rank[mask]
                          .detach()
                          .clone()
                          .to(dtype=torch.int64)
                      )
                  numel_mask = mask.sum().item()
                  graph_partition.sizes[rank_offset][rank] = numel_mask

                  tmp_ids = torch.arange(
                      _num_local_indices,
                      _num_local_indices + numel_mask,
                      device=input_device,
                      dtype=dtype,
                  )
                  _num_local_indices += numel_mask
                  tmp_map = global_src_ids_on_rank[mask]
                  _map_global_src_ids_to_local_graph[tmp_map] = tmp_ids

              local_indices = _map_global_src_ids_to_local_graph[partitioned_src_ids]
              graph_partition.num_indices_in_each_partition[rank] = local_indices.size(0)

              if rank == partition_rank:
                  graph_partition.local_offsets = local_offsets
                  graph_partition.local_indices = local_indices
                  graph_partition.num_local_indices = graph_partition.local_indices.size(0)
                  graph_partition.num_local_dst_nodes = num_dst_nodes_in_each_partition[rank]
                  graph_partition.num_local_src_nodes = global_src_ids_on_rank.size(0)

                  graph_partition.map_partitioned_src_ids_to_global = (
                      src_nodes_in_each_partition[rank]
                  )
                  graph_partition.map_partitioned_dst_ids_to_global = (
                      dst_nodes_in_each_partition[rank]
                  )
                  graph_partition.map_partitioned_edge_ids_to_global = partitioned_edge_ids

          for r in range(graph_partition.partition_size):
              err_msg = "error in graph partition: list containing sizes of exchanged indices does not match the tensor of indices to be exchanged"
              if (
                  graph_partition.sizes[graph_partition.partition_rank][r]
                  != graph_partition.scatter_indices[r].numel()
              ):
                  raise AssertionError(err_msg)

          graph_partition = graph_partition.to(device=device)

          return graph_partition


      def partition_graph_nodewise(
          global_offsets: torch.Tensor,
          global_indices: torch.Tensor,
          partition_size: int,
          partition_rank: int,
          device: torch.device,
      ) -> GraphPartition:

          num_global_src_nodes = global_indices.max().item() + 1
          num_global_dst_nodes = global_offsets.size(0) - 1
          num_dst_nodes_per_partition = (
              num_global_dst_nodes + partition_size - 1
          ) // partition_size
          num_src_nodes_per_partition = (
              num_global_src_nodes + partition_size - 1
          ) // partition_size

          mapping_dst_ids_to_ranks = (
              torch.arange(
                  num_global_dst_nodes,
                  dtype=global_offsets.dtype,
                  device=global_offsets.device,
              )
              // num_dst_nodes_per_partition
          )
          mapping_src_ids_to_ranks = (
              torch.arange(
                  num_global_src_nodes,
                  dtype=global_offsets.dtype,
                  device=global_offsets.device,
              )
              // num_src_nodes_per_partition
          )

          return partition_graph_with_id_mapping(
              global_offsets,
              global_indices,
              mapping_src_ids_to_ranks,
              mapping_dst_ids_to_ranks,
              partition_size,
              partition_rank,
              device,
          )


      def partition_graph_by_coordinate_bbox(
          global_offsets: torch.Tensor,
          global_indices: torch.Tensor,
          src_coordinates: torch.Tensor,
          dst_coordinates: torch.Tensor,
          coordinate_separators_min: List[List[Optional[float]]],
          coordinate_separators_max: List[List[Optional[float]]],
          partition_size: int,
          partition_rank: int,
          device: torch.device,
      ) -> GraphPartition:

          dim = src_coordinates.size(-1)
          if dst_coordinates.size(-1) != dim:
              raise ValueError()
          if len(coordinate_separators_min) != partition_size:
              raise ValueError()
          if len(coordinate_separators_max) != partition_size:
              raise ValueError()

          for rank in range(partition_size):
              if len(coordinate_separators_min[rank]) != dim:
                  raise ValueError()
              if len(coordinate_separators_max[rank]) != dim:
                  raise ValueError()

          num_global_src_nodes = global_indices.max().item() + 1
          num_global_dst_nodes = global_offsets.size(0) - 1

          mapping_dst_ids_to_ranks = torch.zeros(
              num_global_dst_nodes, dtype=global_offsets.dtype, device=global_offsets.device
          )
          mapping_src_ids_to_ranks = torch.zeros(
              num_global_src_nodes,
              dtype=global_offsets.dtype,
              device=global_offsets.device,
          )

          def _assign_ranks(mapping, coordinates):
              for p in range(partition_size):
                  mask = torch.ones_like(mapping).to(dtype=torch.bool)
                  for d in range(dim):
                      min_val, max_val = (
                          coordinate_separators_min[p][d],
                          coordinate_separators_max[p][d],
                      )
                      if min_val is not None:
                          mask = mask & (coordinates[:, d] >= min_val)
                      if max_val is not None:
                          mask = mask & (coordinates[:, d] < max_val)
                  mapping[mask] = p

          _assign_ranks(mapping_src_ids_to_ranks, src_coordinates)
          _assign_ranks(mapping_dst_ids_to_ranks, dst_coordinates)

          return partition_graph_with_id_mapping(
              global_offsets,
              global_indices,
              mapping_src_ids_to_ranks,
              mapping_dst_ids_to_ranks,
              partition_size,
              partition_rank,
              device,
          )


      class DistributedGraph:
          def __init__(
              self,
              global_offsets: torch.Tensor,
              global_indices: torch.Tensor,
              partition_size: int,
              graph_partition_group_name: str = None,
              graph_partition: Optional[GraphPartition] = None,
          ):

              dist_manager = DistributedManager()
              self.device = dist_manager.device
              self.partition_rank = dist_manager.group_rank(name=graph_partition_group_name)
              self.partition_size = dist_manager.group_size(name=graph_partition_group_name)
              if self.partition_size != partition_size:
                  raise AssertionError()
              self.process_group = dist_manager.group(name=graph_partition_group_name)

              if graph_partition is None:
                  self.graph_partition = partition_graph_nodewise(
                      global_offsets,
                      global_indices,
                      self.partition_size,
                      self.partition_rank,
                      self.device,
                  )
              else:
                  if graph_partition.partition_size != self.partition_size:
                      raise AssertionError()
                  if graph_partition.device != self.device:
                      raise AssertionError()
                  self.graph_partition = graph_partition

              dist.barrier(self.process_group)

          def get_src_node_features_in_partition(
              self,
              global_node_features: torch.Tensor,
              scatter_features: bool = False,
              src_rank: int = 0,
          ) -> torch.Tensor:  
              if scatter_features:
                  global_node_features = global_node_features[
                      self.graph_partition.map_concatenated_local_src_ids_to_global
                  ]
                  return scatter_v(
                      global_node_features,
                      self.graph_partition.num_src_nodes_in_each_partition,
                      dim=0,
                      src=src_rank,
                      group=self.process_group,
                  )

              return global_node_features.to(device=self.device)[
                  self.graph_partition.map_partitioned_src_ids_to_global, :
              ]

          def get_src_node_features_in_local_graph(
              self, partitioned_src_node_features: torch.Tensor
          ) -> torch.Tensor: 
              return indexed_all_to_all_v(
                  partitioned_src_node_features,
                  indices=self.graph_partition.scatter_indices,
                  sizes=self.graph_partition.sizes,
                  use_fp32=True,
                  dim=0,
                  group=self.process_group,
              )

          def get_dst_node_features_in_partition(
              self,
              global_node_features: torch.Tensor,
              scatter_features: bool = False,
              src_rank: int = 0,
          ) -> torch.Tensor:  
              if scatter_features:
                  global_node_features = global_node_features.to(device=self.device)[
                      self.graph_partition.map_concatenated_local_dst_ids_to_global
                  ]
                  return scatter_v(
                      global_node_features,
                      self.graph_partition.num_dst_nodes_in_each_partition,
                      dim=0,
                      src=src_rank,
                      group=self.process_group,
                  )

              return global_node_features.to(device=self.device)[
                  self.graph_partition.map_partitioned_dst_ids_to_global, :
              ]

          def get_dst_node_features_in_local_graph(
              self,
              partitioned_dst_node_features: torch.Tensor,
          ) -> torch.Tensor: 
              return partitioned_dst_node_features

          def get_edge_features_in_partition(
              self,
              global_edge_features: torch.Tensor,
              scatter_features: bool = False,
              src_rank: int = 0,
          ) -> torch.Tensor:  
              if scatter_features:
                  global_edge_features = global_edge_features[
                      self.graph_partition.map_concatenated_local_edge_ids_to_global
                  ]
                  return scatter_v(
                      global_edge_features,
                      self.graph_partition.num_indices_in_each_partition,
                      dim=0,
                      src=src_rank,
                      group=self.process_group,
                  )

              return global_edge_features.to(device=self.device)[
                  self.graph_partition.map_partitioned_edge_ids_to_global, :
              ]

          def get_edge_features_in_local_graph(
              self, partitioned_edge_features: torch.Tensor
          ) -> torch.Tensor:  
              return partitioned_edge_features

          def get_global_src_node_features(
              self,
              partitioned_node_features: torch.Tensor,
              get_on_all_ranks: bool = True,
              dst_rank: int = 0,
          ) -> torch.Tensor:  
              if partitioned_node_features.device != self.device:
                  raise AssertionError()

              if not get_on_all_ranks:
                  global_node_feat = gather_v(
                      partitioned_node_features,
                      self.graph_partition.num_src_nodes_in_each_partition,
                      dim=0,
                      dst=dst_rank,
                      group=self.process_group,
                  )
                  if self.graph_partition.partition_rank == dst_rank:
                      global_node_feat = global_node_feat[
                          self.graph_partition.map_global_src_ids_to_concatenated_local
                      ]

                  return global_node_feat

              global_node_feat = all_gather_v(
                  partitioned_node_features,
                  self.graph_partition.num_src_nodes_in_each_partition,
                  dim=0,
                  use_fp32=True,
                  group=self.process_group,
              )
              global_node_feat = global_node_feat[
                  self.graph_partition.map_global_src_ids_to_concatenated_local
              ]
              return global_node_feat

          def get_global_dst_node_features(
              self,
              partitioned_node_features: torch.Tensor,
              get_on_all_ranks: bool = True,
              dst_rank: int = 0,
          ) -> torch.Tensor:  
              if partitioned_node_features.device != self.device:
                  raise AssertionError()

              if not get_on_all_ranks:
                  global_node_feat = gather_v(
                      partitioned_node_features,
                      self.graph_partition.num_dst_nodes_in_each_partition,
                      dim=0,
                      dst=dst_rank,
                      group=self.process_group,
                  )
                  if self.graph_partition.partition_rank == dst_rank:
                      global_node_feat = global_node_feat[
                          self.graph_partition.map_global_dst_ids_to_concatenated_local
                      ]

                  return global_node_feat

              global_node_feat = all_gather_v(
                  partitioned_node_features,
                  self.graph_partition.num_dst_nodes_in_each_partition,
                  dim=0,
                  use_fp32=True,
                  group=self.process_group,
              )
              global_node_feat = global_node_feat[
                  self.graph_partition.map_global_dst_ids_to_concatenated_local
              ]
              return global_node_feat

          def get_global_edge_features(
              self,
              partitioned_edge_features: torch.Tensor,
              get_on_all_ranks: bool = True,
              dst_rank: int = 0,
          ) -> torch.Tensor: 
              if partitioned_edge_features.device != self.device:
                  raise AssertionError()

              if not get_on_all_ranks:
                  global_edge_feat = gather_v(
                      partitioned_edge_features,
                      self.graph_partition.num_indices_in_each_partition,
                      dim=0,
                      dst=dst_rank,
                      group=self.process_group,
                  )
                  if self.graph_partition.partition_rank == dst_rank:
                      global_edge_feat = global_edge_feat[
                          self.graph_partition.map_global_edge_ids_to_concatenated_local
                      ]
                  return global_edge_feat

              global_edge_feat = all_gather_v(
                  partitioned_edge_features,
                  self.graph_partition.num_indices_in_each_partition,
                  dim=0,
                  use_fp32=True,
                  group=self.process_group,
              )
              global_edge_feat = global_edge_feat[
                  self.graph_partition.map_global_edge_ids_to_concatenated_local
              ]
              return global_edge_feat

  skills:

    build_distributed_graph:

      description: 根据系统的物理特性选择合适的图切分策略并初始化分布式图

      inputs:
        - partition_strategy (nodewise or bbox)
        - global_graph_tensors
        - process_group_info

      prompt_template: |
        如果节点具有空间物理坐标，请调用 partition_graph_by_coordinate_bbox ；
        如果只是抽象拓扑图，调用 partition_graph_nodewise。
        使用返回的 GraphPartition 实例化 DistributedGraph。

    diagnose_distributed_gnn_issues:

      description: 排查分布式图通信或张量设备不匹配的问题

      checks:
        - device_mismatch_between_partitioned_features_and_rank_device
        - graph_partition_size_mismatch_with_process_group_size

  knowledge:

    usage_patterns:

      distributed_forward_pass:

        pipeline:
          - dist_graph.get_src_node_features_in_local_graph()
          - Message Passing
          - dist_graph.get_global_dst_node_features()

    design_patterns:

      cpu_preprocessing_and_drop:

        structure:
          - 在 CPU 上进行切分和映射计算。生成包含局部子图的 GraphPartition 后，原始全局大张量可以从内存释放，避免在 GPU 上组装引发 OOM。

    hot_models:

      - model: MeshGraphNet (Distributed)
        year: 2020
        role: 复杂物理动力学求解器
        architecture: message_passing_gnn
        attention_type: None

    model_usage_details:

      Spatial Physics Modeling:

        usage: 在流体力学或天气预报中，按空间边界框切分（coordinate_bbox）能使大多数连边成为“内部边”，极大降低跨卡通信带来的带宽开销。

    best_practices:

      - 使用前须确保映射没有导致某个 Rank 分配到 0 个节点，否则梯度同步会死锁。
      - 传入 DistributedGraph 的张量设备必须与当前 Rank 的设备严格一致。

    anti_patterns:

      - 将整个 global_indices 移动到 GPU 后再计算掩码切分，极易引发显存溢出。
      - 每次前向传播重复计算切分。对于静态图，应只在初始化阶段执行一次并缓存 GraphPartition。

    paper_references:

      - title: "GraphSAGE: Inductive Representation Learning on Large Graphs"
        authors: Hamilton et al.
        year: 2017

      - title: "Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks"
        authors: Wang et al.
        year: 2019

  graph:

    is_a:
      - DistributedManager
      - GraphPartitionUtility

    part_of:
      - DistributedGNNFramework
      - AI_for_Science_Simulator

    depends_on:
      - torch.distributed
      - indexed_all_to_all_v

    variants:
      - DGL Distributed Graph
      - PyG Distributed

    used_in_models:
      - 分布式流体力学模型

    compatible_with:

      inputs:
        - Tensor

      outputs:
        - Tensor