component:

  meta:
    name: MeshEdgeBlock
    alias: EdgeBlock
    version: 1.0
    domain: deep_learning
    category: graph_neural_network
    subcategory: edge_update
    author: OneScience
    license: Apache-2.0
    tags:
      - graphcast
      - meshgraphnet
      - edge_update
      - residual_connection
      - message_passing


  concept:

    description: >
      MeshEdgeBlock 是图神经网络中的边更新模块，基于边特征和边两端节点特征计算新的边表示。
      模块内部调用边 MLP（拼接式或加和式），并通过残差连接将更新结果与原始边特征相加，
      最终输出更新后的边特征与原节点特征。

    intuition: >
      每条边可以看作“节点间关系”的载体，边更新的目标是融合 src/dst 节点上下文后重写关系表示。
      残差连接保留原边信息，避免深层消息传递时关系特征退化。

    problem_it_solves:
      - 在消息传递前后对边关系进行可学习更新
      - 支持两种边 MLP 形式以平衡表达能力与显存占用
      - 通过残差结构提高训练稳定性
      - 兼容 DGLGraph 与 CuGraphCSC 图存储形式


  theory:

    formula:

      edge_update:
        expression: |
          e' = MLP(e, h_{src}, h_{dst})
          e_{new} = e' + e

    variables:

      e:
        name: EdgeFeature
        shape: [E, C_edge]
        description: 输入边特征

      h_{src}:
        name: SourceNodeFeature
        shape: [N, C_node]
        description: 源节点特征

      h_{dst}:
        name: DestinationNodeFeature
        shape: [N, C_node]
        description: 目标节点特征

      e_{new}:
        name: UpdatedEdgeFeature
        shape: [E, C_edge]
        description: 残差更新后的边特征


  structure:

    architecture: residual_edge_processor

    pipeline:

      - name: MLPSelection
        operation: concat_or_sum_based_edge_mlp

      - name: EdgeMessageFusion
        operation: edge_mlp_forward_with_graph_topology

      - name: ResidualAdd
        operation: add_input_edge_feature

      - name: ReturnTuple
        operation: return_updated_edge_and_original_node


  interface:

    parameters:

      input_dim_nodes:
        type: int
        description: 输入节点特征维度

      input_dim_edges:
        type: int
        description: 输入边特征维度

      output_dim:
        type: int
        description: 输出边特征维度

      hidden_dim:
        type: int
        description: 边 MLP 隐层维度

      hidden_layers:
        type: int
        description: 边 MLP 隐层层数

      activation_fn:
        type: nn.Module
        description: 激活函数实例

      norm_type:
        type: str
        description: 归一化类型

      do_concat_trick:
        type: bool
        description: 是否采用低显存加和式实现

      recompute_activation:
        type: bool
        description: 是否启用激活重计算

    inputs:

      efeat:
        type: EdgeEmbedding
        shape: [E, C_edge]
        dtype: float32
        description: 输入边特征

      nfeat:
        type: NodeEmbedding
        shape: [N, C_node]
        dtype: float32
        description: 输入节点特征

      graph:
        type: GraphStructure
        description: 图拓扑对象（DGLGraph 或 CuGraphCSC）

    outputs:

      edge_output:
        type: EdgeEmbedding
        shape: [E, C_edge]
        description: 更新后的边特征

      node_passthrough:
        type: NodeEmbedding
        shape: [N, C_node]
        description: 原样透传的节点特征


  types:

    EdgeEmbedding:
      shape: [E, C_edge]
      description: 边 embedding 表示

    NodeEmbedding:
      shape: [N, C_node]
      description: 节点 embedding 表示

    GraphStructure:
      description: 图连接关系的结构体对象


  implementation:

    framework: pytorch

    code: |

      from typing import Union

      import torch
      import torch.nn as nn
      from dgl import DGLGraph
      from torch import Tensor

      from onescience.modules.mlp.mesh_graph_mlp import MeshGraphEdgeMLPConcat, MeshGraphEdgeMLPSum
      from onescience.modules.utils.gnnlayer_utils import CuGraphCSC

      class MeshEdgeBlock(nn.Module):
          def __init__(
              self,
              input_dim_nodes: int = 512,
              input_dim_edges: int = 512,
              output_dim: int = 512,
              hidden_dim: int = 512,
              hidden_layers: int = 1,
              activation_fn: nn.Module = nn.SiLU(),
              norm_type: str = "LayerNorm",
              do_concat_trick: bool = False,
              recompute_activation: bool = False,
          ):
              super().__init__()

              MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat

              self.edge_mlp = MLP(
                  efeat_dim=input_dim_edges,
                  src_dim=input_dim_nodes,
                  dst_dim=input_dim_nodes,
                  output_dim=output_dim,
                  hidden_dim=hidden_dim,
                  hidden_layers=hidden_layers,
                  activation_fn=activation_fn,
                  norm_type=norm_type,
                  recompute_activation=recompute_activation,
              )

          @torch.jit.ignore()
          def forward(
              self,
              efeat: Tensor,
              nfeat: Tensor,
              graph: Union[DGLGraph, CuGraphCSC],
          ) -> Tensor:
              efeat_new = self.edge_mlp(efeat, nfeat, graph)
              efeat_new = efeat_new + efeat
              return efeat_new, nfeat


  skills:

    build_edge_block:

      description: 构建基于图拓扑的边更新残差模块

      inputs:
        - input_dim_nodes
        - input_dim_edges
        - output_dim
        - hidden_dim
        - hidden_layers
        - do_concat_trick

      prompt_template: |

        构建 MeshEdgeBlock 模块。

        参数：
        input_dim_nodes = {{input_dim_nodes}}
        input_dim_edges = {{input_dim_edges}}
        output_dim = {{output_dim}}
        hidden_dim = {{hidden_dim}}
        hidden_layers = {{hidden_layers}}
        do_concat_trick = {{do_concat_trick}}

        要求：
        使用边 MLP 更新边特征，并保留残差连接 e_new = e' + e。


    diagnose_edge_block:

      description: 分析边更新模块在图学习中的数值与效率问题

      checks:
        - shape_mismatch_between_edge_and_output_dim
        - graph_topology_type_incompatibility
        - memory_pressure_from_concat_mode
        - ineffective_residual_when_output_dim_changed


  knowledge:

    usage_patterns:

      graphcast_message_passing:

        pipeline:
          - EdgeUpdate(MeshEdgeBlock)
          - NodeAggregation
          - NodeUpdate

      memory_optimized_edge_update:

        pipeline:
          - MeshGraphEdgeMLPSum
          - ResidualAdd


    hot_models:

      - model: GraphCast
        year: 2023
        role: 在网格图天气建模中使用边-节点交替更新
        architecture: mesh_graph_network

      - model: MeshGraphNet
        year: 2021
        role: 提供通用网格图消息传递框架
        architecture: graph_neural_network


    best_practices:

      - output_dim 与 input_dim_edges 保持一致以保证残差可加。
      - 显存紧张时优先启用 do_concat_trick 使用加和式边 MLP。
      - 在异构图后端间切换时，先验证图对象类型兼容性。


    anti_patterns:

      - 改变输出维度但仍直接残差相加，导致维度不匹配。
      - 在大规模边数场景盲目使用拼接式 MLP，造成显存溢出。


    paper_references:

      - title: "GraphCast: Learning skillful medium-range global weather forecasting"
        authors: Lam et al.
        year: 2023

      - title: "Learning Mesh-Based Simulation with Graph Networks"
        authors: Pfaff et al.
        year: 2021


  graph:

    is_a:
      - NeuralNetworkComponent
      - EdgeUpdateModule

    part_of:
      - GraphNeuralNetworks
      - MeshGraphModels

    depends_on:
      - MeshGraphEdgeMLPConcat
      - MeshGraphEdgeMLPSum
      - ResidualConnection

    variants:
      - EdgeBlockWithoutResidual
      - AttentionEdgeBlock

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