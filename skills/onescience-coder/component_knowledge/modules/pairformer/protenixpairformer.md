component:

  meta:
    name: ProtenixPairformer
    alias: PairformerStack
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: pair_single_fusion
    author: OneScience
    license: Apache-2.0
    tags:
      - pairformer
      - alphafold3
      - triangular_updates
      - checkpointing
      - pair_bias_attention


  concept:

    description: >
      ProtenixPairformer 实现 AF3 风格 pair/single 表示更新，包含 ProtenixPairformerBlock 与 ProtenixPairformerStack。
      单个 block 由三角乘法更新、三角注意力、pair transition，以及可选 single 路径（pair-bias attention + transition）构成；
      stack 负责多层堆叠并支持按块 checkpoint 与推理阶段显存清理。

    intuition: >
      pair 表示更新强调几何一致性：
      三角路径在 pair-pair 关系上建模闭环约束，single 分支通过 pair bias 回收 pair 上下文，
      最终在多层堆叠中逐步精化结构相关特征。

    problem_it_solves:
      - 迭代更新 pair 表示中的高阶关系
      - 将 pair 信息反馈到 single 表示
      - 在长序列下提供内存友好执行路径
      - 对齐 AF3 中 Pairformer 结构范式


  theory:

    formula:

      pair_update:
        expression: |
          z \leftarrow z + TriMulOut(z) + TriMulIn(z)
          z \leftarrow z + TriAttStart(z) + TriAttEnd(z)
          z \leftarrow z + Transition_z(z)

      single_update:
        expression: |
          s \leftarrow s + AttnPairBias(s, z)
          s \leftarrow s + Transition_s(s)

      stacked_execution:
        expression: |
          (s,z) \leftarrow Block_1 \circ \cdots \circ Block_L (s,z)

    variables:

      s:
        name: SingleEmbedding
        shape: [n_token, c_s]
        description: single 表示

      z:
        name: PairEmbedding
        shape: [n_token, n_token, c_z]
        description: pair 表示

      L:
        name: NumBlocks
        description: Pairformer block 数量

      TriMul/TriAtt:
        name: TriangularOperators
        description: 三角乘法与三角注意力模块


  structure:

    architecture: pairformer_block_stack

    pipeline:

      - name: TriangularMultiplication
        operation: outgoing_then_incoming

      - name: TriangularAttention
        operation: start_node_then_end_node

      - name: PairTransition
        operation: feedforward_transition_on_z

      - name: OptionalSingleUpdate
        operation: pair_bias_attention_and_transition

      - name: BlockStacking
        operation: checkpointed_multi_block_execution


  interface:

    parameters:

      n_heads:
        type: int
        description: pair-bias attention 头数

      c_z:
        type: int
        description: pair 表示维度

      c_s:
        type: int
        description: single 表示维度

      c_hidden_mul:
        type: int
        description: triangular multiplication 隐层维度

      c_hidden_pair_att:
        type: int
        description: triangular attention 隐层维度

      no_heads_pair:
        type: int
        description: triangular attention 头数

      dropout:
        type: float
        description: dropout 比例

      n_blocks:
        type: int
        description: stack 中 block 数量

      blocks_per_ckpt:
        type: int|null
        description: checkpoint 粒度

    inputs:

      s:
        type: SequenceEmbedding
        shape: [n_token, c_s]
        dtype: float32
        description: single 输入

      z:
        type: PairEmbedding
        shape: [n_token, n_token, c_z]
        dtype: float32
        description: pair 输入

      pair_mask:
        type: Mask
        shape: [n_token, n_token]
        description: pair 掩码

    outputs:

      s_out:
        type: SequenceEmbedding
        description: 更新后的 single 表示

      z_out:
        type: PairEmbedding
        description: 更新后的 pair 表示


  types:

    SequenceEmbedding:
      shape: [n_token, c]
      description: single embedding

    PairEmbedding:
      shape: [n_token, n_token, c]
      description: pair embedding

    Mask:
      shape: [n_token, n_token]
      description: 掩码张量


  implementation:

    framework: pytorch

    code: |

      class ProtenixPairformerBlock(nn.Module):
          # tri_mul_out/in + tri_att_start/end + pair_transition + (optional) single update
          ...

      class ProtenixPairformerStack(nn.Module):
          # 多个 ProtenixPairformerBlock 堆叠，使用 checkpoint_blocks 执行
          ...


  skills:

    build_pairformer_stack:

      description: 构建 AF3 风格 Pairformer block 与堆栈

      inputs:
        - n_blocks
        - c_z
        - c_s
        - n_heads
        - blocks_per_ckpt

      prompt_template: |

        构建 ProtenixPairformerStack 模块。

        参数：
        n_blocks = {{n_blocks}}
        c_z = {{c_z}}
        c_s = {{c_s}}
        n_heads = {{n_heads}}
        blocks_per_ckpt = {{blocks_per_ckpt}}

        要求：
        block 内包含 triangular 更新、pair transition 与可选 single 更新。


    diagnose_pairformer_memory:

      description: 分析 Pairformer 在长序列训练/推理中的效率问题

      checks:
        - excessive_memory_without_checkpointing
        - incorrect_pair_mask_transpose
        - inplace_safe_path_numerical_drift
        - dropout_row_misuse


  knowledge:

    usage_patterns:

      af3_pair_update:

        pipeline:
          - TriangularMultiplication
          - TriangularAttention
          - PairTransition
          - SingleUpdate

      scalable_execution:

        pipeline:
          - PrepareBlocksWithPartial
          - checkpoint_blocks
          - clear_cache_between_blocks


    hot_models:

      - model: AlphaFold3
        year: 2024
        role: Pairformer 设计来源
        architecture: pair_single_hybrid_stack

      - model: OpenFold-derived Pipelines
        year: 2022
        role: 三角模块实现来源
        architecture: evoformer_family


    best_practices:

      - 长序列场景启用 blocks_per_ckpt 降低峰值显存。
      - 保持 pair_mask 与 end-node attention 的转置逻辑一致。
      - 对 inplace_safe 分支进行数值回归测试。


    anti_patterns:

      - 在大 token 推理时关闭缓存清理导致 OOM。
      - 忽略 c_s=0 条件仍执行 single 更新路径。


    paper_references:

      - title: "AlphaFold 3"
        authors: Abramson et al.
        year: 2024

      - title: "Highly accurate protein structure prediction with AlphaFold"
        authors: Jumper et al.
        year: 2021


  graph:

    is_a:
      - NeuralNetworkComponent
      - PairSingleFusionModule

    part_of:
      - ProtenixModels
      - AF3StylePipelines

    depends_on:
      - TriangleMultiplicationOutgoing
      - TriangleMultiplicationIncoming
      - TriangleAttention
      - AttentionPairBias

    variants:
      - ProtenixPairformerBlock
      - ProtenixPairformerStack

    used_in_models:
      - AlphaFold3-like Models
      - Protenix Pipelines
      - Structure Prediction Backbones

    compatible_with:

      inputs:
        - SequenceEmbedding
        - PairEmbedding
        - Mask

      outputs:
        - SequenceEmbedding
        - PairEmbedding