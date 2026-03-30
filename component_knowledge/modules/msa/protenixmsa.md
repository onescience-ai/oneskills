component:

  meta:
    name: ProtenixMSAModule
    alias: ProtenixMSA
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: msa_pair_fusion
    author: OneScience
    license: Apache-2.0
    tags:
      - msa
      - pairformer
      - alphafold3
      - checkpointing
      - memory_efficient


  concept:

    description: >
      ProtenixMSAModule 实现了 AF3 风格的 MSA-对表示交互流程。
      模块从输入特征构造 MSA embedding，经多个 ProtenixMSABlock 迭代更新 pair 表示 z；
      每个 block 包含 OPM 通信、可选 MSA stack 以及 pair stack（Pairformer）。
      为适配超大 MSA/N_token，代码集成了采样、chunk、checkpoint 与显存回收策略。

    intuition: >
      模块把“MSA 信息注入 pair 表示”拆成可重复的 block：
      先用 outer-product 把多序列统计写入 z，再用 pairformer 深化 pair 关系；
      通过分块和检查点机制在大规模结构预测中平衡显存与速度。

    problem_it_solves:
      - 将 MSA 特征有效融合进 pair 表示
      - 支持大规模 MSA 采样与推理显存控制
      - 通过分块前向与梯度检查点降低内存峰值
      - 与 AF3 风格 Pairformer 堆栈对接


  theory:

    formula:

      msa_pair_weighted_avg:
        expression: |
          v = W_v LN(m),\ b = W_b LN(z),\ g = \sigma(W_g m)
          w = softmax(b)\ \text{(over msa dim)}
          o = g \odot (w \otimes v)
          m' = W_o o

      communication_to_pair:
        expression: |
          z \leftarrow z + OPM(m)

      stacked_update:
        expression: |
          (m, z) \leftarrow \text{MSABlock}_1 \circ \cdots \circ \text{MSABlock}_K (m, z)

    variables:

      m:
        name: MSAEmbedding
        shape: [n_msa, n_token, c_m]
        description: MSA 表示

      z:
        name: PairEmbedding
        shape: [n_token, n_token, c_z]
        description: 对表示

      OPM:
        name: OuterProductMean
        description: MSA 到 pair 的通信算子

      K:
        name: NumberOfBlocks
        description: MSA block 堆叠层数


  structure:

    architecture: msa_pairformer_hybrid_stack

    pipeline:

      - name: MSAFeatureSampling
        operation: random_without_replacement

      - name: MSAEmbeddingProjection
        operation: one_hot_concat_and_linear_projection

      - name: BlockPreparation
        operation: partial_bind_runtime_flags

      - name: RepeatedMSABlocks
        operation: opm_then_msa_stack_then_pair_stack

      - name: CheckpointExecution
        operation: checkpoint_blocks

      - name: OutputPair
        operation: return_updated_z


  interface:

    parameters:

      n_blocks:
        type: int
        description: MSA block 数量

      c_m:
        type: int
        description: MSA 表示维度

      c_z:
        type: int
        description: Pair 表示维度

      c_s_inputs:
        type: int
        description: 输入 single 表示维度

      msa_dropout:
        type: float
        description: MSA 路径 dropout

      pair_dropout:
        type: float
        description: pair 路径 dropout

      blocks_per_ckpt:
        type: int|null
        description: 每个 checkpoint 包含的 block 数

      msa_chunk_size:
        type: int
        description: MSA 分块尺寸

      msa_max_size:
        type: int
        description: 训练阶段 MSA 预留上限

      msa_configs:
        type: dict
        description: MSA 采样和截断配置

    inputs:

      input_feature_dict:
        type: dict
        description: 含 msa/has_deletion/deletion_value 等输入特征

      z:
        type: PairEmbedding
        shape: [n_token, n_token, c_z]
        dtype: float32
        description: 输入 pair 表示

      s_inputs:
        type: SequenceEmbedding
        shape: [n_token, c_s_inputs]
        dtype: float32
        description: 输入 single 表示

      pair_mask:
        type: Mask
        shape: [n_token, n_token]
        description: pair mask

    outputs:

      z_out:
        type: PairEmbedding
        shape: [n_token, n_token, c_z]
        description: 更新后的 pair 表示


  types:

    PairEmbedding:
      shape: [n_token, n_token, c_z]
      description: pair 表示

    SequenceEmbedding:
      shape: [n_token, c]
      description: single 表示

    Mask:
      shape: [n_token, n_token]
      description: 布尔或0/1掩码


  implementation:

    framework: pytorch

    code: |

      class ProtenixMSAPairWeightedAveraging(nn.Module):
          # v = Wv m, b = Wb z, g = sigmoid(Wg m), o = g * einsum(softmax(b), v)
          ...

      class ProtenixMSAStack(nn.Module):
          # 包含 pair-weighted averaging + transition，支持 chunk_forward/inference_forward
          ...

      class ProtenixMSABlock(nn.Module):
          # z <- z + OPM(m); m <- MSAStack(m,z); (s,z) <- Pairformer(s,z)
          ...

      class ProtenixMSAModule(nn.Module):
          # 采样MSA -> one_hot+线性投影 -> checkpoint_blocks 执行多个MSABlock -> 返回z
          ...


  skills:

    build_protenix_msa:

      description: 构建 AF3 风格 MSA-Pair 融合模块

      inputs:
        - n_blocks
        - c_m
        - c_z
        - msa_chunk_size
        - blocks_per_ckpt

      prompt_template: |

        构建 ProtenixMSAModule。

        参数：
        n_blocks = {{n_blocks}}
        c_m = {{c_m}}
        c_z = {{c_z}}
        msa_chunk_size = {{msa_chunk_size}}
        blocks_per_ckpt = {{blocks_per_ckpt}}

        要求：
        实现 MSA 到 pair 的通信、分块前向和 checkpoint 执行。


    diagnose_msa_memory:

      description: 分析 MSA 模块的显存与稳定性问题

      checks:
        - one_hot_memory_burst_on_large_msa
        - chunk_size_misalignment
        - missing_clear_cache_on_inference
        - checkpoint_configuration_error


  knowledge:

    usage_patterns:

      af3_msa_pipeline:

        pipeline:
          - InputFeatureEmbed
          - ProtenixMSAModule
          - PairformerStack
          - StructureModule

      memory_efficient_inference:

        pipeline:
          - MSA_sampling_cutoff
          - chunk_forward
          - selective_cuda_empty_cache


    hot_models:

      - model: AlphaFold3
        year: 2024
        role: MSA 与 pair 联合更新范式参考
        architecture: msa_pairformer_hybrid

      - model: OpenFold
        year: 2022
        role: 三角更新与注意力模块来源
        architecture: evoformer_family


    best_practices:

      - 大 token 场景下使用较小 chunk_size 并启用 checkpoint。
      - 推理时按阈值清理中间变量，避免 CUDA 峰值暴涨。
      - 对 MSA 采样策略设上下界，稳定训练分布。


    anti_patterns:

      - 对超大 MSA 直接 one_hot 全量展开而不采样。
      - blocks_per_ckpt 设置不当导致速度和显存双重退化。


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
      - MSAPairFusionModule

    part_of:
      - ProtenixModels
      - AF3StylePipelines

    depends_on:
      - OuterProductMean
      - Pairformer
      - CheckpointBlocks
      - MSAFeatureSampling

    variants:
      - ProtenixMSAPairWeightedAveraging
      - ProtenixMSAStack
      - ProtenixMSABlock

    used_in_models:
      - AlphaFold3-like Models
      - Protenix Pipelines
      - Biomolecular Structure Predictors

    compatible_with:

      inputs:
        - PairEmbedding
        - SequenceEmbedding
        - Mask

      outputs:
        - PairEmbedding