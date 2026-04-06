component:

  meta:
    name: ProtenixTransformer
    alias: AlphaFold3DiffusionTransformer
    version: 1.0
    domain: computational_biology
    category: neural_network
    subcategory: diffusion_transformer
    author: OneScience
    license: Apache-2.0
    tags:
      - alphafold3
      - protenix
      - diffusion_transformer
      - protein_structure
      - computational_biology


  concept:

    description: >
      Protenix Transformer 是用于生物大分子结构预测的核心扩散网络模块（对应 AlphaFold 3 中的 Algorithm 7, 23, 25）。
      它通过引入对特征 (Pair Representation) 偏置的局部交叉注意力机制，
      以及由单体特征 (Single Representation) 调制的条件过渡块 (Conditioned Transition Block)，
      实现了原子级别高保真三维坐标的生成与去噪。

    intuition: >
      在预测蛋白质 3D 结构时，原子之间并非孤立存在。
      标准的 Transformer 只看一维序列，而这个模块在计算注意力时，会额外参考“对特征 (z)”（类似于两两原子之间的距离或化学键信息）来作为注意力权重的偏置 (Bias)，从而赋予模型极强的 3D 空间感知能力。
      同时，在特征映射阶段，它还会不断利用“单体特征 (s)”（类似于整个氨基酸或核苷酸的宏观背景）来微调和修正原子局部的状态。

    problem_it_solves:
      - 将 1D 序列信息、2D 对偶交互信息与 3D 扩散去噪过程进行深度融合
      - 解决传统模型在处理大规模蛋白质复合物时，微观原子碰撞和物理违规的问题
      - 极其高效的显存管理（通过重计算 checkpointing 和按需释放缓存）以支持数千个原子的长序列推理


  theory:

    formula:

      attention_with_pair_bias:
        expression: $$a_{attn} = a + \text{Attention}(Q=a, K=a, V=a, \text{Bias}=z, \text{Condition}=s)$$

      conditioned_transition:
        expression: |
          $$a_{norm} = \text{AdaLN}(a_{attn}, s)$$
          $$b = \text{SiLU}(\text{Linear}_{no\_bias}(a_{norm})) \odot \text{Linear}_{no\_bias}(a_{norm})$$
          $$a_{out} = a_{attn} + \sigma(\text{Linear}_s(s)) \odot \text{Linear}_b(b)$$

    variables:

      a / q:
        name: AtomActivations / Queries
        shape: [..., n_queries, c_atom]
        description: 微观原子的特征激活值，在扩散过程中不断被去噪更新

      s / c:
        name: SingleRepresentation / Context
        shape: [..., c_atom]
        description: 宏观单体（如残基）的特征，用于提供全局的上下文条件（AdaLN 和门控）

      z / p:
        name: PairRepresentation
        shape: [..., n_queries, n_keys, c_atompair]
        description: 宏观或微观对偶特征，用于在注意力机制中提供空间几何偏置


  structure:

    architecture: conditioned_diffusion_transformer

    pipeline:

      - name: PairBiasedAttention
        operation: protenix_attention_with_local_attn (整合 a, s, z)

      - name: AttnResidual
        operation: add

      - name: AdaptiveLayerNorm
        operation: adaln (基于 s 调制 a)

      - name: SwiGLU_Variant
        operation: silu + hadamard_product (计算中间状态 b)

      - name: ContextGating
        operation: sigmoid_gate (利用 s 对 b 进行门控输出)

      - name: FFNResidual
        operation: add


  interface:

    parameters:

      c_atom:
        type: int
        default: 128
        description: 原子特征的通道维度

      c_atompair:
        type: int
        default: 16
        description: 对偶特征（Pair Representation）的通道维度

      n_blocks:
        type: int
        default: 3
        description: 扩散 Transformer 的层数

      n_heads:
        type: int
        default: 4
        description: 注意力机制的头数

      n_queries / n_keys:
        type: int
        description: 注意力计算中查询和键的序列长度（对应原子的数量）

    inputs:

      q:
        type: Tensor
        description: 待去噪的原子 Query 特征 (对应代码中的 a)

      c:
        type: Tensor
        description: 单体上下文特征 (对应代码中的 s)

      p:
        type: Tensor
        description: 对偶偏置特征 (对应代码中的 z)

    outputs:

      output:
        type: Tensor
        description: 更新后的原子特征，随后用于预测三维坐标增量


  implementation:

    framework: pytorch

    code: |

      # 核心逻辑参考 ProtenixConditionedTransitionBlock
      class ProtenixConditionedTransitionBlock(nn.Module):
          def __init__(self, c_a: int, c_s: int, n: int = 2, biasinit: float = -2.0):
              super().__init__()
              self.adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
              self.linear_nobias_a1 = ProtenixLinearNoBias(c_a, n * c_a, initializer="relu")
              self.linear_nobias_a2 = ProtenixLinearNoBias(c_a, n * c_a, initializer="relu")
              self.linear_nobias_b = ProtenixLinearNoBias(n * c_a, c_a)
              self.linear_s = ProtenixBiasInitLinear(c_s, c_a, bias=True, biasinit=biasinit)

          def forward(self, a: torch.Tensor, s: torch.Tensor):
              a = self.adaln(a, s)
              b = F.silu((self.linear_nobias_a1(a))) * self.linear_nobias_a2(a)
              a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)
              return a


  skills:

    build_protenix_transformer:

      description: 构建用于生物分子结构扩散生成的 Transformer

      inputs:
        - c_atom
        - c_atompair
        - n_blocks

      prompt_template: |

        请实例化 ProtenixAtomTransformer。
        确保传入的 q (queries), c (context), p (pair) 特征的最后一维分别与 c_atom, c_atom, c_atompair 完全对齐。


    diagnose_protenix_oom:

      description: 排查大分子复合物推理时的显存溢出问题 (OOM)

      checks:
        - attention_chunking_failure (未正确启用 chunk_size 导致 $O(N^2)$ Pair 特征显存爆炸)
        - gradient_checkpointing_bypass (训练时 blocks_per_ckpt 设置不当导致显存峰值过高)



  knowledge:

    usage_patterns:

      alphafold3_diffusion_pipeline:

        pipeline:
          - InputFeatureEmbedder (序列、模板等特征化)
          - Pairformer (抽提宏观 s 和 z)
          - NoiseSchedule (扩散加噪)
          - ProtenixAtomTransformer (逐步去噪预测坐标)


    hot_models:

      - model: AlphaFold 3
        year: 2024
        role: 统一了所有生命分子（蛋白质、核酸、小分子配体、离子）的结构预测模型
        architecture: Diffusion Transformer (DiT) with Pair Bias

      - model: RoseTTAFold All-Atom
        year: 2024
        role: 与 AF3 同期的全原子结构预测竞争模型


    best_practices:

      - `ProtenixDiffusionTransformer` 内置了极其严苛的显存控制逻辑（`clear_cache_between_blocks`）。当序列长度（原子数）大于 2000 时，系统会强制在 Block 之间清理 CUDA 缓存。在部署时应密切关注此特性对推理速度的潜在影响。
      - `inplace_safe=True` 可以在不需要反向传播的情况下，通过原地操作（In-place addition）大幅节省中间张量带来的显存开销。


    anti_patterns:

      - 在小分子或极短的肽链（少于 50 个原子）上进行推理时开启 `clear_cache_between_blocks`，会导致 CUDA 内核频繁同步，严重拖慢推理速度。


    paper_references:

      - title: "Accurate structure prediction of biomolecular interactions with AlphaFold 3"
        authors: Abramson et al. (Google DeepMind)
        year: 2024



  graph:

    is_a:
      - DiffusionTransformer
      - NeuralNetworkComponent

    part_of:
      - AlphaFold3
      - ProtenixModel

    depends_on:
      - AdaptiveLayerNorm
      - ProtenixAttentionPairBiasWithLocalAttn
      - Checkpointing

    variants:
      - PairformerBlock

    used_in_models:
      - AlphaFold 3 (Protenix implementation)

    compatible_with:

      inputs:
        - AtomActivations
        - SingleRepresentation
        - PairRepresentation

      outputs:
        - DenoisedAtomActivations