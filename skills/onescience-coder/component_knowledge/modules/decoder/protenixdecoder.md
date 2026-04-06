component:

  meta:
    name: ProtenixAtomAttentionDecoder
    alias: AlphaFold3 Atom Attention Decoder
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: protein_decoder
    author: OneScience
    license: Apache-2.0
    tags:
      - protein_structure
      - alpha_fold
      - atom_attention
      - coordinate_prediction
      - molecular_modeling

  concept:

    description: >
      ProtenixAtomAttentionDecoder是AlphaFold3中的原子注意力解码器，实现Algorithm 6。
      它将token级别的表示解码为原子坐标更新，通过原子transformer层处理原子间的相互作用，
      最终输出每个原子的3D坐标增量。该模块是蛋白质结构预测中的关键组件，
      负责从序列表示到原子级空间坐标的转换。

    intuition: >
      就像分子生物学家从氨基酸序列推断蛋白质的3D结构：先理解每个氨基酸的整体特征（token表示），
      然后考虑原子间的相互作用（原子注意力），最后精确定位每个原子的空间位置（坐标更新）。
      跳跃连接保留了编码器中的原子级信息，确保解码过程不丢失细节。

    problem_it_solves:
      - 从token级别表示恢复原子级坐标
      - 蛋白质结构预测中的坐标精修
      - 原子间相互作用的建模
      - 多尺度生物分子表示的统一
      - 高精度分子结构预测

  theory:

    formula:

      token_to_atom_broadcast:
        expression: |
          q_{atom} = \text{BroadcastTokenToAtom}(W_{token} \cdot a_{token}) + q_{skip}

      atom_transformer:
        expression: |
          q'_{atom} = \text{AtomTransformer}(q_{atom}, c_{skip}, p_{skip})

      coordinate_prediction:
        expression: |
          \Delta r = W_{out} \cdot \text{LayerNorm}(q'_{atom})

    variables:

      a_{token}:
        name: TokenRepresentations
        shape: [..., N_token, c_token]
        description: Token级别的聚合表示

      q_{skip}:
        name: AtomQueriesSkip
        shape: [..., N_atom, c_atom]
        description: 来自编码器的原子查询跳跃连接

      c_{skip}:
        name: AtomFeaturesSkip
        shape: [..., N_atom, c_atom]
        description: 来自编码器的原子特征跳跃连接

      p_{skip}:
        name: AtomPairFeaturesSkip
        shape: [..., n_blocks, n_queries, n_keys, c_atompair]
        description: 来自编码器的原子对特征跳跃连接

      \Delta r:
        name: CoordinateUpdates
        shape: [..., N_atom, 3]
        description: 预测的原子坐标更新量

  structure:

    architecture: atom_attention_decoder

    pipeline:

      - name: TokenToAtomBroadcast
        operation: linear_projection + broadcast + skip_connection

      - name: AtomTransformerProcessing
        operation: atom_transformer_blocks

      - name: CoordinatePrediction
        operation: layer_norm + linear_projection

  interface:

    parameters:

      n_blocks:
        type: int
        description: 原子transformer块的数量

      n_heads:
        type: int
        description: 注意力头的数量

      c_token:
        type: int
        description: Token表示维度

      c_atom:
        type: int
        description: 原子表示维度

      c_atompair:
        type: int
        description: 原子对表示维度

      n_queries:
        type: int
        description: 局部注意力窗口中的查询原子数

      n_keys:
        type: int
        description: 局部注意力窗口中的键原子数

      blocks_per_ckpt:
        type: Optional[int]
        description: 每个激活检查点的块数，None表示不使用检查点

    inputs:

      input_feature_dict:
        type: FeatureDictionary
        description: 包含'atom_to_token_idx'映射的字典

      a:
        type: TokenRepresentations
        shape: [..., N_token, c_token]
        description: Token级别的聚合表示

      q_skip:
        type: AtomQueries
        shape: [..., N_atom, c_atom]
        description: 编码器原子查询的跳跃连接

      c_skip:
        type: AtomFeatures
        shape: [..., N_atom, c_atom]
        description: 编码器原子特征的跳跃连接

      p_skip:
        type: AtomPairFeatures
        shape: [..., n_blocks, n_queries, n_keys, c_atompair]
        description: 编码器原子对特征的跳跃连接

      inplace_safe:
        type: bool
        description: 是否允许原地操作，默认为False

      chunk_size:
        type: Optional[int]
        description: 内存高效操作的块大小，None表示不分块

    outputs:

      r:
        type: CoordinateUpdates
        shape: [..., N_atom, 3]
        description: 原子坐标更新

  types:

    TokenRepresentations:
      shape: [..., num_tokens, token_dim]
      description: Token级别的特征表示

    AtomCoordinates:
      shape: [..., num_atoms, 3]
      description: 原子的3D坐标

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.models.openfold.primitives import ProtenixLayerNorm
      from onescience.linear.onelinear import ProtenixLinearNoBias
      from onescience.models.protenix.utils import broadcast_token_to_atom

      class ProtenixAtomAttentionDecoder(nn.Module):
          def __init__(self, n_blocks=3, n_heads=4, c_token=384, c_atom=128,
                       c_atompair=16, n_queries=32, n_keys=128, blocks_per_ckpt=None):
              super().__init__()
              self.n_blocks = n_blocks
              self.n_heads = n_heads
              self.c_token = c_token
              self.c_atom = c_atom
              
              # Token到原子的投影
              self.linear_no_bias_a = ProtenixLinearNoBias(c_token, c_atom)
              
              # 坐标预测头
              self.layernorm_q = ProtenixLayerNorm(c_atom, create_offset=False)
              self.linear_no_bias_out = ProtenixLinearNoBias(c_atom, 3, precision=torch.float32)
              
              # 原子Transformer
              from onescience.modules.transformer.protenixtransformer import ProtenixAtomTransformer
              self.atom_transformer = ProtenixAtomTransformer(
                  n_blocks=n_blocks, n_heads=n_heads, c_atom=c_atom,
                  c_atompair=c_atompair, n_queries=n_queries, n_keys=n_keys,
                  blocks_per_ckpt=blocks_per_ckpt
              )

          def forward(self, input_feature_dict, a, q_skip, c_skip, p_skip,
                     inplace_safe=False, chunk_size=None):
              # Token到原子的广播
              q = (
                  broadcast_token_to_atom(
                      x_token=self.linear_no_bias_a(a),
                      atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                  ) + q_skip
              )
              
              # 原子Transformer处理
              q = self.atom_transformer(
                  q, c_skip, p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size
              )
              
              # 坐标预测
              r = self.linear_no_bias_out(self.layernorm_q(q))
              return r

  skills:

    build_atom_decoder:

      description: 构建用于蛋白质结构预测的原子注意力解码器

      inputs:
        - n_blocks
        - n_heads
        - c_token
        - c_atom
        - c_atompair

      prompt_template: |

        构建Protenix原子注意力解码器，实现AlphaFold3 Algorithm 6。

        参数：
        n_blocks = {{n_blocks}}
        n_heads = {{n_heads}}
        c_token = {{c_token}}
        c_atom = {{c_atom}}
        c_atompair = {{c_atompair}}

        要求：
        1. 实现token到原子的特征广播
        2. 支持局部原子注意力机制
        3. 包含跳跃连接
        4. 输出3D坐标更新

    optimize_structure_prediction:

      description: 优化蛋白质结构预测的计算精度

      checks:
        - coordinate_precision (坐标预测的数值精度)
        - memory_efficiency (大分子处理的显存使用)
        - attention_locality (局部注意力的有效性)

  knowledge:

    usage_patterns:

      protein_structure_refinement:

        pipeline:
          - Token Encoding: 氨基酸序列编码
          - Atom Broadcasting: Token特征扩散到原子
          - Atom Attention: 原子间相互作用建模
          - Coordinate Update: 3D坐标精修

      molecular_modeling:

        pipeline:
          - Sequence Features: 序列级别特征
          - Atomic Features: 原子级别特征
          - Spatial Reasoning: 空间关系推理
          - Structure Prediction: 结构预测

    hot_models:

      - model: AlphaFold3
        year: 2024
        role: Google DeepMind的蛋白质结构预测模型
        architecture: diffusion + attention

      - model: RoseTTAFold
        year: 2023
        role: 三轨注意力蛋白质结构预测
        architecture: three-track attention

      - model: ESMFold
        year: 2023
        role: Meta的蛋白质结构预测模型
        architecture: language model + folding

    best_practices:

      - 使用float32精度确保坐标预测的数值稳定性
      - 局部注意力窗口大小应根据分子尺度调整
      - 跳跃连接对保持编码器信息至关重要
      - 激活检查点在处理大分子时可以节省显存

    anti_patterns:

      - 忽略数值精度导致坐标预测不稳定
      - 注意力窗口设置不当影响局部相互作用建模
      - 缺少跳跃连接导致信息丢失
      - 在大分子处理时忽略内存优化

    paper_references:

      - title: "AlphaFold 3: Accurate structure prediction of biomolecular interactions and complexes"
        authors: Abramson et al.
        year: 2024

      - title: "Highly accurate protein structure prediction with AlphaFold"
        authors: Jumper et al.
        year: 2021

      - title: "RoseTTAFold: Protein structure prediction using end-to-end attention"
        authors: Baek et al.
        year: 2021

  graph:

    is_a:
      - NeuralNetworkDecoder
      - AttentionModule
      - ProteinStructurePredictor

    part_of:
      - AlphaFold3Architecture
      - MolecularModelingSystems
      - BioinformaticsModels

    depends_on:
      - ProtenixAtomTransformer
      - ProtenixLayerNorm
      - ProtenixLinearNoBias
      - TokenToAtomBroadcast

    variants:
      - StandardAtomDecoder (无局部注意力)
      - GraphAtomDecoder (基于图的原子解码器)
      - DiffusionDecoder (基于扩散的解码器)

    used_in_models:
      - AlphaFold3
      - 其他蛋白质结构预测模型

    compatible_with:

      inputs:
        - TokenRepresentations
        - AtomicFeatures
        - PairwiseFeatures

      outputs:
        - CoordinateUpdates
        - AtomicPositions
        - MolecularStructure
