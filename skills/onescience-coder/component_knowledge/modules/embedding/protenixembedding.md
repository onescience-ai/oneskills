component:

  meta:
    name: ProtenixEmbeddingModules
    alias: AF3Embeddings
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: embedding
    author: OneScience
    license: Apache-2.0
    tags:
      - embedding
      - alphafold3
      - biology_molecules
      - atom_attention
      - structural_biology

  concept:

    description: >
      ProtenixEmbedding 包含了用于支持类似 AlphaFold 3 蛋白质结构及生物分子图谱预测架构的核心特征提取。
      其中包含三大模块：`ProtenixInputFeatureEmbedder` 负责依据原子编码（AtomAttentionEncoder）抽提微观特征，再叠加上宏观序列态；
      `ProtenixTemplateEmbedder` 负责计算基于三级结构模板掩码对掩图信息的特征传递；
      `ProtenixFourierEmbedding` 提供一个扩散噪声模型级别下的特有傅里叶时间随机正弦波偏移偏置。

    intuition: >
      在折叠高分子时我们需要知道三个信息：
      “这个细胞里面都有啥原子组成的（InputFeature）？”—— 把极其基础的分类化作深维表达；
      “这里过去有一个类似折纸的模板是什么样的（TemplateEmbedder）？”——抽取折叠规则对距离、方位构图作参考；
      “我现在撒了多少灰尘以及应该剥离多少（Fourier Noise）？”——加入包含时间特征频率噪声用于驱除模型重组。

    problem_it_solves:
      - 高度遵循 AlphaFold 3 (Algorithm 2, 16, 22) 的张量映射
      - 对微观高分子序列在宏观特征（restype）与底层（atom features）实现平滑拼合
      - 利用固化分布实现扩散噪声时间步长的特有非学习型周期注入


  theory:

    formula:

      fourier_noise_embedding:
        expression: |
          PE(t) = \cos(2\pi \cdot (t \cdot W + b))
          \text{Where } W, b \sim \mathcal{N}(0, 1)

      input_feature_fusion:
        expression: |
          S_{in} = \text{Concat}(\text{AtomEmb}, \text{ResTypeEmb}, \text{ProfileEmb}, \dots)

    variables:

      t:
        name: NoiseLevel
        description: t_hat 扩散步骤的随机噪音等级特征标量

      W, b:
        name: RandomGenerators
        description: 通过固定种子随机初始化的常数参数偏置，不允许包含反向梯度


  structure:

    architecture: biomolecular_embedding_systems

    pipeline:

      - name: AtomLevelAttention
        operation: encoder_for_atom_inputs

      - name: TemplateProcessing
        operation: pairformer_stack_and_layernorm

      - name: NoiseTimeExtraction
        operation: cosine_fourier_projection


  interface:

    parameters:

      c_atom:
        type: int
        description: 原子特征初始深度 (默认 128)

      c_token:
        type: int
        description: 单个残基/分子令牌的深维尺度 (默认 384)

      seed:
        type: int
        description: 控制傅里叶常量参数层的随机种子数 (默认 42)

    inputs:

      input_feature_dict:
        type: dict[str, Any]
        description: 汇聚多级字符串特征的蛋白质属性张量集合

      t_hat_noise_level:
        type: Tensor
        description: 在去噪过程中的连续时间域或步长索引

    outputs:

      s_inputs:
        type: Tensor
        shape: [..., N_token, dims]
        description: 融合好的令牌层级表征供主注意力层使用


  types:

    MolecularDictionary:
      shape: [Hash Map]
      description: 大分子系统专用的结构属性字段映射字典


  implementation:

    framework: pytorch

    code: |
      """
      Protenix Embedding Modules
      Implements embedding layers for Protenix (AlphaFold3)
      Reference: Algorithm 2, 3, 16, 22 in AF3
      """
      from typing import Any, Optional
      
      import torch
      import torch.nn as nn
      import torch.nn.functional as F
      
      from onescience.models.openfold.primitives import ProtenixLayerNorm
      from onescience.modules.encoder.protenixencoding import ProtenixAtomAttentionEncoder
      from onescience.modules.pairformer.protenixpairformer import ProtenixPairformerStack
      from onescience.modules.linear.protenixlinear import ProtenixLinearNoBias
      
      
      class ProtenixInputFeatureEmbedder(nn.Module):
          """
          Implements Algorithm 2 in AF3
          Input feature embedder for token representation.
          """
      
          def __init__(
              self,
              c_atom: int = 128,
              c_atompair: int = 16,
              c_token: int = 384,
          ) -> None:
              """
              Args:
                  c_atom: Atom embedding dim. Defaults to 128.
                  c_atompair: Atom pair embedding dim. Defaults to 16.
                  c_token: Token embedding dim. Defaults to 384.
              """
              super().__init__()
              self.c_atom = c_atom
              self.c_atompair = c_atompair
              self.c_token = c_token
              self.atom_attention_encoder = ProtenixAtomAttentionEncoder(
                  c_atom=c_atom,
                  c_atompair=c_atompair,
                  c_token=c_token,
                  has_coords=False,
              )
              # Line2
              self.input_feature = {"restype": 32, "profile": 32, "deletion_mean": 1}
      
          def forward(
              self,
              input_feature_dict: dict[str, Any],
              inplace_safe: bool = False,
              chunk_size: Optional[int] = None,
          ) -> torch.Tensor:
              """
              Args:
                  input_feature_dict: Dict of input features
                  inplace_safe: Whether it is safe to use inplace operations. Defaults to False.
                  chunk_size: Chunk size for memory-efficient operations. Defaults to None.
      
              Returns:
                  Token embedding [..., N_token, 384 (c_token) + 32 + 32 + 1 :=449]
              """
              # Embed per-atom features.
              a, _, _, _ = self.atom_attention_encoder(
                  input_feature_dict=input_feature_dict,
                  inplace_safe=inplace_safe,
                  chunk_size=chunk_size,
              )  # [..., N_token, c_token]
              # Concatenate the per-token features.
              batch_shape = input_feature_dict["restype"].shape[:-1]
              s_inputs = torch.cat(
                  [a]
                  + [
                      input_feature_dict[name].reshape(*batch_shape, d)
                      for name, d in self.input_feature.items()
                  ],
                  dim=-1,
              )
              return s_inputs
      
      
      class ProtenixTemplateEmbedder(nn.Module):
          """
          Implements Algorithm 16 in AF3
          Template embedder for pair representation.
          """
      
          def __init__(
              self,
              n_blocks: int = 2,
              c: int = 64,
              c_z: int = 128,
              dropout: float = 0.25,
              blocks_per_ckpt: Optional[int] = None,
          ) -> None:
              """
              Args:
                  n_blocks: Number of blocks for TemplateEmbedder. Defaults to 2.
                  c: Hidden dim of TemplateEmbedder. Defaults to 64.
                  c_z: Hidden dim for pair embedding. Defaults to 128.
                  dropout: Dropout ratio for PairformerStack. Defaults to 0.25.
                      Note this value is missed in Algorithm 16.
                  blocks_per_ckpt: Number of TemplateEmbedder/Pairformer blocks in each activation
                      checkpoint. If None, no checkpointing is performed.
              """
              super().__init__()
              self.n_blocks = n_blocks
              self.c = c
              self.c_z = c_z
              self.input_feature1 = {
                  "template_distogram": 39,
                  "b_template_backbone_frame_mask": 1,
                  "template_unit_vector": 3,
                  "b_template_pseudo_beta_mask": 1,
              }
              self.input_feature2 = {
                  "template_restype_i": 32,
                  "template_restype_j": 32,
              }
              self.distogram = {"max_bin": 50.75, "min_bin": 3.25, "no_bins": 39}
              self.inf = 100000.0
      
              self.linear_no_bias_z = ProtenixLinearNoBias(in_features=self.c_z, out_features=self.c)
              self.layernorm_z = ProtenixLayerNorm(self.c_z)
              self.linear_no_bias_a = ProtenixLinearNoBias(
                  in_features=sum(self.input_feature1.values())
                  + sum(self.input_feature2.values()),
                  out_features=self.c,
              )
              self.pairformer_stack = ProtenixPairformerStack(
                  c_s=0,
                  c_z=c,
                  n_blocks=self.n_blocks,
                  dropout=dropout,
                  blocks_per_ckpt=blocks_per_ckpt,
              )
              self.layernorm_v = ProtenixLayerNorm(self.c)
              self.linear_no_bias_u = ProtenixLinearNoBias(in_features=self.c, out_features=self.c_z)
      
          def forward(
              self,
              input_feature_dict: dict[str, Any],
              z: torch.Tensor,
              pair_mask: torch.Tensor = None,
              use_memory_efficient_kernel: bool = False,
              use_deepspeed_evo_attention: bool = False,
              use_lma: bool = False,
              inplace_safe: bool = False,
              chunk_size: Optional[int] = None,
          ) -> torch.Tensor:
              """
              Args:
                  input_feature_dict: Input feature dict
                  z: Pair embedding [..., N_token, N_token, c_z]
                  pair_mask: Pair masking [..., N_token, N_token]. Default to None.
      
              Returns:
                  Template feature [..., N_token, N_token, c_z]
              """
              # In this version, we do not use TemplateEmbedder by setting n_blocks=0
              if "template_restype" not in input_feature_dict or self.n_blocks < 1:
                  return 0
              return 0
      
      class ProtenixFourierEmbedding(nn.Module):
          """
          Implements Algorithm 22 in AF3
          Fourier embedding for noise level in diffusion.
          """
      
          def __init__(self, c: int, seed: int = 42) -> None:
              """
              Args:
                  c: Embedding dim.
                  seed: Random seed for reproducibility.
              """
              super().__init__()
              self.c = c
              self.seed = seed
              generator = torch.Generator()
              generator.manual_seed(seed)
              w_value = torch.randn(size=(c,), generator=generator)
              self.w = nn.Parameter(w_value, requires_grad=False)
              b_value = torch.randn(size=(c,), generator=generator)
              self.b = nn.Parameter(b_value, requires_grad=False)
      
          def forward(self, t_hat_noise_level: torch.Tensor) -> torch.Tensor:
              """
              Args:
                  t_hat_noise_level: Noise level [..., N_sample]
      
              Returns:
                  Fourier embedding [..., N_sample, c]
              """
              return torch.cos(
                  input=2 * torch.pi * (t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)
              )
      

  skills:

    build_protenix_fourier:

      description: 构建基于 AF3 规范配置的扩散噪音编码特征

      inputs:
        - c
        - seed

      prompt_template: |

        请实现具有不被反向传播更新权重的随机扰动 Fourier 时间节点嵌入模块。
        需要利用确定的 seed 以及 generator 生成常闭的 `self.w` 与 `self.b` 参数。


    diagnose_protenix_embedding:

      description: 分析复杂分子参数配置及特征层级的输入不当失效模式

      checks:
        - random_seed_leakage_or_divergence
        - incorrect_gradient_graph_on_noise_emb
        - missing_dictionary_keys_in_forward


  knowledge:

    usage_patterns:

      alphafold_structure_prediction:

        pipeline:
          - Data Parsing (MSA / Templates / Atoms)
          - Protenix Embedders(Features + Timestep)
          - Pairformer Trunk
          - Diffusion Decoders

    hot_models:

      - model: AlphaFold 3
        year: 2024
        role: 指导和规范了这套带有原子和噪音的输入构建法则
        architecture: Diffusion Molecular Graph Model

    best_practices:

      - 对于需要确定性但是不需要被网络学习拟模化的位置扰动，将参数的 `requires_grad=False` 必须明确设置，否则随着扩散收敛，噪声域自身的标尺会遭到破坏。
      - 当 `n_blocks < 1` 的边界情境下可以直接打断 `TemplateEmbedder` 的输出节省显存（这在代码中也由 `return 0` 直接切断）。


    anti_patterns:

      - 在序列特征向宏观 Token 拼接合并时，没有将基础属性（如 restype 长度为32的 one-hot 编码）展平到最后隐特征维度上，引发对齐报错。

    paper_references:

      - title: "Accurate structure prediction of biomolecular interactions with AlphaFold 3"
        authors: Abramson et al.
        year: 2024

  graph:

    is_a:
      - EmbeddingSystem
      - MolecularFeatures

    part_of:
      - AlphaFold3
      - Protenix

    depends_on:
      - AtomAttentionEncoder
      - Pairformer

    variants:
      - EvoformerEmbeddings

    used_in_models:
      - OpenFold
      - Protenix

    compatible_with:

      inputs:
        - DictionaryOfBiomolecularProperties

      outputs:
        - TokenRepresentations