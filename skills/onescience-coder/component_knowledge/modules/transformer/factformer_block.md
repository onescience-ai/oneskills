component:

  meta:
    name: FactformerBlock
    alias: FactformerEncoderBlock
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: transformer_encoder
    author: OneScience
    license: Apache-2.0
    tags:
      - factformer
      - factorized_attention
      - transformer_block
      - pre_norm
      - structured_grid


  concept:

    description: >
      FactFormer 编码器块是 FactFormer 模型的基础构建单元。
      它采用标准的 Pre-Norm Transformer 架构，但引入了对不同几何结构（1D、2D、3D）数据的原生支持。
      通过 geotype 参数，模块可以动态选择使用标准的多头注意力或分解注意力（FactAttention），
      从而在处理高维网格数据时极大地降低计算复杂度和显存占用。

    intuition: >
      如果把标准的注意力机制比作“让教室里每个人同时和其他所有人说话”（计算量大），
      分解注意力（FactAttention）就像是“先让每一排的人互相交流，再让每一列的代表互相交流”。
      对于 2D 或 3D 的结构化网格数据，沿着独立的轴（如高、宽、深度）分别进行注意力计算，
      能够以近似线性的复杂度达到接近全局注意力的感受野效果。

    problem_it_solves:
      - 高维网格数据（如高分辨率图像、3D气象场、医学影像）全局自注意力计算复杂度爆炸的问题
      - 统一了 1D（无结构/序列）、2D（平面网格）和 3D（立体网格）特征提取的代码接口
      - 提供了网络最后一层的便捷输出映射（Linear Projection）


  theory:

    formula:

      residual_attention:
        expression: |
          x_attn = x + FactAttention_nD(LayerNorm(x))

      residual_mlp:
        expression: |
          x_out = x_attn + StandardMLP(LayerNorm(x_attn))

      optional_last_layer:
        expression: |
          final_out = Linear(LayerNorm(x_out))

    variables:

      FactAttention_nD:
        name: FactorizedAttention
        description: 根据 geotype 选择的 1D/2D/3D 分解注意力层

      StandardMLP:
        name: PointwiseFeedForward
        description: 逐点前馈神经网络，通常包含特征维度的扩展与激活（如 GELU）


  structure:

    architecture: pre_norm_factorized_transformer

    pipeline:

      - name: AttnPreNorm
        operation: layer_norm

      - name: FactAttention
        operation: one_attention (动态分发为 FactAttention2D/3D 或 MHA)

      - name: AttnResidual
        operation: add

      - name: MlpPreNorm
        operation: layer_norm

      - name: FeedForward
        operation: one_mlp

      - name: MlpResidual
        operation: add

      - name: OptionalFinalNorm
        operation: layer_norm (仅当 last_layer=True 时激活)

      - name: OptionalProjection
        operation: linear_projection (仅当 last_layer=True 时激活)


  interface:

    parameters:

      num_heads:
        type: int
        description: 注意力头的数量

      hidden_dim:
        type: int
        description: 隐藏层特征维度 (Embedding Dimension)

      dropout:
        type: float
        description: 注意力矩阵和特征映射的 Dropout 概率

      act:
        type: str
        default: "gelu"
        description: MLP 层的激活函数类型

      mlp_ratio:
        type: int
        default: 4
        description: MLP 隐藏层维度相对于 hidden_dim 的扩展倍数

      last_layer:
        type: bool
        default: False
        description: 是否为网络的最后一层。开启后会附加 LayerNorm 和线性投影输出。

      out_dim:
        type: int
        default: 1
        description: 当 last_layer 为 True 时的最终输出通道维度

      geotype:
        type: str
        default: "structured_2D"
        description: 几何类型，可选 ["structured_1D", "structured_2D", "structured_3D"]

      shapelist:
        type: list[int]
        default: null
        description: 输入网格的形状列表，如 (H, W) 或 (D, H, W)，FactAttention 必需参数

    inputs:

      fx:
        type: GridFeatureMap
        shape: "[batch, seq_len/pixels/voxels, hidden_dim] 或 [batch, hidden_dim, H, W]"
        dtype: float32
        description: 输入的特征张量，形状取决于具体的 Attention 实现和 geotype

    outputs:

      output:
        type: Tensor
        shape: "[batch, ..., hidden_dim] 或 [batch, ..., out_dim]"
        description: 输出特征，维度通常与输入保持一致；若是 last_layer，则最后一维为 out_dim


  types:

    GridFeatureMap:
      description: 结构化网格数据的特征表示，支持展平的一维序列或保留空间维度的多维张量


  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from typing import Optional, List, Union
      from onescience.modules.mlp.onemlp import OneMlp
      from onescience.modules.attention.oneattention import OneAttention

      class Factformer_block(nn.Module):
          """FactFormer 编码器块"""
          def __init__(
              self, num_heads: int, hidden_dim: int, dropout: float, act: str = "gelu",
              mlp_ratio: int = 4, last_layer: bool = False, out_dim: int = 1,
              geotype: str = "structured_2D", shapelist: Optional[List[int]] = None,
          ):
              super().__init__()
              self.last_layer = last_layer
              
              # 1. Pre-Norm
              self.ln_1 = nn.LayerNorm(hidden_dim)

              # 2. Attention Mechanism 
              if geotype == "structured_2D":
                  attn_style = "FactAttention2D"
              elif geotype == "structured_3D":
                  attn_style = "FactAttention3D"
              else:
                  attn_style = "MultiHeadAttention"

              self.Attn = OneAttention(
                  style=attn_style, dim=hidden_dim, heads=num_heads,
                  dim_head=hidden_dim // num_heads, dropout=dropout, shapelist=shapelist,
              )
              
              # 3. MLP Block
              self.ln_2 = nn.LayerNorm(hidden_dim)
              self.mlp = OneMlp(
                  style="StandardMLP", input_dim=hidden_dim, output_dim=hidden_dim,
                  hidden_dims=[hidden_dim * mlp_ratio], activation=act, use_bias=True, 
              )
              
              # 4. Optional Last Layer Projection
              if self.last_layer:
                  self.ln_3 = nn.LayerNorm(hidden_dim)
                  self.mlp2 = nn.Linear(hidden_dim, out_dim)

          def forward(self, fx):
              # Attention Residual
              fx = self.Attn(self.ln_1(fx)) + fx
              # MLP Residual
              fx = self.mlp(self.ln_2(fx)) + fx
              
              if self.last_layer:
                  return self.mlp2(self.ln_3(fx))
              else:
                  return fx


  skills:

    build_factformer_block:

      description: 根据不同的物理场维度（1D/2D/3D）构建对应的 Factformer 编码器

      inputs:
        - num_heads
        - hidden_dim
        - geotype
        - shapelist

      prompt_template: |

        请初始化一个 Factformer_block。
        如果是图像或地表数据，geotype="structured_2D"，并传入 shapelist=[H, W]。
        如果是高空3D数据，geotype="structured_3D"，并传入 shapelist=[D, H, W]。
        如果是最后一层用于输出预测结果，请将 last_layer 设为 True，并指定 out_dim。


    diagnose_factformer:

      description: 排查分解注意力过程中的形状和显存问题

      checks:
        - shapelist_tensor_mismatch (shapelist的乘积与输入序列长度不一致)
        - geotype_routing_error (错误配置geotype导致退化为普通O(N^2)注意力，显存溢出)



  knowledge:

    usage_patterns:

      factformer_encoder_stack:

        pipeline:
          - PatchEmbedding
          - [Factformer_block (last_layer=False)] x (N-1)
          - Factformer_block (last_layer=True, out_dim=Target_Channels)

      axial_attention_optimization:

        structure:
          - 在 H 和 W 轴上交替进行注意力计算，大幅减少长序列计算成本。


    hot_models:

      - model: FactFormer (Architecture)
        role: 针对多维结构化网格数据设计的通用分解注意力模型
        architecture: Factorized/Axial Attention

      - model: CCNet (Criss-Cross Attention)
        year: 2019
        role: 早期在视觉领域验证轴向分解注意力有效性的模型

      - model: Axial-DeepLab
        year: 2020
        role: 完全使用轴向注意力替代卷积层的全注意力模型


    best_practices:

      - 在实例化 `geotype="structured_2D"` 或 `structured_3D` 时，**必须**正确提供 `shapelist`，否则底层的 `FactAttention` 无法得知如何在不同的空间轴上进行特征重排（Reshape）。
      - 如果作为解码器或预测头使用，可以直接利用模块自带的 `last_layer=True` 和 `out_dim` 功能，避免在模块外部额外手写一层 LayerNorm 和 Linear，保持代码整洁。


    anti_patterns:

      - 对完全没有空间拓扑关系的序列（如纯文本或无序点云）强行指定 `structured_2D` 并随意捏造 `shapelist`，会导致模型学习到错误的虚假空间位置关系。


    paper_references:

      - title: "Axial Attention in Multidimensional Transformers"
        authors: Ho et al.
        year: 2019

      - title: "Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation"
        authors: Wang et al.
        year: 2020



  graph:

    is_a:
      - TransformerBlock
      - PreNormEncoder

    part_of:
      - FactFormerModel
      - AIforScienceGridModels

    depends_on:
      - OneAttention (FactAttention)
      - OneMlp
      - LayerNorm
      - Linear

    variants:
      - AxialAttentionBlock

    used_in_models:
      - FactFormer

    compatible_with:

      inputs:
        - GridFeatureMap

      outputs:
        - GridFeatureMap
        - TargetPrediction (if last_layer=True)