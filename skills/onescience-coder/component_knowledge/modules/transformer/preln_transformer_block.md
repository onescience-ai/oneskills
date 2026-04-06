component:

  meta:
    name: PreLNTransformerBlock
    alias: PreNormConcatPosBlock
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: transformer_block
    author: OneScience
    license: Apache-2.0
    tags:
      - transformer_block
      - pre_norm
      - positional_encoding_concat
      - latent_space
      - cluster_routing


  concept:

    description: >
      带有预归一化 (Pre-LayerNorm) 的 Transformer 注意力块。
      该模块采用标准的 Pre-Norm 结构以保证深层网络的梯度稳定。
      其最大特色在于计算注意力之前，不采用传统的特征与位置编码相加策略，
      而是将特征与显式的位置编码在通道维度上进行拼接 (Concat)，并在注意力计算后通过线性层降维还原。

    intuition: >
      在处理一些强空间依赖的物理模型时，如果把“特征”和“坐标位置”直接加在一起，
      网络在后面很容易搞混“这到底是因为它本身的值大，还是因为它的坐标值大？”。
      这个模块采用拼接策略，相当于给每个人发了一张“身份证”（位置编码），
      并且要求大家在开会（Attention）交流时，必须把特征和身份证并排放在桌子上。
      这样其他人在计算相似度时，能清晰地区分出内容特征和位置特征。

    problem_it_solves:
      - 标准相加式位置编码在深层网络中被特征数值淹没的问题
      - 潜空间 (Latent Space) 或簇级别 (Cluster-level) 节点间全局交互时拓扑结构的严重丢失
      - 提高对不规则物理网格或图节点的空间感知锐度


  theory:

    formula:

      attention_with_concat_pos:
        expression: |
          W'_1 = \text{LayerNorm}_1(W)
          W'_{pos} = \text{Concat}(W'_1, \text{posenc}, \text{dim}=-1)
          W_{attn} = \text{MultiheadAttention}(Q=W'_{pos}, K=W'_{pos}, V=W'_{pos})
          W_3 = W + \text{Linear}_{proj}(W_{attn})

      mlp_update:
        expression: |
          W_4 = \text{LayerNorm}_2(W_3)
          W_{out} = W_3 + \text{MLP}(W_4)

    variables:

      W:
        name: NodeFeatures
        shape: [batch, K, w_size]
        description: 簇或序列节点的特征向量

      posenc:
        name: PositionalEncoding
        shape: [batch, K, 4 * pos_length]
        description: 显式注入的空间位置编码

      Linear_{proj}:
        name: DimensionReduction
        description: 由于拼接操作导致维度膨胀，注意力计算后需要投影回 w_size 维度


  structure:

    architecture: pre_norm_concat_pos_transformer

    pipeline:

      - name: PreNorm_1
        operation: layer_norm

      - name: PositionInjection
        operation: torch.cat (在最后一维拼接特征与位置编码)

      - name: SelfAttention
        operation: nn.MultiheadAttention (基于扩充后的维度计算注意力)

      - name: DimensionRestore
        operation: linear_projection (恢复至 w_size)

      - name: AttnResidual
        operation: add

      - name: PreNorm_2
        operation: layer_norm

      - name: FeedForward
        operation: one_mlp (激活函数为 ReLU)

      - name: MlpResidual
        operation: add


  interface:

    parameters:

      w_size:
        type: int
        description: 输入和输出的特征维度 (即簇特征的通道数)

      pos_length:
        type: int
        description: 基础位置编码的长度（代码内固定假定实际传入的位置编码维度为 4 * pos_length）

      n_heads:
        type: int
        description: 多头注意力机制的头数，要求 (w_size + 4 * pos_length) 必须能被 n_heads 整除

    inputs:

      W:
        type: Tensor
        shape: [batch, K, w_size]
        dtype: float32
        description: 当前层的输入节点/序列特征

      attention_mask:
        type: Tensor
        shape: "[batch * n_heads, K, K] 或 [K, K]"
        dtype: float32 / bool
        description: 传给 MultiheadAttention 的掩码矩阵

      posenc:
        type: Tensor
        shape: [batch, K, 4 * pos_length]
        dtype: float32
        description: 待拼接的位置编码

    outputs:

      output:
        type: Tensor
        shape: [batch, K, w_size]
        description: 经过全局注意力交互与 MLP 映射后的更新特征


  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules.mlp.onemlp import OneMlp

      class PreLNTransformerBlock(nn.Module):
          """带有预归一化和 Concat 位置编码的 Transformer 块"""
          def __init__(self, w_size, pos_length, n_heads):    
              super(PreLNTransformerBlock, self).__init__()
              self.ln1 = nn.LayerNorm(w_size)

              # 核心：扩展注意力计算的嵌入维度
              embed_dim = w_size + 4 * pos_length 
              
              self.attention = nn.MultiheadAttention(
                  embed_dim=embed_dim, num_heads=n_heads, batch_first=True
              )
              self.linear = nn.Linear(embed_dim, w_size)
              self.ln2 = nn.LayerNorm(w_size)

              self.mlp = OneMlp(
                  style="StandardMLP", input_dim=w_size, hidden_dims=[w_size], 
                  output_dim=w_size, activation='relu',
              )

          def forward(self, W, attention_mask, posenc):
              W1 = self.ln1(W)
              
              # 在通道维度拼接位置编码
              W1_posenc = torch.cat([W1, posenc], dim=-1)

              attn_out = self.attention(W1_posenc, W1_posenc, W1_posenc, attn_mask=attention_mask)[0]
              
              W3 = W + self.linear(attn_out)
              W4 = self.ln2(W3)
              W5 = self.mlp(W4)
              W6 = W3 + W5

              return W6


  skills:

    build_preln_concat_block:

      description: 构建一个通过拼接保护位置信息的 Transformer 块

      inputs:
        - w_size
        - pos_length
        - n_heads

      prompt_template: |

        请实例化 PreLNTransformerBlock。
        请仔细检查 n_heads 是否能整除 (w_size + 4 * pos_length)，如果不能则会触发 PyTorch MultiheadAttention 的初始化错误。


    diagnose_concat_attention:

      description: 排查由于位置拼接引发的注意力计算和维度错误

      checks:
        - head_dimension_indivisible (拼接后的维度无法被 n_heads 整除)
        - posenc_shape_mismatch (传入的 posenc 最后一维不是 4 * pos_length 导致计算错位)



  knowledge:

    usage_patterns:

      latent_cluster_routing:

        pipeline:
          - Encoder (将物理场压缩为 K 个簇/Latent Tokens)
          - Generate 4x Positional Encodings
          - Multiple PreLNTransformerBlocks (W 与 posenc 拼接交互)
          - Decoder (映射回物理场)


    best_practices:

      - 这种模块非常适合处理图数据（Graphs）或点云（Point Clouds），因为无序节点对位置信息极端敏感，相加策略容易让节点“迷路”，拼接策略则能提供最强的保真度。
      - 当 `w_size` 较小而 `pos_length` 较大时，注意力计算主要由位置相似度主导；反之则由特征相似度主导。可据此调整超参数。


    anti_patterns:

      - 在标准的 NLP 文本任务中使用此模块。NLP 词向量维数（通常 768 或更高）已经通过训练学习了非常好的表达，强制扩充维度会大幅增加 `nn.Linear` 降维时的参数量，且容易过拟合。



  graph:

    is_a:
      - TransformerBlock
      - SpatialAttentionModule

    part_of:
      - GraphNeuralNetworks
      - LatentSpaceModels

    depends_on:
      - MultiheadAttention
      - LayerNorm
      - OneMlp

    variants:
      - AdditivePositionalTransformer

    used_in_models:
      - Cluster-based Neural Operators
      - Point Cloud Transformers

    compatible_with:

      inputs:
        - NodeFeatures
        - PositionalEncoding

      outputs:
        - NodeFeatures