component:

  meta:
    name: MultiHeadAttention
    alias: Standard Multi-Head Self-Attention
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: attention_mechanism
    author: OneScience
    license: Apache-2.0
    tags:
      - multi_head_attention
      - self_attention
      - causal_masking
      - transformer
      - sequence_modeling

  concept:

    description: >
      多头自注意力机制 (Multi-Head Self-Attention) 是 Transformer 架构的基础核心模块。
      该模块实现了标准的缩放点积注意力，并支持多头并行计算。值得注意的是，
      此实现采用了一种特殊的预分头结构：在通过线性投影层生成 Q、K、V 之前，
      首先将输入张量 reshape 为多头格式，这意味着每个注意力头拥有完全独立且隔离的线性变换权重矩阵 (dim_head, dim_head)。

    intuition: >
      如果只用一个“注意力头”去阅读一句话，模型可能只能注意到词与词之间的主谓关系；
      而“多头”机制就像是雇佣了多个具有不同专业视角的阅读理解专家。有的专家专门寻找时间状语，
      有的专家专门分析情感色彩。这些专家（注意力头）在平行的子空间中各自计算注意力分数，
      最后将他们的意见拼接在一起，使得模型能够全面地捕捉序列中丰富且不同的上下文特征。

    problem_it_solves:
      - 解决传统循环神经网络 (RNN/LSTM) 在处理长序列时存在的长距离依赖衰减问题。
      - 突破序列数据处理的串行计算瓶颈，实现高度可并行化的矩阵运算。
      - 解决单头注意力无法同时关注多个不同表征子空间信息的问题。

  theory:

    formula:
      
      scaled_dot_product:
        expression: Attention(Q, K, V) = Softmax(Q * K^T / sqrt(d_k) + Mask) * V

      multi_head_concatenation:
        expression: MultiHead(X) = Concat(head_1, ..., head_h) * W^O

    variables:

      d_k:
        name: DimensionOfKey
        description: 键向量的维度（即 dim_head），用于缩放点积以防止 Softmax 陷入梯度消失的饱和区。

      Mask:
        name: AttentionMask
        description: 掩码矩阵，对于 Padding 为 -inf，对于自回归的未来词也为 -inf，以阻止信息泄漏。

  structure:

    architecture: standard_transformer_attention

    pipeline:

      - name: InputReshape
        operation: reshape_and_permute (将展平的输入预先切分为多头独立张量)

      - name: IndependentQKVProjection
        operation: linear_projection (对各个头独立执行映射)

      - name: ScaledDotProduct
        operation: matmul_and_scale (计算 Q 和 K^T 的点积并乘以缩放因子)

      - name: Masking
        operation: masked_fill (注入因果掩码 Causal Mask 或普通 Padding Mask，将其设为极小值)

      - name: AttentionWeights
        operation: softmax_and_dropout (将点积转化为概率分布)

      - name: ValueAggregation
        operation: matmul (利用注意力权重对 V 进行加权求和)

      - name: HeadConcatenationAndOutput
        operation: rearrange_and_linear (利用 einops 将多个头拼接合并，并通过最后的输出映射层)

  interface:

    parameters:

      dim:
        type: int
        description: 输入和最终输出的特征总维度

      heads:
        type: int
        default: 8
        description: 注意力头的数量

      dim_head:
        type: int
        default: 64
        description: 每个独立的注意力头所处理的子空间特征维度

      dropout:
        type: float
        default: 0.0
        description: 对注意力分布概率矩阵和最终输出应用的随机丢弃率

      scale:
        type: float
        default: null
        description: 自定义的点积缩放因子。若不提供，默认使用 `1 / sqrt(dim_head)`

      is_causal:
        type: bool
        default: false
        description: 是否开启因果掩码（Causal Masking），用于 GPT 风格的自回归生成任务

    inputs:

      x:
        type: Tensor
        shape: [B, N, C]
        dtype: float32
        description: 批次序列输入（C 必须等于 heads * dim_head）

      mask:
        type: Tensor
        shape: "[B, N] 或 [B, 1, 1, N]"
        default: null
        description: 布尔类型或 0/1 掩码，用于标定哪些 token 是有效的，哪些需要被屏蔽

    outputs:

      output:
        type: Tensor
        shape: [B, N, dim]
        description: 注意力机制融合上下文特征后的输出序列

  types:

    SequenceFeatures:
      shape: [B, N, C]
      description: 通用的文本或信号序列特征张量

  implementation:

    framework: pytorch, einops

    code: |
      import torch
      import torch.nn as nn
      from einops import rearrange

      class MultiHeadAttention(nn.Module):
          """
          多头自注意力机制 (Multi-Head Self-Attention) 。

          实现了标准的点积注意力：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V。
          此实现保留了原始代码的特殊结构：即在通过线性层之前先将输入 Reshape 为多头格式。
          这通常意味着输入 Linear 层的权重矩阵是分头独立的 (H, D_head, D_head)。

          Args:
              dim (int): 输入特征维度 (注意：此参数在当前特定实现中仅用于 to_out 的输出维度，
                         输入投影层的维度由 heads * dim_head 隐式决定)。
              heads (int, optional): 注意力头数。默认值: 8。
              dim_head (int, optional): 每个注意力头的维度。默认值: 64。
              dropout (float, optional): Dropout 概率。默认值: 0.0。
              scale (float, optional): 自定义缩放因子。如果为 None，则使用 dim_head ** -0.5。默认值: None。
              is_causal (bool, optional): 是否应用因果掩码（Causal Masking），用于自回归任务。默认值: False。

          形状:
              输入 x: (B, N, C)，其中 C 必须等于 heads * dim_head (基于原代码逻辑)。
              输入 mask (可选): (B, N) 或 (B, 1, 1, N) 的布尔掩码。
              输出: (B, N, dim)。

          Example:
              >>> attn = MultiHeadAttention(dim=128, heads=8, dim_head=16)
              >>> # 注意：输入维度必须匹配 heads * dim_head = 8 * 16 = 128
              >>> x = torch.randn(8, 100, 128)
              >>> out = attn(x)
              >>> out.shape
              torch.Size([8, 100, 128])
          """
          def __init__(
              self, 
              dim, 
              heads=8, 
              dim_head=64, 
              dropout=0., 
              scale=None,
              is_causal=False,
              **kwargs
          ):
              super().__init__()
              inner_dim = dim_head * heads
              self.dim_head = dim_head
              self.heads = heads
              self.scale = scale if scale is not None else dim_head ** -0.5
              self.is_causal = is_causal
              
              self.softmax = nn.Softmax(dim=-1)
              self.dropout = nn.Dropout(dropout)
              
              self.to_q = nn.Linear(dim_head, dim_head, bias=False)
              self.to_k = nn.Linear(dim_head, dim_head, bias=False)
              self.to_v = nn.Linear(dim_head, dim_head, bias=False)
              
              self.to_out = nn.Sequential(
                  nn.Linear(inner_dim, dim),
                  nn.Dropout(dropout)
              )

          def forward(self, x, mask=None):
              """
              前向传播。
              Args:
                  x (Tensor): 输入张量 [B, N, C]
                  mask (Tensor, optional): 注意力掩码。True 表示保留，False 表示屏蔽 (或相反，取决于具体实现习惯，这里通常处理为加性掩码)。
              """
              # B N C
              B, N, C = x.shape
              
              # x: [B, N, C] -> [B, N, H, D] -> [B, H, N, D]
              x = x.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() 
              
              # 投影
              q = self.to_q(x)
              k = self.to_k(x)
              v = self.to_v(x)
              
              # 点积注意力
              dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
              
              # 处理掩码
              if mask is not None or self.is_causal:
                  mask_value = -torch.finfo(dots.dtype).max

                  # 因果掩码
                  if self.is_causal:
                      i, j = dots.shape[-2:]
                      causal_mask = torch.ones(i, j, device=x.device).triu(j - i + 1).bool()
                      dots.masked_fill_(causal_mask, mask_value)

                  if mask is not None:
                      if mask.ndim == 2:
                          mask = mask.unsqueeze(1).unsqueeze(1)
                      dots.masked_fill_(~mask.bool(), mask_value)

              attn = self.softmax(dots)
              attn = self.dropout(attn)
              
              # 加权求和
              res = torch.matmul(attn, v)  # B H N D
              
              # 拼接多头
              res = rearrange(res, 'b h n d -> b n (h d)')
              
              return self.to_out(res)

  skills:

    build_standard_attention:

      description: 构建适用于基础特征提取或自回归生成的注意力模块

      inputs:
        - dim
        - heads
        - dim_head
        - is_causal

      prompt_template: |
        创建一个多头注意力层。
        参数：
        要求输入维度和输出维度一致，dim = {{dim}}。
        设置头数 {{heads}} 以及每个头的维度 {{dim_head}}。
        如果用于 Decoder（如 GPT），确保 is_causal = {{is_causal}}。

    diagnose_masking_bugs:

      description: 诊断自注意力中由于掩码逻辑错误导致的信息泄漏或 NaN

      checks:
        - mask_value_too_large (检查用于屏蔽的负数极值 `mask_value` 在使用 fp16 混合精度时是否导致了下溢/NaN，代码中使用 `-torch.finfo(dots.dtype).max` 是安全的做法)
        - causal_mask_logic_error (检查下三角掩码是否被错误反转，导致历史 token 被屏蔽而未来 token 被泄露)

  knowledge:

    usage_patterns:

      autoregressive_decoder:
        pipeline:
          - LayerNorm
          - MultiHeadAttention (is_causal=True)
          - LayerNorm
          - MLP

      bidirectional_encoder:
        pipeline:
          - MultiHeadAttention (is_causal=False, with Padding Mask)
          - LayerNorm
          - MLP

    design_patterns:

      pre_reshape_projection:
        structure:
          - 典型的 PyTorch 实现是先经过一个大的 `nn.Linear(dim, dim*3)`，再 `view` 切分成多头。
          - 该代码先将 `[B, N, C]` 变形为 `[B, H, N, D]`，再经过 `nn.Linear(dim_head, dim_head)`。这使得参数矩阵大小仅为 `(dim_head, dim_head)`，在某些需要严格限制通道混合范围或执行参数共享的特殊架构中更有优势。

    hot_models:

      - model: Transformer
        year: 2017
        role: 注意力机制的开山之作，现代大模型的基石
        architecture: Encoder-Decoder
        attention_type: Multi-Head Self-Attention / Cross-Attention

      - model: BERT
        year: 2018
        role: 顶级的自然语言理解双向编码器
        architecture: Encoder-only
        attention_type: Multi-Head Self-Attention (is_causal=False)

      - model: GPT-2 / GPT-3
        year: 2019-2020
        role: 自回归语言模型的范式定义者
        architecture: Decoder-only
        attention_type: Multi-Head Self-Attention (is_causal=True)

    model_usage_details:

      Transformer_Base:
        dim: 512
        heads: 8
        dim_head: 64

      GPT_3_Small:
        dim: 768
        heads: 12
        dim_head: 64

    best_practices:
      - 无论在何种任务中，`scale` 因子必须存在。如果没有 `* self.scale`，点积后的数值方差会随着 `dim_head` 的增大而激增，使得 Softmax 函数的梯度极度趋近于 0，导致模型无法训练。
      - 当 `mask` 被激活时，使用 `~mask.bool()` 和 `triu` 来安全地进行掩码替换是工业界的标准防错写法。

    anti_patterns:
      - 未能保证输入的特征维度 `C` 等于 `heads * dim_head`，由于此代码采用了提前 Reshape 的特殊结构，维度不匹配将导致前向传播时直接引发运行时异常 (RuntimeError)。

    paper_references:

      - title: "Attention Is All You Need"
        authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
        year: 2017
        journal: NeurIPS

  graph:

    is_a:
      - AttentionMechanism
      - NeuralNetworkComponent
      - StandardTransformerComponent

    part_of:
      - TransformerBlock
      - SequenceToSequenceModel

    depends_on:
      - Softmax
      - MatMul
      - CausalMask
      - einops.rearrange

    compatible_with:
      inputs:
        - SequenceFeatures
      outputs:
        - SequenceFeatures