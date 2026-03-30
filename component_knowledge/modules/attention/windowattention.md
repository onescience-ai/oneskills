component:

  meta:
    name: SelfAttention
    alias: Efficient Linear Self-Attention
    version: 1.0
    domain: efficient_deep_learning
    category: neural_network
    subcategory: attention_mechanism
    author: OneScience
    license: Apache-2.0
    tags:
      - self_attention
      - efficient_attention
      - linear_complexity
      - dual_softmax
      - einsum

  concept:

    description: >
      SelfAttention 是一个基于高效计算的高层自注意力封装模块。
      底层利用了基于爱因斯坦求和约定 (einsum) 的线性注意力算法 (`linear_attn`)。
      它通过分别对 Query（在特征维度）和 Key（在序列维度）执行 Softmax，然后利用矩阵乘法结合律优先计算 K 和 V 的乘积，
      成功避开了 O(N^2) 的注意力矩阵计算。此版本还专门修复了 Padding 掩码传递的问题，并内置了将多头拆分计算的架构潜力。

    intuition: >
      传统的 Attention 是让每一个 Query 都去和所有的 Key 打分，计算出 N x N 的关系网。
      这个模块的思路是：我们先让所有的 Key 和 Value 内部开个会，由于 Key 在序列维度做过 Softmax，
      它相当于一个全局的“重要性分布”，开会的结果就是提炼出一个尺寸固定为 D x D 的“全局知识浓缩包 (Context)”。
      接下来，所有的 Query 只需要去这个浓缩包里按需提取信息就可以了。这种“先聚合后分发”的方式，把计算量从 O(N^2) 降到了 O(N)。

    problem_it_solves:
      - 解决长序列建模中 O(N^2) 复杂度的显存与计算时间双重瓶颈。
      - 修复了某些开源线性注意力实现中，全局聚合阶段无法正确屏蔽 Padding 噪声（导致无效 token 污染全局 Context）的缺陷。
      - 通过 `split_at_index` 机制，为未来实现“一部分头关注局部，一部分头关注全局”的混合注意力（Hybrid Attention）预留了接口设计。

  theory:

    formula:
      
      dual_softmax_linear_attention:
        expression: Output = Softmax(Q_{scaled}, dim=-1) * (Softmax(K, dim=-2)^T * V)

    variables:

      Context (K^T * V):
        name: 全局上下文矩阵
        shape: [B, H, D, D]
        description: 融合了 Key 的空间分布和 Value 的特征表示的固定大小上下文特征。

      kv_mask:
        name: 键值掩码
        description: 用于在计算 Softmax(K) 之前，将无效的 Padding token 填充为极小负值，防止其参与 Context 聚合。

  structure:

    architecture: efficient_attention_with_head_splitting

    pipeline:

      - name: QKVProjection
        operation: linear_projection (生成 Q, K, V，并拆分多头)

      - name: HeadSplitting
        operation: split_at_index (在头维度划分局部注意力头和全局注意力头)

      - name: KeyValueMasking
        operation: masked_fill (注入 kv_mask，用极大负值屏蔽无关的 K 特征)

      - name: DualSoftmaxNormalization
        operation: softmax_on_different_dims (对 Q 在特征维度做 Softmax，对 K 在序列维度做 Softmax)

      - name: ContextAggregation
        operation: einsum_matmul (将 K 和 V 收缩为 D x D 维度的 Context 矩阵)

      - name: QueryExtraction
        operation: einsum_matmul (将 Q 与 Context 矩阵相乘，得出每个 token 的更新特征)

      - name: HeadMergingAndOutput
        operation: concat_and_linear (拼接被拆分的注意力头，并通过输出投影层)

  interface:

    parameters:

      dim:
        type: int
        description: 输入和输出的总特征维度

      heads:
        type: int
        description: 注意力头的数量

      dim_head:
        type: int
        default: null
        description: 每个注意力头的维度。如果为 null，则自动推断为 dim // heads

      dropout:
        type: float
        default: 0.0
        description: 输出特征的随机丢弃率

      scale:
        type: float
        default: null
        description: 自定义缩放因子。如果不提供，默认采用 1 / sqrt(dim_head) 并对全局输出进行校正。

    inputs:

      x:
        type: Tensor
        shape: [B, N, dim]
        dtype: float32
        description: 输入序列特征

      mask:
        type: Tensor
        shape: [B, N]
        default: null
        description: 布尔类型掩码。True 表示有效 token，False 表示需要被屏蔽的 Padding。

    outputs:

      output:
        type: Tensor
        shape: [B, N, dim]
        description: 经过线性注意力聚合后的序列特征

  types:

    SequenceFeatures:
      shape: [B, N, dim]
      description: 包含长程上下文信息的序列特征表示

  implementation:

    framework: pytorch, einops (implicitly via torch.einsum)

    code: |
      import torch
      import torch.nn as nn
      from functools import partial
      from torch import einsum

      def exists(val):
          return val is not None

      def default(value, d):
          return d if not exists(value) else value

      def max_neg_value(tensor):
          return -torch.finfo(tensor.dtype).max

      def linear_attn(q, k, v, kv_mask = None):
          dim = q.shape[-1]

          if exists(kv_mask):
              mask_value = max_neg_value(q)
              mask = kv_mask[:, None, :, None]
              k = k.masked_fill_(~mask, mask_value)
              v = v.masked_fill_(~mask, 0.)
              del mask

          q = q.softmax(dim=-1)
          k = k.softmax(dim=-2)

          q = q * dim ** -0.5

          context = einsum('bhnd,bhne->bhde', k, v)
          attn = einsum('bhnd,bhde->bhne', q, context)
          return attn.reshape(*q.shape)

      def split_at_index(dim, index, t):
          pre_slices = (slice(None),) * dim
          l = (*pre_slices, slice(None, index))
          r = (*pre_slices, slice(index, None))
          return t[l], t[r]

      class SelfAttention(nn.Module):
          """
          基于高效计算的自注意力封装 (增强版)。

          该模块是对 linear_attn 的高层封装。
          它保留了原始代码独特的 Split 机制（将头分为局部/全局两部分，尽管默认配置下全部为全局），
          并修复了原版无法传递 Mask 的问题。

          增强特性：
          1. Mask 支持：允许传入 (B, N) 的掩码，防止 Padding 干扰全局上下文聚合。
          2. 自定义 Scale：允许覆盖默认的 1/sqrt(dim) 缩放因子。

          Args:
              dim (int): 输入特征维度。
              heads (int): 注意力头数。
              dim_head (int, optional): 每个注意力头的维度。如果为 None，则默认为 dim // heads。
              dropout (float, optional): Dropout 概率。默认值: 0.0。
              scale (float, optional): 自定义缩放因子。如果为 None，默认使用 dim_head ** -0.5。

          形状:
              输入 x: (B, N, C)。
              输入 mask (可选): (B, N) 的布尔掩码。
              输出: (B, N, C)。

          Example:
              >>> sa = SelfAttention(dim=64, heads=8)
              >>> x = torch.randn(8, 256, 64)
              >>> mask = torch.ones(8, 256).bool() # 假设全有效
              >>> out = sa(x, mask=mask)
          """
          def __init__(self, dim, heads, dim_head=None, dropout=0., scale=None):
              super().__init__()
              assert dim_head or (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
              
              d_heads = default(dim_head, dim // heads)

              self.heads = heads
              self.d_heads = d_heads
              
              self.default_scale = d_heads ** -0.5
              self.custom_scale = scale
              
              self.global_attn_heads = heads
              self.global_attn_fn = linear_attn 

              self.to_q = nn.Linear(dim, d_heads * heads, bias=False)
              self.kv_heads = heads
              self.to_k = nn.Linear(dim, d_heads * heads, bias=False)
              self.to_v = nn.Linear(dim, d_heads * heads, bias=False)

              self.to_out = nn.Linear(d_heads * heads, dim)
              self.dropout = nn.Dropout(dropout)

          def forward(self, x, mask=None):
              """
              Args:
                  x (Tensor): 输入特征 [B, N, C]
                  mask (Tensor, optional): 掩码 [B, N]。True 表示有效，False 表示 Padding。
              """
              # 投影
              q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

              # 提取维度信息
              b, t, e, h, dh = *q.shape, self.heads, self.d_heads

              # Reshape & Transpose: [B, N, H*D] -> [B, N, H, D] -> [B, H, N, D]
              merge_heads = lambda x: x.reshape(b, t, h, dh).transpose(1, 2)
              q, k, v = map(merge_heads, (q, k, v))

              out = []

              split_index_fn = partial(split_at_index, 1, 0)

              (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))

              _, has_global = map(lambda x: x.shape[1] > 0, (lq, q))

              if has_global:
                  global_out = self.global_attn_fn(q, k, v, kv_mask=mask)
                  
                  if self.custom_scale is not None:
                      scale_correction = self.custom_scale / self.default_scale
                      global_out = global_out * scale_correction
                      
                  out.append(global_out)

              # 拼接
              attn = torch.cat(out, dim=1)
              # Reshape back: [B, H, N, D] -> [B, N, H*D]
              attn = attn.transpose(1, 2).reshape(b, t, -1)
              
              return self.dropout(self.to_out(attn))

  skills:

    build_efficient_self_attention:

      description: 搭建能够有效处理长序列的线性注意力基础模块

      inputs:
        - dim
        - heads
        - dim_head

      prompt_template: |
        创建一个 SelfAttention 层（基于线性高效计算的版本）。
        输入通道为 {{dim}}，头数配置为 {{heads}}。

    diagnose_linear_attention_masking:

      description: 诊断线性注意力中的掩码屏蔽失败与全局污染问题

      checks:
        - mask_in_value_tensor (确保 `v = v.masked_fill_(~mask, 0.)`，如果不将 Padding 的 Value 设为 0，即便 K 被设置了极小值，Softmax 的分母计算仍会导致全局 Context 被微弱的噪声污染)
        - max_neg_value_overflow (确保使用 `-torch.finfo(tensor.dtype).max` 而非硬编码 `-1e9`，以防止在 fp16 混合精度下溢出)

  knowledge:

    usage_patterns:

      efficient_long_sequence_processing:
        pipeline:
          - TokenEmbedding
          - PositionalEncoding
          - SelfAttention (O(N) Complexity)
          - FFN

    design_patterns:

      head_splitting_architecture:
        structure:
          - 通过 `partial(split_at_index, 1, 0)`，架构保留了在头维度（`dim=1`）切分 Q, K, V 的能力。
          - 这种设计允许我们在未来扩展时，比如让前一半的头（Local Heads）去执行标准 O(N^2) 局部滑动窗口注意力提取精细高频特征，后一半的头（Global Heads）执行 `linear_attn` 提取低频全局特征。

      dual_softmax_attention:
        structure:
          - 对 Query 执行 `q.softmax(dim=-1)` 提取通道权重，对 Key 执行 `k.softmax(dim=-2)` 提取空间重要性权重，这是 Efficient Attention 论文定义的一种极其优雅的正交归一化方法。

    hot_models:

      - model: Efficient Attention (EA)
        year: 2021
        role: 首批提出双重 Softmax 与矩阵结合律优化复杂度的架构
        architecture: Linear Attention

    model_usage_details:

      Long_Document_Encoder:
        dim: 512
        heads: 8
        scale: custom_value (如需调整温度)

    best_practices:
      - 强烈建议在包含大量 Padding 的变长序列 batch 训练中，显式传入 `mask` 参数，否则由于 O(N) 注意力的全局汇聚特性，Padding 部分的信息会比标准 Attention 更容易污染有效 token 的特征。

    anti_patterns:
      - 在序列极短（如 N < 128）的场景下使用此模块。因为 D x D 大小的矩阵乘法在此时比 N x N 更加耗时且低效。
      - 误用 `mask` 参数的布尔逻辑（输入了 False 代表有效，True 代表 Padding），代码内部使用的是 `~mask`，所以期望的逻辑必须是 True 为有效区域。

    paper_references:

      - title: "Efficient Attention: Attention with Linear Complexities"
        authors: Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, Hongsheng Li
        year: 2021
        journal: WACV

  graph:

    is_a:
      - AttentionMechanism
      - EfficientTransformerComponent
      - LinearComplexityAttention

    part_of:
      - EfficientTransformers
      - SequenceModeling

    depends_on:
      - torch.einsum
      - dual_softmax
      - split_at_index

    compatible_with:
      inputs:
        - SequenceFeatures
      outputs:
        - SequenceFeatures