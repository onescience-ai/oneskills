component:

  meta:
    name: FlashAttention
    alias: Memory-Efficient Exact Attention
    version: 2.0
    domain: deep_learning
    category: neural_network
    subcategory: attention_mechanism
    author: OneScience
    license: Apache-2.0
    tags:
      - flash_attention
      - sdpa
      - memory_efficient
      - io_awareness
      - transformer

  concept:

    description: >
      FlashAttention 是一种通过感知 IO（内存读写）来实现快速且节省显存的精确注意力机制模块。
      该代码是对 PyTorch 2.0+ 内部的 `F.scaled_dot_product_attention` 的高级封装。
      它不改变传统注意力机制的数学等价性，而是通过软硬件协同优化，大幅减少了在 GPU 高带宽内存（HBM）和片上静态随机存储器（SRAM）之间的数据搬运。

    intuition: >
      在传统的注意力计算中，模型会把一个巨大的 N x N 注意力分数矩阵（例如 8K 序列就是 8000x8000）完整地写入显存（HBM），然后再读取出来做 Softmax，这不仅非常慢，而且极易导致显存溢出（OOM）。
      FlashAttention 像是一个极其聪明的“流水线工人”，它采用分块（Tiling）和重计算（Recomputation）策略，
      把 Q、K、V 切成小块放入访问速度极快的 SRAM 中，算完一小块直接累加结果，最终一气呵成输出最终的张量，彻底消灭了 N x N 中间矩阵的显存占用。

    problem_it_solves:
      - 解决传统自注意力计算中时间和空间复杂度随序列长度呈二次方 O(N^2) 增长导致的显存溢出 (OOM) 难题。
      - 突破 Transformer 处理超长文本（Long-context, 如 32K, 128K）的硬件限制。
      - 提升大模型前向和反向传播的绝对计算速度，提升 GPU 算力利用率 (MFU)。

  theory:

    formula:

      hardware_optimized_attention:
        expression: Output = SDPA(Q, K, V, mask, dropout, is_causal) 
        # SDPA 在底层使用 Tiling 将 Softmax(Q*K^T) * V 融合为一个内核 (Kernel) 操作

    variables:

      HBM:
        name: High_Bandwidth_Memory
        description: 显卡上的全局显存（容量大但读写慢，FlashAttention 极力减少对其的访问）

      SRAM:
        name: Static_Random_Access_Memory
        description: 流式多处理器 (SM) 上的超快速缓存（容量极小但速度极快，用于核心点积计算）

  structure:

    architecture: io_aware_attention

    pipeline:

      - name: QKVProjection
        operation: linear_projection (分别生成 Q, K, V)

      - name: ReshapeAndTranspose
        operation: einops_rearrange (将张量变形为多头注意力所需的 [B, H, N, D] 格式，并确保连续性)

      - name: FlashAttentionCore
        operation: scaled_dot_product_attention (调用底层融合算子完成点积、掩码、Softmax和V值聚合)

      - name: OutputReshape
        operation: einops_rearrange (展平多头特征)

      - name: OutputProjection
        operation: linear_and_dropout (最终的线性变换)

  interface:

    parameters:

      dim:
        type: int
        description: 输入序列特征的总维度大小

      heads:
        type: int
        default: 8
        description: 多头注意力的头数

      dim_head:
        type: int
        default: 64
        description: 每个注意力头的维度特征数

      dropout:
        type: float
        default: 0.0
        description: 注意力权重的随机丢弃率

      scale:
        type: float
        default: null
        description: 缩放因子。若不填，默认采用 1 / sqrt(dim_head)

      is_causal:
        type: bool
        default: false
        description: 是否开启因果掩码（Causal Mask），若开启，序列中的 token 将无法看到其未来的 token

    inputs:

      x:
        type: Tensor
        shape: [B, N, dim]
        dtype: float32_or_fp16_bf16
        description: 批次序列特征输入

      mask:
        type: Tensor
        shape: "[B, N] 或 [B, 1, 1, N]"
        default: null
        description: 布尔类型或浮点类型的注意力掩码

    outputs:

      output:
        type: Tensor
        shape: [B, N, dim]
        description: 提取特征后的序列张量

  types:

    SequenceEmbedding:
      shape: [B, N, dim]
      description: 批次文本或序列的嵌入表达

  implementation:

    framework: pytorch, einops

    code: |
      import torch
      import torch.nn as nn
      import torch.nn.functional as F
      from einops import rearrange, repeat

      import torch
      import torch.nn as nn
      import torch.nn.functional as F
      from einops import rearrange

      class FlashAttention(nn.Module):
          """
          FlashAttention 模块。

          利用 PyTorch 的 F.scaled_dot_product_attention 实现的高效注意力机制。
          该实现完全兼容 PyTorch 2.0+ 的 SDPA 加速（FlashAttention V2, Memory-Efficient Attention, C++ Math）。
          它通过 Tiling 技术减少 HBM (高带宽内存) 访问次数，显著提升长序列的训练和推理速度。

          Args:
              dim (int): 输入特征维度。
              heads (int, optional): 注意力头数。默认值: 8。
              dim_head (int, optional): 每个注意力头的维度。默认值: 64。
              dropout (float, optional): Dropout 概率。默认值: 0.0。
              scale (float, optional): 自定义缩放因子。如果为 None，则使用 dim_head ** -0.5。默认值: None。
              is_causal (bool, optional): 是否应用因果掩码（Causal Masking），用于自回归任务。默认值: False。

          形状:
              输入 x: (B, N, C)，其中 B 是批次大小，N 是序列长度，C 是特征维度。
              输入 mask (可选): (B, N) 或 (B, 1, 1, N) 的布尔/浮点掩码。
              输出: (B, N, C)。

          Example:
              >>> attn = FlashAttention(dim=64, heads=8, dim_head=8)
              >>> x = torch.randn(8, 128, 64)
              >>> out = attn(x)
              >>> print(out.shape)
              torch.Size([8, 128, 64])
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
              self.dropout_p = dropout 
              self.is_causal = is_causal
              
              self.to_q = nn.Linear(dim, inner_dim, bias=False)
              self.to_k = nn.Linear(dim, inner_dim, bias=False)
              self.to_v = nn.Linear(dim, inner_dim, bias=False)
              
              self.to_out = nn.Sequential(
                  nn.Linear(inner_dim, dim),
                  nn.Dropout(dropout)
              )

          def forward(self, x, mask=None):
              """
              前向传播。
              
              Args:
                  x (Tensor): 输入张量 [B, N, C]
                  mask (Tensor, optional): 注意力掩码。
              """
              # x shape: [batch_size, seq_len, dim]
              batch_size, seq_len, _ = x.shape
              
              # Get query, key, value projections
              q = self.to_q(x)
              k = self.to_k(x)
              v = self.to_v(x)
              
              # Reshape for multi-head attention: [B, N, H*D] -> [B, H, N, D]
              q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads).contiguous()
              k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads).contiguous()
              v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads).contiguous()
              
              # 处理 Mask
              if mask is not None:
                  if mask.ndim == 2: # [B, N] -> [B, 1, 1, N]
                      mask = mask.unsqueeze(1).unsqueeze(1)
                  pass 

              # Flash attention implementation
              attn_output = F.scaled_dot_product_attention(
                  query=q, 
                  key=k, 
                  value=v,
                  attn_mask=mask,
                  dropout_p=self.dropout_p if self.training else 0.0,
                  is_causal=self.is_causal,
                  scale=self.scale
              )

              # Reshape back: [B, H, N, D] -> [B, N, H*D]
              out = rearrange(attn_output, 'b h n d -> b n (h d)')    
              return self.to_out(out)

  skills:

    build_flash_attention:

      description: 构建内存优化级别的 Transformer 核心部件

      inputs:
        - dim
        - heads
        - dim_head
        - is_causal

      prompt_template: |
        创建一个使用 PyTorch SDPA 的 FlashAttention 层。
        参数：
        输入维度 = {{dim}}
        如果是自然语言生成（GPT类自回归模型），请将 is_causal 设为 {{is_causal}}。

    diagnose_sdpa_fallback:

      description: 分析 `scaled_dot_product_attention` 未能调用底层 Flash Cuda Kernel 而是退化为普通 Math 模式的原因

      checks:
        - non_contiguous_tensors (检查 Q, K, V 是否在 reshape 后调用了 `.contiguous()`，否则后端无法加速)
        - precision_mismatch (检查计算是否使用了 fp16/bf16，因为纯 fp32 在许多硬件架构上可能无法启用最快的 Flash Kernel)

  knowledge:

    usage_patterns:

      long_context_llm_training:
        pipeline:
          - RMSNorm
          - FlashAttention (替代标准 MHA)
          - RMSNorm
          - SwiGLU / MLP

    design_patterns:

      pytorch_sdpa_delegation:
        structure:
          - 不要在 Python 层面实现矩阵乘法和 Softmax
          - 利用 `F.scaled_dot_product_attention` 接口，将复杂的内存块分配、硬件调度自动交由 PyTorch 底层 C++ / CUDA 引擎决定（它会自动在 FlashAttention-2、xFormers 的 Memory-Efficient Attention 或 standard math 中选择最优解）。

    hot_models:

      - model: LLaMA-3 / GPT-4
        year: 2024
        role: 顶级语言大模型
        architecture: Decoder-only
        attention_type: FlashAttention-2 / 3

    model_usage_details:

      Long_Context_Pretraining:
        sequence_length: 128000
        attention_backend: FlashAttention-2
        precision: bfloat16

    best_practices:
      - 强烈推荐结合半精度（FP16 或 BF16）运行此代码，只有在混合精度下，FlashAttention 才能最大化发挥其 Tensor Core 的计算性能。
      - 在自回归生成时，如果需要传入因果掩码，可以直接设置 `is_causal=True` 而不需要手动生成下三角矩阵 `mask`，这样运行效率最高。

    anti_patterns:
      - 在 PyTorch 2.0 之前的旧版本环境运行此代码，会因为缺少 `F.scaled_dot_product_attention` 函数而抛出异常。
      - 在设置 `is_causal=True` 的同时，又手动传入一个下三角矩阵作为 `mask` 参数，会导致 API 发生冲突并报错。

    paper_references:

      - title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
        authors: Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, Christopher Ré
        year: 2022
        journal: NeurIPS

      - title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
        authors: Tri Dao
        year: 2023
        journal: ICLR

  graph:

    is_a:
      - AttentionMechanism
      - HardwareAcceleratedComponent

    part_of:
      - EfficientTransformer
      - LargeLanguageModels

    depends_on:
      - F.scaled_dot_product_attention
      - einops.rearrange

    compatible_with:
      inputs:
        - SequenceEmbedding
      outputs:
        - SequenceEmbedding