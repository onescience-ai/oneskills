component:

  meta:
    name: NystromAttention
    alias: Nyström Low-Rank Attention
    version: 1.0
    domain: efficient_deep_learning_and_ai4science
    category: neural_network
    subcategory: attention_mechanism
    author: OneScience
    license: Apache-2.0
    tags:
      - nystrom_method
      - linear_attention
      - low_rank_approximation
      - moore_penrose_inverse
      - long_sequence

  concept:

    description: >
      NystromAttention 是一种基于 Nyström 方法对标准自注意力矩阵进行低秩近似的高效注意力机制。
      它通过在长序列中提取少量的“地标节点” (Landmarks) 作为锚点，重构出全局的注意力分布。
      该模块采用 Moore-Penrose 伪逆的迭代逼近算法来保证数值稳定性，成功将标准 Transformer 的
      时间与空间复杂度从 O(N^2) 降低到线性复杂度 O(N)。

    intuition: >
      如果把计算 N 个 token 两两之间的注意力比作“让百万级群体互相握手交换信息”，计算量是灾难性的。
      Nyström 方法的思路是选举机制：先均匀选出少数具有代表性的“代表” (Landmarks)。让所有人只和代表握手，
      代表之间再互相交流。通过这种“间接传递”，模型以极小的误差和极快的速度近似还原了全局的信息交互网络。

    problem_it_solves:
      - 彻底解决标准 Transformer 在处理超长序列（如长文档、长音频）或超高分辨率物理场网格时面临的 O(N^2) 计算灾难和显存溢出问题。
      - 避免直接计算特征矩阵逆矩阵时 O(k^3) 的极高时间复杂度与数值不稳定性（通过迭代逼近伪逆实现）。

  theory:

    formula:
      
      nystrom_approximation:
        expression: Output = Softmax(Q * K_L^T) * (Softmax(Q_L * K_L^T))^+ * Softmax(Q_L * K^T) * V  # 其中 + 表示 Moore-Penrose 伪逆

      iterative_pseudo_inverse:
        expression: Z_{n+1} = 0.25 * Z_n * (13 * I - Z_n * X * (15 * I - Z_n * X * (7 * I - Z_n * X))) # 牛顿-舒尔茨高阶迭代逼近法

    variables:

      Landmarks (Q_L, K_L):
        name: 代表性地标节点
        description: 选取的代表性节点特征，通过对原始序列进行分段平均池化 (Segment Mean Pooling) 获得。

      moore_penrose_iter_pinv:
        name: 迭代伪逆计算器
        description: 不使用 torch.pinverse 直接求逆，而是通过迭代法稳定求解代表节点间的注意力矩阵之逆矩阵。

  structure:

    architecture: nystrom_low_rank_attention

    pipeline:

      - name: SequencePadding
        operation: pad_to_multiple (如果序列长度 N 无法被地标数量 m 整除，先进行尾部零填充)

      - name: QKVGeneration
        operation: linear_projection (生成 Query, Key, Value)

      - name: LandmarkExtraction
        operation: segment_mean_pooling (通过 einops.reduce 对 Q 和 K 进行分段求和/平均，提取地标特征)

      - name: SimilarityComputation
        operation: matmul_and_softmax (分别计算 Q与K_L, Q_L与K_L, Q_L与K 的相似度矩阵)

      - name: IterativePseudoInverse
        operation: newton_schulz_iteration (迭代求解中心矩阵 Softmax(Q_L * K_L^T) 的伪逆)

      - name: MatrixMultiplicationChain
        operation: chained_matmul (链式相乘还原全局注意力分布并结合 V)

      - name: ResidualConvolution
        operation: depthwise_conv2d (可选，直接加在输出上的深度残差卷积，用于增强局部信息的连续性感知)

  interface:

    parameters:

      dim:
        type: int
        description: 输入和输出特征的总维度大小

      dim_head:
        type: int
        default: 64
        description: 每个独立的注意力头所处理的子空间特征维度

      heads:
        type: int
        default: 8
        description: 注意力头数

      num_landmarks:
        type: int
        default: 256
        description: 抽取的代表性地标节点数量（即低秩矩阵的秩 k）

      pinv_iterations:
        type: int
        default: 6
        description: 伪逆迭代逼近的次数（通常 6 次足以收敛）

      residual:
        type: bool
        default: true
        description: 是否在最终输出上叠加一个基于 Value 提取的深度卷积残差分支

      residual_conv_kernel:
        type: int
        default: 33
        description: 残差卷积的核大小（推荐使用大卷积核以感受更大局部范围）

      eps:
        type: float
        default: 1e-8
        description: 用于防止除零错误的极小值平滑项

      dropout:
        type: float
        default: 0.0
        description: 最终输出的 Dropout 概率

    inputs:

      x:
        type: Tensor
        shape: [B, N, dim]
        dtype: float32
        description: 长序列或超高分辨率网格特征输入

      mask:
        type: Tensor
        shape: [B, N]
        default: null
        description: 布尔类型的掩码，用于标定哪些 token 是有效的

    outputs:

      out:
        type: Tensor
        shape: [B, N, dim]
        description: 提取特征后的序列张量（自动裁减掉初始的 Padding 部分）

  types:

    GridOrSequenceFeatures:
      shape: [B, N, dim]
      description: 包含长程依赖的展平后序列或网格特征

  implementation:

    framework: pytorch, einops

    code: |
      from math import ceil
      import torch
      from torch import nn, einsum
      import torch.nn.functional as F

      from einops import rearrange, reduce

      # helper functions
      def exists(val):
          return val is not None

      def moore_penrose_iter_pinv(x, iters = 6):
          device = x.device

          abs_x = torch.abs(x)
          col = abs_x.sum(dim = -1)
          row = abs_x.sum(dim = -2)
          z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

          I = torch.eye(x.shape[-1], device = device)
          I = rearrange(I, 'i j -> () i j')

          for _ in range(iters):
              xz = x @ z
              z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

          return z

      # main attention class
      class NystromAttention(nn.Module):
          """
          Nystrom 注意力机制 (Nyström Attention)。

          该模块通过 Nyström 方法对标准自注意力矩阵进行低秩近似，从而将 Transformer 的
          时间与空间复杂度从 $O(N^2)$ 降低到 $O(N)$。它通过提取少量的 Landmark (地标节点) 
          来重构全局的注意力矩阵，并使用 Moore-Penrose 伪逆的迭代逼近算法来保证数值稳定性。
          非常适用于处理超长序列或高分辨率网格的物理场数据。

          Args:
              dim (int): 输入和输出的特征维度。
              dim_head (int, optional): 每个注意力头的维度。默认值: 64。
              heads (int, optional): 注意力头的数量。默认值: 8。
              num_landmarks (int, optional): 用于近似的地标节点数量。默认值: 256。
              pinv_iterations (int, optional): 伪逆迭代逼近的次数。默认值: 6。
              residual (bool, optional): 是否在 Value 上添加深度卷积残差。默认值: True。
              residual_conv_kernel (int, optional): 残差卷积的核大小。默认值: 33。
              eps (float, optional): 防止除零的极小值。默认值: 1e-8。
              dropout (float, optional): Dropout 概率。默认值: 0.0。

          形状:
              输入 x: (B, N, C)，其中 N 为序列长度，C 为特征维度 (dim)。
              输入 mask: (B, N)，布尔类型的掩码。
              输出 out: (B, N, C)，形状与输入 x 保持一致。

          Example:
              >>> attn = NystromAttention(dim=128, heads=4, num_landmarks=64)
              >>> x = torch.randn(2, 1024, 128)
              >>> out = attn(x)
              >>> out.shape
              torch.Size([2, 1024, 128])
          """
          def __init__(
              self,
              dim,
              dim_head = 64,
              heads = 8,
              num_landmarks = 256,
              pinv_iterations = 6,
              residual = True,
              residual_conv_kernel = 33,
              eps = 1e-8,
              dropout = 0.
          ):
              super().__init__()
              self.eps = eps
              inner_dim = heads * dim_head

              self.num_landmarks = num_landmarks
              self.pinv_iterations = pinv_iterations

              self.heads = heads
              self.scale = dim_head ** -0.5
              self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

              self.to_out = nn.Sequential(
                  nn.Linear(inner_dim, dim),
                  nn.Dropout(dropout)
              )

              self.residual = residual
              if residual:
                  kernel_size = residual_conv_kernel
                  padding = residual_conv_kernel // 2
                  self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

          def forward(self, x, mask = None, return_attn = False, return_attn_matrices = False):
              b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

              remainder = n % m
              if remainder > 0:
                  padding = m - (n % m)
                  x = F.pad(x, (0, 0, padding, 0), value = 0)

                  if exists(mask):
                      mask = F.pad(mask, (padding, 0), value = False)

              q, k, v = self.to_qkv(x).chunk(3, dim = -1)
              q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

              if exists(mask):
                  mask = rearrange(mask, 'b n -> b () n')
                  q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

              q = q * self.scale

              l = ceil(n / m)
              landmark_einops_eq = '... (n l) d -> ... n d'
              q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
              k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

              divisor = l
              if exists(mask):
                  mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
                  divisor = mask_landmarks_sum[..., None] + eps
                  mask_landmarks = mask_landmarks_sum > 0

              q_landmarks = q_landmarks / divisor
              k_landmarks = k_landmarks / divisor

              einops_eq = '... i d, ... j d -> ... i j'
              sim1 = einsum(einops_eq, q, k_landmarks)
              sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
              sim3 = einsum(einops_eq, q_landmarks, k)

              if exists(mask):
                  mask_value = -torch.finfo(q.dtype).max
                  sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
                  sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
                  sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

              attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
              attn2_inv = moore_penrose_iter_pinv(attn2, iters)

              out = (attn1 @ attn2_inv) @ (attn3 @ v)

              if self.residual:
                  out = out + self.res_conv(v)

              out = rearrange(out, 'b h n d -> b n (h d)', h = h)
              out = self.to_out(out)
              out = out[:, -n:]

              if return_attn_matrices:
                  return out, (attn1, attn2_inv, attn3)
              elif return_attn:
                  attn = attn1 @ attn2_inv @ attn3
                  return out, attn

              return out

  skills:

    build_nystrom_attention:

      description: 为百万级极长序列任务构建低秩注意力机制

      inputs:
        - dim
        - heads
        - num_landmarks

      prompt_template: |
        创建一个 NystromAttention 层。
        参数：
        输入维度 = {{dim}}，头数 = {{heads}}。
        设置低秩逼近的地标节点数 num_landmarks = {{num_landmarks}}。

    diagnose_iterative_divergence:

      description: 诊断伪逆计算由于数值异常导致的模型崩溃或 NaN 问题

      checks:
        - moore_penrose_instability (由于 `iters` 设定过小导致伪逆矩阵发散未收敛)
        - zero_division_in_landmarks (如果使用了全掩码序列，确保 `eps` 生效，防止被 0 除)

  knowledge:

    usage_patterns:

      ultra_long_document_modeling:
        pipeline:
          - EmbeddingLayer
          - NystromAttention (替代所有标准注意力)
          - FFN
          - ClassificationHead

    design_patterns:

      iterative_pseudo_inverse:
        structure:
          - 坚决避免使用 O(N^3) 的传统矩阵求逆（如 `torch.inverse` 或 `torch.linalg.pinv`），因为在深度学习图的自动求导中它们不仅极其耗时且极不稳定。
          - 模块采用了高阶牛顿-舒尔茨法逼近，通过有限次简单的矩阵相乘加法（通常 6 次）获得极其稳定的伪逆结果，实现了彻底的并行加速。

    hot_models:

      - model: Nyströmformer
        year: 2021
        role: 高效的线性注意力语言模型与长序列视觉模型
        architecture: Nyström Approximation
        attention_type: NystromAttention

    model_usage_details:

      Long_Sequence_Standard_Config:
        num_landmarks: 64 到 256 之间 (视原始长度而定)
        pinv_iterations: 6
        residual: True

    best_practices:
      - 地标节点数 `num_landmarks` 的选取至关重要：如果过小（如 8 或 16），低秩矩阵会严重丢失原序列的信息，导致注意力退化；如果过大，又会失去 O(N) 复杂度的加速意义。推荐设为 64 或 128。
      - `residual_conv_kernel` (残差一维卷积) 能在一定程度上弥补因为地标离散化而丢失的“局部高频平滑性”，尤其在 CV 或 1D 连续信号处理中务必保持开启 `residual=True`。

    anti_patterns:
      - 在处理极短的序列（如 N < 512）时盲目套用此模型。对于短序列，提取 Landmark 和多重矩阵相乘以及迭代求伪逆带来的额外算力开销，远远超过了它所节省的 $O(N^2)$ 计算量，导致“加速反被降速”。

    paper_references:

      - title: "Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention"
        authors: Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, Vikas Singh
        year: 2021
        journal: AAAI

  graph:

    is_a:
      - AttentionMechanism
      - EfficientTransformerComponent
      - LowRankApproximation

    part_of:
      - Nyströmformer
      - LongSequenceModels

    depends_on:
      - moore_penrose_iter_pinv
      - einops.reduce
      - nn.Conv2d

    compatible_with:
      inputs:
        - GridOrSequenceFeatures
      outputs:
        - GridOrSequenceFeatures