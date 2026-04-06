component:

  meta:
    name: FactAttention
    alias: Factorized Attention (2D/3D)
    version: 1.0
    domain: computer_vision_and_ai4science
    category: neural_network
    subcategory: attention_mechanism
    author: OneScience
    license: Apache-2.0
    tags:
      - factorized_attention
      - axial_attention
      - einsum
      - 3d_vision
      - computational_efficiency

  concept:

    description: >
      FactAttention (因子化注意力) 是一种专为 2D 图像或 3D 体素/网格数据设计的高效注意力机制。
      它不直接在展平的全局序列上计算复杂度为 O(N^2) 的标准注意力，而是将特征分解为 X、Y（及 Z）轴，
      分别计算各轴向的注意力权重，然后通过爱因斯坦求和约定 (Einstein Summation, einsum) 进行特征的顺次投影与聚合。

    intuition: >
      想象你要在一个巨大的 3D 房间（网格）里寻找与其他所有点相关的线索。如果让每个点都与其他所有点比对，计算量会大到无法承受。
      因子化注意力的思路是：“降维打击”。它先让点只跟同在 X 轴（同一行）的点交流，然后再把结果跟 Y 轴（同一列）的点交流，最后是 Z 轴。
      通过这种“正交方向上的接力传递”，模型以极小的计算代价（从 O((H*W*D)^2) 降至 O(H*W*D * (H+W+D))），间接实现了全局信息的融合。

    problem_it_solves:
      - 解决在高分辨率 2D 图像或 3D 体素/流体网格数据上，直接应用全局自注意力机制导致的显存溢出 (OOM) 和计算灾难。
      - 克服传统卷积在处理高维空间数据时感受野受限的问题，以线性增长的成本提供全局感受野。

  theory:

    formula:
      
      axis_pooling:
        expression: X_{axis} = MLP(MeanPooling_over_other_axes(X))

      axis_attention_weight:
        expression: Attn_{axis} = Softmax(Q_{axis} * K_{axis}^T / sqrt(d_k))

      einsum_aggregation_2d:
        expression: |
          Res_X = Einsum(Attn_X, V)
          Res_Y = Einsum(Attn_Y, Res_X)
          Output = Linear(Rearrange(Res_Y))

    variables:

      shapelist:
        name: GridShape
        description: 数据的原始空间拓扑结构，2D 为 (H, W)，3D 为 (H, W, D)。

      inner_dim:
        name: HiddenDimension
        description: 内部计算的隐藏层维度，严格等于 heads * dim_head。

  structure:

    architecture: factorized_axial_attention

    pipeline:

      - name: AxisDecomposition
        operation: rearrange (使用 einops 将数据从 1D 序列还原并置换为目标轴向的 3D/4D 张量)

      - name: FeatureReduction
        operation: pooling_reducer (通过均值池化压缩非当前轴的特征，并经过 MLP 映射)

      - name: WeightCalculation
        operation: fact_attn_weight (计算单个正交轴向上的注意力概率矩阵)

      - name: SequentialContraction
        operation: einsum (利用 torch.einsum 将注意力权重与 Value 矩阵或上一个轴的输出进行张量收缩)

      - name: OutputFusion
        operation: rearrange_and_linear (将分步计算的特征重新拼接并投影回原始维度)

  interface:

    parameters:

      dim:
        type: int
        description: 输入特征的总维度（必须等于 heads * dim_head）

      heads:
        type: int
        default: 8
        description: 注意力头数

      dim_head:
        type: int
        default: 64
        description: 每个注意力头的维度

      dropout:
        type: float
        default: 0.0
        description: Dropout 丢弃率

      shapelist:
        type: tuple_or_list
        description: 必须提供的参数，用于指明输入数据的网格形状 (H, W) 或 (H, W, D)

    inputs:

      x:
        type: Tensor
        shape: "[B, N, C] (其中 N = H*W 或 H*W*D)"
        dtype: float32
        description: 展平的网格特征序列

    outputs:

      output:
        type: Tensor
        shape: [B, N, C]
        description: 经过多轴因子化注意力融合后的特征序列

  types:

    GridSequence:
      shape: [B, H*W*D, C]
      description: 隐式包含空间拓扑关系的一维序列特征

  implementation:

    framework: pytorch, einops

    code: |
      import torch
      import torch.nn as nn
      from einops import rearrange
      from einops.layers.torch import Rearrange

      # ==========================================
      # 辅助组件 (Internal Utilities)
      # ==========================================

      class _PoolingReducer(nn.Module):
          """
          内部组件：用于降维和特征压缩的池化层。
          """
          def __init__(self, in_dim, hidden_dim, out_dim):
              super().__init__()
              self.to_in = nn.Linear(in_dim, hidden_dim, bias=False)
              self.out_ffn = nn.Sequential(
                  nn.LayerNorm(hidden_dim),
                  nn.Linear(hidden_dim, hidden_dim),
                  nn.GELU(),
                  nn.Linear(hidden_dim, out_dim)
              )

          def forward(self, x):
              x = self.to_in(x)
              ndim = len(x.shape)
              if ndim > 3:
                  x = x.mean(dim=tuple(range(2, ndim - 1)))
              x = self.out_ffn(x)
              return x 


      class _FactAttnWeight(nn.Module):
          """
          内部组件：因子化注意力权重计算器。
          """
          def __init__(self, heads=8, dim_head=64, dropout=0.0):
              super().__init__()
              self.dim_head = dim_head
              self.heads = heads
              self.scale = dim_head**-0.5
              self.softmax = nn.Softmax(dim=-1)
              self.dropout = nn.Dropout(dropout)
              self.to_q = nn.Linear(dim_head, dim_head, bias=False)
              self.to_k = nn.Linear(dim_head, dim_head, bias=False)

          def forward(self, x):
              B, N, C = x.shape
              x = x.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
              q = self.to_q(x)
              k = self.to_k(x)
              dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
              attn = self.softmax(dots)
              return attn 


      # ==========================================
      # 因子化注意力 (Factorized Attention)
      # ==========================================

      class FactAttention2D(nn.Module):
          """
          2D 因子化注意力 (Factorized Attention 2D)。

          专为 2D 网格结构数据设计的高效注意力机制。
          它不直接在展开的序列上计算 O((HW)^2) 的注意力，而是将特征分解为 X 轴和 Y 轴特征，
          分别计算注意力权重，然后通过爱因斯坦求和 (einsum) 进行特征聚合。

          **注意**：由于内部实现涉及 Reshape 操作，输入维度 `dim` 必须等于 `heads * dim_head`。

          Args:
              dim (int): 输入特征维度。
              heads (int, optional): 注意力头数。默认值: 8。
              dim_head (int, optional): 每个头的维度。默认值: 64。
              dropout (float, optional): Dropout 概率。默认值: 0.0。
              shapelist (tuple or list): 输入数据的网格形状 (H, W)。**必须提供**。

          形状:
              输入 x: (B, N, C)，其中 N = H * W，且 C = heads * dim_head。
              输出: (B, N, C)。

          Example:
              >>> # 示例：dim=128, heads=4, dim_head=32 (4*32=128)
              >>> fact_attn = FactAttention2D(dim=128, heads=4, dim_head=32, shapelist=(32, 32))
              >>> x = torch.randn(2, 32*32, 128)
              >>> out = fact_attn(x)
              >>> out.shape
              torch.Size([2, 1024, 128])
          """
          def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, shapelist=None):
              super().__init__()
              assert shapelist is not None and len(shapelist) == 2, "FactAttention2D 需要 shapelist=(H, W)"
              assert dim == heads * dim_head, f"Input dim {dim} must equal heads {heads} * dim_head {dim_head}"
              
              inner_dim = dim_head * heads
              self.dim_head = dim_head
              self.heads = heads
              self.H, self.W = shapelist
              
              self.attn_x = _FactAttnWeight(heads, dim_head, dropout)
              self.attn_y = _FactAttnWeight(heads, dim_head, dropout)
              
              self.to_v = nn.Linear(dim_head, dim_head, bias=False)
              
              self.to_x = nn.Sequential(_PoolingReducer(inner_dim, inner_dim, inner_dim))
              
              self.to_y = nn.Sequential(
                  Rearrange("b nx ny c -> b ny nx c"),
                  _PoolingReducer(inner_dim, inner_dim, inner_dim),
              )
              
              self.to_out = nn.Sequential(
                  nn.Linear(inner_dim, dim), 
                  nn.Dropout(dropout)
              )

          def forward(self, x):
              B, N, C = x.shape
              assert N == self.H * self.W, f"Sequence length {N} does not match shape {self.H}*{self.W}"

              x = x.reshape(B, self.H, self.W, C).contiguous()
              
              v = (
                  self.to_v(
                      x.reshape(B, self.H, self.W, self.heads, self.dim_head).contiguous()
                  )
                  .permute(0, 3, 1, 2, 4)
                  .contiguous()
              )
              
              res_x = torch.einsum("bhij,bhjmc->bhimc", self.attn_x(self.to_x(x)), v)
              res_y = torch.einsum("bhlm,bhimc->bhilc", self.attn_y(self.to_y(x)), res_x)
              
              res = rearrange(res_y, "b h i l c -> b (i l) (h c)", h=self.heads)
              return self.to_out(res)


      class FactAttention3D(nn.Module):
          """
          3D 因子化注意力 (Factorized Attention 3D)。

          类似于 2D 版本，但处理 3D 体素/网格数据。
          它将注意力分解为 X, Y, Z 三个维度的计算，极大地节省了 3D 数据的显存占用。
          
          **注意**：输入维度 `dim` 必须等于 `heads * dim_head`。

          Args:
              dim (int): 输入特征维度。
              heads (int, optional): 注意力头数。默认值: 8。
              dim_head (int, optional): 每个头的维度。默认值: 64。
              dropout (float, optional): Dropout 概率。默认值: 0.0。
              shapelist (tuple or list): 输入数据的网格形状 (H, W, D)。**必须提供**。

          形状:
              输入 x: (B, N, C)，其中 N = H * W * D，且 C = heads * dim_head。
              输出: (B, N, C)。

          Example:
              >>> # 示例：dim=512, heads=8, dim_head=64 (8*64=512)
              >>> fact_attn = FactAttention3D(dim=512, heads=8, dim_head=64, shapelist=(16, 16, 16))
              >>> x = torch.randn(2, 16**3, 512)
              >>> out = fact_attn(x)
              >>> out.shape
              torch.Size([2, 4096, 512])
          """
          def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, shapelist=None):
              super().__init__()
              assert shapelist is not None and len(shapelist) == 3, "FactAttention3D 需要 shapelist=(H, W, D)"
              assert dim == heads * dim_head, f"Input dim {dim} must equal heads {heads} * dim_head {dim_head}"
              
              inner_dim = dim_head * heads
              self.dim_head = dim_head
              self.heads = heads
              self.H, self.W, self.D = shapelist
              
              self.attn_x = _FactAttnWeight(heads, dim_head, dropout)
              self.attn_y = _FactAttnWeight(heads, dim_head, dropout)
              self.attn_z = _FactAttnWeight(heads, dim_head, dropout)
              
              self.to_v = nn.Linear(dim_head, dim_head, bias=False)
              
              self.to_x = nn.Sequential(_PoolingReducer(inner_dim, inner_dim, inner_dim))
              
              self.to_y = nn.Sequential(
                  Rearrange("b nx ny nz c -> b ny nx nz c"),
                  _PoolingReducer(inner_dim, inner_dim, inner_dim),
              )
              
              self.to_z = nn.Sequential(
                  Rearrange("b nx ny nz c -> b nz nx ny c"),
                  _PoolingReducer(inner_dim, inner_dim, inner_dim),
              )
              
              self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

          def forward(self, x):
              B, N, C = x.shape
              assert N == self.H * self.W * self.D, f"Sequence length {N} != {self.H}*{self.W}*{self.D}"
              
              x = x.reshape(B, self.H, self.W, self.D, C).contiguous()
              
              v = (
                  self.to_v(
                      x.reshape(
                          B, self.H, self.W, self.D, self.heads, self.dim_head
                      ).contiguous()
                  )
                  .permute(0, 4, 1, 2, 3, 5)
                  .contiguous()
              )

              res_x = torch.einsum("bhij,bhjmsc->bhimsc", self.attn_x(self.to_x(x)), v)
              res_y = torch.einsum("bhlm,bhimsc->bhilsc", self.attn_y(self.to_y(x)), res_x)
              res_z = torch.einsum("bhrs,bhilsc->bhilrc", self.attn_z(self.to_z(x)), res_y)
              
              res = rearrange(res_z, "b h i l r c -> b (i l r) (h c)", h=self.heads)
              return self.to_out(res)

  skills:

    build_factorized_attention:

      description: 构建应对高维数据的因子化注意力层

      inputs:
        - dim
        - heads
        - dim_head
        - shapelist

      prompt_template: |
        构建一个处理多维网格数据的 Factorized Attention 层。
        参数：
        输入维度 = {{dim}}
        头数 = {{heads}}
        网格拓扑形状 = {{shapelist}}
        务必检查 dim 是否等于 heads * dim_head。

    diagnose_tensor_shape_errors:

      description: 排查因子化计算中的张量重塑和收缩错误

      checks:
        - shape_mismatch_with_sequence_length (传入的序列长度 N 与配置的 shapelist 体积不匹配)
        - dimension_indivisibility (输入维度 dim 无法被 heads 整除，导致内部特征拆解崩溃)

  knowledge:

    usage_patterns:

      high_res_3d_modeling:
        pipeline:
          - 3D_PatchEmbedding
          - FactAttention3D (大幅节约显存)
          - FeedForward
          - 3D_UpSampling

    design_patterns:

      einsum_tensor_contraction:
        structure:
          - 不使用笨重的 `.matmul()` 循环，而是使用爱因斯坦求和约定 (Einstein Summation)。
          - 例如 `torch.einsum("bhij,bhjmsc->bhimsc", attn_x, v)` 以极其优雅和底层优化的方式完成高维张量的批量乘加运算，这也是此模块执行效率高的核心原因。

    hot_models:

      - model: Axial-DeepLab / CCNet
        year: 2019-2020
        role: 高效计算机视觉分割与特征提取架构
        architecture: Axial Attention / Criss-Cross Attention
        attention_type: Factorized/Axial Attention

    model_usage_details:

      HighRes_Medical_Image_Segmentation_3D:
        shapelist: (32, 32, 32)
        dim: 512
        heads: 8
        dim_head: 64

    best_practices:
      - 必须在初始化模块时就精确传递 `shapelist`，如果是 2D 图像输入，提供 `(H, W)`；如果是 3D 物理场或视频帧输入，提供 `(H, W, D)`。
      - 当 `rearrange` 报错时，优先检查传入前向传播的张量 `x` 在调用 `forward` 前是否含有正确的 Batch 维度和展平的 Sequence 维度。

    anti_patterns:
      - 将未经空间拓扑关联的纯文本序列（如 NLP 数据）传入此模块。因为 NLP 词汇序列没有物理上的 X/Y/Z 轴概念，强行切分为网格会导致注意力逻辑完全错乱。
      - 传入的 `dim` 不等于 `heads * dim_head`，会直接触发 `assert` 断言失败。

    paper_references:

      - title: "Axial Attention in Multidimensional Transformers"
        authors: Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, Tim Salimans
        year: 2019

      - title: "CCNet: Criss-Cross Attention for Semantic Segmentation"
        authors: Zilong Huang, Xinggang Wang, Lichao Huang, Chang Huang, Yunchao Wei, Wenyu Liu
        year: 2019

  graph:

    is_a:
      - AttentionMechanism
      - EfficientTransformerComponent
      - DimensionalityReductionModule

    part_of:
      - 3D_VisionModel
      - AI4Science_GridSolver

    depends_on:
      - torch.einsum
      - einops.rearrange
      - _PoolingReducer
      - _FactAttnWeight

    compatible_with:
      inputs:
        - GridSequence
      outputs:
        - GridSequence