component:

  meta:
    name: FeatureUngroupingAttention
    alias: Global SIE Ungrouping Attention / Cross-Attention Broadcast
    version: 1.0
    domain: meteorological_ai
    category: neural_network
    subcategory: attention_mechanism
    author: OneScience
    license: Apache-2.0
    tags:
      - cross_attention
      - feature_ungrouping
      - perceiver_style_broadcast
      - global_to_local
      - xihe_model

  concept:

    description: >
      FeatureUngroupingAttention 是针对高分辨率物理/气象模型设计的特征解组与广播模块。
      作为全局空间信息提取（Global SIE）的第三步，它的任务是将高度压缩的全局组特征（Group Vectors）
      重新融合回高维度的局部物理网格特征（Patch Tokens）中。
      该模块采用交叉注意力机制，利用局部 Patch 作为 Query，全局 Group 作为 Key 和 Value，并在最后通过特征拼接与映射完成局部-全局的深度融合。

    intuition: >
      如果说前一步的 Grouping 是“选拔少数代表去开会总结全局规律”，那么这一步的 Ungrouping 就是“代表开完会后，向下传达最高指示”。
      每一个基层的网格点（Patch Token）带着自己当前的局部状态（作为 Query），去向代表们（Group Vectors）询问：“针对我这个区域，全局的洋流或气旋大趋势是什么？”
      获取到这些宏观信息后，网格点把它们和自己的局部细节拼接在一起（Concat），从而既懂微观细节，又具备宏观视野。

    problem_it_solves:
      - 解决在提取了低维度的全局宏观特征后，如何将其无缝、自适应地分配并反向映射回高维度的局部物理网格中的难题。
      - 相比于简单的插值上采样（Upsampling）或加法广播，交叉注意力机制能够让不同区域的网格根据自身特征有选择性地吸收全局信息，极大增强了模型对复杂边界（如海岸线）的表达能力。

  theory:

    formula:
      
      cross_attention_broadcast:
        expression: X_{out} = Linear(CrossAttention(Query=Norm(X_{patch}), Key=Norm(G_{group}), Value=Norm(G_{group})))

      residual_fusion:
        expression: Output = Linear(Concat([X_{out}, X_{patch}], dim=-1))

    variables:

      X_{patch} (代码中的 y):
        name: LocalPatchTokens
        shape: [B, N, C]
        description: 维持着高分辨率物理拓扑的局部网格特征序列，数量庞大（N）。

      G_{group} (代码中的 x / G_tilde):
        name: GlobalGroupVectors
        shape: [B, G, C]
        description: 经历了自注意力更新后的全局组特征向量，数量极少（G）。

  structure:

    architecture: perceiver_cross_attention_broadcast

    pipeline:

      - name: DualNormalization
        operation: layer_norm (分别对输入的 Patch Tokens 和 Group Vectors 进行独立的层归一化)

      - name: CrossAttentionInteraction
        operation: multihead_attention (庞大的 Patch Tokens 作为 Q 去向少量的 Group Vectors (K/V) 提取信息)

      - name: AttentionProjection
        operation: linear_and_dropout (对注意力输出的全局提取特征进行初步的线性映射)

      - name: FeatureConcatenation
        operation: concat (将提取到的全局特征 X_out 与原始的局部特征 X 在特征通道维度上进行拼接，通道数变为 2C)

      - name: FinalFusion
        operation: linear_and_dropout (使用线性层将融合特征压缩回原始维度 C，完成解组)

  interface:

    parameters:

      dim:
        type: int
        description: 输入网格特征和组向量的通道维度 C

      num_heads:
        type: int
        default: 12
        description: 交叉注意力的多头数量

      qkv_bias:
        type: bool
        default: true
        description: 是否在交叉注意力的 Q, K, V 投影上使用偏置

      attn_drop:
        type: float
        default: 0.0
        description: 注意力概率矩阵的 Dropout 比例

      proj_drop:
        type: float
        default: 0.0
        description: 投影融合层的 Dropout 比例

    inputs:

      obj:
        type: dict_or_object
        description: 统一的输入载体
        properties:
          y:
            shape: [B, N, C]
            description: 本地的高分辨率 Patch Tokens（局部特征）
          x:
            shape: [B, G, C]
            description: 更新完毕的全局 Group Vectors（全局特征）

    outputs:

      output:
        type: Tensor
        shape: [B, N, dim]
        description: 深度融合了全局与局部信息的高分辨率网格特征，准备传递给后续的层

  types:

    FusedGridFeatures:
      shape: [B, N, dim]
      description: 既包含局部高频细节又具备全局低频感受野的混合物理场特征

  implementation:

    framework: pytorch

    code: |
      import torch
      import torch.nn as nn
      from onescience.modules.func_utils import Mlp
      class FeatureUngroupingAttention(nn.Module):
          """
          全局 SIE 的第三步：特征解组模块，将全局 group 特征融合回局部 patch tokens。

          Args:
              dim (int): 输入通道数 C。
              num_heads (int): 多头注意力的头数，默认为 12。
              qkv_bias (bool): 是否在 QKV 上添加可学习的偏置，默认为 True。
              attn_drop (float): 注意力权重的 dropout 比例，默认为 0.0。
              proj_drop (float): 输出投影的 dropout 比例，默认为 0.0。
              LN (nn.Module): 归一化层类型，默认为 nn.LayerNorm。
              drop_layer (nn.Module): dropout 层类型，默认为 nn.Dropout。

          形状:
              输入 obj: 包含以下属性的对象
                  - y (torch.Tensor): 原始 patch tokens，形状为 (B, N, C)，其中 N = Pl × Lat × Lon
                  - x (torch.Tensor): 经过 Group Propagation 的 group vectors，形状为 (B, G, C)
              输出 x_out (torch.Tensor): 融合全局信息后的 patch tokens，形状为 (B, N, C)

          Example:
              >>> ungrouping = FeatureUngroupingAttention(
              ...     dim=192,
              ...     num_heads=12
              ... )
              >>> from types import SimpleNamespace
              >>> B, N, C = 2, 425984, 192  # N = 13*128*256
              >>> G = 32  # group 数量
              >>> x_patch = torch.randn(B, N, C)  # patch tokens
              >>> G_tilde = torch.randn(B, G, C)  # group vectors
              >>> obj = SimpleNamespace(y=x_patch, x=G_tilde)
              >>> out = ungrouping(obj)
              >>> out.shape
              torch.Size([2, 425984, 192])
          """

          def __init__(
              self,
              dim,
              num_heads=12,
              qkv_bias=True,
              attn_drop=0.0,
              proj_drop=0.0,
              LN=nn.LayerNorm,
              drop_layer=nn.Dropout,
          ):
              super().__init__()
              self.dim = dim
              self.num_heads = num_heads     
           
              self.norm_x = LN(dim)  # 对 patch tokens 做归一化
              self.norm_g = LN(dim)  # 对 group vectors 做归一化
              # Cross-Attention (Q=patch tokens, K/V=group vectors)
              self.attn = nn.MultiheadAttention(
                  embed_dim=dim, num_heads=num_heads, bias=qkv_bias, dropout=attn_drop,batch_first=True
              )
              
              # 注意力输出的投影层
              self.attn_proj = nn.Linear(dim, dim)
               # 拼接后的融合层
              self.concat_proj = nn.Linear(2 * dim, dim)
              self.proj_drop = drop_layer(proj_drop)

          def forward(self, obj,mask=None):
              """
              x: (B, N, C)  patch tokens
              G_tilde: (B, G, C)  group vectors
              """
              # x=obj.y
              # G_tilde=obj.x

              if isinstance(obj, dict):
                  # 字典方式访问
                  x=obj["y"]
                  G_tilde=obj["x"]
          
              # 判断是否为对象（非字典的其他类型）
              else:
                  # 对象方式访问        
                  x=obj.y
                  G_tilde=obj.x
              
              B, N, C = x.shape
              _, G, _ = G_tilde.shape

              # 归一化
              x_norm = self.norm_x(x)
              G_norm = self.norm_g(G_tilde)

              x_out, _ = self.attn(query=x_norm, key=G_norm, value=G_norm)
              x_out = self.proj_drop(self.attn_proj(x_out))
              
              # 拼接 [U, x] 并线性映射回原维度 C
              x_concat = torch.cat([x_out, x], dim=-1)   # (B, N, 2C)   
              x_out = self.proj_drop(self.concat_proj(x_concat))  # (B, N, C)


              return x_out

  skills:

    build_feature_ungrouping_attention:

      description: 构建负责全局特征向下广播与融合的注意力解码模块

      inputs:
        - dim
        - num_heads

      prompt_template: |
        创建一个 FeatureUngroupingAttention 层。
        参数：
        输入通道维度 = {{dim}}，多头头数为 {{num_heads}}。
        必须能够接收包含局部特征和全局组特征的复合对象输入。

    diagnose_feature_fusion_errors:

      description: 分析局部与全局特征融合时的维度越界或内存泄漏问题

      checks:
        - key_value_dimension_mismatch (检查传入的 `x` 和 `y` 是否搞反了。庞大的 N 作为 Query，小巧的 G 作为 Key/Value，否则计算量将从 O(NG) 变成更可怕的 O(N^2))
        - memory_spike_on_concat (由于 `torch.cat` 会在内存中显式分配 `2 * dim` 的张量，这通常是显存 OOM 的高发区，必要时检查 batch_size 或启用 checkpoint)

  knowledge:

    usage_patterns:

      xihe_global_spatial_information_extraction:
        pipeline:
          - XihelocalTransformer (Local Extraction)
          - FeatureGroupingAttention (将 N 聚合成 G)
          - GlobalTransformerBlock (G 维度的全局自注意力)
          - FeatureUngroupingAttention (将更新后的 G 融合回 N)
          - XihelocalTransformer (后续的 Local Extraction)

    design_patterns:

      cross_attention_broadcast:
        structure:
          - 类似于 Perceiver IO 的解码器。它没有使用硬性的上采样网络（Up-sampling / Deconv），而是完全依赖 Cross-Attention 的数据自适应性来解码全局信息。

      residual_concatenation_fusion:
        structure:
          - 在融合局部与全局信息时，没有采用简单的加性残差 `x_out + x`，而是使用 `torch.cat([x_out, x], dim=-1)` 再经过 `concat_proj`，这种“晚期显式拼流”给予了模型极大的自由度来决定局部细节与宏观趋势的权重分配。

    hot_models:

      - model: XiHe
        year: 2024
        role: 高分辨率数据驱动的全球海洋涡旋分辨预报模型
        architecture: Local-Global Dual-Path Transformer
        attention_type: Feature Ungrouping Cross-Attention

      - model: Perceiver IO
        year: 2021
        role: 可以处理并输出任意尺寸/维度数据的通用架构
        architecture: Encoder-Decoder
        attention_type: Cross-Attention Decoder

    model_usage_details:

      XiHe_Global_SIE_Module:
        dim: 192
        num_heads: 12
        fusion_method: "concat_and_linear"

    best_practices:
      - 尽管 `G`（组数）很小，但由于 `N`（网格数）极大，此处的注意力计算复杂度仍然是 $O(N \times G)$。在模型设计时，这部分的计算成本应当与 Local Window Attention 联合评估。
      - `nn.MultiheadAttention` 设置了 `batch_first=True`，在推理阶段传入的 `obj` 数据结构必须严格遵守 Batch 在首位的约束。

    anti_patterns:
      - 在调用过程中混淆了 `obj` 字典中代表局部特征和全局特征的 key (`y` 为局部 N，`x` 为全局 G)，会导致底层 Cross-Attention 的 Q 和 K/V 错位，不仅破坏逻辑还会引发维度错误。

    paper_references:

      - title: "XiHe: A Data-Driven Model for Global Ocean Eddy-Resolving Forecasting"
        authors: Xiang Wang, Renzhi Wang, Junqiang Song, et al.
        year: 2024

      - title: "Perceiver IO: A General Architecture for Structured Inputs & Outputs"
        authors: Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, et al.
        year: 2021
        journal: ICLR

  graph:

    is_a:
      - AttentionMechanism
      - CrossAttention
      - DecoderComponent
      - FeatureFusionLayer

    part_of:
      - XiHeModel
      - GlobalSIE

    depends_on:
      - nn.MultiheadAttention
      - LayerNorm
      - torch.cat

    compatible_with:
      inputs:
        - LocalGridFeatures (as Query)
        - CompactGroupFeatures (as Key/Value)
      outputs:
        - FusedGridFeatures