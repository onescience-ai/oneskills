component:

  meta:
    name: SwinTransformerBlock
    alias: SwinBlock
    version: 1.0
    domain: computer_vision
    category: neural_network
    subcategory: vision_transformer
    author: OneScience
    license: Apache-2.0
    tags:
      - swin_transformer
      - window_attention
      - shifted_window
      - computer_vision
      - linear_complexity


  concept:

    description: >
      Swin Transformer 块是 Swin Transformer 架构的基础单元。
      它实现了带有移动窗口机制的多头自注意力层。根据 `shift_size` 的不同，
      它可以作为标准的局部窗口自注意力层 (W-MSA, shift_size=0) 或移动窗口自注意力层 (SW-MSA, shift_size>0)。
      它限制了注意力计算只在局部窗口内进行，从而大幅降低了计算复杂度。

    intuition: >
      传统的 ViT 就像是让图片上的每个像素点同时去和其他所有像素点“相亲”，图片越大，相亲次数呈平方级爆炸。
      Swin Transformer 则像是把图片划分成一个个“相亲小包厢（Window）”，大家只在包厢内交流。
      为了防止不同包厢之间的人永远不认识，下一层网络会把包厢的隔板“平移（Shift）”一下，
      这样上一层在不同包厢边界的人，在这一层就被分到了同一个包厢里，从而巧妙地实现了全局信息的传递。

    problem_it_solves:
      - 解决标准自注意力机制在处理高分辨率图像时计算量和显存随像素数呈平方 $O(N^2)$ 增长的瓶颈
      - 克服局部注意力网络（如单纯的块计算）缺乏跨块全局信息交互的缺陷
      - 使得 Transformer 能够像 CNN 一样构建层次化（Hierarchical）的特征图提取架构


  theory:

    formula:

      w_msa_forward:
        expression: |
          x_hat = x + DropPath(W-MSA(LayerNorm(x)))
          x_out = x_hat + DropPath(MLP(LayerNorm(x_hat)))

      sw_msa_forward:
        expression: |
          x_hat = x_out + DropPath(SW-MSA(LayerNorm(x_out)))
          x_final = x_hat + DropPath(MLP(LayerNorm(x_hat)))

    variables:

      W-MSA:
        name: WindowMultiHeadSelfAttention
        description: 标准局部窗口自注意力，shift_size=0

      SW-MSA:
        name: ShiftedWindowMultiHeadSelfAttention
        description: 带有循环移位和掩码（Mask）的移动窗口自注意力

      attn_mask:
        name: AttentionMask
        description: 用于防止在循环移位（torch.roll）后，原本在图像物理边界两端的不相邻像素发生错误的注意力交互


  structure:

    architecture: shifted_window_attention_block

    pipeline:

      - name: PreNorm_1
        operation: layer_norm

      - name: WindowShift
        operation: torch.roll (仅当 shift_size > 0 时触发)

      - name: SpatialPartition
        operation: window_partition (划分为不重叠的局部窗口)

      - name: LocalAttention
        operation: one_attention (结合 attn_mask 执行自注意力)

      - name: SpatialReverse
        operation: window_reverse (还原为整图空间排列)

      - name: WindowUnshift
        operation: torch.roll_back (仅当 shift_size > 0 时触发)

      - name: AttnResidual
        operation: drop_path + residual_add

      - name: PreNorm_2
        operation: layer_norm

      - name: FeedForward
        operation: one_mlp

      - name: MlpResidual
        operation: drop_path + residual_add


  interface:

    parameters:

      dim:
        type: int
        description: 输入特征的通道数 (特征维度)

      input_resolution:
        type: tuple[int]
        description: 输入特征图的 2D 空间分辨率 (H, W)

      num_heads:
        type: int
        description: 注意力机制的头数

      window_size:
        type: int
        default: 7
        description: 局部注意力窗口的大小（通常为 7 或 8）

      shift_size:
        type: int
        default: 0
        description: 窗口移动的大小。对于 SW-MSA，通常设置为 window_size // 2

      mlp_ratio:
        type: float
        default: 4.0
        description: MLP 中隐藏层维度相对于输入维度的扩展比例

      fused_window_process:
        type: bool
        default: False
        description: 是否使用融合的 CUDA 算子来加速窗口划分与循环移位操作

    inputs:

      x:
        type: VisionFeatureMap
        shape: [batch, L, dim]  # L = H * W
        dtype: float32
        description: 展平后的视觉或网格特征张量

    outputs:

      output:
        type: VisionFeatureMap
        shape: [batch, L, dim]
        description: 输出形状与输入完全一致的更新特征


  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from timm.layers import DropPath, to_2tuple
      from onescience.modules.mlp.onemlp import OneMlp
      from onescience.modules.attention.oneattention import OneAttention
      
      # 注：window_partition 和 window_reverse 函数见源文件。

      class SwinTransformerBlock(nn.Module):
          """Swin Transformer 块"""
          def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, drop_path=0.0):
              super().__init__()
              self.dim = dim
              self.input_resolution = input_resolution
              self.num_heads = num_heads
              self.window_size = window_size
              self.shift_size = shift_size

              self.norm1 = nn.LayerNorm(dim)
              self.attn = OneAttention(style="WindowAttention", dim=dim, window_size=to_2tuple(window_size), num_heads=num_heads)
              self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
              self.norm2 = nn.LayerNorm(dim)
              self.mlp = OneMlp(style="StandardMLP", input_dim=dim, output_dim=dim, hidden_dims=[int(dim * mlp_ratio)], activation="gelu")

              # 生成 Shifted Window 的掩码，防止跨边界交互
              if self.shift_size > 0:
                  H, W = self.input_resolution
                  img_mask = torch.zeros((1, H, W, 1))  
                  h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
                  w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
                  cnt = 0
                  for h in h_slices:
                      for w in w_slices:
                          img_mask[:, h, w, :] = cnt
                          cnt += 1
                  mask_windows = window_partition(img_mask, self.window_size)
                  mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
                  attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                  attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
              else:
                  attn_mask = None
              self.register_buffer("attn_mask", attn_mask)

          def forward(self, x):
              H, W = self.input_resolution
              B, L, C = x.shape
              shortcut = x
              
              x = self.norm1(x).view(B, H, W, C)

              if self.shift_size > 0:
                  shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
              else:
                  shifted_x = x
                  
              x_windows = window_partition(shifted_x, self.window_size) 
              x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

              attn_windows = self.attn(x_windows, mask=self.attn_mask)
              attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

              if self.shift_size > 0:
                  shifted_x = window_reverse(attn_windows, self.window_size, H, W)
                  x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
              else:
                  x = window_reverse(attn_windows, self.window_size, H, W)
                  
              x = x.view(B, H * W, C)
              x = shortcut + self.drop_path(x)
              x = x + self.drop_path(self.mlp(self.norm2(x)))

              return x


  skills:

    build_swin_block:

      description: 构建具有线性复杂度的移动窗口注意力层

      inputs:
        - dim
        - input_resolution
        - num_heads

      prompt_template: |

        请实例化 SwinTransformerBlock。
        强烈建议成对使用此模块：第一层设置 shift_size=0，第二层设置 shift_size=window_size//2。
        注意：输入张量 x 在传入前必须展平为 [Batch, H*W, Channels]。


    diagnose_swin_mask:

      description: 排查循环移位机制中的边缘特征污染问题

      checks:
        - mask_generation_error (h_slices 与 w_slices 未严格对齐导致 attn_mask 数值错误)
        - non_divisible_resolution (input_resolution 无法被 window_size 整除，导致 forward 时 shape 突变)



  knowledge:

    usage_patterns:

      hierarchical_vision_transformer:

        pipeline:
          - PatchMerging (空间下采样)
          - [W-MSA SwinBlock, SW-MSA SwinBlock] x N
          - GlobalAveragePooling


    hot_models:

      - model: Swin Transformer
        year: 2021
        role: ICCV 2021 最佳论文，极大地推动了 Transformer 在密集预测任务（检测、分割）中的应用
        architecture: Hierarchical Vision Transformer

      - model: Video Swin Transformer
        year: 2022
        role: 3D 视频特征提取的时空扩展版本


    best_practices:

      - 必须保证 `input_resolution` 能被 `window_size` 完美整除。如果遇到不规则的分辨率，必须在模块外部对特征图进行预先的 Padding，并在输出后进行 Crop。
      - 当模型很大或分辨率极高时，强烈建议将 `fused_window_process` 设置为 True，利用底层 CUDA 算子替换原生的 `torch.roll` 和 `view`，能显著降低显存碎片并加速。


    anti_patterns:

      - 连续堆叠多个 `shift_size=0` 的 Block。这会使得不同窗口之间的特征“老死不相往来”，导致模型全局感受野严重受限，性能大幅下降。


    paper_references:

      - title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
        authors: Liu et al.
        year: 2021



  graph:

    is_a:
      - TransformerBlock
      - VisionTransformerComponent

    part_of:
      - SwinTransformer
      - HierarchicalEncoder

    depends_on:
      - OneAttention (WindowAttention)
      - OneMlp
      - DropPath

    variants:
      - SwinTransformerV2Stage
      - EarthTransformer2DBlock

    used_in_models:
      - Swin Transformer

    compatible_with:

      inputs:
        - VisionFeatureMap

      outputs:
        - VisionFeatureMap