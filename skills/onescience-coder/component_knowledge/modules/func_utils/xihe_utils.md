component:

  meta:
    name: XiheMaskUtilities
    alias: xihe_utils
    version: 1.0
    domain: ai_for_science
    category: utility
    subcategory: data_processing
    author: OneScience
    license: Apache-2.0
    tags:
      - xihe
      - land_sea_mask
      - downsampling
      - broadcasting
      - multi_scale

  concept:

    description: >
      该模块包含羲和（Xihe）气象大模型中用于动态调整和生成海陆掩码（Land-Sea Mask）的核心辅助函数 `change_mask`。
      它的主要作用是根据模型当前层特征图的空间分辨率（h_out, w_out），将输入的全局高分辨率静态海陆掩码下采样（Coarsening）为对应的粗粒度掩码，
      并自动将其扩展（Broadcast）以匹配当前计算批次（Batch Size）的维度。

    intuition: >
      在多尺度网络（如包含下采样的层级 Vision Transformer 或 U-Net 架构）中，网络每一层处理的空间分辨率是不断缩小的。
      由于海洋和陆地的物理特性差异巨大，模型经常需要海陆掩码（通常约定海洋=1，陆地=0）作为辅助特征或路由条件。
      当分辨率降低时，这个工具就像一个保守的“特征保留器”：只要对应的高分辨率网格块（Patch）内存在哪怕一点点海洋（`torch.any(patch > 0.5)`），
      就将降采样后的粗网格也标记为海洋。这确保了在宏观尺度上海洋的边界不会被轻易平滑掉。

    problem_it_solves:
      - 解决气象大模型中静态全分辨率地表特征（如海陆掩码）与网络深层低分辨率特征图的空间维度对齐问题
      - 提供了一种基于“存在性（Any）”的非线性二值化池化机制，避免了常规平均池化（Average Pooling）导致掩码变成连续模糊值的问题
      - 自动处理掩码张量在不同计算设备（CPU/GPU）、数据类型（dtype）以及批次大小（Batch Size）上的无缝对齐

  theory:

    formula:

      patch_size_calculation:
        expression: P_h = \lceil H / h_{out} \rceil, \quad P_w = \lceil W / w_{out} \rceil

      mask_coarsening:
        expression: M_{\text{coarse}}[i, j] = \begin{cases} 1.0, & \text{if } \exists p \in M_{\text{full}}[i \cdot P_h : (i+1) \cdot P_h, j \cdot P_w : (j+1) \cdot P_w] \text{ s.t. } p > 0.5 \\ 0.0, & \text{otherwise} \end{cases}

    variables:

      M_{\text{full}}:
        name: FullResolutionMask
        shape: [H, W]
        description: 原始的高分辨率海陆二值掩码

      h_{out}, w_{out}:
        name: TargetResolution
        description: 目标下采样后的空间高度和宽度

      P_h, P_w:
        name: PatchSize
        description: 对应到高分辨率图上的局部窗口尺寸

      M_{\text{coarse}}:
        name: CoarseMask
        shape: [B, 1, h_{out}, w_{out}]
        description: 下采样并广播后的粗粒度特征掩码

  structure:

    architecture: functional_utility

    pipeline:

      - name: TensorConversion
        operation: ensure_input_is_tensor_and_float32

      - name: PatchSizeComputation
        operation: calculate_ceil_division

      - name: IterativeCoarsening
        operation: nested_loop_any_pooling

      - name: DeviceAndDtypeAlignment
        operation: cast_to_reference_tensor_device

      - name: BatchBroadcasting
        operation: unsqueeze_and_repeat

  interface:

    parameters:

      h_out:
        type: int
        description: 目标输出特征图的高度

      w_out:
        type: int
        description: 目标输出特征图的宽度

    inputs:

      mask_full:
        type: Union[Tensor, numpy.ndarray]
        shape: [H, W]
        description: 原始的高分辨率掩码矩阵

      x:
        type: Tensor
        shape: "[B, C, ...]"
        description: 当前层的参考特征张量，用于获取 Batch Size (B)、Device 和 dtype

    outputs:

      mask_coarse:
        type: Tensor
        shape: [B, 1, h_out, w_out]
        description: 转换并广播后的掩码张量，可以直接与特征图拼接 (Concat) 或相乘

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      import math
      import os
      import torch
      import sys
      import numpy as np
      import torch.nn as nn

      def change_mask(mask_full, x, h_out, w_out):
          
          #根据当前层特征分辨率，自动生成掩码（海洋=1，陆地=0）
          if not torch.is_tensor(mask_full):
              mask_full = torch.tensor(mask_full, dtype=torch.float32)
          else:
              mask_full = mask_full

          H, W = mask_full.shape
          patch_h = math.ceil(H / h_out)
          patch_w = math.ceil(W / w_out)

          mask_coarse = torch.zeros((h_out, w_out), dtype=torch.float32)
          for i in range(h_out):
              for j in range(w_out):
                  h0, h1 = i * patch_h, min((i + 1) * patch_h, H)
                  w0, w1 = j * patch_w, min((j + 1) * patch_w, W)
                  patch = mask_full[h0:h1, w0:w1]
                  mask_coarse[i, j] = 1.0 if torch.any(patch > 0.5) else 0.0
                  
          mask_coarse = mask_coarse.to(x.device, dtype=x.dtype) 
          B = x.shape[0]                
          mask_coarse = mask_coarse.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1) #broadcast
          return mask_coarse  

  skills:

    generate_multi_scale_mask:

      description: 为分层网络生成与当前特征图分辨率匹配的地表掩码

      inputs:
        - original_high_res_mask
        - reference_feature_tensor
        - target_height
        - target_width

      prompt_template: |
        调用 change_mask 函数，传入 `mask_full`, `x`, `h_out`, `w_out`。
        确保 `x` 是当前正在处理的张量，以便正确同步 Batch Size 和 GPU 设备环境。

    diagnose_mask_generation_bottleneck:

      description: 分析 Python 嵌套循环处理高分辨率网格时的性能与设备内存转移问题

      checks:
        - cpu_to_gpu_transfer_overhead_during_mask_coarse_to
        - inefficient_nested_for_loops_for_large_h_out_and_w_out

  knowledge:

    usage_patterns:

      hierarchical_feature_fusion:

        pipeline:
          - Extract Features (Resolution: H/4, W/4)
          - Generate Coarse Mask (change_mask to H/4, W/4)
          - Concatenate (Features, Mask)
          - Next Layer (Downsample to H/8, W/8)
          - Generate Coarse Mask (change_mask to H/8, W/8)
          - ...

    design_patterns:

      existence_based_pooling:

        structure:
          - 不使用 `F.max_pool2d` 或 `F.avg_pool2d` 等标准的卷积池化层
          - 使用显式的网格切割和 `torch.any(patch > 0.5)` 来判定，这在逻辑上等价于带有特定二值化激活函数的最大池化，目的是严格保证海陆边界的非黑即白性质

    hot_models:

      - model: Xihe (羲和)
        year: 2023-2024
        role: 高分辨率全球气象预报模型
        architecture: hierarchical_transformer / unet
        attention_type: None

    model_usage_details:

      Xihe Model Surface Guidance:

        usage: 气象预测中，由于海洋和陆地的热动力学性质（如潜热通量、显热通量）完全不同，向网络显式提供当前分辨率下的准确 Land-Sea Mask 是极其关键的。`change_mask` 会在 Decoder 或 Encoder 的每个不同尺度的层中被动态调用。

    best_practices:

      - `mask_coarse.to(x.device, dtype=x.dtype)` 这一句非常关键。在分布式训练中，不同进程的张量被放置在不同的 GPU 上（例如 `cuda:0`, `cuda:1`），通过参考 `x` 的属性可以避免 Device mismatch 导致的报错。
      - 输出的张量通过 `.unsqueeze(0).unsqueeze(0)` 添加了 Batch 维和 Channel 维，这使得它可以直接和典型的 4D 图像特征张量 `(B, C, H, W)` 进行按通道维度（dim=1）的拼接。

    anti_patterns:

      - （性能隐患）目前的实现采用了双层纯 Python 的 `for i in range(h_out): for j in range(w_out):` 嵌套循环。如果 `h_out` 和 `w_out` 较大（如 512x1024），这在 PyTorch 训练中会成为严重的 CPU 性能瓶颈。建议在工程化时使用支持可变核大小的 `F.max_pool2d` 或者张量切片/Unfold 操作进行向量化替代。
      - 每次前向传播都重新计算静态掩码。最佳实践是：由于 `mask_full` 和各个层级的 `h_out`, `w_out` 是固定的，应该在模型初始化阶段或首个 epoch 中预计算并缓存各尺度的 `mask_coarse`，以避免无谓的重复计算。

    paper_references:

      - title: "Xihe: A Data-Driven Model for Global Weather Forecasting"
        authors: Xihe Research Team
        year: 2024

  graph:

    is_a:
      - HelperFunctions
      - DataProcessingUtility
      - SpatialPooling

    part_of:
      - XiheWeatherModel
      - MultiScaleNetwork

    depends_on:
      - torch.tensor
      - torch.any
      - math.ceil

    variants:
      - Max Pooling (与之逻辑类似的标准算子)
      - Nearest Neighbor Interpolation

    used_in_models:
      - Xihe (羲和)

    compatible_with:

      inputs:
        - Tensor
        - Numpy Array

      outputs:
        - Tensor (4D)