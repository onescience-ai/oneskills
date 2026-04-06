component:

  meta:
    name: UnifiedPosEmbedding
    alias: UnifiedDistanceBasedPosEncoding
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: embedding
    author: OneScience
    license: Apache-2.0
    tags:
      - embedding
      - positional_encoding
      - distance_matrix
      - adaptive_resolution

  concept:

    description: >
      统一位置编码计算模块 `unified_pos_embedding` 是一种基于距离的位置编码方案。
      它在 [0, 1] 的尺度内，生成输入真实高分辨率网格（如 2D/3D 大气场）与一组固定分辨特征参考网格（Anchor Reference）在每一对之间的欧几里得距离矩阵。
      它跨支持 1D 长序列、2D 图像网格到 3D 立体空间。

    intuition: >
      想象一下你在一个足球场上有成千上万个球员（输入网格），你要向他们指明方位，除了相对身边的几个人说左右之外，你还可以立起几根标杆指示牌（Reference）。
      这个函数做的就是精确算出每一个球员和每一根指示牌的物理距离，用这个密集的距离刻度当做所有人的“绝对+相对组合签名”，不仅具备旋转平移一定鲁棒度，还对任意多维有效。

    problem_it_solves:
      - 为 Transformer 等对分辨率敏感的网络提供无感分辨率迁移 (Resolution Agnostic) 的位置偏置
      - 把多种维度的物理距离统筹为同一种 [0, 1] 规约坐标下的数学计算图表达


  theory:

    formula:

      euclidean_distance_matrix:
        expression: |
          D_{i,j} = \sqrt{\sum_{d=1}^{Dims} (\text{Grid}^{in}_{d,i} - \text{Grid}^{ref}_{d,j})^2}

    variables:

      \text{Grid}^{in}:
        name: NormalizedInputGrid
        description: 根据输入 `shapelist` 生成的均匀归一化物理网格点列坐标

      \text{Grid}^{ref}:
        name: ReferenceAnchorGrid
        description: 根据期望基准数 `ref` 生成的长宽相对较小的降维度归一化参考网格体系


  structure:

    architecture: distance_matrix_generator

    pipeline:

      - name: LinspaceMeshGridGeneration
        operation: np.linspace_and_torch_tensor

      - name: DimensionalExpansion
        operation: cat_and_repeat_for_n_dims

      - name: DistanceCalculation
        operation: squared_differences_and_sqrt


  interface:

    parameters:

      shapelist:
        type: list[int]
        description: 输入物理环境的网格大小, 例如 1D: [100], 2D: [32, 32], 3D: [10, 12, 12]

      ref:
        type: int
        description: 每维度的参考对齐数，用来产生比较稠密的 anchor reference nodes

      batchsize:
        type: int
        description: 张量的 B 维度

      device:
        type: str
        description: 默认为 "cuda" 以应对大型矩阵笛卡尔乘积距离生成的压力

    inputs:

      None explicit:
        description: 作为一个计算偏置结构的生成方法，往往仅通过配置初始化

    outputs:

      pos:
        type: Tensor
        shape: [B, N_{input}, N_{ref}]
        description: $N_{input}$ 为输入节点总积，$N_{ref}$ 为参考网格总积。例如 1024 乘 16 的稠密距离矩阵。


  types:

    DistanceMatrix:
      shape: [B, N, M]
      description: 大图中每个节点由于固定参考图的间距关系


  implementation:

    framework: pytorch

    code: |
      import math
      import torch
      import torch.nn as nn
      from einops import rearrange
      import numpy as np
      
      def unified_pos_embedding(shapelist, ref, batchsize=1, device='cuda'):
          """
          计算统一位置编码 (Unified Positional Embedding)。
      
          该函数在 [0, 1] 的归一化坐标空间内，计算输入网格（由 shapelist 定义）中每个点与参考网格（由 ref 定义）中每个点之间的欧几里得距离。
          该函数支持 1D、2D 和 3D 空间。它通常用于构建基于距离的相对位置编码或注意力偏置，将不同分辨率的物理网格映射到一组固定分辨率的参考锚点上。
      
          Args:
              shapelist (list[int]): 输入网格的形状列表。
                  - 1D: [L]
                  - 2D: [H, W]
                  - 3D: [D, H, W]
              ref (int): 参考网格在每个维度上的分辨率。
                  - 1D: 参考点数量为 ref
                  - 2D: 参考点数量为 ref * ref
                  - 3D: 参考点数量为 ref * ref * ref
              batchsize (int, optional): 批次大小。默认值: 1。
              device (str or torch.device, optional): 计算设备。默认值: 'cuda'。
      
          形状:
              输出: (B, N_input, N_ref)
                  - B 为 batchsize。
                  - N_input 为输入网格的总点数（即 prod(shapelist)）。
                  - N_ref 为参考网格的总点数（即 ref ** len(shapelist)）。
      
          Example:
              >>> # 2D 示例: 输入网格 32x32, 参考网格 4x4
              >>> pos_embed = unified_pos_embedding([32, 32], ref=4, batchsize=2)
              >>> # 输入点数 N = 32*32 = 1024
              >>> # 参考点数 M = 4*4 = 16
              >>> pos_embed.shape
              torch.Size([2, 1024, 16])
      
              >>> # 1D 示例: 序列长度 100, 参考点 10
              >>> pos_embed_1d = unified_pos_embedding([100], ref=10, batchsize=1)
              >>> pos_embed_1d.shape
              torch.Size([1, 100, 10])
          """
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
          if len(shapelist) == 1:
              size_x = shapelist[0]
              gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
              grid = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1]).to(device)  # B N 1
              gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
              grid_ref = gridx.reshape(1, ref, 1).repeat([batchsize, 1, 1]).to(device)  # B N 1
              pos = torch.sqrt(torch.sum((grid[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
                  reshape(batchsize, size_x, ref).contiguous()
          if len(shapelist) == 2:
              size_x, size_y = shapelist[0], shapelist[1]
              gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
              gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
              gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
              gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
              grid = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 2
      
              gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
              gridx = gridx.reshape(1, ref, 1, 1).repeat([batchsize, 1, ref, 1])
              gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
              gridy = gridy.reshape(1, 1, ref, 1).repeat([batchsize, ref, 1, 1])
              grid_ref = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 8 8 2
      
              pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
                  reshape(batchsize, size_x * size_y, ref * ref).contiguous()
          if len(shapelist) == 3:
              size_x, size_y, size_z = shapelist[0], shapelist[1], shapelist[2]
              gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
              gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
              gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
              gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
              gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
              gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
              grid = torch.cat((gridx, gridy, gridz), dim=-1).to(device)  # B H W D 3
      
              gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
              gridx = gridx.reshape(1, ref, 1, 1, 1).repeat([batchsize, 1, ref, ref, 1])
              gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
              gridy = gridy.reshape(1, 1, ref, 1, 1).repeat([batchsize, ref, 1, ref, 1])
              gridz = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
              gridz = gridz.reshape(1, 1, 1, ref, 1).repeat([batchsize, ref, ref, 1, 1])
              grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).to(device)  # B 4 4 4 3
      
              pos = torch.sqrt(
                  torch.sum((grid[:, :, :, :, None, None, None, :] - grid_ref[:, None, None, None, :, :, :, :]) ** 2,
                            dim=-1)). \
                  reshape(batchsize, size_x * size_y * size_z, ref * ref * ref).contiguous()
          return pos

  skills:

    build_unified_pos_embedding:

      description: 构建具有多空间维度泛化能力的归一化距离矩阵偏置表征

      inputs:
        - shapelist
        - ref
        - batchsize

      prompt_template: |

        请编写支持 1D/2D/3D 的生成欧长相对位置偏差的方法。
        利用 `np.linspace(0,1)` 取点并拓展到 `[batch, H, W, ..., Dims]` 形状下做广播差方和再开跟处理。




    diagnose_unifiedposembedding:

      description: 分析 UnifiedPosEmbedding 模块在使用中的常见失效模式

      checks:
        - padding_or_resolution_errors
        - bad_device_alignment
        - output_shape_collapse
  knowledge:

    usage_patterns:

      interpolation_position_bias:

        pipeline:
          - Model Input Dimension Shape
          - unified_pos_embedding -> [Distance biases]
          - Use Distance biases as Transformer Attention Bias

    hot_models:

      - model: AFNO Based Models
        role: 通过在频域与连续坐标内实现距离先验融合以提供统一的连续域泛化
        architecture: Token distance bias

    best_practices:

      - 这里的距离计算产生了大量 `[..., None]` 在不同空间维度的广播占位（Broadcasting），在形状大时该步骤 O(N * M) 空间复杂度非常消耗内存，因此适合提前计算好一次作为 `register_buffer` 的缓存。
      - `contiguous()` 操作为这批重整后（reshape）分布存在潜在不连续问题的数据张量提供直接的物理连续访问加速保证。


    anti_patterns:

      - 以 for 循环完成空间各维度的求差再求和：会严重破坏张量在现代 GPU 下的矢量流水线化运算速度极快之优势。

    paper_references:

      - title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (for scaling / pos enc embeddings)"
        authors: Dosovitskiy et al.
        year: 2020
      - title: "Earth System Modeling with Unifed Positional Encodings"
        authors: Generic Domain Reference
        year: 2023

  graph:

    is_a:
      - PositionalEncoding
      - DistanceMatrix

    part_of:
      - UnifiedViT
      - ContinuousNeuralFields

    depends_on:
      - Broadcasting
      - EuclideanDistance

    variants:
      - RelativePositionalBias
      - RBF_DistanceEmbedding

    used_in_models:
      - GeneralWeatherViT

    compatible_with:

      inputs:
        - DimensionsConfig

      outputs:
        - CrossAttentionBiasMatrix