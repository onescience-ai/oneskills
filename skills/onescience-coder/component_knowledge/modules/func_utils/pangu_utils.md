component:

  meta:
    name: PanguUtilities
    alias: pangu_utils
    version: 1.0
    domain: ai_for_science
    category: utility
    subcategory: model_components_and_spatial_ops
    author: OneScience
    license: Apache-2.0
    tags:
      - pangu_weather
      - swin_transformer
      - earth_specific_topology
      - tensor_model_parallelism
      - distributed_training
      - window_partition
      - relative_position_bias

  concept:

    description: >
      pangu_utils 是盘古气象大模型（Pangu-Weather）的综合工具集，主要提供三大类功能：
      1. 基于 Megatron-LM 的张量并行多层感知机（DistributedMlp）和普通 MLP，支持大规模分布式训练。
      2. 针对地球气象场 2D/3D 网格的尺寸对齐工具（Padding、Cropping、Truncated Normal Initialization、DropPath）。
      3. 针对 3D Earth-Specific Transformer 定制的局部窗口注意力辅助函数：包括窗口划分（window_partition/reverse）、
      位置偏置索引（get_earth_position_index）以及极其重要的环形移动窗口掩码（get_shift_window_mask）。

    intuition: >
      标准的 Swin Transformer 是为处理平面的图片设计的，边界就是硬边界。
      但地球是一个球体，经度（Longitude）的最左端和最右端在物理上是相连的（例如 -180° 和 +180°）。
      因此，在进行 Shifted Window Attention 时，掩码（Mask）生成逻辑必须体现出“环形跨越”的特性，
      将切分在矩阵两端的半个窗口无缝“缝合”成一个完整的物理窗口。
      此外，为了应对高分辨率全球网格带来的显存爆炸，引入 Megatron 的列并行和行并行策略来切分 MLP。

    problem_it_solves:
      - 解决传统 CV 窗口注意力在气象网格经度边界处的物理规律截断问题
      - 解决气象大模型（极大参量和高分辨率输入）在单卡甚至单机上的显存溢出（OOM）难题
      - 统一管理基于窗口的 3D Transformer 中的张量形变、边界填充与还原恢复

  theory:

    formula:

      normal_truncation:
        expression: X \sim \mathcal{N}(\text{mean}, \text{std}^2) \quad \text{s.t.} \quad a \le X \le b

      tensor_parallel_mlp:
        expression: y = \text{RowParallel}(\text{Dropout}(\text{GELU}(\text{ColumnParallel}(x))))

    variables:

      window_size:
        name: WindowDimensions
        description: 局部注意力窗口的维度，2D 为 (win_lat, win_lon)，3D 为 (win_pl, win_lat, win_lon)

      shift_size:
        name: ShiftDimensions
        description: 在计算 Shifted-Window Attention 时，网格滚动的步长（通常为窗口大小的一半）

      position_index:
        name: RelativePositionBiasIndex
        description: 预先计算的二维矩阵，用于在窗口内快速索引相对位置的偏置参数

  structure:

    architecture: functional_utilities_and_modules

    pipeline:

      - name: ParallelComputation
        operation: distributed_mlp_using_megatron

      - name: SpatialFormatting
        operation: pad_and_crop (get_pad3d, crop3d, etc.)

      - name: WindowAttentionRouting
        operation: partition_shift_and_mask (get_shift_window_mask)

  interface:

    parameters:

      input_resolution:
        type: tuple[int]
        description: 当前张量在空间轴上的分辨率

      window_size:
        type: tuple[int]
        description: 窗口在各个空间轴上的尺寸

      ndim:
        type: int
        default: 3
        description: 张量的空间维度标识（2 表示二维表面场，3 表示三维高空场）

    inputs:
      # 包含众多工具函数，此处以核心的 window_partition 为例
      x:
        type: Tensor
        shape: "[B, Pl, Lat, Lon, C]"
        description: 未划分的全局特征图

    outputs:

      windows:
        type: Tensor
        shape: "[B * num_windows, win_pl, win_lat, win_lon, C]"
        description: 划分后的局部窗口堆叠张量

  types:

    Tensor:
      shape: dynamic
      description: PyTorch 或 Megatron-LM 分布式张量

  implementation:

    framework: pytorch, megatron-lm

    code: |
      from torch import nn
      import torch
      import math
      import warnings

      from onescience.distributed.megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
      from onescience.distributed.megatron.core.utils import init_method_normal, scaled_init_method_normal

      class DistributedMlp(nn.Module):
          def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, config=None):
              super().__init__()
              self.tp_size = config.tensor_model_parallel_size
              out_features = out_features or in_features
              hidden_features = hidden_features or in_features

              sigma = 0.01
              init_method = init_method_normal(sigma)
              out_init = scaled_init_method_normal(sigma, num_layers=config.num_layers)
              
              self.fc1 = ColumnParallelLinear(
                  input_size = in_features,
                  output_size = hidden_features,
                  config = config,
                  init_method = init_method,
                  bias=True
              )
              self.act = act_layer()

              self.fc2 = RowParallelLinear(
                  input_size = hidden_features,
                  output_size = out_features,
                  config = config,
                  init_method = out_init,
                  bias = True,
                  input_is_parallel = True,
                  skip_bias_add = False
              )
              self.drop = nn.Dropout(drop)

          def forward(self, x: torch.Tensor):
              x = self.fc1(x)
              x = self.act(x[0])
              x = self.drop(x)
              x = self.fc2(x)
              x = self.drop(x[0])
              return x

      class Mlp(nn.Module):
          def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
              super().__init__()
              # (标准 MLP 实现，与前文 PanguLayer 中的 Mlp 完全一致)

      def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
          """按样本丢弃路径（随机深度 Stochastic Depth）"""
          if drop_prob == 0.0 or not training:
              return x
          keep_prob = 1 - drop_prob
          shape = (x.shape[0],) + (1,) * (x.ndim - 1)
          random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
          if keep_prob > 0.0 and scale_by_keep:
              random_tensor.div_(keep_prob)
          return x * random_tensor

      class DropPath(nn.Module):
          # (Wrapper for drop_path)

      def get_earth_position_index(window_size, ndim=3):
          """构建位置索引以复用位置偏置的对称参数"""
          # (详见源代码，生成用于 Relative Position Bias 的索引矩阵)

      def get_pad3d(input_resolution, window_size):
          # (详见源代码，计算3D空间的对称 padding 参数)

      def crop3d(x: torch.Tensor, resolution):
          # (详见源代码，基于目标分辨率对3D张量进行中心裁剪)

      def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
          """使用截断正态分布为输入张量填充值"""
          with torch.no_grad():
              return _trunc_normal_(tensor, mean, std, a, b)

      def window_partition(x: torch.Tensor, window_size, ndim=3):
          """将输入特征图划分为不重叠的局部窗口"""
          if ndim == 3:
              B, Pl, Lat, Lon, C = x.shape
              win_pl, win_lat, win_lon = window_size
              x = x.view(B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
              windows = (x.permute(0, 5, 1, 3, 2, 4, 6, 7).contiguous()
                         .view(-1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C))
              return windows
          # (2D 分支略)

      def window_reverse(windows, window_size, Pl=1, Lat=1, Lon=1, ndim=3):
          """window_partition 的逆操作，将局部窗口还原为全局特征图"""
          # (详见源代码)

      def get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3):
          """生成考虑了地球经度循环边界的移动窗口掩码"""
          # (核心逻辑：利用 slice 划分区域，并标记区域 ID，相邻且满足环形拓扑的区域不被掩盖)
          # ...

  skills:

    build_distributed_mlp:

      description: 为大规模气象模型构建分布式的多层感知机（基于 Megatron-LM）

      inputs:
        - in_features
        - hidden_features
        - megatron_config

      prompt_template: |
        如果配置支持张量并行 (tensor_model_parallel_size > 1)，
        请使用 pangu_utils 中的 DistributedMlp 来代替标准的 Mlp，以利用列并行和行并行分散显存压力。

    diagnose_earth_topology_issues:

      description: 分析由于未考虑气象环形拓扑导致的接缝处预测异常

      checks:
        - incorrect_mask_in_shifted_window_attention
        - artifacts_at_meridian_boundaries

  knowledge:

    usage_patterns:

      earth_specific_shifted_window_attention:

        pipeline:
          - Apply get_pad3d (对齐网格)
          - torch.roll (按照 shift_size 滚动特征张量，使边界跨越缝合)
          - window_partition (切分窗口)
          - get_shift_window_mask (获取掩码，确保缝合处仅让真实相邻的物理点进行 Attention)
          - Self-Attention + Mask
          - window_reverse (还原全局视角)
          - torch.roll (逆向滚动回原坐标系)
          - crop3d (去除 padding)

    design_patterns:

      tensor_model_parallelism:

        structure:
          - `ColumnParallelLinear`：将全连接层的权重矩阵按列切分，分发到不同的 GPU 上，输出无需通信即为完整的局部结果。
          - `RowParallelLinear`：将上一层的局部结果和本层权重矩阵按行切分进行计算，最后使用 All-Reduce 汇总得到完整输出。这种组合极大地降低了单卡的显存负担。

    hot_models:

      - model: Pangu-Weather
        year: 2023
        role: 华为推出的 3D 高分辨率全球天气预报模型
        architecture: 3D_earth_specific_transformer

      - model: Megatron-LM
        year: 2019+
        role: 英伟达推出的大规模模型分布式训练底层框架
        architecture: pipeline_and_tensor_parallelism

    model_usage_details:

      Pangu-Weather 3D Block:

        usage: 与 CV 领域的 Swin Transformer 最大的区别就在于 `get_shift_window_mask` 函数。在 CV 中，左上角和右上角的窗口毫无关联；而在 Pangu 中，经度 0° 的左侧紧邻经度 359°，该掩码精确地反映了这一环面（Torus）物理规律。

    best_practices:

      - 必须保证 `DistributedMlp` 中 `fc1(x)[0]` 语法的正确使用。Megatron 并行层前向传播通常返回一个元组 `(output, bias_tensor)`，需要正确提取第一个元素送入激活函数。
      - `get_earth_position_index` 返回的值是用于在预先定义的相对位置偏置参数表（Parameter Table）中查找索引的，这个表需要在模型初始化时被注册为可学习的参数。
      - 在分布式环境下训练时，使用 `trunc_normal_` 初始化权重必须谨慎，确保不同进程上的随机数种子已同步，否则会导致不同卡的权重不同步崩溃。Megatron 提供了特定的并行初始化方法（如代码中引用的 `init_method_normal`）。

    anti_patterns:

      - 在不跨越经线或非周期性网格数据（如局地区域天气预报，如仅覆盖北美）上使用 `get_shift_window_mask` 的环形边界逻辑，会导致模型错误地将大西洋和太平洋的特征强行缝合产生严重干扰。

    paper_references:

      - title: "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast"
        authors: Bi et al.
        year: 2023

      - title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
        authors: Shoeybi et al.
        year: 2019

  graph:

    is_a:
      - HelperFunctions
      - DistributedTrainingComponents
      - SpatialTopologyUtility

    part_of:
      - PanguWeatherModel
      - DistributedTransformer

    depends_on:
      - Megatron-LM Core
      - PyTorch Math Ops

    variants:
      - Swin Transformer Utils (非环形边界的传统实现)

    used_in_models:
      - Pangu-Weather
      - WeatherLearn (底层依赖)

    compatible_with:

      inputs:
        - Tensor
        - DistributedConfig

      outputs:
        - Tensor
        - Indices