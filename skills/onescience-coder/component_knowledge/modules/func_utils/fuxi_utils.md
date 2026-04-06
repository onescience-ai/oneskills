component:

  meta:
    name: FuxiPaddingUtilities
    alias: fuxi_utils
    version: 1.0
    domain: ai_for_science
    category: utility
    subcategory: spatial_padding
    author: OneScience
    license: Apache-2.0
    tags:
      - padding
      - window_attention
      - swin_transformer
      - fuxi
      - 2d_pad
      - 3d_pad

  concept:

    description: >
      该模块包含伏羲（FuXi）气象大模型中用于空间对齐的核心辅助函数 `get_pad2d` 和 `get_pad3d`。
      它们的作用是计算任意分辨率的 2D/3D 网格张量在进行基于窗口（Window-based）的操作（如 Swin Transformer 中的局部窗口注意力）时，
      为了使张量的空间维度能被目标窗口大小（window_size）完美整除，所需要向网格四周对称填充（Padding）的像素/网格数。

    intuition: >
      在基于窗口的模型中，数据被划分为一个个固定大小的窗口（例如 8x8）。如果输入的图像/气象场的分辨率（如 721x1440）不能被 8 整除，
      在划分窗口时就会在边缘截断报错。这就像铺地砖，如果房间长宽不是地砖的整数倍，我们需要在边缘“补”一点面积。
      这两个函数精确计算了需要“补”多少面积，并且将其尽可能对称地分配到上下、左右（或前后）两侧，以防止物理场的中心发生严重偏移。

    problem_it_solves:
      - 解决气象/物理场网格分辨率无法被模型 Patch/Window 尺寸整除的维度不匹配问题
      - 为 `torch.nn.functional.pad` 提供格式完全对齐的 6 元组 (3D) 或 4 元组 (2D) 填充参数
      - 在多维空间特征提取前，确保空间结构的整齐划分，并在处理后（结合前文的 Recovery 模块）进行对应裁剪

  theory:

    formula:

      padding_calculation:
        expression: P_{\text{total}} = \begin{cases} W - (D \pmod W) & \text{if } D \pmod W \neq 0 \\ 0 & \text{otherwise} \end{cases}

      symmetric_split:
        expression: P_{\text{front}} = \lfloor P_{\text{total}} / 2 \rfloor, \quad P_{\text{back}} = P_{\text{total}} - P_{\text{front}}

    variables:

      D:
        name: InputDimension
        description: 输入张量在某一空间轴上的分辨率（如 Pl, Lat, Lon）

      W:
        name: WindowSize
        description: 该空间轴对应的局部窗口大小（如 win_pl, win_lat, win_lon）

      P_{\text{total}}:
        name: TotalPadding
        description: 该轴上总共需要补充的像素量

      P_{\text{front}}, P_{\text{back}}:
        name: PaddingDistribution
        description: 分配到该轴两侧（如左/右、上/下）的具体填充量

  structure:

    architecture: mathematical_utility_functions

    pipeline:

      - name: RemainderCheck
        operation: compute_modulo_division

      - name: PaddingTotalCalculation
        operation: subtract_remainder_from_window

      - name: SymmetricDistribution
        operation: split_padding_into_two_sides

  interface:

    parameters:

      input_resolution:
        type: tuple[int]
        description: "输入特征的空间分辨率。3D 为 (Pl, Lat, Lon)，2D 为 (Lat, Lon)"

      window_size:
        type: tuple[int]
        description: "模型要求的窗口大小。3D 为 (win_pl, win_lat, win_lon)，2D 为 (win_lat, win_lon)"

    inputs:
      # 该模块仅包含逻辑计算函数，不直接接收张量，输入为表示尺寸的元组
      None:
        type: null

    outputs:

      padding:
        type: tuple[int]
        description: >
          返回可以直接传给 F.pad() 的填充元组。
          对于 3D: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
          对于 2D: (pad_left, pad_right, pad_top, pad_bottom)

  types:

    Tuple:
      shape: 1D
      description: Python 标准元组

  implementation:

    framework: pytorch

    code: |
      import torch

      def get_pad3d(input_resolution, window_size):
          """
          Args:
              input_resolution (tuple[int]): (Pl, Lat, Lon)
              window_size (tuple[int]): (Pl, Lat, Lon)
          Returns:
              padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
          """
          Pl, Lat, Lon = input_resolution
          win_pl, win_lat, win_lon = window_size

          padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
          pl_remainder = Pl % win_pl
          lat_remainder = Lat % win_lat
          lon_remainder = Lon % win_lon

          if pl_remainder:
              pl_pad = win_pl - pl_remainder
              padding_front = pl_pad // 2
              padding_back = pl_pad - padding_front
          if lat_remainder:
              lat_pad = win_lat - lat_remainder
              padding_top = lat_pad // 2
              padding_bottom = lat_pad - padding_top
          if lon_remainder:
              lon_pad = win_lon - lon_remainder
              padding_left = lon_pad // 2
              padding_right = lon_pad - padding_left

          return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back


      def get_pad2d(input_resolution, window_size):
          """
          Args:
              input_resolution (tuple[int]): Lat, Lon
              window_size (tuple[int]): Lat, Lon
          Returns:
              padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
          """
          input_resolution = [2] + list(input_resolution)
          window_size = [2] + list(window_size)
          padding = get_pad3d(input_resolution, window_size)
          return padding[: 4]

  skills:

    calculate_padding_for_window:

      description: 为基于窗口的神经网络结构计算张量边缘填充参数

      inputs:
        - resolution
        - window_size

      prompt_template: |
        调用 get_pad2d/3d 函数，传入当前特征图分辨率 {{resolution}} 和目标窗口大小 {{window_size}}，
        将返回的 padding 传入 `torch.nn.functional.pad(x, padding)`。

    diagnose_tensor_pad_issues:

      description: 分析在使用 F.pad 时出现的张量维度截断或顺序错误

      checks:
        - ensure_padding_order_matches_f_pad_requirements
        - verify_window_size_is_not_larger_than_resolution_in_certain_edge_cases

  knowledge:

    usage_patterns:

      swin_transformer_pre_padding:

        pipeline:
          - Extract Current Shape (Lat, Lon)
          - get_pad2d (获取填充元组)
          - F.pad(x, padding)
          - Swin Attention (在整齐的网格上滑动窗口)
          - (可选) Crop / 裁剪恢复到原始尺寸

    design_patterns:

      dry_code_reuse:

        structure:
          - `get_pad2d` 并没有重新写一遍冗长的计算逻辑。
          - 它通过巧妙地向输入列表中插入一个虚构的深度轴 `[2]`（由于 `2 % 2 == 0`，因此前轴不会产生 padding），
          - 直接复用 `get_pad3d`，然后截取 `padding[:4]` 返回，完美符合 DRY（Don't Repeat Yourself）原则。

    hot_models:

      - model: FuXi (伏羲)
        year: 2023
        role: 高精度全球气象预报模型
        architecture: swin_transformer_based

      - model: Swin Transformer
        year: 2021
        role: 基于窗口移位注意力的视觉模型骨干
        attention_type: Window Attention

    model_usage_details:

      Swin/FuXi:

        usage: 在计算局部窗口的 Self-Attention 时，如果特征图尺寸不规范，会在 window partition 步骤崩溃。此工具正是为在计算 Attention 之前临时填充数据而设计的。

    best_practices:

      - `torch.nn.functional.pad` 接收 padding 参数的顺序是从最后一个维度开始向前计算的。即：`(pad_left, pad_right)` 对应最后一个维度（Lon），`(pad_top, pad_bottom)` 对应倒数第二个维度（Lat）。本代码的返回值元组顺序完美对齐了 PyTorch 的底层规范，调用时可以直接展开。
      - 当 `input_resolution` 本身就能被 `window_size` 整除时，`remainder` 为 0，函数会返回全 0 元组，`F.pad` 将变为无副作用的透传操作。
      - 在模型前向传播结束或进入下一层（如恢复层 OneRecovery）前，必须记得根据这里计算出的 padding 值将多余的像素切除，以防止特征在多层传递中无谓扩张或物理坐标系偏移。

    anti_patterns:

      - 手动计算 Padding 且只向一侧（例如只在 right 或 bottom）进行不对称补充，这在图像分类中可能影响不大，但在地理和物理场中会导致预测的空间场整体向左上方偏移（Shift Bias）。此模块的对称分割 `pad // 2` 避免了这一问题。

    paper_references:

      - title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
        authors: Liu et al.
        year: 2021

      - title: "FuXi: A cascade machine learning forecasting system for 15-day global weather forecast"
        authors: Chen et al.
        year: 2023

  graph:

    is_a:
      - HelperFunctions
      - DimensionAlignmentUtility

    part_of:
      - FuxiWeatherModel
      - WindowAttentionPreparation

    depends_on:
      - PyTorch Math Ops

    variants:
      - numpy.pad (等效逻辑的 NumPy 实现)

    used_in_models:
      - FuXi
      - Swin Transformer

    compatible_with:

      inputs:
        - Tuple of Integers

      outputs:
        - Tuple of Integers