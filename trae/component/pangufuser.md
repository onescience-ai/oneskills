component:
  meta:
    name: PanguFuser
    中文名称: 三维特征融合模块
    别名: 3DEST特征融合层, 三维Transformer融合块
    version: 1.0
    领域: 气象AI
    分类: 神经网络组件
    子类: 特征融合层
    作者: OneScience
    tags:
      - Pangu-Weather
      - 三维Transformer
      - 特征融合

  concept:
    描述: >
      PanguFuser 是基于 3D Earth-specific Transformer 的三维特征融合模块，
      在给定的三维网格上堆叠多层 3D Transformer 块，对多时刻、多高度和经纬空间的气象特征进行联合建模与融合。
    直觉理解: >
      可以将 PanguFuser 想象成把一叠随时间和高度变化的全球天气“立体切片”放入一个三维搅拌器，
      通过多层带地球先验的自注意力和前馈网络，将不同时间、高度和空间位置上的信息充分交换后再输出形状不变但语义更丰富的特征。
    解决的问题:
      - 三维相关性弱建模: 在单一二维平面或简单通道堆叠下难以同时捕捉高度、经度和纬度之间的复杂依赖关系，PanguFuser 通过 3D 体建模与自注意力增强三维相关性建模能力。
      - 多尺度信息整合: 结合 Pangu-Weather 的 U-Net 式多尺度结构，PanguFuser 所在的 3DEST 主干负责在压缩后的 3D 体上整合大尺度与局地特征，提高中期预报精度。

  theory:
    核心公式:
      三维Transformer特征融合:
        表达式: >
          x_out = F_L(⋯F_2(F_1(x_in)))，[基于通用知识补充]
        变量说明:
          x_in:
            名称: 输入三维特征序列
            形状: [B, T * H * W, dim]
            描述: >
              扁平化后的三维网格特征序列，B 为样本批大小，
              T 为时间或层维长度，H/W 为经纬方向 patch 网格尺寸，dim 为通道维度。
          x_out:
            名称: 输出三维特征序列
            形状: [B, T * H * W, dim]
            描述: >
              经过 L 层 Earth-specific 3D Transformer 块融合后的特征序列，
              在形状上与 x_in 一致，但编码了更强的三维相关性信息。
          F_l:
            名称: 第 l 层 3D Transformer 块
            形状: [B, T * H * W, dim] → [B, T * H * W, dim]
            描述: >
              由窗口化自注意力（带 Earth-specific positional bias）、
              MLP 与残差结构组成的 3DEST 子层，对三维体内局部窗口执行注意力并在通道维上进行非线性映射。
          L:
            名称: Transformer 层数
            形状: 标量
            描述: >
              PanguFuser 中堆叠的 EarthTransformer3DBlock 层数，对应初始化参数 depth。
          B:
            名称: 批大小
            形状: 标量
            描述: >
              同一批次内并行处理的样本数。
          T:
            名称: 三维网格第一维长度
            形状: 标量
            描述: >
              代表时间步或垂直层方向上的离散索引，具体物理含义由上游模块定义。
          H:
            名称: 三维网格高度方向长度
            形状: 标量
            描述: >
              对应经度或纬度方向的 patch 网格尺寸之一，与 Pangu-Weather 中 360/181 等规格相适配。
          W:
            名称: 三维网格宽度方向长度
            形状: 标量
            描述: >
              对应经度或纬度方向的另一维 patch 网格尺寸。
          dim:
            名称: 通道维度
            形状: 标量
            描述: >
              单个三维网格单元的特征维度，Pangu-Weather 主干中典型取值为 192 及其倍数。

  structure:
    架构类型: 三维Transformer模块
    计算流程:
      - 接收形状为 (B, T * H * W, dim) 的三维特征序列，并假定与 input_resolution 对应。
      - 依次将特征输入多个 EarthTransformer3DBlock，每层在三维局部窗口内执行带 Earth-specific positional bias 的自注意力与 MLP 更新。
      - 在偶数层和奇数层之间交替使用非移位与移位窗口，实现跨窗口的信息交换（由 OneTransformer 内部实现）。
      - 逐层叠加残差更新后的特征，保持整体形状不变。
      - 输出与输入形状一致的 (B, T * H * W, dim) 序列，用于下游解码或进一步 3D 建模。
    计算流程图: |
      x_in (B, T*H*W, dim)
          |
          v
      +-----------------------+
      |  EarthTransformer3D   |  F_1
      +-----------------------+
          |
          v
      +-----------------------+
      |  EarthTransformer3D   |  F_2 (shifted window)
      +-----------------------+
          |
         ...
          |
          v
      +-----------------------+
      |  EarthTransformer3D   |  F_L
      +-----------------------+
          |
          v
      x_out (B, T*H*W, dim)

  interface:
    参数:
      dim:
        类型: int
        默认值: null
        描述: 特征通道维度，需与输入 x 的最后一维一致。
      input_resolution:
        类型: tuple[int, int, int]
        默认值: null
        描述: 三维网格尺寸 (T, H, W)，用于在内部构造 3D 窗口与位置偏置信息。
      depth:
        类型: int
        默认值: null
        描述: 堆叠的 EarthTransformer3DBlock 层数，对应理论公式中的 L。
      num_heads:
        类型: int
        默认值: null
        描述: 多头自注意力的头数，影响每层注意力的表示能力和计算开销。
      window_size:
        类型: tuple[int, int, int]
        默认值: null
        描述: 三维窗口大小 (Wt, Wh, Ww)，控制 3D 窗口化自注意力在时间/层与空间上的局部感受野。
      drop_path:
        类型: float | Sequence[float]
        默认值: 0.0
        描述: 每层的 Stochastic Depth 丢弃比例，可为标量或长度为 depth 的序列。
      mlp_ratio:
        类型: float
        默认值: 4.0
        描述: MLP 隐藏维度与 dim 的比例，典型取值 4.0。
      qkv_bias:
        类型: bool
        默认值: true
        描述: 是否在 Q/K/V 线性投影中使用偏置。
      qk_scale:
        类型: float | null
        默认值: null
        描述: QK 点积缩放因子，若为 null 则采用默认 1/sqrt(dim_head) 缩放。[基于通用知识补充]
      drop:
        类型: float
        默认值: 0.0
        描述: 特征上的 dropout 比例。
      attn_drop:
        类型: float
        默认值: 0.0
        描述: 注意力权重上的 dropout 比例。
      norm_layer:
        类型: nn.Module
        默认值: torch.nn.LayerNorm
        描述: 用于块内归一化的层类型，需接受通道维 dim 作为特征大小。
    输入:
      x:
        类型: FeatureTokens3D
        形状: [B, T * H * W, dim]
        描述: >
          三维网格上的特征序列，已按 input_resolution 展平，
          其中 (T, H, W) 对应 input_resolution。
    输出:
      out:
        类型: FeatureTokens3D
        形状: [B, T * H * W, dim]
        描述: >
          经过多层 EarthTransformer3DBlock 融合后的特征序列，
          保持与输入相同的三维网格大小与通道维度。

  types:
    FeatureVolume3D:
      形状: [T, H, W, dim]
      描述: >
        三维网格体表示，T 为时间或层索引，H/W 为经纬 patch 网格尺寸，dim 为通道维度。
    FeatureTokens3D:
      形状: [B, T * H * W, dim]
      描述: >
        由三维体展平得到的 token 序列表示，是 PanguFuser 接收与输出的主要数据类型。

  constraints:
    shape_constraints:
      - 规则: x.shape[1] == input_resolution[0] * input_resolution[1] * input_resolution[2]
        描述: 输入 token 序列长度必须与三维网格体素总数一致。
    parameter_constraints:
      - 规则: depth > 0 且 num_heads > 0
        描述: Transformer 层数和注意力头数应为正整数。[基于通用知识补充]
      - 规则: window_size 各维大于等于 1
        描述: 三维窗口尺寸需为正整数，以保证窗口划分合法。[基于通用知识补充]
    compatibility_rules:
      - 输入类型: FeatureTokens3D
        输出类型: FeatureTokens3D
        描述: 当输入与输出的 input_resolution 与 dim 匹配时，PanguFuser 可插入到任意 3D Transformer 主干中作为特征融合层使用。

  implementation:
    框架: pytorch
    示例代码: |
      from torch import nn
      from onescience.modules.transformer.onetransformer import OneTransformer
      from collections.abc import Sequence

      class PanguFuser(nn.Module):
          def __init__(
              self,
              dim,
              input_resolution,
              depth,
              num_heads,
              window_size,
              drop_path=0.0,
              mlp_ratio=4.0,
              qkv_bias=True,
              qk_scale=None,
              drop=0.0,
              attn_drop=0.0,
              norm_layer=nn.LayerNorm,
          ):
              super().__init__()
              self.dim = dim
              self.input_resolution = input_resolution
              self.depth = depth

              self.blocks = nn.ModuleList(
                  [
                      OneTransformer(
                          style="EarthTransformer3DBlock",
                          dim=dim,
                          input_resolution=input_resolution,
                          num_heads=num_heads,
                          window_size=window_size,
                          shift_size=(0, 0, 0) if i % 2 == 0 else None,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          drop=drop,
                          attn_drop=attn_drop,
                          drop_path=drop_path[i]
                          if isinstance(drop_path, Sequence)
                          else drop_path,
                          norm_layer=norm_layer,
                      )
                      for i in range(depth)
                  ]
              )

          def forward(self, x):
              for blk in self.blocks:
                  x = blk(x)
              return x
    Attention结构诊断:
      描述: >
        PanguFuser 内部使用的 EarthTransformer3DBlock 继承了 3DEST 的窗口化注意力结构，
        可通过检查窗口划分、Earth-specific positional bias 索引和多头输出稳定性来诊断注意力是否正常工作。[基于通用知识补充]
      检查项:
        - 窗口大小与 input_resolution 是否匹配，padding 与 crop 是否正确避免空间错位。[基于通用知识补充]
        - 注意力权重在不同高度与纬向上的分布是否合理反映大气层次和纬度差异。[基于通用知识补充]

  usage_examples:
    - pl: null
      描述: 基于三维网格特征的标准特征融合用法
      示例代码: |

        import torch

        dim = 256
        input_resolution = (10, 181, 360)
        fuser = PanguFuser(
            dim=dim,
            input_resolution=input_resolution,
            depth=4,
            num_heads=8,
            window_size=(2, 6, 12),
        )

        B, T, H, W = 2, 10, 181, 360
        x = torch.randn(B, T * H * W, dim)
        out = fuser(x)
        # out.shape → (2, T * H * W, dim)

  knowledge:
    应用说明: >
      PanguFuser 所属的三维 Transformer 特征融合模块类别，位于“3D 体建模策略”与
      “Transformer 范式”的交汇处：其输入通常是形如 [N_pl, N_lon, N_lat, C] 或抽象为
      [T, H, W, C] 的三维体张量（见 3D 体建模策略），通过窗口化自注意力和 U-Net 式多尺度
      结构在 3D 体上反复执行特征重排与融合（见 Transformer 范式与 U-Net 式下采样/上采样范式），
      以在有限计算预算下捕捉跨高度、跨经纬度乃至多时间步的长程相关性，并保持与规整 0.25°
      经纬网格和层次化时间演化策略兼容（见层次化时间模型范式）。
    热点模型:
      - 模型: Pangu-Weather
        年份: 2022 [基于通用知识补充]
        场景: >
          全球中期数值天气预报，在规整 0.25° 经纬网格和多层大气变量上进行 1–24 小时领时预测。
        方案: >
          在 Pangu-Weather 中，3D Earth-specific Transformer 主干接收由 3D/2D Patch Embedding
          拼接而成的统一体 volume [8, 360, 181, C]，在该体上堆叠多层 3DEST Block 形成编码–解码
          结构。PanguFuser 类型的三维特征融合模块对应这些堆叠的 3D Transformer 层，
          通过局部窗口注意力与 Earth-specific positional bias 在各层中对体内 token 进行融合和重分布。
        作用: >
          该模块类别负责在单一步 3DEST 前向传播中，将高度、经纬方向和上空/地表变量的耦合关系显式
          编码到潜在特征中，为 Patch Recovery、分层时间聚合以及极端事件追踪等下游流程提供高质量三维表示。
        创新: >
          相比二维 Swin 主干，Pangu-Weather 的三维特征融合模块在 3D 体上扩展了 Swin 风格窗口注意力，
          并引入 Earth-specific positional bias 区分经度周期性与纬度非周期性，使得融合过程同时遵守球面几何
          与垂直分层结构的物理先验。
      - 模型: FuXi
        年份: 未知
        场景: >
          基于 2D Swin V2 U-Transformer 的全球中期天气预报，在二维网格和多变量通道上进行多步自回归预测。
        方案: >
          FuXi 采用 Cube embedding 将多变量场映射到统一 2D 特征网格，通过 Down Block 与 Up Block 组成的
          U-Transformer 主干堆叠多层 Swin V2 块，在多尺度特征图之间执行窗口化自注意力与 MLP，
          其主干在范式上与 PanguFuser 一类的 Transformer 特征融合模块等价。
        作用: >
          该模块类别在 FuXi 中作为核心空间骨干，将卷积式下/上采样与窗口化 Transformer 结合，实现大尺度结构
          与局地细节的多尺度融合。
        创新: >
          FuXi 在 2D 场景下采用 Swin V2 的 scaled cosine attention 与改进位置编码，在 U-Transformer 结构中实现
          稳定训练与高分辨率建模，为类似 PanguFuser 的特征融合模块在二维网格上提供了一种更稳定的实现路径。
    最佳实践:
      - 在 3D 体建模中，将高度或层维显式纳入 input_resolution，并在 PanguFuser 内部保持三维形状的一致性，有助于充分利用 Earth-specific positional bias 编码的几何先验。[基于通用知识补充]
      - 结合 U-Net 式多尺度结构使用 PanguFuser，将其放置在编码–解码通路的主干位置，有利于在低分辨率 3D 体上高效整合行星尺度与局地尺度的大气结构信息。[基于通用知识补充]
      - 为不同任务（如多时间步融合 vs 纯垂直层融合）选择合适的 input_resolution 映射策略，明确 T 维是时间还是层，以避免语义混淆。[基于通用知识补充]
    常见错误:
      - 忽略 x.shape[1] 与 input_resolution 三维体素总数的一致性，导致在内部重排为 [T, H, W] 时发生越界或形状错配。[基于通用知识补充]
      - 在未考虑经纬分辨率和窗口大小关系的情况下随意设置 window_size，可能导致过多 padding 或窗口过大带来的显存开销激增。[基于通用知识补充]
    论文参考:
      - 标题: Pangu-Weather
        作者: 未知
        年份: 未知
        摘要: null

  graph:
    类型关系:
      - 神经网络层
      - 三维Transformer特征融合模块
    所属结构:
      - 3D Earth-specific Transformer 主干
      - U-Net 式编码–解码结构
    依赖组件:
      - EarthTransformer3DBlock
      - 3D Patch Embedding 与 Patch Recovery
      - Earth-specific Positional Bias
    变体组件:
      - 二维 Swin Transformer 特征融合层
      - 纯时间轴 1D Transformer 融合层
      - 三维卷积特征融合块
    使用模型:
      - Pangu-Weather
      - FuXi
    类型兼容:
      输入:
        - FeatureTokens3D
      输出:
        - FeatureTokens3D

