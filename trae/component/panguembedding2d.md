component:
  meta:
    name: PanguEmbedding2D
    中文名称: 二维面片嵌入层
    别名: 2D Patch Embedding, Surface Patch Embedding
    version: 1.0
    领域: 深度学习/气象
    分类: 神经网络组件
    子类: 嵌入层
    作者: OneScience
    tags:
      - 气象AI
      - Patch Embedding
      - 地表变量处理
      - Pangu-Weather

  concept:
    描述: >
      将二维地表气象场划分为不重叠的面片(patch)并通过卷积映射到高维潜在空间，为后续Transformer处理提供token化表示。
    直觉理解: >
      就像把一张高分辨率地图切成小方块，每个方块用一个特征向量来概括其内容，这样Transformer就能像处理文字一样处理气象场。
    解决的问题:
      - 高分辨率场的计算复杂度: 直接在全分辨率(1440×721)上执行注意力计算代价极高，通过patch降采样将空间分辨率压缩到可处理范围
      - 地表与上空变量的统一表示: 将地表二维场映射到与上空三维体相同的通道维度(C=192)，便于在层维度拼接形成统一3D体
      - 局部空间结构保留: 通过卷积而非全连接进行嵌入，保留patch内的局部空间相关性

  theory:
    核心公式:
      Patch划分与嵌入:
        表达式: Y = Conv2D(Pad(X), kernel=patch_size, stride=patch_size)
        变量说明:
          X:
            名称: 输入地表场
            形状: [B, C_in, H, W]
            描述: 批次×输入通道×纬度×经度，对应地表变量(MSLP/T2M/U10/V10)及常数掩膜(地形/陆海/土壤类型)
          Pad(X):
            名称: 零填充后的输入
            形状: [B, C_in, H_pad, W_pad]
            描述: 当H或W不能被patch_size整除时，在边界进行零填充以确保完整划分
          Y:
            名称: 嵌入后的token网格
            形状: [B, embed_dim, H', W']
            描述: H'=⌈H/patch_h⌉, W'=⌈W/patch_w⌉，每个位置对应原始场中一个patch的高维表示
          patch_size:
            名称: 面片大小
            形状: (patch_h, patch_w)
            描述: 每个patch在纬度和经度方向的栅格数，Pangu-Weather中为(4,4)

  structure:
    架构类型: 卷积嵌入网络
    计算流程:
      - 检查输入尺寸是否能被patch_size整除
      - 若不能整除，计算所需填充量并在边界进行零填充
      - 使用stride=patch_size的卷积提取不重叠patch特征
      - 可选地应用LayerNorm归一化(若norm_layer非空)
      - 输出形状为[B, embed_dim, H', W']的token网格
    计算流程图: |
      输入地表场 [B, C_in, H, W]
        ↓
      边界零填充 (若H或W不整除patch_size)
        ↓
      Conv2d(kernel=patch_size, stride=patch_size)
        ↓
      [可选] LayerNorm归一化
        ↓
      输出token网格 [B, embed_dim, H', W']

  interface:
    参数:
      img_size:
        类型: tuple[int, int]
        默认值: (721, 1440)
        描述: 输入图像尺寸(纬度×经度)，对应ERA5的0.25°分辨率网格
      patch_size:
        类型: tuple[int, int]
        默认值: (4, 4)
        描述: 每个patch的大小(patch_h, patch_w)，决定下采样倍数
      in_chans:
        类型: int
        默认值: 7
        描述: 输入通道数，包含4个地表变量(MSLP/T2M/U10/V10)和3个常数掩膜(地形/陆海/土壤类型)
      embed_dim:
        类型: int
        默认值: 192
        描述: 嵌入后的特征维度，需与上空3D嵌入的通道维度一致以便拼接
      norm_layer:
        类型: nn.Module
        默认值: null
        描述: 归一化层类型，常用nn.LayerNorm，若为None则不进行归一化
    输入:
      x:
        类型: SurfaceField
        形状: [B, C_in, H, W]
        描述: 批次×通道×纬度×经度的地表气象场张量，通道包含地表变量和常数掩膜
    输出:
      embedded_tokens:
        类型: TokenGrid2D
        形状: [B, embed_dim, H', W']
        描述: 嵌入后的二维token网格，H'=⌈H/patch_h⌉, W'=⌈W/patch_w⌉，用于与上空体在层维拼接

  types:
    SurfaceField:
      形状: [B, C_in, H, W]
      描述: 地表气象场，包含4个预报变量(MSLP/T2M/U10/V10)和3个静态掩膜(地形/陆海/土壤类型)，H=721, W=1440对应ERA5的0.25°网格
    TokenGrid2D:
      形状: [B, embed_dim, H', W']
      描述: 二维token网格，每个token对应原始场中一个patch的高维表示，H'和W'为patch网格尺寸

  constraints:
    shape_constraints:
      - 规则: H_pad % patch_h == 0 且 W_pad % patch_w == 0
        描述: 填充后的尺寸必须能被patch_size整除，以确保完整的patch划分
      - 规则: embed_dim == 上空3D嵌入的通道维度
        描述: 地表嵌入的通道维度必须与上空嵌入一致，以便在层维度拼接形成统一3D体
    parameter_constraints:
      - 规则: patch_h > 0 且 patch_w > 0
        描述: patch尺寸必须为正整数
      - 规则: in_chans == 地表变量数 + 常数掩膜数
        描述: 输入通道数应匹配实际输入的变量数量(Pangu中为4+3=7)
    compatibility_rules:
      - 输入类型: SurfaceField
        输出类型: TokenGrid2D
        描述: 将地表气象场转换为token网格，输出可与上空3D token体在层维拼接

  implementation:
    框架: pytorch
    示例代码: |
      import torch
      from torch import nn

      class PanguEmbedding2D(nn.Module):
          def __init__(self, img_size=(721, 1440), patch_size=(4, 4),
                       embed_dim=192, in_chans=7, norm_layer=None):
              super().__init__()
              height, width = img_size
              h_patch_size, w_patch_size = patch_size

              # 计算填充量
              h_remainder = height % h_patch_size
              w_remainder = width % w_patch_size
              padding_top = padding_bottom = padding_left = padding_right = 0

              if h_remainder:
                  h_pad = h_patch_size - h_remainder
                  padding_top = h_pad // 2
                  padding_bottom = h_pad - padding_top

              if w_remainder:
                  w_pad = w_patch_size - w_remainder
                  padding_left = w_pad // 2
                  padding_right = w_pad - padding_left

              self.pad = nn.ZeroPad2d((padding_left, padding_right,
                                       padding_top, padding_bottom))
              self.proj = nn.Conv2d(in_chans, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)
              self.norm = norm_layer(embed_dim) if norm_layer else None

          def forward(self, x):
              x = self.pad(x)
              x = self.proj(x)
              if self.norm is not None:
                  x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
              return x

  usage_examples:
    - 标准地表嵌入 (721×1440 → 181×360):
        示例代码: |
          embed = PanguEmbedding2D(
              img_size=(721, 1440),
              patch_size=(4, 4),
              in_chans=7,
              embed_dim=192
          )
          x = torch.randn(2, 7, 721, 1440)  # 2个样本，7通道地表场
          out = embed(x)
          # out.shape → (2, 192, 181, 360)

    - 带归一化的嵌入:
        示例代码: |
          embed = PanguEmbedding2D(
              img_size=(721, 1440),
              patch_size=(4, 4),
              in_chans=7,
              embed_dim=192,
              norm_layer=nn.LayerNorm
          )
          x = torch.randn(4, 7, 721, 1440)
          out = embed(x)
          # out.shape → (4, 192, 181, 360)，已归一化

  knowledge:
    应用说明: >
      PanguEmbedding2D属于"2D通道堆叠策略"中的Patch Embedding模块，在气象AI建模中负责将高分辨率地表场转换为可处理的token表示。该策略将多变量气象场组织为二维网格上的通道堆叠张量，通过patch划分降低空间分辨率，为后续Transformer/卷积网络提供统一输入格式。核心问题是平衡空间分辨率与计算效率，创新点在于通过卷积嵌入保留局部空间结构，同时实现显著的维度压缩。
    热点模型:
      - 模型: Pangu-Weather
        年份: 2023
        场景: 全球中期天气预报，需要处理0.25°分辨率(1440×721)的地表变量场
        方案: 使用4×4 patch将地表场从1440×721下采样到360×181，通过卷积映射到192维潜在空间，然后与上空3D体(7×360×181×192)在层维拼接形成统一8层3D体(8×360×181×192)，为3DEST编码-解码网络提供输入
        作用: 将地表二维场与上空三维体统一到相同的空间分辨率和通道维度，使模型能在单一3D Transformer中联合建模上空与地表变量的耦合关系
        创新: 通过2D patch嵌入与3D patch嵌入的层维拼接，实现了地表-上空变量的统一表示，这是Pangu-Weather区别于纯2D模型(如FourCastNet)的关键设计之一

      - 模型: FourCastNet
        年份: 2022
        场景: 全球天气预报，处理721×1440分辨率的多层大气变量
        方案: 将所有气压层变量作为通道堆叠为[721,1440,20]张量，使用8×8 patch嵌入后送入AFNO主干网络
        作用: 通过patch嵌入将高分辨率场压缩为可处理的token网格，为频域AFNO算子提供输入
        创新: 采用较大的8×8 patch以适配AFNO的全局频域混合机制，在保持全球视野的同时控制计算复杂度

      - 模型: FuXi
        年份: 2023
        场景: 全球0-15天天气预报，处理70通道(13层×5变量+5地表)的气象场
        方案: 使用Cube Embedding(结合时间和空间的3D卷积)将[70,721,1440]场嵌入为[C,180,360]，然后送入Swin Transformer V2主干
        作用: 通过patch嵌入实现空间下采样，同时在嵌入阶段融合时间信息(两个时间步)
        创新: 将时间维纳入嵌入过程，使用3D卷积同时处理时空patch，增强对时间演化的建模能力

    最佳实践:
      - 选择patch_size时需权衡空间分辨率与计算效率，过大会丢失细节，过小会导致token数量过多
      - 当输入尺寸不能被patch_size整除时，使用对称零填充而非截断，以保持空间中心对齐
      - embed_dim应与后续网络的隐藏维度一致，避免额外的维度转换开销
      - 对于气象场，建议使用LayerNorm归一化以稳定不同变量间的尺度差异
      - 使用stride=patch_size的卷积而非全连接，可保留patch内的局部空间结构并减少参数量

    常见错误:
      - 忘记处理尺寸不整除情况，导致边界信息丢失或形状不匹配
      - 在多GPU训练时未同步填充策略，导致不同设备上的输出形状不一致
      - 混淆通道顺序(NCHW vs NHWC)，特别是在应用LayerNorm时需要正确的维度置换
      - patch_size设置过大导致下采样后丢失重要的中小尺度天气系统信息

    论文参考:
      - 标题: Accurate medium-range global weather forecasting with 3D neural networks
        作者: Kaifeng Bi et al.
        年份: 2023
        摘要: Pangu-Weather论文，提出3D Earth-specific Transformer和2D/3D patch嵌入策略

  graph:
    类型关系:
      - 神经网络层
      - 嵌入层
      - 卷积层
    所属结构:
      - 3D Patch Embedding模块
      - Pangu-Weather编码器
    依赖组件:
      - nn.Conv2d
      - nn.ZeroPad2d
      - nn.LayerNorm
    变体组件:
      - PanguEmbedding3D (三维上空变量嵌入)
      - Cube Embedding (FuXi的时空嵌入)
      - ViT Patch Embedding (标准视觉Transformer嵌入)
    使用模型:
      - Pangu-Weather
    类型兼容:
      输入:
        - SurfaceField
      输出:
        - TokenGrid2D

