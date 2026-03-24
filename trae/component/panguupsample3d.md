component:
  meta:
    name: PanguUpSample3D
    中文名称: 三维上采样层
    别名: 3D上采样模块, Pangu三维上采样
    version: 1.0
    领域: 深度学习/气象
    分类: 神经网络组件
    子类: 上采样层/解码器
    作者: OneScience
    tags:
      - 气象AI
      - 上采样
      - 3D建模
      - Pangu-Weather

  concept:
    描述: >
      三维大气变量上采样模块，通过像素重排（pixel shuffle）将低分辨率特征图在水平方向（纬度、经度）上采样2倍，
      同时保持或调整气压层维度，是 PanguDownSample3D 的逆操作，用于 Pangu-Weather 解码器中恢复空间分辨率。
    直觉理解: >
      类似于将一张压缩的气象图放大，把每个格点"拆分"成4个子格点（2×2），让预报场从粗糙网格恢复到精细网格。
      就像把一个低分辨率的天气预报图放大到高分辨率，但不是简单插值，而是通过学习到的线性变换重新分配信息。
    解决的问题:
      - 分辨率恢复: 在 U-Net 式编码-解码结构中，将编码器下采样后的低分辨率特征恢复到原始或目标分辨率
      - 通道维度调整: 将高维潜在特征（如384维）映射回较低维度（如192维），同时完成空间上采样
      - 三维体结构保持: 在处理三维大气体数据时，保持气压层-经度-纬度的结构化表示

  theory:
    核心公式:
      像素重排上采样:
        表达式: x' = Reshape(Linear(x), [B, pl, H×2, W×2, C'])
        变量说明:
          x:
            名称: 输入特征
            形状: [B, pl×H×W, C]
            描述: 批次大小B，气压层pl，水平分辨率H×W（纬度×经度），输入通道C
          x':
            名称: 输出特征
            形状: [B, out_pl×H'×W', C']
            描述: 上采样后的特征，水平分辨率提升至H'×W'（通常为H×2和W×2经裁剪后），输出通道C'
          Linear:
            名称: 线性变换
            形状: [C] → [C'×4]
            描述: 将输入通道扩展4倍，为2×2像素重排准备
          Reshape:
            名称: 重排操作
            形状: [pl, H, W, 2, 2, C'] → [pl, H×2, W×2, C']
            描述: 将通道维度重排为空间维度，实现上采样

  structure:
    架构类型: 像素重排上采样网络
    计算流程:
      - 线性扩展：通过 Linear1 将输入通道 C 扩展到 C'×4
      - 重排变换：将扩展后的特征重排为 [B, pl, H×2, W×2, C']
      - 气压层裁剪：取前 out_pl 层
      - 中心裁剪：若上采样后尺寸超过目标，从中心裁剪到目标分辨率
      - 归一化：LayerNorm 标准化
      - 线性映射：通过 Linear2 映射到输出通道维度
    计算流程图: |
      输入 x [B, pl×H×W, C]
        ↓
      Linear1: C → C'×4
        ↓
      Reshape: [B, pl, H, W, 2, 2, C'/2] → [B, pl, H×2, W×2, C']
        ↓
      气压层切片: [:, :out_pl, :, :, :]
        ↓
      中心裁剪: 移除多余的边界像素
        ↓
      Flatten: [B, out_pl×H'×W', C']
        ↓
      LayerNorm
        ↓
      Linear2: C' → C'
        ↓
      输出 [B, out_pl×H'×W', C']

  interface:
    参数:
      in_dim:
        类型: int
        默认值: 384
        描述: 输入token的通道数，通常为下采样后的高维特征（如192×2=384）
      out_dim:
        类型: int
        默认值: 192
        描述: 输出token的通道数，恢复到编码前的维度
      input_resolution:
        类型: tuple[int, int, int]
        默认值: null
        描述: 输入特征图的空间分辨率 (pl, lat, lon)，表示气压层数、纬度格点数、经度格点数
      output_resolution:
        类型: tuple[int, int, int]
        默认值: null
        描述: 目标输出分辨率 (out_pl, out_lat, out_lon)，水平方向应满足 out_lat ≤ in_lat×2 且 out_lon ≤ in_lon×2
    输入:
      x:
        类型: Tensor3D_Flattened
        形状: [B, pl×lat×lon, in_dim]
        描述: 扁平化的三维大气特征，批次B，空间token数为气压层×纬度×经度的乘积，通道数为in_dim
    输出:
      output:
        类型: Tensor3D_Flattened
        形状: [B, out_pl×out_lat×out_lon, out_dim]
        描述: 上采样后的三维大气特征，空间分辨率提升，通道数降至out_dim

  types:
    Tensor3D_Flattened:
      形状: [B, N, C]
      描述: 扁平化的三维体张量，N = pl×lat×lon 为空间token总数，C为特征通道数

  constraints:
    shape_constraints:
      - 规则: out_lat ≤ in_lat × 2
        描述: 输出纬度不能超过输入纬度的2倍
      - 规则: out_lon ≤ in_lon × 2
        描述: 输出经度不能超过输入经度的2倍
      - 规则: out_pl ≤ in_pl
        描述: 输出气压层数不能超过输入气压层数
      - 规则: N = pl × lat × lon
        描述: 输入token数必须等于三个空间维度的乘积
    parameter_constraints:
      - 规则: in_dim % 2 == 0
        描述: 输入通道数必须为偶数，以便进行2×2像素重排
      - 规则: out_dim > 0
        描述: 输出通道数必须为正整数
    compatibility_rules:
      - 输入类型: Tensor3D_Flattened
        输出类型: Tensor3D_Flattened
        描述: 保持扁平化三维体结构，仅改变空间分辨率和通道数

  implementation:
    框架: pytorch
    示例代码: |
      import torch
      from torch import nn

      class PanguUpSample3D(nn.Module):
          def __init__(self, in_dim=384, out_dim=192,
                       input_resolution=None, output_resolution=None):
              super().__init__()
              self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
              self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
              self.norm = nn.LayerNorm(out_dim)
              self.input_resolution = input_resolution
              self.output_resolution = output_resolution

          def forward(self, x):
              B, N, C = x.shape
              in_pl, in_lat, in_lon = self.input_resolution
              out_pl, out_lat, out_lon = self.output_resolution

              # 线性扩展到4倍通道
              x = self.linear1(x)
              # 重排为2×2像素
              x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2)
              x = x.permute(0, 1, 2, 4, 3, 5, 6)
              x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)

              # 裁剪到目标分辨率
              pad_h, pad_w = in_lat * 2 - out_lat, in_lon * 2 - out_lon
              pad_top, pad_left = pad_h // 2, pad_w // 2
              x = x[:, :out_pl, pad_top:in_lat*2-pad_h+pad_top,
                    pad_left:in_lon*2-pad_w+pad_left, :]

              x = x.reshape(B, -1, x.shape[-1])
              x = self.norm(x)
              x = self.linear2(x)
              return x

  usage_examples:
    - 场景: Pangu-Weather 解码器标准配置（8层，91×180 → 181×360）
      示例代码: |
        upsample = PanguUpSample3D(
            in_dim=384,
            out_dim=192,
            input_resolution=(8, 91, 180),
            output_resolution=(8, 181, 360),
        )
        x = torch.randn(2, 131040, 384)  # (B, 8*91*180, 384)
        out = upsample(x)
        # out.shape → (2, 521280, 192)

    - 场景: 13气压层整除上采样（64×128 → 128×256）
      示例代码: |
        upsample = PanguUpSample3D(
            in_dim=384,
            out_dim=192,
            input_resolution=(13, 64, 128),
            output_resolution=(13, 128, 256),
        )
        x = torch.randn(2, 106496, 384)  # (B, 13*64*128, 384)
        out = upsample(x)
        # out.shape → (2, 425984, 192)

  knowledge:
    应用说明: >
      PanguUpSample3D 属于 U-Net 式编码-解码架构中的上采样模块，在气象AI建模中用于解码器阶段恢复空间分辨率。
      该组件是多尺度建模范式的核心部分，通过逐层上采样将编码器压缩的低分辨率特征恢复到原始分辨率，
      同时结合跳跃连接融合不同尺度的特征，以保留细节信息。在全球天气预报任务中，上采样模块需要处理
      三维大气体数据（气压层×纬度×经度），并保持球面几何结构的一致性。
    热点模型:
      - 模型: Pangu-Weather
        年份: 2023
        场景: 全球中期天气预报（7天及以上），需要在三维大气体上进行多尺度建模
        方案: >
          在3DEST编码-解码主干中，解码器的后6层使用PanguUpSample3D将低分辨率特征（8×180×91×2C）
          逐步上采样回高分辨率（8×360×181×C）。通过像素重排而非转置卷积实现上采样，避免棋盘效应。
          同时配合跳跃连接，将第二编码层的输出与第七解码层的输出在通道维拼接，融合多尺度信息。
        作用: 恢复空间分辨率，将编码器提取的抽象特征映射回原始网格尺度，用于最终的patch recovery
        创新: 将2D像素重排上采样扩展到3D体数据，保持气压层-经纬度的结构化表示，配合Earth-specific positional bias实现地球几何感知的上采样
      - 模型: FuXi
        年份: 2023
        场景: 0-15天全球天气预报，采用级联U-Transformer架构
        方案: >
          在U-Transformer的Up Block中使用上采样模块将特征从低分辨率（90×180）恢复到高分辨率（180×360）。
          虽然FuXi采用2D通道堆叠策略（将气压层作为通道），但上采样机制与PanguUpSample3D类似，
          都基于线性扩展+重排的像素重排方法，并配合Swin Transformer V2的窗口注意力机制。
        作用: 在U-Transformer解码路径中恢复空间细节，配合跳跃连接实现多尺度特征融合
        创新: 结合scaled cosine attention和log-spaced相对位置编码，提升上采样过程中的训练稳定性

    最佳实践:
      - 配合跳跃连接使用，将编码器对应层的特征与上采样后的特征在通道维拼接，保留细节信息
      - 使用LayerNorm而非BatchNorm，避免在小批次或长序列自回归中的不稳定性
      - 输入输出分辨率应提前规划，确保下采样和上采样路径对称，避免尺寸不匹配
      - 对于非整除的分辨率变化，采用中心裁剪而非零填充，保持特征图的中心区域信息
      - 在气象应用中，注意经度方向的周期性，裁剪时应考虑边界连续性
    常见错误:
      - 输入token数与input_resolution不匹配，导致reshape失败
      - output_resolution超过input_resolution的2倍，导致裁剪后出现负索引
      - 忘记设置input_resolution和output_resolution参数，导致forward时报错
      - 在多步自回归预测中，上采样后的分辨率与下一层输入不匹配
      - 通道数设置不当，in_dim不是偶数或out_dim×4超过显存限制
    论文参考:
      - 标题: Accurate medium-range global weather forecasting with 3D neural networks
        作者: Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, Qi Tian
        年份: 2023
        摘要: >
          提出Pangu-Weather模型，采用3D Earth-specific Transformer架构进行全球中期天气预报。
          在编码-解码结构中使用3D上采样和下采样模块，配合Earth-specific positional bias，
          在ERA5数据上所有评估变量的RMSE和ACC均优于ECMWF-IFS和FourCastNet。

  graph:
    类型关系:
      - 神经网络层
      - 上采样模块
      - 解码器组件
    所属结构:
      - 3DEST解码器
      - U-Net式编码-解码架构
      - Pangu-Weather主干网络
    依赖组件:
      - Linear层
      - LayerNorm
      - PanguDownSample3D（逆操作）
    变体组件:
      - PanguUpSample2D（二维版本，仅处理地表变量）
      - PatchRecovery3D（最终重建层）
    使用模型:
      - Pangu-Weather
      - FuXi（类似机制）
    类型兼容:
      输入:
        - Tensor3D_Flattened
      输出:
        - Tensor3D_Flattened
