component:
  meta:
    name: PanguDownSample3D
    中文名称: 盘古三维下采样层
    别名: 3D Downsampling Layer, Pangu 3D DownSample
    version: 1.0
    领域: 深度学习/气象AI
    分类: 神经网络组件
    子类: 下采样层
    作者: OneScience
    tags:
      - 气象AI
      - 下采样
      - 3D建模
      - Swin-Transformer

  concept:
    描述: >
      Pangu-Weather 风格的三维大气变量下采样模块，对 (气压层, 纬度, 经度) 三维特征图进行空间下采样。
      仅对水平方向（纬度、经度）做 2 倍下采样，气压层维度保持不变，同时将通道数加倍（从 C 到 2C），
      实现空间分辨率与特征维度的权衡。
    直觉理解: >
      类似于将一张高分辨率的三维气象"体积图"压缩成低分辨率版本：想象把地球表面的网格从精细变粗糙，
      每 2×2 个相邻格点合并成 1 个格点，但保留更丰富的特征信息（通道数翻倍）。垂直方向的气压层数不变，
      就像保持蛋糕的层数，只是把每层的平面尺寸缩小了。
    解决的问题:
      - 高分辨率计算负担: 在 Pangu-Weather 的层次化编码器中，需要逐步降低空间分辨率以扩大感受野并减少计算量，同时保持垂直结构完整性
      - 多尺度特征提取: 通过降低水平分辨率并增加通道维度，使模型能够在不同尺度上捕捉大气场的层次化特征
      - 3D 体结构适配: 针对 Pangu 的三维体建模策略，需要在保持气压层维度的前提下对水平方向进行下采样

  theory:
    核心公式:
      Token合并与线性映射:
        表达式: y = Linear(LayerNorm(Merge4Tokens(Pad(x))))
        变量说明:
          x:
            名称: 输入三维特征体
            形状: [B, pl*lat*lon, C]
            描述: 批次大小 B，气压层数 pl，纬度格点数 lat，经度格点数 lon，输入通道数 C
          Pad(x):
            名称: 零填充后的特征体
            形状: [B, pl, lat_padded, lon_padded, C]
            描述: 对纬度和经度方向进行零填充，使其能被 2 整除；气压层维度不填充
          Merge4Tokens(·):
            名称: 四邻域 Token 合并
            形状: [B, pl*out_lat*out_lon, 4C]
            描述: 将水平方向每 2×2 个相邻 token 合并为 1 个，通道维从 C 扩展到 4C
          y:
            名称: 输出特征体
            形状: [B, pl*out_lat*out_lon, 2C]
            描述: 经过 LayerNorm 和线性层后，通道数降至 2C，实现 2 倍下采样

  structure:
    架构类型: 基于 Token 合并的下采样模块
    计算流程:
      - 将输入从 [B, N, C] 重塑为 [B, pl, lat, lon, C] 的三维体结构
      - 对纬度和经度方向进行零填充，确保能被 2 整除（气压层不填充）
      - 将水平方向每 2×2 个相邻位置的 token 合并，形成 [B, pl*out_lat*out_lon, 4C]
      - 应用 LayerNorm 归一化
      - 通过线性层将通道数从 4C 降至 2C
    计算流程图: |
      输入 x: [B, pl*lat*lon, C]
        ↓ reshape
      三维体: [B, pl, lat, lon, C]
        ↓ ZeroPad3d (仅水平方向)
      填充体: [B, pl, lat_pad, lon_pad, C]
        ↓ reshape + permute (合并 2×2 邻域)
      合并体: [B, pl*out_lat*out_lon, 4C]
        ↓ LayerNorm
      归一化: [B, pl*out_lat*out_lon, 4C]
        ↓ Linear(4C → 2C)
      输出 y: [B, pl*out_lat*out_lon, 2C]

  interface:
    参数:
      input_resolution:
        类型: tuple[int, int, int]
        默认值: null
        描述: 输入特征图的空间分辨率 (pl, lat, lon)，其中 pl 为气压层数，lat 为纬度格点数，lon 为经度格点数
      output_resolution:
        类型: tuple[int, int, int]
        默认值: null
        描述: 输出特征图的空间分辨率 (pl, out_lat, out_lon)，气压层维度与输入一致，水平方向约为输入的 1/2
      in_dim:
        类型: int
        默认值: 192
        描述: 输入 token 的通道数，输出通道数为 in_dim * 2
    输入:
      x:
        类型: Tensor3D_Flattened
        形状: [B, pl*lat*lon, C]
        描述: 展平的三维特征体，其中 B 为批次大小，pl*lat*lon 为总 token 数，C 为输入通道数（等于 in_dim）
    输出:
      y:
        类型: Tensor3D_Downsampled
        形状: [B, pl*out_lat*out_lon, 2C]
        描述: 下采样后的特征体，气压层数不变，水平分辨率减半，通道数加倍

  types:
    Tensor3D_Flattened:
      形状: [B, pl*lat*lon, C]
      描述: 三维气象体在空间维度上展平的 token 序列，保留批次和通道维度
    Tensor3D_Downsampled:
      形状: [B, pl*out_lat*out_lon, 2C]
      描述: 下采样后的三维体 token 序列，水平分辨率减半，通道数加倍

  constraints:
    shape_constraints:
      - 规则: out_lat * 2 >= in_lat
        描述: 输出纬度的 2 倍应大于等于输入纬度，以确保填充量非负
      - 规则: out_lon * 2 >= in_lon
        描述: 输出经度的 2 倍应大于等于输入经度，以确保填充量非负
      - 规则: out_pl == in_pl
        描述: 输出气压层数必须等于输入气压层数，该模块仅对水平方向下采样
    parameter_constraints:
      - 规则: in_dim > 0
        描述: 输入通道数必须为正整数
      - 规则: all(r > 0 for r in input_resolution)
        描述: 输入分辨率的所有维度必须为正整数
    compatibility_rules:
      - 输入类型: Tensor3D_Flattened
        输出类型: Tensor3D_Downsampled
        描述: 输入必须是展平的三维体 token 序列，输出为下采样后的 token 序列

  implementation:
    框架: pytorch
    示例代码: |
      import torch
      from torch import nn

      class PanguDownSample3D(nn.Module):
          def __init__(self, input_resolution, output_resolution, in_dim=192):
              super().__init__()
              self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
              self.norm = nn.LayerNorm(4 * in_dim)
              self.input_resolution = input_resolution
              self.output_resolution = output_resolution

              in_pl, in_lat, in_lon = input_resolution
              out_pl, out_lat, out_lon = output_resolution

              h_pad = out_lat * 2 - in_lat
              w_pad = out_lon * 2 - in_lon
              pad_top = h_pad // 2
              pad_bottom = h_pad - pad_top
              pad_left = w_pad // 2
              pad_right = w_pad - pad_left

              self.pad = nn.ZeroPad3d((pad_left, pad_right, pad_top, pad_bottom, 0, 0))

          def forward(self, x):
              B, N, C = x.shape
              in_pl, in_lat, in_lon = self.input_resolution
              out_pl, out_lat, out_lon = self.output_resolution

              x = x.reshape(B, in_pl, in_lat, in_lon, C)
              x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
              x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
              x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)

              x = self.norm(x)
              x = self.linear(x)
              return x

  usage_examples:
    - 场景: 典型 Pangu-Weather 大气变量配置，气压层保持 8 层，水平分辨率 181×360 → 91×180
      示例代码: |
        downsample = PanguDownSample3D(
            input_resolution=(8, 181, 360),
            output_resolution=(8, 91, 180),
            in_dim=192,
        )
        x = torch.randn(2, 521280, 192)  # (B, 8*181*360, C)
        out = downsample(x)
        # out.shape → torch.Size([2, 131040, 384])  # (B, 8*91*180, 2C)

    - 场景: 水平整除格点的标准 2 倍下采样，pl=13, 128×256 → 64×128
      示例代码: |
        downsample2 = PanguDownSample3D(
            input_resolution=(13, 128, 256),
            output_resolution=(13, 64, 128),
            in_dim=192,
        )
        x2 = torch.randn(2, 13 * 128 * 256, 192)
        out2 = downsample2(x2)
        # out2.shape → torch.Size([2, 106496, 384])  # (B, 13*64*128, 2*192)

  knowledge:
    应用说明: >
      PanguDownSample3D 属于 Swin-Transformer 架构中的层次化多尺度建模模块。在气象 AI 领域建模中，
      多尺度建模是捕捉不同空间尺度大气现象的核心策略：通过逐步降低空间分辨率并增加特征通道数，
      模型能够在较低分辨率上以更大的感受野捕捉行星尺度环流，同时在高分辨率上保留局地细节。
      该组件解决了高分辨率全球气象场计算负担过重的问题，通过 U-Net 式的编码-解码结构实现多尺度特征提取与融合。
    热点模型:
      - 模型: Pangu-Weather
        年份: 2023
        场景: 全球中期天气预报（7天及以上），需要在三维大气体上进行层次化编码
        方案: 在 3DEST 编码器的前 2 层保持分辨率 (8×360×181×C)，后 6 层通过 PanguDownSample3D 将水平分辨率降至 (8×180×91×2C)，在低分辨率上执行大感受野的 3D 窗口注意力，解码器对称地通过上采样恢复分辨率
        作用: 实现三维体的层次化多尺度表示，在控制计算量的同时捕捉从局地到行星尺度的大气结构
        创新: 将 Swin-Transformer 的层次化下采样扩展到三维空间体，仅对水平方向下采样而保持气压层维度不变，符合大气垂直分层的物理特性

      - 模型: FuXi
        年份: 2023
        场景: 全球 0-15 天中长期天气预报，采用级联模型结构
        方案: 使用 Swin Transformer V2 构建 U-Transformer 主干，通过 Down Block 将特征从 (C, 180, 360) 下采样到 (2C, 90, 180)，在低分辨率上执行窗口注意力，再通过 Up Block 上采样并与编码器特征跳跃连接
        作用: 在二维空间网格上实现多尺度特征提取，平衡计算效率与长程依赖建模能力
        创新: 结合 Swin V2 的 scaled cosine attention 与 U-Net 结构，通过跳跃连接保留细节信息

    最佳实践:
      - 下采样比例通常为 2 倍，过大的下采样比例会导致信息丢失
      - 在下采样的同时将通道数加倍（C → 2C），以补偿空间分辨率降低带来的表达能力损失
      - 对于三维气象体，应保持垂直维度（气压层）不变，仅对水平方向下采样，以保留大气垂直分层结构
      - 使用 LayerNorm 而非 BatchNorm，以适应气象数据的时空变异性
      - 配合跳跃连接使用，在解码器中融合编码器的高分辨率特征

    常见错误:
      - 对气压层维度进行下采样，破坏了大气垂直分层的物理意义
      - 下采样时未相应增加通道数，导致特征表达能力不足
      - 输入分辨率不满足输出分辨率的 2 倍关系，导致填充计算错误
      - 忘记在下采样前进行零填充，导致不能整除时出现形状不匹配

    论文参考:
      - 标题: "Accurate medium-range global weather forecasting with 3D neural networks"
        作者: Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, Qi Tian
        年份: 2023
        摘要: Pangu-Weather 论文，提出 3D Earth-specific Transformer (3DEST) 架构，将高度维度显式建模为空间维度，通过层次化编码-解码结构实现多尺度三维大气建模

  graph:
    类型关系:
      - 神经网络层
      - 下采样模块
      - Transformer组件
    所属结构:
      - 3DEST编码器
      - U-Net式层次化网络
      - Swin-Transformer主干
    依赖组件:
      - LayerNorm
      - Linear层
      - ZeroPad3d
    变体组件:
      - PanguDownSample2D
      - PanguUpSample3D
      - SwinDownSample
    使用模型:
      - Pangu-Weather
      - FuXi
      - 其他基于Swin-Transformer的气象模型
    类型兼容:
      输入:
        - Tensor3D_Flattened
      输出:
        - Tensor3D_Downsampled
