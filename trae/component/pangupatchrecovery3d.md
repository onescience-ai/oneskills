component:
  meta:
    name: PanguPatchRecovery3D
    中文名称: 三维面片重建层
    别名: 3D Patch Recovery, 三维反卷积重建模块
    version: 1.0
    领域: 深度学习/气象
    分类: 神经网络组件
    子类: 重建层/解码器
    作者: OneScience
    tags:
      - 气象AI
      - 三维重建
      - 反卷积
      - Pangu-Weather

  concept:
    描述: >
      三维面片重建层是Pangu-Weather模型解码阶段的核心组件，通过三维反卷积操作将编码器输出的低分辨率三维patch特征还原为原始层数与空间分辨率的三维大气场，并自动裁剪补零边界，实现从潜在表示空间到物理变量空间的精确映射。
    直觉理解: >
      就像用放大镜将压缩的拼图块还原成完整图片。编码器将大气场切成小块并压缩特征，重建层则像逆向操作，把这些压缩块"展开"并拼接回原始的三维网格，同时去掉填充的多余边界，确保输出尺寸与输入完全一致。
    解决的问题:
      - 潜在空间到物理空间的映射: 将编码器学习到的高维抽象特征（如384通道）映射回原始物理变量维度（如5个上空变量），实现从特征表示到可解释物理量的转换
      - 空间分辨率恢复: 将下采样后的低分辨率三维体（如7×181×360）精确还原到原始高分辨率网格（如13×721×1440），保证预报场与输入场的空间一致性
      - 边界填充处理: 自动识别并裁剪patch划分时引入的零填充区域，确保输出场的有效区域与ERA5数据的实际覆盖范围严格对齐

  theory:
    核心公式:
      三维反卷积变换:
        表达式: output = ConvTranspose3d(x; kernel=patch_size, stride=patch_size)
        变量说明:
          x:
            名称: 输入特征体
            形状: [B, in_chans, L', H', W']
            描述: 批次大小B，输入通道in_chans（编码器输出特征维度），下采样后的层数L'、纬度H'、经度W'
          output:
            名称: 重建后的三维场
            形状: [B, out_chans, Pl, Lat, Lon]
            描述: 批次大小B，输出通道out_chans（物理变量数），重建后的层数Pl、纬度Lat、经度Lon（含填充）
          patch_size:
            名称: 三维patch尺寸
            形状: [patch_l, patch_h, patch_w]
            描述: 垂直层、纬度、经度三个维度的patch大小，同时作为反卷积的kernel_size和stride
      边界裁剪:
        表达式: y = output[:, :, padding_front:Pl-padding_back, padding_top:Lat-padding_bottom, padding_left:Lon-padding_right]
        变量说明:
          padding_front/back:
            名称: 垂直层维度填充量
            形状: 标量
            描述: 前后填充量，通过(Pl - img_size[0])计算并均分
          padding_top/bottom:
            名称: 纬度维度填充量
            形状: 标量
            描述: 上下填充量，通过(Lat - img_size[1])计算并均分
          padding_left/right:
            名称: 经度维度填充量
            形状: 标量
            描述: 左右填充量，通过(Lon - img_size[2])计算并均分
          y:
            名称: 最终输出场
            形状: [B, out_chans, img_size[0], img_size[1], img_size[2]]
            描述: 裁剪后精确匹配目标尺寸的三维物理场

  structure:
    架构类型: 三维反卷积网络
    计算流程:
      - 三维反卷积：使用ConvTranspose3d将输入特征体从[B,in_chans,L',H',W']上采样到[B,out_chans,Pl,Lat,Lon]
      - 计算填充量：分别计算三个空间维度的总填充量(Pl-img_size[0], Lat-img_size[1], Lon-img_size[2])
      - 均分填充：将每个维度的总填充量均分为前/后、上/下、左/右两部分
      - 边界裁剪：使用切片操作去除填充区域，输出精确尺寸[B,out_chans,img_size[0],img_size[1],img_size[2]]
    计算流程图: |
      输入特征体 x [B, in_chans, L', H', W']
        ↓
      ConvTranspose3d(kernel=patch_size, stride=patch_size)
        ↓
      重建体 output [B, out_chans, Pl, Lat, Lon] (含填充)
        ↓
      计算三维填充量: pl_pad, lat_pad, lon_pad
        ↓
      均分填充: padding_front/back, padding_top/bottom, padding_left/right
        ↓
      切片裁剪: output[:,:, front:Pl-back, top:Lat-bottom, left:Lon-right]
        ↓
      输出 y [B, out_chans, img_size[0], img_size[1], img_size[2]]

  interface:
    参数:
      img_size:
        类型: tuple[int, int, int]
        默认值: (13, 721, 1440)
        描述: 输出目标场的尺寸(L, H, W)，分别对应垂直层数、纬度网格数、经度网格数，默认为ERA5上空13层的全球0.25°分辨率
      patch_size:
        类型: tuple[int, int, int]
        默认值: (2, 4, 4)
        描述: 三维patch大小(patch_l, patch_h, patch_w)，即反卷积的kernel_size与stride，决定上采样倍率
      in_chans:
        类型: int
        默认值: 384
        描述: 输入特征通道数，通常为编码器最后一层的输出维度（192*2表示经过下采样后通道数翻倍）
      out_chans:
        类型: int
        默认值: 5
        描述: 输出场通道数，对应物理变量数（上空5变量：Z/Q/T/U/V）
    输入:
      x:
        类型: Tensor3D
        形状: [B, in_chans, L', H', W']
        描述: 编码器输出的三维特征体，其中L'=ceil(img_size[0]/patch_size[0])，H'=ceil(img_size[1]/patch_size[1])，W'=ceil(img_size[2]/patch_size[2])
    输出:
      y:
        类型: Tensor3D
        形状: [B, out_chans, img_size[0], img_size[1], img_size[2]]
        描述: 重建后的三维物理场，精确匹配目标尺寸，可直接用于损失计算或后续预报

  types:
    Tensor3D:
      形状: [B, C, L, H, W]
      描述: 五维张量，表示批次×通道×层数×纬度×经度的三维大气场数据

  constraints:
    shape_constraints:
      - 规则: img_size[0] <= L' * patch_size[0]
        描述: 目标层数不能超过反卷积后的最大层数，确保裁剪操作有效
      - 规则: img_size[1] <= H' * patch_size[1]
        描述: 目标纬度网格数不能超过反卷积后的最大纬度数
      - 规则: img_size[2] <= W' * patch_size[2]
        描述: 目标经度网格数不能超过反卷积后的最大经度数
    parameter_constraints:
      - 规则: in_chans > 0 and out_chans > 0
        描述: 输入输出通道数必须为正整数
      - 规则: all(p > 0 for p in patch_size)
        描述: patch_size三个维度必须均为正整数
      - 规则: all(s > 0 for s in img_size)
        描述: img_size三个维度必须均为正整数
    compatibility_rules:
      - 输入类型: Tensor3D
        输出类型: Tensor3D
        描述: 输入输出均为五维张量，但通道数和空间维度发生变化

  implementation:
    框架: pytorch
    示例代码: |
      import torch
      from torch import nn

      class PanguPatchRecovery3D(nn.Module):
          def __init__(self, img_size=(13, 721, 1440),
                       patch_size=(2, 4, 4),
                       in_chans=192*2,
                       out_chans=5):
              super().__init__()
              self.img_size = img_size
              self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

          def forward(self, x: torch.Tensor):
              output = self.conv(x)
              _, _, Pl, Lat, Lon = output.shape

              pl_pad = Pl - self.img_size[0]
              lat_pad = Lat - self.img_size[1]
              lon_pad = Lon - self.img_size[2]

              padding_front = pl_pad // 2
              padding_back = pl_pad - padding_front
              padding_top = lat_pad // 2
              padding_bottom = lat_pad - padding_top
              padding_left = lon_pad // 2
              padding_right = lon_pad - padding_left

              return output[
                  :, :,
                  padding_front : Pl - padding_back,
                  padding_top : Lat - padding_bottom,
                  padding_left : Lon - padding_right,
              ]

  usage_examples:
    - 场景: 标准上空变量重建（13层×721×1440，ERA5全球0.25°分辨率）
      示例代码: |
        recovery3d = PanguPatchRecovery3D(
            img_size=(13, 721, 1440),
            patch_size=(2, 4, 4),
            in_chans=384,
            out_chans=5,
        )
        x = torch.randn(2, 384, 7, 181, 360)
        y = recovery3d(x)
        # y.shape → torch.Size([2, 5, 13, 721, 1440])

    - 场景: 地表变量重建（单层×721×1440）
      示例代码: |
        recovery_surface = PanguPatchRecovery3D(
            img_size=(1, 721, 1440),
            patch_size=(1, 4, 4),
            in_chans=192,
            out_chans=4,
        )
        x_surface = torch.randn(2, 192, 1, 181, 360)
        y_surface = recovery_surface(x_surface)
        # y_surface.shape → torch.Size([2, 4, 1, 721, 1440])

    - 场景: 低分辨率重建（用于多尺度解码）
      示例代码: |
        recovery_low = PanguPatchRecovery3D(
            img_size=(13, 360, 720),
            patch_size=(2, 2, 2),
            in_chans=768,
            out_chans=5,
        )
        x_low = torch.randn(2, 768, 7, 180, 360)
        y_low = recovery_low(x_low)
        # y_low.shape → torch.Size([2, 5, 13, 360, 720])

  knowledge:
    应用说明: >
      PanguPatchRecovery3D属于解码器/重建层模块，在深度学习气象预报的编码-解码架构中扮演关键角色。该组件位于Pangu-Weather模型的输出端，负责将3DEST编码器学习到的抽象特征表示还原为可解释的物理变量场。在气象AI建模范式中，重建层是连接潜在表示空间与物理观测空间的桥梁，其设计直接影响预报场的精度和物理一致性。三维重建策略相比二维方法能更好地保持垂直层间的物理关联性。
    热点模型:
      - 模型: Pangu-Weather
        年份: 2023
        场景: 全球中期天气预报（7天及以上），需要将3D Earth-specific Transformer编码的高维特征还原为13层×5变量的上空场和4变量的地表场
        方案: 采用三维反卷积(ConvTranspose3d)作为核心算子，kernel_size和stride均设为patch_size(2,4,4)，实现精确的逆patch操作。通过自动边界裁剪机制处理不能整除情况，确保输出尺寸与ERA5数据严格对齐。上空和地表分别使用独立的PatchRecovery3D实例，分别重建5通道和4通道输出
        作用: 将编码器输出的8×360×181×C三维体分离为上空7层和地表1层，分别通过重建层映射回13×1440×721×5和1440×721×4的原始分辨率，支持后续的变量级损失计算和预报输出
        创新: 在三维空间体上实现patch recovery，相比传统二维重建能更好地保持垂直层间的连续性和物理一致性；通过对称的patch embedding和patch recovery设计，确保编码-解码过程的信息无损传递
      - 模型: FuXi
        年份: 2023
        场景: 0-15天全球天气预报，采用级联U-Transformer架构，需要在多尺度特征图上进行上采样重建
        方案: 使用Swin Transformer V2的上采样块(Up Block)结合线性层实现特征重建，通过逐步上采样和跳跃连接恢复空间分辨率。虽未显式使用三维反卷积，但在U-Transformer的解码路径中实现类似的分辨率恢复功能
        作用: 在U-Transformer解码器中，将低分辨率特征(90×180)上采样回高分辨率(180×360)，并与编码器的跳跃连接融合，最终通过线性层映射到70通道的物理变量输出
        创新: 结合Swin V2的scaled cosine attention和patch merging/expanding机制，在保持计算效率的同时实现高质量的多尺度重建
    最佳实践:
      - 确保img_size与patch_size的兼容性，避免过度裁剪导致信息丢失
      - 对于不同物理变量（上空vs地表），使用独立的重建层实例以适配不同的通道数和空间尺寸
      - 在训练时监控重建层的梯度流，避免反卷积层出现梯度消失或爆炸
      - 重建层的输出应直接用于损失计算，避免额外的插值或变换操作以保持精度
      - 对于多尺度架构，在不同分辨率层级使用对应尺寸的重建层，保持编码-解码的对称性
    常见错误:
      - 错误设置img_size导致裁剪后尺寸不匹配，引发张量形状错误
      - 忘记patch_size必须与编码阶段的patch embedding保持一致，导致重建失真
      - 在多GPU训练时未正确同步batch维度，导致不同设备上的裁剪结果不一致
      - 混淆in_chans和out_chans的含义，将编码器输出维度错误地设为out_chans
      - 对填充量计算使用错误的整除方式，导致前后/上下/左右裁剪不对称
    论文参考:
      - 标题: Accurate medium-range global weather forecasting with 3D neural networks
        作者: Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, Qi Tian
        年份: 2023
        摘要: Pangu-Weather论文，提出3D Earth-specific Transformer和对应的3D patch embedding/recovery机制，在ERA5数据上实现超越ECMWF-IFS的中期预报精度

  graph:
    类型关系:
      - 神经网络层
      - 解码器组件
      - 三维卷积模块
    所属结构:
      - Pangu-Weather解码器
      - 3DEST主干网络
      - Patch-based编码解码架构
    依赖组件:
      - ConvTranspose3d
      - 3D Patch Embedding
    变体组件:
      - PanguPatchRecovery2D
      - 线性投影重建层
      - PixelShuffle上采样
    使用模型:
      - Pangu-Weather
    类型兼容:
      输入:
        - Tensor3D
      输出:
        - Tensor3D

