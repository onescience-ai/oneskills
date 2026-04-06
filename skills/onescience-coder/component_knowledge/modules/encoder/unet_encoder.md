component:

  meta:
    name: UNetEncoder
    alias: U-Net Encoder Family
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: encoder
    author: OneScience
    license: Apache-2.0
    tags:
      - unet
      - feature_extraction
      - semantic_segmentation
      - multi_scale
      - down_sampling

  concept:

    description: >
      U-Net编码器家族包含1D、2D、3D三个变体，用于提取输入数据的多尺度特征。
      通过初始双卷积层和多个下采样层，逐步降低空间分辨率同时增加特征通道数，
      为解码器提供层次化的特征表示。

    intuition: >
      就像医生分析医学影像时，先看整体轮廓（浅层特征），然后逐步放大关注细节（深层特征）。
      每次下采样就像缩小视野但能看到更大的模式，特征通道数的增加则表示学习到更复杂的特征组合。

    problem_it_solves:
      - 多尺度特征的层次化提取
      - 编码器-解码器架构中的特征压缩
      - 空间/时间/体积数据的降维处理
      - 语义分割中的特征学习

  theory:

    formula:

      feature_extraction:
        expression: |
          x_0 = \text{DoubleConv}(x_{input})
          x_i = \text{DownSample}(x_{i-1}) \quad \text{for } i = 1, 2, ..., N
          \text{features} = [x_0, x_1, ..., x_N]

      channel_evolution:
        expression: |
          \text{channels}_i = \text{base\_channels} \times 2^i
          \text{resolution}_i = \frac{\text{input\_resolution}}{2^i}

  structure:

    architecture: unet_encoder_family

    pipeline:

      - name: InitialConvolution
        operation: double_conv (initial feature extraction)

      - name: ProgressiveDownsampling
        operation: down1d/down2d/down3d (multi-stage)

      - name: FeatureCollection
        operation: feature_list (for skip connections)

  interface:

    parameters:

      in_channels:
        type: int
        description: 输入通道数

      base_channels:
        type: int
        description: 初始特征通道数

      num_stages:
        type: int
        description: 下采样的层数

      bilinear:
        type: bool
        description: 解码器是否使用双线性/三线性插值

      normtype:
        type: str
        description: 归一化类型 ('bn' 或 'in')

      kernel_size:
        type: int
        description: 卷积核大小，必须为奇数

    inputs:

      x:
        type: InputData
        shape: 
          - 1D: [batch, in_channels, length]
          - 2D: [batch, in_channels, height, width]
          - 3D: [batch, in_channels, depth, height, width]

    outputs:

      features:
        type: FeatureList
        description: 包含(num_stages + 1)个Tensor的特征列表

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules.layer.unet_layer import (
          DoubleConv1D, Down1D, 
          DoubleConv2D, Down2D, 
          DoubleConv3D, Down3D
      )

      class UNetEncoder1D(nn.Module):
          def __init__(self, in_channels, base_channels=16, num_stages=2, **kwargs):
              super().__init__()
              self.inc = DoubleConv1D(in_channels, base_channels)
              self.down_stages = nn.ModuleList()
              in_ch = base_channels
              for i in range(num_stages):
                  out_ch = in_ch * 2
                  self.down_stages.append(Down1D(in_ch, out_ch))
                  in_ch = out_ch

          def forward(self, x):
              features = [self.inc(x)]
              for stage in self.down_stages:
                  features.append(stage(features[-1]))
              return features

      class UNetEncoder2D(nn.Module):
          def __init__(self, in_channels, base_channels=16, num_stages=2, **kwargs):
              super().__init__()
              self.inc = DoubleConv2D(in_channels, base_channels)
              self.down_stages = nn.ModuleList()
              in_ch = base_channels
              for i in range(num_stages):
                  out_ch = in_ch * 2
                  self.down_stages.append(Down2D(in_ch, out_ch))
                  in_ch = out_ch

          def forward(self, x):
              features = [self.inc(x)]
              for stage in self.down_stages:
                  features.append(stage(features[-1]))
              return features

      class UNetEncoder3D(nn.Module):
          def __init__(self, in_channels, base_channels=16, num_stages=2, **kwargs):
              super().__init__()
              self.inc = DoubleConv3D(in_channels, base_channels)
              self.down_stages = nn.ModuleList()
              in_ch = base_channels
              for i in range(num_stages):
                  out_ch = in_ch * 2
                  self.down_stages.append(Down3D(in_ch, out_ch))
                  in_ch = out_ch

          def forward(self, x):
              features = [self.inc(x)]
              for stage in self.down_stages:
                  features.append(stage(features[-1]))
              return features

  skills:

    build_unet_encoder:

      description: 构建U-Net编码器家族

      inputs:
        - dimension (1D/2D/3D)
        - in_channels
        - base_channels
        - num_stages

      prompt_template: |

        构建{{dimension}} U-Net编码器，实现多尺度特征提取。

        参数：
        in_channels = {{in_channels}}
        base_channels = {{base_channels}}
        num_stages = {{num_stages}}

        要求：
        1. 支持渐进式下采样
        2. 特征通道数倍增
        3. 输出特征金字塔
        4. 适配解码器跳跃连接

  knowledge:

    hot_models:

      - model: U-Net
        year: 2015
        role: 医学图像分割的经典架构
        architecture: encoder-decoder with skip connections

      - model: U-Net++
        year: 2018
        role: 改进的U-Net架构
        architecture: nested skip connections

    best_practices:

      - 确保编码器和解码器的通道数对称
      - 下采样方法应与数据特性匹配
      - 特征金字塔对细节恢复很重要

    paper_references:

      - title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        authors: Ronneberger et al.
        year: 2015

  graph:

    is_a:
      - NeuralNetworkEncoder
      - FeatureExtractor
      - MultiScaleProcessor

    part_of:
      - UNetArchitecture
      - SegmentationModels

    variants:
      - UNetEncoder1D (时序数据)
      - UNetEncoder2D (图像数据)
      - UNetEncoder3D (体积数据)

    used_in_models:
      - U-Net
      - U-Net++
      - 3D U-Net

    compatible_with:

      inputs:
        - ImageData
        - SequenceData
        - VolumeData

      outputs:
        - FeaturePyramid
        - MultiScaleFeatures
