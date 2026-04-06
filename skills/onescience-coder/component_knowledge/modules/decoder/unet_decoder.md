component:

  meta:
    name: UNetDecoder
    alias: U-Net Decoder Family
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: decoder
    author: OneScience
    license: Apache-2.0
    tags:
      - unet
      - image_reconstruction
      - semantic_segmentation
      - skip_connection
      - multi_scale

  concept:

    description: >
      U-Net解码器家族包含1D、2D、3D三个变体，用于从编码器输出的多尺度特征中恢复高分辨率表示。
      通过逐层上采样深层特征并与编码器对应层的浅层特征进行跳跃连接融合，实现精细的空间重建。
      该架构广泛应用于图像分割、时序预测、体积重建等任务。

    intuition: >
      就像医生先看CT扫描的概览（深层特征），然后逐步放大关注细节（上采样），同时参考原始扫描的细节信息（跳跃连接），
      最终给出精确的诊断结果。跳跃连接确保在恢复分辨率时不丢失重要的细节信息。

    problem_it_solves:
      - 从低分辨率特征恢复高分辨率表示
      - 编码器-解码器架构中的信息传递
      - 多尺度特征的融合与重建
      - 图像/信号/体积数据的分割与重建
      - 保持空间/时间细节的分辨率恢复

  theory:

    formula:

      upsampling_with_skip:
        expression: |
          x_{i} = \text{UpSample}(x_{i+1}) \oplus \text{skip}_{i}
          \text{where } \oplus \text{ denotes concatenation}

      feature_reconstruction:
        expression: |
          x_{out} = \text{ConvBlock}(x_{1})
          \text{gradually reconstruct from deep to shallow}

    variables:

      x_{i+1}:
        name: DeepFeatures
        description: 来自更深层的特征

      skip_{i}:
        name: SkipConnection
        description: 来自编码器对应层的跳跃连接特征

      x_{i}:
        name: CurrentLevelFeatures
        description: 当前层的重建特征

  structure:

    architecture: unet_decoder_family

    pipeline:

      - name: DeepFeatureInput
        operation: receive_bottleneck_features

      - name: ProgressiveUpsampling
        operation: up1d/up2d/up3d + skip_connection_fusion

      - name: FeatureReconstruction
        operation: convolution_blocks

      - name: HighResolutionOutput
        operation: final_feature_maps

  interface:

    parameters:

      base_channels:
        type: int
        description: 与编码器匹配的初始特征通道数

      num_stages:
        type: int
        description: 上采样的层数，需与编码器一致

      bilinear:
        type: bool
        description: 是否使用线性/双线性/三线性插值进行上采样

      normtype:
        type: str
        description: 归一化类型 ('bn' 或 'in')

      kernel_size:
        type: int
        description: 卷积核大小，必须为奇数

    inputs:

      features:
        type: FeatureList
        description: 由UNetEncoder输出的特征列表

    outputs:

      output:
        type: ReconstructedFeatures
        description: 恢复到输入分辨率的深层特征

  types:

    FeatureList:
      description: 编码器输出的多尺度特征列表

    ReconstructedFeatures:
      description: 解码器重建的高分辨率特征

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules.layer.unet_layer import Up1D, Up2D, Up3D

      class UNetDecoder1D(nn.Module):
          def __init__(self, base_channels=16, num_stages=2, bilinear=True, 
                       normtype="bn", kernel_size=3):
              super().__init__()
              self.up_stages = nn.ModuleList()
              features = [base_channels * (2 ** i) for i in range(num_stages + 1)]
              
              for i in range(num_stages, 0, -1):
                  in_ch = features[i] + features[i-1] 
                  out_ch = features[i-1]
                  self.up_stages.append(Up1D(in_ch, out_ch, bilinear, normtype, kernel_size=kernel_size))

          def forward(self, features):
              x = features[-1]
              skips = reversed(features[:-1])
              for up_stage, skip in zip(self.up_stages, skips):
                  x = up_stage(x, skip)
              return x

      class UNetDecoder2D(nn.Module):
          def __init__(self, base_channels=16, num_stages=2, bilinear=True, 
                       normtype="bn", kernel_size=3):
              super().__init__()
              self.up_stages = nn.ModuleList()
              features = [base_channels * (2 ** i) for i in range(num_stages + 1)]
              
              for i in range(num_stages, 0, -1):
                  if bilinear:
                      in_ch = features[i] + features[i-1]
                  else:
                      in_ch = features[i]
                  out_ch = features[i-1]
                  self.up_stages.append(Up2D(in_ch, out_ch, bilinear, normtype, kernel_size=kernel_size))

          def forward(self, features):
              x = features[-1]
              skips = reversed(features[:-1])
              for up_stage, skip in zip(self.up_stages, skips):
                  x = up_stage(x, skip)
              return x

      class UNetDecoder3D(nn.Module):
          def __init__(self, base_channels=16, num_stages=2, bilinear=True, 
                       normtype="bn", kernel_size=3):
              super().__init__()
              self.up_stages = nn.ModuleList()
              features = [base_channels * (2 ** i) for i in range(num_stages + 1)]
              
              for i in range(num_stages, 0, -1):
                  in_ch = features[i] + features[i-1]
                  out_ch = features[i-1]
                  self.up_stages.append(Up3D(in_ch, out_ch, bilinear, normtype, kernel_size=kernel_size))

          def forward(self, features):
              x = features[-1]
              skips = reversed(features[:-1])
              for up_stage, skip in zip(self.up_stages, skips):
                  x = up_stage(x, skip)
              return x

  skills:

    build_unet_decoder:

      description: 构建U-Net解码器家族

      inputs:
        - dimension (1D/2D/3D)
        - base_channels
        - num_stages
        - bilinear

      prompt_template: |

        构建{{dimension}} U-Net解码器，实现多尺度特征重建。

        参数：
        base_channels = {{base_channels}}
        num_stages = {{num_stages}}
        bilinear = {{bilinear}}
        normtype = {{normtype}}

        要求：
        1. 支持跳跃连接融合
        2. 逐层上采样重建
        3. 保持特征通道数对称
        4. 适配编码器输出格式

    optimize_reconstruction:

      description: 优化特征重建的质量和效率

      checks:
        - channel_consistency (通道数匹配)
        - resolution_recovery (分辨率恢复质量)
        - gradient_flow (梯度在跳跃连接中的流动)

  knowledge:

    usage_patterns:

      image_segmentation_2d:

        pipeline:
          - Encoder Features: 多尺度图像特征
          - Progressive Upsampling: 逐步恢复分辨率
          - Skip Fusion: 融合浅层细节
          - Final Segmentation: 像素级预测

      time_series_forecasting_1d:

        pipeline:
          - Temporal Encoding: 时序特征编码
          - Resolution Recovery: 时间分辨率恢复
          - Detail Fusion: 融合短期模式
          - Sequence Prediction: 时序预测

      volume_reconstruction_3d:

        pipeline:
          - Spatial Encoding: 3D空间特征
          - Volume Upsampling: 体积数据上采样
          - Cross-level Fusion: 跨层级融合
          - 3D Reconstruction: 体积重建

    hot_models:

      - model: U-Net
        year: 2015
        role: 医学图像分割的经典架构
        architecture: encoder-decoder with skip connections

      - model: U-Net++
        year: 2018
        role: 改进的U-Net架构
        architecture: nested skip connections

      - model: 3D U-Net
        year: 2016
        role: 3D医学图像分割
        architecture: 3D extension of U-Net

    best_practices:

      - 确保编码器和解码器的通道数对称
      - 根据数据维度选择合适的上采样方法
      - 跳跃连接对保持细节信息至关重要
      - 归一化类型应与编码器保持一致

    anti_patterns:

      - 编码器和解码器维度不匹配
      - 忽略跳跃连接导致细节丢失
      - 上采样方法选择不当产生伪影
      - 特征通道数计算错误

    paper_references:

      - title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        authors: Ronneberger et al.
        year: 2015

      - title: "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        authors: Çiçek et al.
        year: 2016

      - title: "U-Net++: A Nested U-Net Architecture for Medical Image Segmentation"
        authors: Zhou et al.
        year: 2018

  graph:

    is_a:
      - NeuralNetworkDecoder
      - ReconstructionModule
      - MultiScaleProcessor

    part_of:
      - UNetArchitecture
      - SegmentationModels
      - AutoencoderSystems

    depends_on:
      - Up1D/Up2D/Up3D
      - SkipConnection
      - ConvolutionalBlocks
      - UpsamplingOperations

    variants:
      - UNetDecoder1D (时序数据)
      - UNetDecoder2D (图像数据)
      - UNetDecoder3D (体积数据)

    used_in_models:
      - U-Net
      - U-Net++
      - 3D U-Net
      - 其他分割重建模型

    compatible_with:

      inputs:
        - MultiScaleFeatures
        - EncoderOutputs

      outputs:
        - ReconstructedFeatures
        - SegmentationMaps
        - PredictedSignals
