component:

  meta:
    name: nowcastnet
    alias: NowcastNetUNet
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: evolution_model
    author: OneScience
    license: Apache-2.0
    tags:
      - nowcasting
      - unet
      - evolution
      - optical_flow
      - precipitation


  concept:

    description: >
      nowcastnet 实现了一个双头 U-Net 风格演化网络。
      主干编码-解码路径提取多尺度特征后输出强度场 x，并通过可学习参数 gamma 进行逐通道缩放；
      同时并行分支输出速度/位移相关场 v（通道数为 n_classes * 2），用于后续时空演化建模。

    intuition: >
      该网络将“场值预测”和“运动估计”拆分为共享编码器、双解码器的结构，
      既复用低层空间特征，也允许两个任务在解码阶段学习不同的重建目标。

    problem_it_solves:
      - 从输入气象场提取多尺度空间表征
      - 同时预测强度场与运动场两类输出
      - 通过 U-Net 跳连提升细节保真度
      - 用 gamma 参数控制主输出幅度


  theory:

    formula:

      encoder_decoder:
        expression: |
          x_1 = Enc_1(X),\dots,x_5 = Enc_5(x_4)
          y = Dec(x_5, x_4, x_3, x_2, x_1)

      dual_head_outputs:
        expression: |
          x = OutConv(y) \odot \gamma
          v = OutConv_v(Dec_v(x_5, x_4, x_3, x_2, x_1))

    variables:

      X:
        name: InputField
        shape: [batch, n_channels, H, W]
        description: 输入场数据

      x:
        name: IntensityField
        shape: [batch, n_classes, H, W]
        description: 主输出强度场

      v:
        name: VelocityField
        shape: [batch, n_classes*2, H, W]
        description: 运动相关输出

      \gamma:
        name: ChannelScale
        shape: [1, n_classes, 1, 1]
        description: 可学习通道缩放参数


  structure:

    architecture: dual_head_unet

    pipeline:

      - name: Encoder
        operation: inc_down1_down2_down3_down4

      - name: DecoderIntensity
        operation: up1_up2_up3_up4_outc

      - name: IntensityScaling
        operation: multiply_gamma

      - name: DecoderVelocity
        operation: up1_v_up2_v_up3_v_up4_v_outc_v

      - name: ReturnTuple
        operation: return_intensity_and_velocity


  interface:

    parameters:

      n_channels:
        type: int
        description: 输入通道数

      n_classes:
        type: int
        description: 输出主场通道数

      base_c:
        type: int
        default: 64
        description: 基础通道宽度

      bilinear:
        type: bool
        default: true
        description: 上采样是否使用双线性插值

    inputs:

      x:
        type: FieldEmbedding
        shape: [batch, n_channels, H, W]
        dtype: float32
        description: 输入气象场

    outputs:

      intensity:
        type: FieldEmbedding
        shape: [batch, n_classes, H, W]
        description: 强度场输出

      velocity:
        type: FieldEmbedding
        shape: [batch, n_classes*2, H, W]
        description: 速度/位移场输出


  types:

    FieldEmbedding:
      shape: [batch, channels, H, W]
      description: 二维场数据 embedding


  implementation:

    framework: pytorch

    code: |

      import torch.nn.functional as F
      from .module import *

      class nowcastnet(nn.Module):
          def __init__(self, n_channels, n_classes, base_c=64, bilinear=True):
              super(nowcastnet, self).__init__()
              self.inc = DoubleConv(n_channels, base_c)
              self.down1 = Down(base_c * 1, base_c * 2)
              self.down2 = Down(base_c * 2, base_c * 4)
              self.down3 = Down(base_c * 4, base_c * 8)
              factor = 2 if bilinear else 1
              self.down4 = Down(base_c * 8, base_c * 16 // factor)

              self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
              self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
              self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
              self.up4 = Up(base_c * 2, base_c * 1, bilinear)
              self.outc = OutConv(base_c * 1, n_classes)
              self.gamma = nn.Parameter(torch.zeros(1, n_classes, 1, 1), requires_grad=True)

              self.up1_v = Up(base_c * 16, base_c * 8 // factor, bilinear)
              self.up2_v = Up(base_c * 8, base_c * 4 // factor, bilinear)
              self.up3_v = Up(base_c * 4, base_c * 2 // factor, bilinear)
              self.up4_v = Up(base_c * 2, base_c * 1, bilinear)
              self.outc_v = OutConv(base_c * 1, n_classes * 2)

          def forward(self, x):
              x1 = self.inc(x)
              x2 = self.down1(x1)
              x3 = self.down2(x2)
              x4 = self.down3(x3)
              x5 = self.down4(x4)

              x = self.up1(x5, x4)
              x = self.up2(x, x3)
              x = self.up3(x, x2)
              x = self.up4(x, x1)
              x = self.outc(x) * self.gamma

              v = self.up1_v(x5, x4)
              v = self.up2_v(v, x3)
              v = self.up3_v(v, x2)
              v = self.up4_v(v, x1)
              v = self.outc_v(v)
              return x, v


  skills:

    build_nowcastnet:

      description: 构建双头 U-Net 的演化预测模块

      inputs:
        - n_channels
        - n_classes
        - base_c
        - bilinear

      prompt_template: |

        构建 nowcastnet 模块。

        参数：
        n_channels = {{n_channels}}
        n_classes = {{n_classes}}
        base_c = {{base_c}}
        bilinear = {{bilinear}}

        要求：
        同时输出强度场 x 与速度场 v，其中 x 需乘以可学习参数 gamma。


    diagnose_nowcastnet:

      description: 分析双头 U-Net 演化模型的训练和数值问题

      checks:
        - channel_mismatch_between_heads
        - unstable_gamma_scaling
        - upsample_artifacts
        - skip_connection_shape_mismatch


  knowledge:

    usage_patterns:

      precipitation_nowcasting:

        pipeline:
          - EncodeRadarField
          - DecodeIntensity
          - DecodeMotion

      dual_task_learning:

        pipeline:
          - SharedEncoder
          - IntensityHead
          - VelocityHead


    hot_models:

      - model: NowcastNet
        year: 2022
        role: 降水临近预报中的深度学习框架代表
        architecture: dual_head_unet_variant


    best_practices:

      - 对 intensity 与 velocity 分支分别设置损失权重。
      - 监控 gamma 的数值范围，避免主输出过度缩放。
      - 在高分辨率输入下使用梯度检查点降低显存压力。


    anti_patterns:

      - 忽略双头任务平衡导致某一分支退化。
      - 仅优化像素损失而缺少运动一致性约束。


    paper_references:

      - title: "NowcastNet: Skilful Nowcasting of Extreme Precipitation with NowcastNet"
        authors: Zhang et al.
        year: 2022


  graph:

    is_a:
      - NeuralNetworkComponent
      - EvolutionModel

    part_of:
      - NowcastingSystems
      - WeatherForecastingPipelines

    depends_on:
      - U-NetBlocks
      - SkipConnections
      - DualHeadDecoders

    variants:
      - nowcastnet
      - oneevolution_wrapper

    used_in_models:
      - NowcastNet
      - Radar Nowcasting Models
      - Spatiotemporal Evolution Pipelines

    compatible_with:

      inputs:
        - FieldEmbedding

      outputs:
        - FieldEmbedding