component:

  meta:
    name: FourCastNetFuser
    alias: FourCastNet AFNO Transformer Block
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: feature_fusion
    author: OneScience
    license: Apache-2.0
    tags:
      - weather_forecasting
      - afno
      - transformer
      - frequency_domain
      - double_skip

  concept:

    description: >
      FourCastNet的核心Transformer Block，以AFNO替代自注意力机制的Transformer Block。
      结构为"AFNO频域混合 → MLP通道混合"，并支持双残差连接（double skip）。
      AFNO在频域进行操作，比传统空间注意力更高效，特别适合大规模气象数据。

    intuition: >
      就像音频处理中，有时在频域分析比在时域更高效。AFNO将图像特征转换到频域进行处理，
      就像用傅里叶变换分析天气模式的周期性，然后在频域进行混合，最后转换回空间域。

    problem_it_solves:
      - 大规模气象数据的高效处理
      - 频域特征混合替代空间注意力
      - 深层网络的梯度传播优化
      - 计算效率与模型效果的平衡

  theory:

    formula:

      afno_processing:
        expression: |
          x_{freq} = \text{FFT2D}(x_{norm})
          x_{mixed} = \text{FrequencyMixing}(x_{freq})
          x_{spatial} = \text{IFFT2D}(x_{mixed})

      double_residual:
        expression: |
          x_1 = x + \text{AFNO}(\text{Norm}(x)) \quad \text{(first residual)}
          x_2 = x_1 + \text{MLP}(\text{Norm}(x_1)) \quad \text{(second residual)}

  structure:

    architecture: afno_transformer_block

    pipeline:

      - name: FirstNorm
        operation: layer_norm

      - name: AFNOProcessing
        operation: fourcastnet_afno_2d

      - name: FirstResidual
        operation: addition (if double_skip)

      - name: SecondNorm
        operation: layer_norm

      - name: MLPProcessing
        operation: fourcastnet_fc

      - name: SecondResidual
        operation: addition

  interface:

    parameters:

      dim:
        type: int
        description: 输入token的通道数（嵌入维度）

      mlp_ratio:
        type: float
        description: MLP隐层相对于dim的扩展倍数

      drop:
        type: float
        description: MLP的Dropout比例

      drop_path:
        type: float
        description: Stochastic Depth的比例

      double_skip:
        type: bool
        description: 是否启用双残差连接

      num_blocks:
        type: int
        description: 传递给AFNO的通道分块数

      sparsity_threshold:
        type: float
        description: 传递给AFNO的软阈值

      hard_thresholding_fraction:
        type: float
        description: 传递给AFNO的频率保留比例

    inputs:

      x:
        type: SpatialFeatures
        shape: [batch, H, W, C]
        description: 输入特征图

    outputs:

      x:
        type: FusedFeatures
        shape: [batch, H, W, C]
        description: 输出特征图

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from onescience.modules.fc.onefc import OneFC
      from onescience.modules.afno.oneafno import OneAFNO

      class FourCastNetFuser(nn.Module):
          def __init__(self, dim=768, mlp_ratio=4., double_skip=True,
                       num_blocks=8, sparsity_threshold=0.01, **kwargs):
              super().__init__()
              self.norm1 = nn.LayerNorm(dim)
              self.filter = OneAFNO(style="FourCastNetAFNO2D")
              self.norm2 = nn.LayerNorm(dim)
              mlp_hidden_dim = int(dim * mlp_ratio)
              self.mlp = OneFC(style="FourCastNetFC")
              self.double_skip = double_skip

          def forward(self, x):
              residual = x
              x = self.norm1(x)
              x = self.filter(x)

              if self.double_skip:
                  x = x + residual
                  residual = x

              x = self.norm2(x)
              x = self.mlp(x)
              x = x + residual
              return x

  skills:

    build_afno_fuser:

      description: 构建基于AFNO的特征融合器

  knowledge:

    hot_models:

      - model: FourCastNet
        year: 2023
        role: NVIDIA的气象预报模型
        architecture: AFNO-based transformer

    best_practices:

      - 双残差连接有助于深层网络训练
      - AFNO特别适合大规模空间数据处理

    paper_references:

      - title: "FourCastNet: Forecasting at Scale with High-Resolution Neural Networks"
        authors: Pathak et al.
        year: 2022

  graph:

    is_a:
      - FeatureFusionModule
      - FrequencyDomainProcessor
      - TransformerBlock

    used_in_models:
      - FourCastNet

    compatible_with:

      inputs:
        - SpatialFeatures
        - WeatherFields

      outputs:
        - FusedFeatures
        - FrequencyMixedFeatures
