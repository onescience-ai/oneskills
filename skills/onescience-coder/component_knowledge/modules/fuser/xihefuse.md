component:

  meta:
    name: XiheFuser
    alias: Xihe Ocean Model Feature Fuser
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: ocean_fusion
    author: OneScience
    license: Apache-2.0
    tags:
      - ocean_modeling
      - feature_fusion
      - local_global_attention
      - sea_ice
      - climate_model

  concept:

    description: >
      Xihe模型的海洋特定特征融合模块，包含局部和全局特征提取器。
      通过多个局部SIE模块处理局部海洋特征，再通过全局SIE模块实现全局信息交互，
      特别针对海洋-海冰耦合系统设计，支持海陆掩码处理。

    intuition: >
      就像海洋学家分析海洋系统：先关注局部海域的特征（如局部涡流、温度梯度），
      然后考虑全球洋流的相互影响，最后形成完整的海洋系统认知。
      海陆掩码就像区分海洋和陆地，只在海洋区域进行分析。

    problem_it_solves:
      - 海洋-海冰耦合系统的特征融合
      - 局部和全局海洋特征的统一处理
      - 海陆掩码下的有效计算
      - 多尺度海洋现象的建模

  theory:

    formula:

      local_global_fusion:
        expression: |
          x_{local} = \text{LocalSIEBlocks}(x_{input}, mask)
          x_{global} = \text{GlobalSIEBlocks}(x_{local}, mask)
          x_{output} = x_{global}

      sea_ice_masking:
        expression: |
          x_{masked} = x \odot mask_{sea}
          \text{where } mask_{sea} \in \{0, 1\}^{B \times L}

    variables:

      mask:
        name: SeaLandMask
        shape: [batch, L]
        description: 海陆掩码，1=海洋，0=陆地

      x_{local}:
        name: LocalFeatures
        description: 局部SIE提取的特征

      x_{global}:
        name: GlobalFeatures
        description: 全局SIE融合的特征

  structure:

    architecture: hierarchical_ocean_fusion

    pipeline:

      - name: InputProcessing
        operation: apply_sea_land_mask

      - name: LocalSIEBlocks
        operation: multiple_local_sie_modules

      - name: GlobalSIEBlocks
        operation: global_sie_modules

      - name: OutputFeatures
        operation: fused_ocean_features

  interface:

    parameters:

      dim:
        type: int
        description: 输入通道数

      input_resolution:
        type: tuple[int, int, int]
        description: 输入空间分辨率 (Pl, Lat, Lon)

      num_local:
        type: int
        description: 局部SIE模块的数量

      num_global:
        type: int
        description: 全局SIE模块的数量

      num_heads_local:
        type: int
        description: 局部SIE的注意力头数量

      num_heads_global:
        type: int
        description: 全局SIE的注意力头数量

      window_size:
        type: tuple[int, int, int]
        description: 局部SIE的3D窗口大小

      num_groups:
        type: int
        description: 全局SIE的group数量

    inputs:

      obj:
        type: OceanData
        description: 包含x和mask的对象

      x:
        type: OceanFeatures
        shape: [batch, L, C]
        description: 输入海洋特征

      mask:
        type: SeaLandMask
        shape: [batch, L]
        description: 海陆掩码

    outputs:

      x:
        type: FusedOceanFeatures
        shape: [batch, L, C]
        description: 融合后的海洋特征

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      from .xihelocalsiefuser import XiheLocalSIEFuser
      from .xiheglobalsiefuser import XiheGlobalSIEFuser

      class XiheFuser(nn.Module):
          def __init__(self, dim=192, input_resolution=(13, 128, 256),
                       num_local=2, num_global=1, num_heads_local=6,
                       num_heads_global=12, window_size=(1, 6, 12),
                       num_groups=32, **kwargs):
              super().__init__()
              
              # 局部SIE模块
              self.local_sie_modules = nn.ModuleList([
                  XiheLocalSIEFuser(dim=dim, input_resolution=input_resolution,
                                   num_heads=num_heads_local, window_size=window_size)
                  for _ in range(num_local)
              ])
              
              # 全局SIE模块
              self.global_sie_modules = nn.ModuleList([
                  XiheGlobalSIEFuser(dim=dim, num_heads=num_heads_global,
                                     num_groups=num_groups)
                  for _ in range(num_global)
              ])

          def forward(self, obj):
              x, mask = obj.x, getattr(obj, 'mask', None)
              
              # 应用海陆掩码
              if mask is not None:
                  x = x * mask.unsqueeze(-1)
              
              # 局部SIE处理
              for local_sie in self.local_sie_modules:
                  local_obj = type('Obj', (), {'x': x, 'mask': mask})()
                  x = local_sie(local_obj)
              
              # 全局SIE处理
              for global_sie in self.global_sie_modules:
                  global_obj = type('Obj', (), {'x': x, 'mask': mask})()
                  x = global_sie(global_obj)
              
              return x

  skills:

    build_ocean_fuser:

      description: 构建海洋模型特征融合器

  knowledge:

    hot_models:

      - model: Xihe
        year: 2023
        role: 海洋-海冰耦合模型
        architecture: local-global attention

    best_practices:

      - 合理设置局部和全局模块的数量
      - 窗口大小应考虑海洋现象的空间尺度
      - 海陆掩码对计算效率很重要

    paper_references:

      - title: "Xihe: A large-scale ocean-sea ice model for climate prediction"
        authors: Climate Modeling Team
        year: 2023

  graph:

    is_a:
      - OceanModelingModule
      - FeatureFusionModule
      - HierarchicalProcessor

    used_in_models:
      - Xihe
      - Ocean-ice coupled models

    compatible_with:

      inputs:
        - OceanFeatures
        - SeaIceData
        - ClimateVariables

      outputs:
        - FusedOceanFeatures
        - ClimatePredictions
