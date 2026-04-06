component:

  meta:
    name: ProtenixRelativePositionEncoding
    alias: AlphaFold3 Relative Position Encoder
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: protein_encoder
    author: OneScience
    license: Apache-2.0
    tags:
      - protein_structure
      - alpha_fold
      - position_encoding
      - molecular_modeling

  concept:

    description: >
      ProtenixRelativePositionEncoding是AlphaFold3中的相对位置编码模块，实现Algorithm 3。
      编码token之间的相对位置信息，包括残基索引、token索引和链信息，为蛋白质结构预测提供空间关系建模。

    intuition: >
      就像生物学家理解蛋白质结构时，不仅看每个氨基酸的位置，更重要的是理解它们之间的相对关系。
      相同链上的相邻残基、不同链上的相互作用、同一实体的对称关系等，这些相对位置信息对预测蛋白质3D结构至关重要。

    problem_it_solves:
      - 蛋白质分子中相对位置关系的建模
      - 多链蛋白质复合物的空间关系编码
      - 残基级别和token级别的位置信息融合

  theory:

    formula:

      relative_position_computation:
        expression: |
          d_{residue} = \text{clip}(residue_i - residue_j + r_{max}, 0, 2r_{max} + 1) \cdot b_{same\_chain} + (1 - b_{same\_chain}) \cdot (2r_{max} + 1)
          d_{token} = \text{clip}(token_i - token_j + r_{max}, 0, 2r_{max} + 1) \cdot b_{same\_chain} \cdot b_{same\_residue} + (1 - b_{same\_chain} \cdot b_{same\_residue}) \cdot (2r_{max} + 1)

  structure:

    architecture: relative_position_encoder

    pipeline:

      - name: FeatureExtraction
        operation: extract_meta_features

      - name: MaskComputation
        operation: boolean_masks

      - name: RelativeDistance
        operation: clipped_differences

      - name: OneHotEncoding
        operation: one_hot_encoding

      - name: LinearProjection
        operation: linear_no_bias

  interface:

    parameters:

      r_max:
        type: int
        description: 最大相对位置索引裁剪值

      s_max:
        type: int
        description: 最大相对链索引裁剪值

      c_z:
        type: int
        description: 对表示嵌入维度

    inputs:

      input_feature_dict:
        type: FeatureDictionary
        description: 输入元特征字典

    outputs:

      relative_position_encoding:
        type: PositionEncoding
        shape: [..., N_token, N_token, c_z]

  implementation:

    framework: pytorch

    code: |

      import torch
      import torch.nn as nn
      import torch.nn.functional as F
      from onescience.modules.linear.protenixlinear import ProtenixLinearNoBias

      class ProtenixRelativePositionEncoding(nn.Module):
          def __init__(self, r_max=32, s_max=2, c_z=128):
              super().__init__()
              self.r_max = r_max
              self.s_max = s_max
              self.c_z = c_z
              
              self.linear_no_bias = ProtenixLinearNoBias(
                  in_features=(4 * self.r_max + 2 * self.s_max + 7), 
                  out_features=self.c_z
              )

          def forward(self, input_feature_dict):
              # 计算布尔掩码
              b_same_chain = (
                  input_feature_dict["asym_id"][..., :, None] == 
                  input_feature_dict["asym_id"][..., None, :]
              ).long()
              
              b_same_residue = (
                  input_feature_dict["residue_index"][..., :, None] == 
                  input_feature_dict["residue_index"][..., None, :]
              ).long()

              # 计算相对距离并one-hot编码
              d_residue = torch.clip(
                  input_feature_dict["residue_index"][..., :, None] - 
                  input_feature_dict["residue_index"][..., None, :] + self.r_max,
                  min=0, max=2 * self.r_max,
              ) * b_same_chain + (1 - b_same_chain) * (2 * self.r_max + 1)
              a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))

              # 拼接所有特征并投影
              features = torch.cat([
                  a_rel_pos, b_same_chain.unsqueeze(-1), b_same_residue.unsqueeze(-1)
              ], dim=-1)
              
              return self.linear_no_bias(features)

  skills:

    build_position_encoder:

      description: 构建蛋白质相对位置编码器

  knowledge:

    hot_models:

      - model: AlphaFold3
        year: 2024
        role: Google DeepMind的蛋白质结构预测模型
        architecture: diffusion + attention

    best_practices:

      - r_max应根据蛋白质长度适当调整
      - 位置编码对长距离相互作用建模很重要

    paper_references:

      - title: "AlphaFold 3: Accurate structure prediction of biomolecular interactions and complexes"
        authors: Abramson et al.
        year: 2024

  graph:

    is_a:
      - PositionEncoder
      - ProteinStructureComponent
      - FeatureExtractor

    used_in_models:
      - AlphaFold3

    compatible_with:

      inputs:
        - ProteinMetaFeatures
        - SequenceInformation

      outputs:
        - RelativePositionEncoding
        - PairRepresentation
