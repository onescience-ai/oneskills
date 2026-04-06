component:

  meta:
    name: MaskedMSAHead
    alias: MSAHead
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: prediction_head
    author: OneScience
    license: Apache-2.0
    tags:
      - alphafold
      - masked_msa
      - language_model_head
      - self_supervised_learning
      - co_evolution

  concept:

    description: >
      MaskedMSAHead 是蛋白质结构预测模型（如 AlphaFold 2 / OpenFold）中用于计算遮蔽多序列比对损失（Masked MSA Loss）的预测头组件。
      它接收提取后的 MSA 嵌入表示，通过一个线性投影层，将特征映射到输出类别空间（通常为氨基酸类型的概率分布），
      从而预测并重构输入阶段被随机遮蔽（Masked）的氨基酸残基。

    intuition: >
      该模块的原理与自然语言处理（NLP）中 BERT 的掩码语言模型（Masked Language Modeling, MLM）非常相似。
      把 MSA（多序列比对）看作是一种“蛋白质语言”：模型在输入端故意遮住了进化序列中的某些氨基酸词汇，
      强迫网络在内部处理时去寻找上下文中“共进化（Co-evolution）”的线索。当特征流到网络末端时，
      这个 Head 就像一个答题卡，负责给出这些被遮住空位最可能的氨基酸类别填空答案。

    problem_it_solves:
      - 为蛋白质结构预测网络提供强大的自监督学习（Self-Supervised Learning）信号
      - 强迫网络充分学习和提取多序列比对（MSA）中的残基共进化规律
      - 将高维的隐藏层 MSA 嵌入转换为实际物理/化学意义上的氨基酸类别对数几率（Logits）

  theory:

    formula:

      masked_msa_prediction:
        expression: \text{logits}_{*, s, r, c} = \text{Linear}(m_{*, s, r, m_{in}})

    variables:

      m:
        name: MSAEmbedding
        shape: [*, N_seq, N_res, C_m]
        description: 网络深层输出的多序列比对特征张量

      \text{logits}:
        name: PredictionLogits
        shape: [*, N_seq, N_res, C_out]
        description: 预测的每种氨基酸类别在每个位置的对数几率

      N_{seq}:
        name: NumberOfSequences
        description: 多序列比对中的序列数量

      N_{res}:
        name: NumberOfResidues
        description: 蛋白质序列的残基长度

  structure:

    architecture: linear_prediction_head

    pipeline:

      - name: TokenClassification
        operation: linear_projection_with_final_init

  interface:

    parameters:

      c_m:
        type: int
        description: 输入 MSA 特征的通道维度大小 (MSA channel dimension)

      c_out:
        type: int
        description: 输出通道维度大小，通常对应氨基酸种类的数量（如 AlphaFold 中的 23 类）

    inputs:

      m:
        type: Tensor
        shape: "[*, N_seq, N_res, C_m]"
        dtype: float32
        description: MSA 特征表示（星号 * 代表可能存在的 Batch 等额外维度）

    outputs:

      output:
        type: Tensor
        shape: "[*, N_seq, N_res, C_out]"
        description: 预测的输出类别 Logits，可直接用于与真实 Label 计算交叉熵损失（CrossEntropy Loss）

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      import torch 
      from torch import nn

      class MaskedMSAHead(nn.Module):
          """
          For use in computation of masked MSA loss, subsection 1.9.9
          """

          def __init__(self, c_m, c_out, **kwargs):
              """
              Args:
                  c_m:
                      MSA channel dimension
                  c_out:
                      Output channel dimension
              """
              super(MaskedMSAHead, self).__init__()

              self.c_m = c_m
              self.c_out = c_out

              # 注意: 此处的 Linear 为定制化实现，非标准的 nn.Linear
              self.linear = Linear(self.c_m, self.c_out, init="final")

          def forward(self, m):
              """
              Args:
                  m:
                      [*, N_seq, N_res, C_m] MSA embedding
              Returns:
                  [*, N_seq, N_res, C_out] reconstruction
              """
              # [*, N_seq, N_res, C_out]
              logits = self.linear(m)
              return logits

  skills:

    build_masked_msa_head:

      description: 为蛋白质大模型构建用于自监督学习的预测头

      inputs:
        - c_m
        - c_out (通常为 23)

      prompt_template: |
        构建一个 MaskedMSAHead。
        输入通道维度为 {{c_m}}，输出氨基酸类别数为 {{c_out}}。

    diagnose_masked_msa:

      description: 分析 Masked MSA 训练初期损失爆炸或收敛异常的问题

      checks:
        - verify_final_initialization_zeroes_weights
        - dimension_mismatch_in_unmasked_loss_computation

  knowledge:

    usage_patterns:

      masked_msa_loss_computation:

        pipeline:
          - Extract MSA features
          - MaskedMSAHead (获取 Logits)
          - 结合 mask_positions 选取被掩码的元素
          - F.cross_entropy (计算预测结果与原始 Amino Acid 序列之间的损失)

    design_patterns:

      final_initialization:

        structure:
          - 代码中的 `Linear(..., init="final")` 是一种常见于 AlphaFold 的设计模式。
          - 所谓 "final" 初始化，通常意味着将该线性层的权重和偏置初始化为全零（Zero Initialization）。
          - 这保证了在训练的最初阶段，网络对于任何类别的预测都是均匀分布的（Logits 均为 0），避免了随机初始化带来的初始巨大 Loss 震荡。

    hot_models:

      - model: AlphaFold 2 (AF2)
        year: 2021
        role: 蛋白质折叠预测的里程碑模型
        architecture: evoformer_plus_structure_module
        attention_type: None (此处是附加任务的头部)

      - model: OpenFold
        year: 2022
        role: AF2 的开源高性能复现版本
        architecture: af2_architecture_pytorch

    model_usage_details:

      AlphaFold 2 Supplementary Information:

        usage: 代码注释中的 "subsection 1.9.9" 直接指向 AlphaFold 2 补充材料（Supplementary Methods）中描述 Masked MSA 辅助损失的特定章节。这部分损失对提升最终的三维结构预测精度具有重要的正则化和特征提纯作用。
        c_out: 通常设定为 23，代表 20 种标准氨基酸 + 1 种未知氨基酸 (X) + 1 种 Gap 符号 (-) + 1 种 Mask 符号。

    best_practices:

      - 由于该模块只包含一个 `Linear` 层，计算量非常轻量。在计算 Loss 时，为了进一步节省计算量，可以先在 `N_seq` 和 `N_res` 维度上筛选出实际被 Mask 的那些 Token，然后再仅对这些局部的 Token 应用 `self.linear` 进行投影，而不是全量映射。
      - 确保依赖库中具有支持 `init="final"` 关键字的定制化 `Linear` 算子实现。如果迁移到纯原生 PyTorch，需要在 `__init__` 后手动使用 `nn.init.zeros_(self.linear.weight)`。

    anti_patterns:

      - 使用标准的 `nn.Linear` 默认随机初始化（Kaiming/Xavier）来替代 `init="final"`。在这个分类头中，随机初始化会引起训练初期分类 Loss 的剧烈波动，影响主干 Evoformer 提取结构特征的稳定性。

    paper_references:

      - title: "Highly accurate protein structure prediction with AlphaFold"
        authors: Jumper et al.
        year: 2021
        note: (Specifically Supplementary Methods Subsection 1.9.9 "Masked MSA loss")

  graph:

    is_a:
      - NeuralNetworkComponent
      - PredictionHead
      - TokenClassifier

    part_of:
      - AlphaFold2
      - OpenFold
      - ProteinLanguageModel

    depends_on:
      - CustomLinearOperator

    variants:
      - MLMHead (自然语言处理中的掩码语言模型预测头)

    used_in_models:
      - AlphaFold 2
      - OpenFold
      - ESMFold

    compatible_with:

      inputs:
        - Tensor (MSA Representations)

      outputs:
        - Tensor (Logits)