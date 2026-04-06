component:

  meta:
    name: NeuralSpectralBlock
    alias: LatentSpectralTransformer
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: spectral_neural_operator
    author: OneScience
    license: Apache-2.0
    tags:
      - neural_spectral
      - latent_attention
      - perceiver
      - patchify
      - 3d_spatiotemporal


  concept:

    description: >
      神经谱 Transformer 块 (Neural Spectral Block) 包含 1D, 2D, 3D 三种实现。
      它首先通过 Patchify 将连续的物理网格进行分块，然后使用编码器交叉注意力（Encoder Attention）
      将海量的物理特征投影到极少数的潜在 Token（Latent Tokens）上。
      在紧凑的隐空间内部，利用正余弦基函数进行高频/低频谱域特征的交互，
      最后再通过解码器注意力将演化后的特征还原回高分辨率的物理网格中。

    intuition: >
      想象要在一个有着数万人的大型体育场（高分辨率物理场）里交换信息。
      标准的 Transformer 是让每个人都和其余所有人说话，效率极低且会耗尽显存。
      这个模块的做法是：先把人按区域分组（Patchify），每组选出几个代表（Latent Tokens）。
      这些代表进入一个专用的会议室，利用特定的频率广播频段（谱域基函数）快速交换全局信息。
      会议结束后，代表们回到各自区域，把总结好的全局信息传达给每个人（Decoder Attention）。

    problem_it_solves:
      - 高维时空数据（特别是 3D 数据如视频、气象体素）中注意力计算显存爆炸的问题
      - 在保持严格的 O(N) 计算复杂度的同时，实现真正意义上的全局空间感受野
      - 克服标准卷积或局部 Transformer 感受野受限的物理信息传递壁垒


  theory:

    formula:

      latent_encoding:
        expression: $$Z_{latent} = Z_{latent} + \text{Softmax}(Z_{latent} X_{patch}^T) X_{patch}$$

      spectral_transition:
        expression: |
          $$B_{modes} = \text{Concat}(\sin(\text{modes} \cdot Z_{latent}), \cos(\text{modes} \cdot Z_{latent}))$$
          $$Z'_{latent} = Z_{latent} + B_{modes} \cdot W_{spectral}$$

      latent_decoding:
        expression: $$X'_{patch} = X_{patch} + \text{Softmax}(X_{patch} (Z'_{latent})^T) Z'_{latent}$$

    variables:

      X_{patch}:
        name: PatchifiedInput
        description: 物理空间经过分块（Patchify）后的网格特征

      Z_{latent}:
        name: LatentTokens
        description: 数量固定且极小的隐空间特征向量，作为信息流转的瓶颈

      B_{modes}:
        name: SpectralBasis
        description: 由预设的频率模态生成的正余弦基函数矩阵


  structure:

    architecture: perceiver_spectral_hybrid

    pipeline:

      - name: SpacePatchify
        operation: reshape_and_permute (将 1D/2D/3D 空间划分为 Patch)

      - name: LatentEncoderAttn
        operation: cross_attention (X -> Z)

      - name: SpectralTransition
        operation: basis_generation + complex_multiplication

      - name: LatentDecoderAttn
        operation: cross_attention (Z -> X)

      - name: SpaceDePatchify
        operation: reshape_and_permute (恢复原始物理分辨率)


  interface:

    parameters:

      width:
        type: int
        description: 输入和输出的特征通道数

      num_basis:
        type: int
        description: 隐空间谱处理中使用的基函数（模态）数量

      patch_size:
        type: list[int]
        default: [3, 3] 或 [8, 8, 4]
        description: 物理空间的分块尺寸，列表长度对应数据的维度 (1D/2D/3D)

      num_token:
        type: int
        default: 4
        description: 隐空间中潜在 Token 的数量，决定了计算与信息压缩瓶颈的大小

      n_heads:
        type: int
        default: 8
        description: 编解码交叉注意力机制的头数

    inputs:

      x:
        type: PhysicalGrid
        shape: "[batch, width, L] 或 [batch, width, H, W] 或 [batch, width, H, W, T]"
        dtype: float32
        description: 连续维度的物理场输入

    outputs:

      output:
        type: PhysicalGrid
        shape: 与输入形状完全一致
        description: 经过全局谱域信息交互后的物理场特征


  types:

    PhysicalGrid:
      description: 保持了空间或时空拓扑结构的连续网格数据


  implementation:

    framework: pytorch

    code: |

      # 请参考源文件 Neural_Spectral_Block.py 中 
      # NeuralSpectralBlock1D, NeuralSpectralBlock2D, NeuralSpectralBlock3D 
      # 三个类的完整实现逻辑。代码核心为：
      # 1. patchify: view().permute() 变换
      # 2. latent_encoder_attn: Conv1d/2d/3d 生成 K, V，与固定参数 latent 交互
      # 3. get_basis & compl_mul2d: 隐空间谱特征过滤
      # 4. latent_decoder_attn: 将 latent 映射回物理 Patch
      # 5. de-patchify: 逆变换恢复形状


  skills:

    build_spectral_block:

      description: 根据数据的维度构建对应的高效谱域注意力块

      inputs:
        - width
        - num_basis
        - patch_size
        - dimensionality (1D/2D/3D)

      prompt_template: |

        请实例化 NeuralSpectralBlock{dimensionality}D。
        确保传入的输入序列在各个空间维度上能够被对应的 patch_size 完美整除。
        如果需要处理更复杂的动力学，可以适当增大 num_token 以拓宽隐空间的信息带宽。


    diagnose_spectral_latent:

      description: 排查隐式 Transformer 架构中常见的信息丢失与形状对齐问题

      checks:
        - patch_size_indivisible (输入网格尺寸无法被 patch_size 整除，导致 view 崩溃)
        - latent_bottleneck_collapse (num_token 设置过小，导致高频物理细节在编码压缩时永久丢失)



  knowledge:

    usage_patterns:

      latent_spectral_surrogate:

        pipeline:
          - Data Normalization
          - [NeuralSpectralBlock_nD] x Depth
          - Pointwise Predictor / Linear Head


    hot_models:

      - model: Perceiver / Perceiver IO
        year: 2021
        role: 奠定通过少量 Latent Token 处理超大规模输入序列的架构基础
        architecture: Asymmetric Attention Bottleneck

      - model: Fourier Neural Operator (FNO)
        year: 2020
        role: 奠定在深度网络中利用谱域（频率）截断提取全局连续特征的方法


    best_practices:

      - `num_token` 的大小直接决定了模型的容量上限和计算速度。对于较为平滑的气象场（如气温），较小的 token 数即可；但对于存在剧烈突变的流体场（如风速涡流），应适当增大 `num_token`。
      - 必须保证进入此模块的输入张量维度，其空间边界严格是 `patch_size` 的整数倍。如果存在边角不整除的情况，需要在模块外部使用 ZeroPad 提前填充。


    anti_patterns:

      - 在极小分辨率的网格（如 16x16）上使用此模块。在小网格中，$O(N^2)$ 的自注意力毫无压力，引入 Latent 机制反而会导致信息冗余压缩，降低预测精度。


    paper_references:

      - title: "Perceiver: General Perception with Iterative Attention"
        authors: Jaegle et al.
        year: 2021

      - title: "Fourier Neural Operator for Parametric Partial Differential Equations"
        authors: Li et al.
        year: 2020



  graph:

    is_a:
      - TransformerBlock
      - NeuralOperator
      - LatentArchitecture

    part_of:
      - 3DSpatiotemporalModel
      - AIWeatherModel

    depends_on:
      - Convolution (1D/2D/3D)
      - Softmax
      - SineCosineBasis

    variants:
      - PerceiverIOBlock
      - FNOBlock

    used_in_models:
      - Custom Latent Spectral Solvers

    compatible_with:

      inputs:
        - PhysicalGrid

      outputs:
        - PhysicalGrid