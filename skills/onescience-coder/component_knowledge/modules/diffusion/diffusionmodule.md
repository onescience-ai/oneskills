component:

  meta:
    name: AlphaFold3DiffusionModule
    alias: DiffusionModule
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: generative_model
    author: OneScience
    license: Apache-2.0
    tags:
      - alphafold3
      - diffusion_model
      - protein_structure_prediction
      - atom_attention
      - score_matching

  concept:

    description: >
      DiffusionModule 实现了 AlphaFold3 (AF3) 论文中的 Algorithm 20。
      它是整个蛋白质/分子结构预测架构中的核心生成组件。给定由前置模块（PairFormer）提取的单体特征（Single Representation）和对特征（Pair Representation），
      该模块通过扩散模型（Diffusion Model，基于 EDM 框架）迭代地去噪（Denoise）原子的三维坐标，最终生成高精度的分子三维结构。
      模块内部采用了一种层次化的结构：首先通过 `AtomAttentionEncoder` 将原子级特征聚合为粗粒度的 Token 级特征，
      接着由主干 `DiffusionTransformer` 进行全局自注意力交互，最后通过 `AtomAttentionDecoder` 广播回原子级别并预测坐标更新。

    intuition: >
      你可以将这个模块想象成一个技术精湛的雕刻家在完成作品的最后打磨。
      最初的三维坐标是充满随机噪声的（像一块未成形的石头），雕刻家（DiffusionModule）参考了设计图纸（由 s_trunk 和 z_trunk 提供的单体和对特征）和当前的噪声水平（t_hat_noise_level），
      先从微观的原子视角观察（Atom Encoder），然后退后一步进行宏观的全局思考（Diffusion Transformer），
      最后再聚焦到每一个具体的原子上（Atom Decoder），精准地凿去一点点噪声（r_update），一步步还原出蛋白质最真实的三维折叠状态。

    problem_it_solves:
      - 基于连续时间扩散模型（Continuous-time Diffusion Model, EDM）生成大分子和复合物的高精度三维坐标
      - 解决在全原子级别直接进行全局自注意力计算导致的 $O(N^2)$ 计算量和显存爆炸问题（通过 Token 化层次结构）
      - 将网络预测的“无量纲更新”（Dimensionless Update）通过严谨的 EDM 数学公式重新缩放回物理空间坐标

  theory:

    formula:

      edm_preconditioning_input:
        expression: c_{in} = \frac{1}{\sqrt{\sigma_{data}^2 + \sigma^2}} \quad \Rightarrow \quad r_{noisy} = c_{in} \cdot x_{noisy}

      edm_preconditioning_skip_and_out:
        expression: c_{skip} = \frac{\sigma_{data}^2}{\sigma_{data}^2 + \sigma^2}, \quad c_{out} = \frac{\sigma_{data} \cdot \sigma}{\sqrt{\sigma_{data}^2 + \sigma^2}}

      edm_denoised_output:
        expression: x_{denoised} = c_{skip} \cdot x_{noisy} + c_{out} \cdot F_{\theta}(r_{noisy}, \sigma)

    variables:

      \sigma:
        name: NoiseLevel
        description: 当前时间步的噪声水平（对应代码中的 t_hat_noise_level）

      \sigma_{data}:
        name: DataStandardDeviation
        description: 训练数据坐标的经验标准差（默认为 16.0 埃）

      x_{noisy}:
        name: NoisyCoordinates
        shape: [..., N_sample, N_atom, 3]
        description: 添加了噪声的原子坐标

      F_{\theta}:
        name: NeuralNetworkUpdate
        description: 神经网络（f_forward）在无量纲空间预测的更新量（r_update）

      x_{denoised}:
        name: DenoisedCoordinates
        shape: [..., N_sample, N_atom, 3]
        description: 去噪一步后的目标原子坐标

  structure:

    architecture: hierarchical_diffusion_transformer

    pipeline:

      - name: NoisePreconditioning
        operation: scale_input_coordinates

      - name: DiffusionConditioning
        operation: inject_noise_level_and_features

      - name: AtomLevelEncoding
        operation: atom_attention_encoder (Atom -> Token)

      - name: TokenLevelProcessing
        operation: diffusion_transformer (Global Self-Attention)

      - name: AtomLevelDecoding
        operation: atom_attention_decoder (Token -> Atom)

      - name: CoordinateRescaling
        operation: apply_edm_skip_connection

  interface:

    parameters:

      sigma_data:
        type: float
        default: 16.0
        description: 训练集真实坐标的标准差，用于控制 EDM 预处理的缩放比例

      c_token:
        type: int
        default: 768
        description: Token 级别的特征通道数

      atom_encoder / transformer / atom_decoder:
        type: dict
        description: 控制内部三大核心模块的层数（n_blocks）和注意力头数（n_heads）的配置字典

      blocks_per_ckpt:
        type: int
        default: null
        description: 激活重计算（Gradient Checkpointing）的块大小，用于在训练时用计算时间换取显存空间

    inputs:

      x_noisy:
        type: Tensor
        shape: "[..., N_sample, N_atom, 3]"
        dtype: float32
        description: 当前扩散步的带噪原子三维坐标

      t_hat_noise_level:
        type: Tensor
        shape: "[..., N_sample]"
        dtype: float32
        description: 当前批次样本对应的噪声水平/时间步

      s_trunk / z_trunk:
        type: Tensor
        description: 由 PairFormer 提取出的静态结构特征（Single / Pair）

    outputs:

      x_denoised:
        type: Tensor
        shape: "[..., N_sample, N_atom, 3]"
        description: 执行一步去噪公式后的原子三维坐标预测值

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      import torch
      from torch import nn
      from typing import Optional, Union

      class DiffusionModule(nn.Module):
          """Implements Algorithm 20 in AF3"""
          def __init__(self, sigma_data=16.0, c_atom=128, c_atompair=16, c_token=768, 
                       c_s=384, c_z=128, c_s_inputs=449, 
                       atom_encoder={"n_blocks": 3, "n_heads": 4},
                       transformer={"n_blocks": 24, "n_heads": 16, "drop_path_rate": 0},
                       atom_decoder={"n_blocks": 3, "n_heads": 4},
                       blocks_per_ckpt=None, use_fine_grained_checkpoint=False):
              super().__init__()
              self.sigma_data = sigma_data
              
              # 初始化内部子模块（伪代码表示）
              # self.diffusion_conditioning = DiffusionConditioning(...)
              # self.atom_attention_encoder = AtomAttentionEncoder(...)
              # self.layernorm_s = ProtenixLayerNorm(...)
              # self.linear_no_bias_s = LinearNoBias(...)
              # self.diffusion_transformer = DiffusionTransformer(...)
              # self.layernorm_a = ProtenixLayerNorm(...)
              # self.atom_attention_decoder = AtomAttentionDecoder(...)

          def f_forward(self, r_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk, ...):
              # 1. 注入噪声级别条件
              s_single, z_pair = self.diffusion_conditioning(...)
              
              # 2. 原子到 Token 的特征聚合
              a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(..., r_l=r_noisy, ...)
              
              # 3. 加上残差条件特征
              a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s_single))
              
              # 4. 全局 Token Transformer
              a_token = self.diffusion_transformer(a=a_token, s=s_single, z=z_pair, ...)
              a_token = self.layernorm_a(a_token)
              
              # 5. Token 到原子的特征广播与坐标更新预测
              r_update = self.atom_attention_decoder(..., a=a_token, q_skip=q_skip, c_skip=c_skip, p_skip=p_skip, ...)
              
              return r_update

          def forward(self, x_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk, ...):
              # EDM 缩放: c_in * x_noisy
              r_noisy = x_noisy / torch.sqrt(self.sigma_data**2 + t_hat_noise_level**2)[..., None, None]
              
              # 神经网络预测更新
              r_update = self.f_forward(r_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk, ...)
              
              # EDM 输出重构公式
              s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(r_update.dtype)
              x_denoised = (
                  1 / (1 + s_ratio**2) * x_noisy
                  + t_hat_noise_level[..., None, None] / torch.sqrt(1 + s_ratio**2) * r_update
              ).to(r_update.dtype)

              return x_denoised

  skills:

    build_af3_diffusion:

      description: 构建遵循 AlphaFold3 Algorithm 20 的层次化扩散生成模块

      inputs:
        - sigma_data
        - transformer_config
        - blocks_per_ckpt

      prompt_template: |
        构建一个 AF3 DiffusionModule。
        使用数据标准差 {{sigma_data}}。
        如果处于显存受限的微调阶段，请确保开启 `blocks_per_ckpt`={{blocks_per_ckpt}} 以利用 Gradient Checkpointing。

    diagnose_diffusion_nan:

      description: 分析扩散生成过程中由于噪声水平极值或坐标缩放导致的 NaN 异常

      checks:
        - division_by_zero_in_edm_scaling_due_to_zero_noise_level
        - unstable_gradients_without_gradient_checkpointing_in_deep_transformer

  knowledge:

    usage_patterns:

      structure_generation_loop:

        pipeline:
          - Initialize x_noisy with standard Gaussian noise multiplied by max_sigma
          - Loop for T steps:
            - t = current_noise_level
            - x_denoised = DiffusionModule(x_noisy, t, context_features)
            - Update x_noisy based on ODE/SDE solver (e.g., Euler, Heun) using x_denoised
          - Return final x_denoised as structure

    design_patterns:

      elucidating_diffusion_models_framework:

        structure:
          - 采用 Karras 等人提出的 EDM (Elucidating the Design Space of Diffusion-Based Generative Models) 框架。
          - 核心思想是将神经网络的输入（坐标）和目标输出（更新量）通过方差进行严谨的预缩放（Preconditioning），确保网络内部的激活值始终保持在单位方差附近，极大地提升了模型收敛的稳定性和生成质量。

    hot_models:

      - model: AlphaFold 3
        year: 2024
        role: 革命性的生物大分子结构预测模型
        architecture: pairformer_plus_diffusion_transformer
        attention_type: full_self_attention (Token level) / sequence_local_attention (Atom level)

      - model: EDM (Elucidating Diffusion)
        year: 2022
        role: 提供本模块缩放公式的理论基础模型
        architecture: diffusion_preconditioning

    model_usage_details:

      AF3 Diffusion Stage:

        usage: 在 AF3 中，该模块通常在推理时会被调用 200 次（采样步数）来逐步提纯蛋白质坐标。为了优化显存，`blocks_per_ckpt` 通常在训练期间被启用（如设为 1），否则长序列的 AtomAttention 将导致 OOM。

    best_practices:

      - 在混合精度训练（FP16/BF16）中，必须如代码所示，在传入 `diffusion_transformer` 前将特征（`a_token`, `s_single`, `z_pair`）强制 upcast 为 `torch.float32`，以防止自注意力机制中的数值溢出。
      - 当 `use_fine_grained_checkpoint` 开启时，代码会将 `AtomAttentionEncoder` 和 `AtomAttentionDecoder` 也纳入重计算范围。这对于处理极长序列（如超过 2000 个残基）的复合体训练是必须的。
      - `sigma_data` 参数（默认 16.0）应与训练集中原子坐标到重心的经验距离方差严格匹配。如果将模型迁移到完全不同的尺度（如宇宙星系坐标或纳米材料），必须重新校准该参数。

    anti_patterns:

      - 忽略 `s_ratio` 的数据类型转换，导致后续加法操作时 FP32 的缩放因子与 FP16/BF16 的坐标张量不匹配，引发报错。
      - 在推理阶段（`torch.no_grad()`）仍然保留 `blocks_per_ckpt` 的设定，这会无意义地触发 Checkpointing 逻辑，增加开销。代码中 `if not torch.is_grad_enabled(): blocks_per_ckpt = None` 是正确的防御性编程。

    paper_references:

      - title: "Accurate structure prediction of biomolecular interactions with AlphaFold 3"
        authors: Abramson et al.
        year: 2024

      - title: "Elucidating the Design Space of Diffusion-Based Generative Models"
        authors: Karras et al.
        year: 2022

  graph:

    is_a:
      - NeuralNetworkComponent
      - GenerativeModel
      - DiffusionModel

    part_of:
      - AlphaFold3

    depends_on:
      - AtomAttentionEncoder
      - DiffusionTransformer
      - AtomAttentionDecoder
      - DiffusionConditioning

    variants:
      - DDPM (标准的去噪扩散概率模型)
      - Score-Based SDE

    used_in_models:
      - AlphaFold 3
      - Protenix (开源复现项目)

    compatible_with:

      inputs:
        - Tensor (Coordinates)
        - Tensor (Noise Level)
        - Tensor (Features)

      outputs:
        - Tensor (Denoised Coordinates)