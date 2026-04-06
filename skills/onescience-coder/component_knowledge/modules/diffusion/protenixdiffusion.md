component:

  meta:
    name: ProtenixDiffusionSystem
    alias: ProtenixDiffusion
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: generative_model
    author: OneScience
    license: Apache-2.0
    tags:
      - alphafold3
      - protenix
      - diffusion_model
      - noise_schedule
      - diffusion_conditioning
      - structure_prediction

  concept:

    description: >
      该模块是 AlphaFold 3 (AF3) 扩散生成过程在 Protenix 开源项目中的完整实现。
      它包含了三大核心组件：`ProtenixDiffusionConditioning`（实现 AF3 Algorithm 21，用于注入单体、对特征和噪声级别作为生成条件）、
      `ProtenixDiffusionSchedule`（管理训练时的对数正态噪声采样和推理时的多项式衰减噪声时间表），
      以及 `ProtenixDiffusionModule`（实现 AF3 Algorithm 20，结合原子编码器、全局扩散 Transformer 和原子解码器进行迭代去噪）。

    intuition: >
      如果把蛋白质三维结构的生成比作“雕刻”，`ProtenixDiffusionSchedule` 是一张严格的“时间表”，规定了每一锤子下去应该凿去多少多余的石头（噪声）；
      `ProtenixDiffusionConditioning` 则是“设计图纸”，它将氨基酸序列的特征（s_trunk, z_trunk）与当前时间步的信息融合，告诉雕刻家该往什么方向雕刻；
      而 `ProtenixDiffusionModule` 则是“雕刻家”本人，它先在局部观察原子分布，然后进行全局统筹，最后施加精准的去噪更新。

    problem_it_solves:
      - 基于连续时间扩散模型（EDM 框架）高精度生成大分子及复合物的三维坐标
      - 解决扩散过程中潜变量条件注入（Conditioning）的特征对齐和维度变换问题
      - 统一管理基于 AF3 的复杂噪声采样与推理时间表（Noise Scheduling）

  theory:

    formula:

      inference_noise_schedule:
        expression: \sigma_t = \sigma_{data} \left( s_{max}^{1/p} + t \cdot (s_{min}^{1/p} - s_{max}^{1/p}) \right)^p

      conditioning_noise_embedding:
        expression: n_{emb} = \text{Linear}(\text{LayerNorm}(\text{Fourier}(\frac{\log(t\_hat / \sigma_{data})}{4})))

      edm_denoised_output:
        expression: x_{denoised} = \frac{1}{1 + s_{ratio}^2} x_{noisy} + \frac{t\_hat}{\sqrt{1 + s_{ratio}^2}} F_{\theta}(r_{noisy}, t\_hat)

    variables:

      t\_hat:
        name: NoiseLevel
        description: 当前扩散步的噪声水平（$\sigma$），在代码中为 `t_hat_noise_level`

      s_{max}, s_{min}:
        name: NoiseBoundaries
        description: 噪声调度的最大值（默认 160.0）和最小值（默认 4e-4）

      p:
        name: PolynomialExponent
        description: 控制推理噪声时间表衰减曲线的多项式指数（默认 7.0）

      s_{ratio}:
        name: SigmaRatio
        description: 当前噪声水平与数据标准差的比值 ($t\_hat / \sigma_{data}$)

  structure:

    architecture: conditioned_diffusion_with_schedule

    pipeline:

      - name: NoiseScheduling
        operation: generate_noise_level_for_current_step

      - name: DiffusionConditioning
        operation: fuse_trunk_features_and_noise_embedding (Algorithm 21)

      - name: Preconditioning
        operation: scale_x_noisy_to_r_noisy

      - name: IterativeDenoising
        operation: encoder_transformer_decoder_forward (Algorithm 20)

      - name: OutputRescaling
        operation: edm_skip_connection_and_output

  interface:

    parameters:

      sigma_data:
        type: float
        default: 16.0
        description: 训练数据坐标经验分布的标准差

      s_max:
        type: float
        default: 160.0
        description: 扩散调度允许的最大噪声水平

      dt:
        type: float
        default: 0.005 (1/200)
        description: 推理时的扩散时间步长

      blocks_per_ckpt:
        type: int
        default: null
        description: 梯度检查点（Gradient Checkpointing）的块大小，用于优化显存

    inputs:

      x_noisy:
        type: Tensor
        shape: "[..., N_sample, N_atom, 3]"
        dtype: float32
        description: 当前时间步的带噪原子坐标

      t_hat_noise_level:
        type: Tensor
        shape: "[..., N_sample]"
        dtype: float32
        description: 当前批次样本对应的噪声水平

      s_trunk:
        type: Tensor
        shape: "[..., N_tokens, c_s]"
        description: 来自 PairFormer 的单体特征 (Single Representation)

      z_trunk:
        type: Tensor
        shape: "[..., N_tokens, N_tokens, c_z]"
        description: 来自 PairFormer 的节点对特征 (Pair Representation)

    outputs:

      x_denoised:
        type: Tensor
        shape: "[..., N_sample, N_atom, 3]"
        description: 经过一步去噪处理后的三维坐标预测结果

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      from typing import Optional, Union

      import torch
      import torch.nn as nn

      from onescience.models.openfold.primitives import ProtenixLayerNorm
      from onescience.models.protenix.utils import broadcast_token_to_atom, expand_at_dim
      from onescience.modules.embedding.protenixembedding import ProtenixFourierEmbedding
      from onescience.modules.decoder.protenixdecoder import ProtenixAtomAttentionDecoder
      from onescience.modules.encoder.protenixencoding import ProtenixAtomAttentionEncoder, ProtenixRelativePositionEncoding
      from onescience.modules.transformer.protenixtransformer import ProtenixDiffusionTransformer
      from onescience.utils.openfold.checkpointing import get_checkpoint_fn
      from onescience.models.protenix.modules.primitives import Transition
      from onescience.modules.linear.protenixlinear import ProtenixLinearNoBias

      class ProtenixDiffusionConditioning(nn.Module):
          """Implements Algorithm 21 in AlphaFold3."""
          # (初始化与前向逻辑较长，核心逻辑为将 s_trunk, z_trunk 与傅里叶编码后的噪声水平融合，并通过 Transition 层更新)
          # ... (详见源文件完整代码)

      class ProtenixDiffusionSchedule:
          """Diffusion noise schedule for training and inference."""
          def __init__(self, sigma_data=16.0, s_max=160.0, s_min=4e-4, p=7.0, dt=1/200, p_mean=-1.2, p_std=1.5):
              self.sigma_data, self.s_max, self.s_min, self.p, self.dt = sigma_data, s_max, s_min, p, dt
              self.p_mean, self.p_std = p_mean, p_std
              self.T = int(1 / dt) + 1  

          def get_train_noise_schedule(self) -> torch.Tensor:
              return self.sigma_data * torch.exp(self.p_mean + self.p_std * torch.randn(1))

          def get_inference_noise_schedule(self) -> torch.Tensor:
              time_step_lists = torch.arange(start=0, end=1 + 1e-10, step=self.dt)
              inference_noise_schedule = (
                  self.sigma_data * (self.s_max ** (1 / self.p) + time_step_lists * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))) ** self.p
              )
              return inference_noise_schedule

      class ProtenixDiffusionModule(nn.Module):
          """Main diffusion module for structure prediction. Implements Algorithm 20 in AlphaFold3."""
          # (整合了 ProtenixDiffusionConditioning, ProtenixAtomAttentionEncoder, ProtenixDiffusionTransformer, ProtenixAtomAttentionDecoder，并实现 EDM 缩放算法)
          # ... (详见源文件完整代码)

  skills:

    build_protenix_diffusion:

      description: 构建遵循 Protenix/AF3 架构的完整扩散流水线（含调度、条件和主干）

      inputs:
        - sigma_data
        - blocks_per_ckpt

      prompt_template: |
        构建 ProtenixDiffusionModule 及其对应的 Schedule。
        数据标准差设定为 {{sigma_data}}，
        如果需要进行 Stage 2 微调以节约显存，请确保开启 `blocks_per_ckpt`={{blocks_per_ckpt}} 并且设置 `use_fine_grained_checkpoint=True`。

    diagnose_conditioning_oom:

      description: 分析在输入具有长序列长度的大复合物时 DiffusionConditioning 引发的显存溢出

      checks:
        - extremely_large_z_trunk_shape_causing_cuda_out_of_memory_in_linear_layers
        - failure_to_clear_cuda_cache_during_inference_for_large_pairs

  knowledge:

    usage_patterns:

      inference_sampling_loop:

        pipeline:
          - Schedule.get_inference_noise_schedule() (获取时间表)
          - Sample Initial Noise (从 N(0, s_max^2) 采样)
          - For t in Schedule:
            - Conditioning (Algorithm 21)
            - ProtenixDiffusionModule (Algorithm 20)
            - Update coordinates using Euler/Heun solver
          - Return final coordinates

    design_patterns:

      elucidating_diffusion_models_framework:

        structure:
          - 采用 Karras 等人提出的 EDM 框架理论
          - 在训练阶段使用对数正态分布（Log-Normal）独立采样每一步的噪声水平，而在推理阶段则遵循精心设计的多项式衰减时间表，确保采样稳定和快速收敛

    hot_models:

      - model: AlphaFold 3 (AF3)
        year: 2024
        role: 革命性的生物大分子结构预测模型
        architecture: pairformer_plus_diffusion_transformer

      - model: Protenix
        year: 2024
        role: AlphaFold 3 的高质量开源复现项目
        architecture: af3_architecture_pytorch_implementation

    model_usage_details:

      Protenix Finetuning Phase 2:

        usage: 为了在极长序列（例如 2000 个 tokens 以上）进行训练，必须开启 `blocks_per_ckpt` 和 `use_fine_grained_checkpoint`，这将不仅在全局 Transformer，而且在 `AtomAttentionEncoder` 和 `AtomAttentionDecoder` 内部广泛应用梯度检查点机制。

    best_practices:

      - 代码在 `ProtenixDiffusionConditioning` 前向传播结束时内置了一个显存优化 Trick：`if not self.training and pair_z.shape[-2] > 2000: torch.cuda.empty_cache()`。这对于推理阶段处理超大蛋白质复合物时防止显存碎片化非常有效。
      - 当 `use_conditioning=False` 时，代码会显式地清零 `s_trunk` 和 `z_trunk`。这通常用于 Classifier-Free Guidance (CFG) 中的无条件生成分支计算，此时模型仅依据噪声水平和时间步进行盲目的去噪。
      - `get_train_noise_schedule` 直接从以 `p_mean` (-1.2) 和 `p_std` (1.5) 参数化的分布中进行随机采样，因此训练时同一个 Batch 的各个样本可以且应该处于不同的噪声时间步，这需要正确的广播（Broadcasting）操作支持。

    anti_patterns:

      - 在构建 `t_hat_noise_level` 时，直接传递从 0 到 1 线性递减的值（例如 DDPM 的做法），而不是按照 EDM 的要求传入通过 `get_inference_noise_schedule` 计算出的实际物理尺度标准差数值 $\sigma_t$。
      - 忽略对傅里叶编码（`ProtenixFourierEmbedding`）输入的对数变换：正确的输入应该是 `log(t_hat / sigma_data) / 4`，如果不做变换会导致模型在高噪声水平时注意力特征崩溃。

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
      - DiffusionPipeline

    part_of:
      - Protenix
      - AlphaFold3

    depends_on:
      - ProtenixAtomAttentionEncoder
      - ProtenixDiffusionTransformer
      - ProtenixAtomAttentionDecoder
      - ProtenixFourierEmbedding
      - Transition

    variants:
      - None

    used_in_models:
      - Protenix

    compatible_with:

      inputs:
        - Tensor (Coordinates)
        - Tensor (Trunk Embeddings)

      outputs:
        - Tensor (Coordinates)