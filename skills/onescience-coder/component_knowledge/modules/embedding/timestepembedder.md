component:

  meta:
    name: TimestepEmbedder
    alias: MLPTimeEmbedder
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: embedding
    author: OneScience
    license: Apache-2.0
    tags:
      - embedding
      - timestep
      - diffusion
      - mlp
      - flax/jax

  concept:

    description: >
      TimestepEmbedder (时间步长嵌入器) 是用于将标量的时间刻度特征（通常为扩散模型中的噪音等级步骤）投射为模型可计算的深层次向量特征的模块。
      它先使用底层的正弦时间步表示将标量扩展为多频率的向量特征，然后通过一个包含了 SiLU 激活函数的多层感知机（MLP）进一步将其非线性转换到隐藏状态维度。

    intuition: >
      直接生成的 Sinusoidal Embedding 是固定的频率特征，虽然捕捉了时序信息，但对于神经网络而言往往不够灵活。
      TimestepEmbedder 就像个“翻译官”，拿到了固定模板的频率特征后，用几层带有激活函数的全连接层（MLP），把它翻译成和模型主干网络更契合的形式和分布。

    problem_it_solves:
      - 提供针对时间或步骤标量特征的非线性投影变换
      - 适配模型主干的隐藏特征大小以方便残差连接或注意力加运算
      - 补充简单的正弦时间嵌入无法自适应调整其特征分布的缺陷


  theory:

    formula:

      time_embedding_projection:
        expression: |
          PE(t) = \text{timestep\_embedding}(t)
          h = \text{SiLU}(W_1 \cdot PE(t) + b_1)
          x_{out} = W_2 \cdot h + b_2

    variables:

      t:
        name: TimestepScalar
        description: 原始输入的时间步或时间分数索引

      PE:
        name: PositionalEncodingFunction
        description: 标准正弦时间步编码输出

      W_1, W_2:
        name: DenseWeights
        description: 投射和转换全连接层的权重矩阵


  structure:

    architecture: multilayer_perceptron

    pipeline:

      - name: BasePositionalEncoding
        operation: sinusoidal_timestep_embedding

      - name: LinearProjection1
        operation: dense_layer

      - name: NonLinearActivation
        operation: silu_activation (Swish)

      - name: LinearProjection2
        operation: dense_layer


  interface:

    parameters:

      hidden_size:
        type: int
        description: MLP层隐状态或最终时间步向量的输出维度

      frequency_embedding_size:
        type: int
        description: 内部正弦词嵌入层的特征维度

    inputs:

      t:
        type: Tensor (Jax Array)
        shape: [N]
        description: 包含时间步编号或噪声分数级别的数组

    outputs:

      x:
        type: Tensor (Jax Array)
        shape: [N, hidden_size]
        description: 映射后可以用于加入模型主干的时间步长隐向量表达


  types:

    ScalarTimesteps:
      shape: [N]
      description: 批次时间的标量数值张量


  implementation:

    framework: flax (jax)

    code: |
      import torch
      
      from torch import nn
      
      class TimestepEmbedder(nn.Module):
          """
          Embeds scalar timesteps into vector representations.
          """
          config: ConfigDict
          global_config: ConfigDict
      
          @nn.compact
          def __call__(self, t):
      
              hidden_size = self.config.hidden_size
              arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
              x = self.timestep_embedding(t)
              x = nn.Dense(hidden_size, kernel_init=normal(0.02), dtype=arr_dtype)(x)
              x = nn.silu(x)
              x = nn.Dense(hidden_size, kernel_init=normal(0.02), dtype=arr_dtype)(x)
              return x
      
          # t is between [0, max_period]. It's the INTEGER timestep, not the fractional (0,1).;
          def timestep_embedding(self, t, max_period = 10000):
              """
              Create sinusoidal timestep embeddings.
              :param t: a 1-D Tensor of N indices, one per batch element.
                                These may be fractional.
              :param dim: the dimension of the output.
              :param max_period: controls the minimum frequency of the embeddings.
              :return: an (N, D) Tensor of positional embeddings.
              """
              # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
      
              t = jax.lax.convert_element_type(t, jnp.float32)
              dim = self.config.frequency_embedding_size
              half = dim // 2
              freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
              args = t[:, None] * freqs[None]
              embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1) ### TODO: pi here?
              return embedding

  skills:

    build_timestep_embedder:

      description: 构建将简单的时间步正弦编码映射入模型主状态的主投影结构

      inputs:
        - config

      prompt_template: |

        构建一个带有非线性层的时间步映射。
        要求：
        调用内部函数 timestep_embedding 获得初级特征。然后过一个 `Dense -> SiLU -> Dense`。




    diagnose_timestepembedder:

      description: 分析 TimestepEmbedder 模块在使用中的常见失效模式

      checks:
        - padding_or_resolution_errors
        - bad_device_alignment
        - output_shape_collapse
  knowledge:

    usage_patterns:

      diffusion_unet:

        pipeline:
          - Extract Timestep [t]
          - Pass through TimestepEmbedder -> [t_emb]
          - Broadcast and Add [t_emb] into spatial feature maps

    hot_models:

      - model: DDPM / Glide
        year: 2021
        role: 广泛使用 Dense+Swish(SiLU) 的时序嵌入组件
        architecture: Diffusion Model

    best_practices:

      - MLP 的隐藏层维度常常设为传入主干网络对应通道数的乘数倍（如原来是 C，此处的 hidden_size 可以放大到 4C 后再做二次投影）。
      - 激活函数选择 `SiLU / Swish` 会比 `ReLU` 带来平滑的一阶导数并提升扩散生成任务的质量。


    anti_patterns:

      - 此处如果引入过于深的 MLP 等于是变相削弱时间信息对于生成网络的直接短路径传递，容易丢失精度。


    paper_references:

      - title: "Diffusion Models Beat GANs on Image Synthesis"
        authors: Dhariwal et al.
        year: 2021


  graph:

    is_a:
      - ProjectionModule
      - EmbeddingLayer

    part_of:
      - DiffusionModels

    depends_on:
      - TimestepEmbedding
      - Dense
      - SiLU

    variants:
      - ADAINTimeEmbedder
      - CrossAttentionTimeEmbedder

    used_in_models:
      - DDPM
      - Glide
      - StableDiffusion

    compatible_with:

      inputs:
        - Timesteps

      outputs:
        - DenseFeatureMap