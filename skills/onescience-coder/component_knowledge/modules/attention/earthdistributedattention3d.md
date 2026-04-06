component:

  meta:
    name: EarthDistributedAttention3D
    alias: Distributed 3D Earth Position Bias Window Attention
    version: 1.0
    domain: meteorological_ai
    category: neural_network
    subcategory: distributed_attention_mechanism
    author: OneScience 
    license: Apache-2.0
    tags:
      - megatron
      - tensor_parallel
      - distributed_training
      - 3d_attention
      - pangu_weather

  concept:

    description: >
      EarthDistributedAttention3D 是三维地球位置偏置注意力机制的张量并行（Tensor Parallel, TP）版本。
      该模块深度集成了 Megatron-LM 分布式框架，通过将注意力头（Attention Heads）均分到多张 GPU 上并行计算，
      旨在突破单卡显存瓶颈，支持极高空间分辨率的多层大气气象变量的预训练和推理。

    intuition: >
      想象一下，如果要处理 0.25° 甚至更高分辨率的全球 3D 大气数据，一个巨大的“注意力魔方”单张显卡根本装不下。
      既然多头注意力机制（Multi-Head Attention）的各个“头”在计算 QKV 时是相互独立的，
      这个模块就像是一个“分工系统”：它沿着列方向（Column Parallel）把 QKV 矩阵切开，分配给不同的显卡（每张卡只计算自己分到的 num_heads_per_rank），
      等各自算完注意力之后，再沿着行方向（Row Parallel）通过网络通信（All-Reduce）把结果完美拼装起来。

    problem_it_solves:
      - 解决超高分辨率 3D 全球气象大模型在单节点上显存溢出（OOM）的问题。
      - 解决分布式混合精度（FP16/BF16）训练时，巨大的注意力点积容易导致的数值溢出问题（通过强制 FP32 计算与 Safe Softmax 策略）。

  theory:

    formula:
      
      tensor_parallel_qkv:
        expression: Q_i, K_i, V_i = ColumnParallelLinear(X) # 仅分配当前 GPU (Rank i) 负责的维度
        
      safe_attention_fp32:
        expression: Attn_i = Softmax( (Q_i * K_i^T) - max(Q_i * K_i^T) + EarthPositionBias3D_i ) * V_i

      tensor_parallel_proj:
        expression: Output = AllReduce( RowParallelLinear(Attn_i) )

    variables:

      tp_size:
        name: TensorModelParallelSize
        description: 参与张量并行计算的 GPU 数量

      num_heads_per_rank:
        name: HeadsPerGPU
        description: 每张 GPU 实际负责计算的注意力头数 (num_heads / tp_size)

  structure:

    architecture: megatron_tensor_parallel_attention

    pipeline:

      - name: ParallelQKVGeneration
        operation: column_parallel_linear (将输入 X 投影到当前显卡负责的 QKV 分片空间)

      - name: PrecisionCasting
        operation: cast_to_fp32 (将 Q, K, V 强制转换为 float32 防止点积溢出)

      - name: DistributedEarthBiasAddition
        operation: broadcast_add_bias (查表并叠加属于当前 head 分片的三维位置偏置)

      - name: SafeSoftmaxWithMask
        operation: masked_softmax_with_amax (减去最大值后计算 softmax，并处理无效掩码)

      - name: OutputGathering
        operation: row_parallel_linear (执行投影合并，Megatron 底层自动触发通信操作)

  interface:

    parameters:

      dim:
        type: int
        description: 输入特征图的总通道数

      input_resolution:
        type: tuple[int, int, int]
        description: 填充对齐后的三维空间分辨率 (pl, lat, lon)

      window_size:
        type: tuple[int, int, int]
        description: 三维注意力窗口大小 (Wpl, Wlat, Wlon)

      num_heads:
        type: int
        description: 模型的全局总注意力头数（必须能被 tp_size 整除）

      config:
        type: MegatronConfig
        description: 包含 tensor_model_parallel_size、num_layers 等分布式核心配置信息的对象

      qkv_bias:
        type: bool
        default: true
        description: 是否为并行 QKV 层添加偏置

    inputs:

      x:
        type: Tensor
        shape: [B * num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, C]
        description: 分块展平后的三维大气特征

      mask:
        type: Tensor
        shape: [num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, Wpl * Wlat * Wlon]
        default: null
        description: 分布式环境下的注意力屏蔽掩码

    outputs:

      output:
        type: Tensor
        shape: [B * num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, C]
        description: 经过并行计算并跨卡聚合后的特征输出

  types:

    MegatronConfig:
      description: 兼容 Megatron-Core 的模型并行化全局配置项

  implementation:

    framework: pytorch, megatron-core

    code: |
      import torch
      from torch import nn

      from ..func_utils import (
          get_earth_position_index,
          trunc_normal_,
      )

      from onescience.distributed.megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
      from onescience.distributed.megatron.core.utils import init_method_normal, scaled_init_method_normal

      class EarthDistributedAttention3D(nn.Module):
          """
              用于3D大气变量的地球位置偏置窗口注意力机制。
              
              EarthAttention2D 的三维扩展版本，在气压层（pressure level）维度上
              额外引入位置偏置，用于同时捕捉垂直方向与水平方向的空间关系。适用于
              处理多层大气变量（如位势高度、温度、风场等等压面数据）的注意力计算。
              
              Args:
                  dim (int): 输入通道数（嵌入维度）。
                  input_resolution (tuple[int, int, int]): 输入特征图的空间分辨率
                      (pl, lat, lon)，用于计算窗口数量：
                      - num_pl = pl // Wpl
                      - num_lat = lat // Wlat
                      其中 type_of_windows = num_pl * num_lat，经度方向折叠进 batch 维。
                  window_size (tuple[int, int, int]): 注意力窗口大小 (Wpl, Wlat, Wlon)。
                  num_heads (int): 多头注意力的头数。
                  qkv_bias (bool, optional): 是否为QKV投影添加偏置项，默认为True。
                  qk_scale (float, optional): QK点积的缩放系数，默认为None，
                      此时自动使用 head_dim ** -0.5。
                  attn_drop (float, optional): 注意力权重的Dropout比例，默认为0.0。
                  proj_drop (float, optional): 输出投影的Dropout比例，默认为0.0。
              
              形状:
                  - 输入 x: (B * num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, C)
                      其中 num_pl = pl // Wpl，num_lat = lat // Wlat，num_lon = lon // Wlon
                  - 输入 mask: (num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, Wpl * Wlat * Wlon) 或 None
                  - 输出: (B * num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, C)
              
                  Examples:
                  >>> # 典型Pangu-Weather大气变量配置
                  >>> # 原始气压层数为13，经 get_pad3d padding 后 pl=14（对齐 Wpl=2）
                  >>> # pad_resolution = (14, 128, 256)，window_size = (2, 8, 8)
                  >>> # num_pl  = 14 // 2 = 7
                  >>> # num_lat = 128 // 8 = 16
                  >>> # num_lon = 256 // 8 = 32
                  >>> # type_of_windows = num_pl * num_lat = 7 * 16 = 112
                  >>> # B_ = B * num_lon = 4 * 32 = 128
                  >>> # N = Wpl * Wlat * Wlon = 2 * 8 * 8 = 128
                  >>> attn = EarthAttention3D(
                  ...     dim=192,
                  ...     input_resolution=(14, 128, 256),  # 传入 padding 后的分辨率
                  ...     window_size=(2, 8, 8),
                  ...     num_heads=6,
                  ... )
                  >>> x = torch.randn(128, 112, 128, 192)  # (B*num_lon, num_pl*num_lat, N, C)
                  >>> out = attn(x)
                  >>> out.shape
                  torch.Size([128, 112, 128, 192])
                  
                  >>> # 带mask的前向传播（经度循环边界填充场景）
                  >>> mask = torch.zeros(32, 112, 128, 128)  # (num_lon, num_pl*num_lat, N, N)
                  >>> out = attn(x, mask=mask)
                  >>> out.shape
                  torch.Size([128, 112, 128, 192])
          """
          def __init__(
              self,
              dim,
              input_resolution,
              window_size,
              num_heads,
              qkv_bias=True,
              qk_scale=None,
              attn_drop=0.0,
              proj_drop=0.0,
              config = None
          ):
              super().__init__()
              self.dim = dim
              self.window_size = window_size  # Wpl, Wlat, Wlon
              self.tp_size = config.tensor_model_parallel_size
              self.config = config
              self.num_heads = num_heads
              assert self.num_heads % self.tp_size == 0 ,"num_heads must be devided by tp_size"
              head_dim = dim // num_heads
              self.num_heads_per_rank = self.num_heads // self.tp_size
              self.scale = qk_scale or head_dim**-0.5

              self.type_of_windows = (input_resolution[0] // window_size[0]) * (
                  input_resolution[1] // window_size[1]
              )

              self.earth_position_bias_table = nn.Parameter(
                  torch.zeros(
                      (window_size[0] ** 2)
                      * (window_size[1] ** 2)
                      * (window_size[2] * 2 - 1),
                      self.type_of_windows,
                      # num_heads,
                      self.num_heads_per_rank,
                  )
              )  # Wpl**2 * Wlat**2 * Wlon*2-1, Npl//Wpl * Nlat//Wlat, nH

              earth_position_index = get_earth_position_index(
                  window_size
              )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
              self.register_buffer("earth_position_index", earth_position_index)

              sigma = 0.01
              init_method = init_method_normal(sigma)
              out_init = scaled_init_method_normal(sigma, num_layers=config.num_layers)
              # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
              self.qkv = ColumnParallelLinear(
                  input_size = dim,
                  output_size = dim * 3,
                  config = config,
                  init_method = init_method,
                  bias=qkv_bias
              )
              self.attn_drop = nn.Dropout(attn_drop)
              # self.proj = nn.Linear(dim, dim)
              self.proj = RowParallelLinear(
                  input_size = dim,
                  output_size = dim,
                  config = config,
                  init_method = out_init,
                  bias = True,
                  input_is_parallel = True,
                  skip_bias_add = False
              )
              self.proj_drop = nn.Dropout(proj_drop)

              self.earth_position_bias_table = trunc_normal_(
                  self.earth_position_bias_table, std=0.01
              )
              self.softmax = nn.Softmax(dim=-1)

          def forward(self, x: torch.Tensor, mask=None):
              """
              参数:
                  x: 输入特征张量，形状为 (B * num_lon, num_pl*num_lat, N, C)
                  mask: 取值为 0 或 -∞，形状为 (num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon)
              """
              B_, nW_, N, C = x.shape
              x = x.reshape(-1, C)
              qkv, _ = self.qkv(x)
              qkv = qkv.reshape(B_, nW_, N, C*3//self.tp_size)

              qkv = (
                  qkv
                  .reshape(B_, nW_, N, 3, self.num_heads_per_rank, C // self.num_heads)
                  .permute(3, 0, 4, 1, 2, 5)
              )
              #print("after qvk:",qkv.shape)
              q, k, v = qkv[0], qkv[1], qkv[2]

              q = (q * self.scale).float()
              k = k.float()
              v = v.float()

              attn = q @ k.transpose(-2, -1)
              earth_position_bias = self.earth_position_bias_table[
                  self.earth_position_index.view(-1)
              ].view(
                  self.window_size[0] * self.window_size[1] * self.window_size[2],
                  self.window_size[0] * self.window_size[1] * self.window_size[2],
                  self.type_of_windows,
                  -1,
              )
              earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
              attn = attn + earth_position_bias.unsqueeze(0).to(attn.dtype)

              if mask is not None:
                  nLon = mask.shape[0]
                  mask32 = mask.to(torch.float32)

                  if mask32.max() > 0:
                      mask32 = torch.where(mask32 > 0, torch.full_like(mask32, float("-inf")), torch.zeros_like(mask32))
                  attn = attn.view(B_ // nLon, nLon, self.num_heads_per_rank, nW_, N, N) + \
                         mask32.unsqueeze(1).unsqueeze(0)
                  attn = attn.view(-1, self.num_heads_per_rank, nW_, N, N)

              attn = attn - attn.amax(dim=-1, keepdim=True)
              attn = torch.softmax(attn, dim=-1)

              attn = self.attn_drop(attn)

              out = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C // self.tp_size)
              out = out.to(qkv.dtype)

              s1, s2, s3, s4 = out.shape
              out = out.reshape(-1, s4)
              out, _ = self.proj(out)
              out = out.reshape(s1, s2, s3, -1)
              out = self.proj_drop(out)
              return out

  skills:

    build_distributed_3d_attention:

      description: 根据模型并行策略构建分布式地球注意力机制

      inputs:
        - dim
        - input_resolution
        - num_heads
        - config

      prompt_template: |
        请基于 Megatron 的配置项 config 构建 EarthDistributedAttention3D。
        全局头数为 {{num_heads}}，确保它能被 config.tensor_model_parallel_size 完美整除。
        输入分辨率为 {{input_resolution}}。

    diagnose_tensor_parallel_errors:

      description: 排查分布式切分导致的常见崩溃问题

      checks:
        - num_heads_not_divisible_by_tp_size (若全局头数无法被 GPU 数量整除，将触发断言失败)
        - precision_overflow_in_fp16 (如果去掉了强制 float32 转换的代码，在大 batch 或高分辨率下可能产生 NaN)

  knowledge:

    usage_patterns:

      distributed_pangu_pretraining:
        pipeline:
          - MegatronInitialization
          - EarthDistributedAttention3D (Replaces standard EarthAttention3D)
          - DistributedMLP (e.g., RowParallel + ColumnParallel)

    design_patterns:

      safe_softmax_for_meteorology:
        structure:
          - `attn = attn - attn.amax(dim=-1, keepdim=True)`：经典的防止指数爆炸策略，这在处理大尺度气象变量导致的极端数值分布时尤为关键。
          - 强制高精度运算：在 `q @ k.transpose()` 执行前显式使用 `.float()` 以保障数值稳定性，计算完毕后通过 `out = out.to(qkv.dtype)` 还原回原始精度，平衡了稳定性与显存带宽。

    hot_models:

      - model: Megatron-LM / Megatron-Core
        role: 业界公认的大规模 Transformer 分布式训练底层框架
        architecture: Tensor Parallelism, Pipeline Parallelism

    model_usage_details:

      High_Resolution_Pangu_Training:
        tp_size: 4 # 或 8，取决于单节点 GPU 数量
        precision: bfloat16 (内部自动通过 float32 防止溢出)

    best_practices:
      - 在实例化模块之前，必须确保 Megatron-Core 的分布式环境已经正确初始化 (`initialize_megatron`)。
      - `earth_position_bias_table` 的初始化中最后一个维度变为了 `num_heads_per_rank`，它意味着每一张 GPU 只维护属于自己的那一部分偏置权重，这能进一步节省显存。

    anti_patterns:
      - 错误配置 `tp_size` 导致 `num_heads % tp_size != 0`，比如 14 个头分配给 4 张卡，会导致切分不均进而引发程序崩溃。
      - 在推理部署（单卡环境）中直接调用此模块；单卡推理应切回普通的 `EarthAttention3D` 或将 `tp_size` 置为 1。

    paper_references:

      - title: "Accurate medium-range global weather forecasting with 3D neural networks"
        authors: Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, Qi Tian
        year: 2023
        journal: Nature
        
      - title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
        authors: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro
        year: 2019

  graph:

    is_a:
      - AttentionMechanism
      - DistributedComponent

    part_of:
      - DistributedPanguWeather
      - MegatronParallelSystem

    depends_on:
      - ColumnParallelLinear
      - RowParallelLinear
      - get_earth_position_index

    compatible_with:
      inputs:
        - WindowedAtmosphericFeatures
      outputs:
        - WindowedAtmosphericFeatures