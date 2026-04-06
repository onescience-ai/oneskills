# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions — 模型与组件知识提取

> 说明：AERIS 是基于像素级 Swin Transformer 的大规模扩散生成式地球系统预测模型（1.3B–80B 参数），并提出 SWiPe（Sequence-Window Parallelism）并行策略，在 Aurora 超算上实现 10.21 ExaFLOPS 级训练性能。以下按照整体模型和组件两大维度进行结构化整理。

---

## 维度一：整体模型知识（Model-Level）

### 1. 核心任务与需求

- 核心建模任务：
  - 构建一个数据驱动的、生成式的全球地球系统模型，学习 ERA5 再分析资料上的状态转移 $p(x_{i+1}\mid x_i)$，实现：
    - 6 小时、24 小时步长的中期天气预报（1–14 天）。
    - 延伸到 90 天的季节–次季节（S2S）尺度稳定预报。
    - 利用扩散模型自然生成集合预报并量化不确定性。

- 解决的主要问题：
  - 传统 NWP：
    - 小尺度物理过程（云物理、辐射等）参数化不确定性大，计算代价高；
    - 精度提升主要依赖专家改进物理和数值算法，而非“数据量增加”；
    - 高分辨率与多集合、多变量、多前兆范围时，成本爆炸。
  - 既有确定性深度学习模型（GraphCast、FourCastNet 等）：
    - 在中期预报上接近/超过 NWP，但在集合和长时间滚动上表现不足：
      - 频谱偏差导致场景变得平滑、集合分布校准不足；
      - 对初始扰动不敏感，集合发散不够。
  - 既有扩散天气模型（如 GenCast）：
    - 能改善小尺度变率和集合校准，但在 0.25° 高分辨率下多步求解易在两周后失稳；
    - 基于 GNN 的骨干在大规模并行训练上不如 Transformer 友好。

### 2. 解决方案

- 怎么解决的（核心思路）：
  - 模型方面：
    - 使用非层级（non-hierarchical）、像素级（patch size $1\times 1$）Swin Transformer 作为骨干，在像素空间上直接建模 ERA5 全分辨率网格，结合扩散模型（TrigFlow 参数化）实现生成式预报：
      - 6 小时模型：$30\times 30$ 窗口；
      - 24 小时模型：$60\times 60$ 窗口。
    - 采用 TrigFlow 统一 EDM 和 Flow Matching，进行 $v$-prediction（速度场）学习，生成条件于历史状态和强迫的扩散轨迹。
    - 对多变量、多层次大气–海洋状态使用纬度–高度加权的物理加权损失，以强调近地面和重要气象变量。
  - 训练/系统方面：
    - 提出 SWiPe：将窗口并行（Window Parallelism, WP）与序列并行（SP）、流水线并行（PP）组合成层次化并行策略：
      - 将输入图像按 Swin 的非重叠窗口划分，在空间上分配给不同节点/设备；
      - 每个节点内再使用 Ulysses Sequence Parallelism 沿序列维度切分；
      - 结合流水线并行沿层深分段，实现 $WP \times PP \times SP$ 的三维模型并行；
      - 降低每设备激活内存和通信负载，使得在 $0.25^\circ$ 分辨率和 1.3B–80B 参数级别下稳定训练。

- 结构映射（解决思路 → 架构模块）：
  - 像素级 Swin Transformer + 扩散：对应 AERIS 主干网络与扩散目标（Section 5.2, 6.2）。
  - TrigFlow 扩散目标 + PF-ODE + DPM-Solver++：对应训练目标与推理求解器（Section 6.2）。
  - 纬度/变量加权损失：对应物理加权 loss 层（Equation (2)）。
  - SWiPe（WP + SP + PP）：对应分布式训练策略与 SWiPe 通信/内存优化模块（Section 5.1, 6.1, 6.3）。

### 3. 模型架构概览

- 整体范式：
  - 扩散生成式、像素级 Swin Transformer，autoregressive 逐步时间推进：
    - 扩散时间 $t$：通过 TrigFlow 在 $[0, \pi/2]$ 上对噪声强度采样。
    - 物理时间 index $i$：从 $x_{i-1}$ → $x_i$，以 6h/24h 步长滚动 14 天或 90 天。

- 前向流程（单时间步训练）：
  1. 输入构造：
     - 观测/再分析场 $x_{i-1}$（上一时间全场状态），和强迫量 $x_f$（太阳辐射、地形、陆海掩膜等），按通道拼接。
     - 从真实目标 $x_i$ 构造残差 $x_0 = x_i - x_{i-1}$。
     - 采样噪声 $z \sim \mathcal{N}(0, \sigma_d^2 I)$，构造插值噪声样本：

       $$x_t = \cos(t) x_0 + \sin(t) z$$

     - 条件输入：$\hat{x}_t = [x_t, x_{i-1}, x_f]$（按通道拼接）。
  2. 预处理与嵌入：
     - 对每个变量使用训练集统计作 z-score 标准化；
     - 叠加二维正弦位置编码；
     - 线性层投影到隐藏维度 $d$，得到 token 特征图；
     - 按 $30\times30$ 或 $60\times60$ 分块为局部窗口。
  3. Swin 层堆叠（共 $N$ 层）：
     - 每层包含若干 Transformer block：
       - pre-RMSNorm → 多头自注意力（窗口内 attention，使用 2D 旋转位置嵌入 RoPE）→ 残差；
       - pre-RMSNorm → SwiGLU 前馈网络 → 残差。
     - 每隔一层执行窗口平移（Shifted Window），实现全局感受野。
  4. Diffusion 时间嵌入：
     - $t$ 经共享线性层编码后广播到每层，再经层内线性层投影为自适应层归一化的 $(\alpha, \beta, \gamma)$ 参数：
       - 在 Adaptive LayerNorm 中调整均值/方差与缩放偏置。
  5. 输出解码：
     - 经过最终归一化与线性投影回到像素空间，得到对速度场 $v_t$ 的估计；
     - 与 TrigFlow 真值 $v_t$ 比较，计算物理加权损失。

- 推理流程：
  - 使用训练好的 $F_\theta$，集成概率流 ODE：

    $$
    \frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \sigma_d F_\theta(\mathbf{x}_t / \sigma_d, t)
    $$

  - 利用二阶 DPM-Solver++ 2S（10 步）在 $t \in [0, \pi/2]$ 上积分，配合 TrigFlow 的三角形“Langevin-like churn”注入噪声，提高样本质量与集合离散度；
  - 在 $t = \pi/2$ 处重新采样噪声生成新集合成员，每个输出作为下一步时间的初始条件，进行自回归滚动到 90 天。

### 4. 创新与未来

- 论文自称的主要创新点：
  - 第一个亿级参数、像素级（$1\times 1$ patch）扩散天气气候模型：
    - 在 0.25° ERA5 上训练，参数规模 1.3B–80B；
    - 在 Aurora 上实现混合精度 10.21 ExaFLOPS 持续性能、11.21 ExaFLOPS 峰值；
    - 相比 IFS ENS，在中期范围内 RMSE/CRPS 更优或相当，并在 90 天 S2S 滚动中保持稳定谱结构。
  - SWiPe 并行策略：
    - 利用 Swin 窗口独立性的窗口并行（WP），与序列并行、流水线并行组合；
    - 无需增加额外通信点或全局 batch size，即可扩展高分辨率、超大模型；
    - 明显降低激活内存、减少 all-to-all/sendrev 消息大小，提升弱/强缩放效率。
  - 非层级、像素级 Swin 结构：
    - 区别于原 Swin 分类架构的多尺度下采样层级结构，更适合时空建模；
    - 采纳 pre-RMSNorm + SwiGLU、2D RoPE 等大模型技术。

- 后续研究方向（论文中提出）：
  - 提升集合发散与 SSR：
    - 当前集合略显“under-dispersive”（SSR<1），拟通过更丰富的初始扰动策略和 TrigFlow 噪声调度（churn schedule）改进。
  - Consistency distillation：
    - 利用扩散到一致性蒸馏，压缩模型并将多步求解压缩为单步，大幅降低推理成本。
  - 多步微调（multi-step finetuning）：
    - 在一致性模型框架下采用多步监督以提升长期技巧。
  - 更长时间与更高分辨率：
    - 在更高分辨率输入和其他资料集上扩展模型，延伸预测时间和变量范围。
  - SWiPe 改进：
    - 引入零气泡流水线（zero-bubble pipeline）等技术减小现有 1F1B 方案中的空泡时间。

### 5. 实现细节与代码逻辑

#### 5.1 扩散训练目标（TrigFlow）

- 噪声构造：

$$
\mathbf{x}_t = \cos(t)\, \mathbf{x}_0 + \sin(t)\, \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \sigma_d^2 I),\; \sigma_d = 1
$$

- $t$ 采样：

$$
\tau = (1 - u)\log \sigma_{\min} + u \log \sigma_{\max}, \quad u \sim \mathcal{U}(0,1) \\
\sigma_{\min} = 0.2,\; \sigma_{\max} = 500 \\
t = \arctan (e^{\tau} / \sigma_d) \in [0, \pi/2]
$$

- 模型参数化：

$$
\mathbf{f}_\theta(\mathbf{x}_t, t) = F_\theta(\mathbf{x}_t / \sigma_d, t)
$$

- 目标速度：

$$
\mathbf{v}_t = \cos(t)\, \mathbf{z} - \sin(t)\, \mathbf{x}_0
$$

- 扩散损失：

$$
\ell^{\text{Diff}}(\theta) = \mathbb{E}_{\mathbf{x}_0, \mathbf{z}, t} \Big[ \big\| \sigma_d F_\theta(\hat{\mathbf{x}}_t / \sigma_d, t) - \mathbf{v}_t \big\|_2^2 \Big]\tag{1}
$$

其中 $\hat{\mathbf{x}}_t = [x_t, x_{i-1}, x_f]$ 为条件输入。

- 物理加权总损失：

$$
\mathcal{L}(\theta) = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \sum_{v \in \mathcal{V}} \kappa(v)\, \alpha(s)\, \ell^{\text{Diff}}_{v,s}(\theta)\tag{2}
$$

其中：
  - $\alpha(s)$：纬度/气压层加权，修正球面重格点的非均匀性，并强调近地面层；
  - $\kappa(v)$：变量级权重，对 T2m、MSLP、近地面风等关键要素赋更大权重。

#### 5.2 训练与推理伪代码

```pseudo
# 基于文中描述生成的伪代码（单 step 训练）
for batch in dataloader:
    # 1) 读取并标准化数据
    x_prev = batch.x_{i-1}      # 上一时刻全场
    x_curr = batch.x_i          # 当前时刻全场
    x_forc = batch.forcings     # 太阳辐射、地形、陆海

    x0 = x_curr - x_prev        # 残差
    x0 = zscore_normalize(x0)

    # 2) 采样 t 与噪声
    tau = (1 - u) * log(sigma_min) + u * log(sigma_max)
    t   = atan(exp(tau) / sigma_d)
    z   = normal(0, sigma_d)

    x_t = cos(t) * x0 + sin(t) * z
    x_hat = concat_channels(x_t, x_prev, x_forc)

    # 3) 前向
    v_model = F_theta(x_hat / sigma_d, t)
    v_true  = cos(t) * z - sin(t) * x0

    # 4) 物理加权损失
    loss = 0
    for variable v in V:
        for spatial_index s in S:
            w = kappa(v) * alpha(s)
            loss += w * ||sigma_d * v_model[v,s] - v_true[v,s]||^2
    loss /= |S|

    # 5) 反向与优化
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

```pseudo
# 基于文中描述生成的伪代码（单步推理，用 PF-ODE + DPM-Solver++）
function forecast_step(x_prev, forcings):
    # 初始化 t = pi/2, 对应最大噪声
    t_start = pi / 2
    x_t = sample_gaussian_like(x_prev)   # 初始化样本

    # 使用 DPM-Solver++ 2S 在 [0, t_start] 上积分 10 步
    for (t_k, t_{k+1}) in time_schedule(t_start, 0, steps=10):
        # 可选：TrigFlow churn 注入噪声
        x_t = x_t + churn_noise_if_needed(t_k)

        # 计算速度
        v = sigma_d * F_theta(x_t / sigma_d, t_k)

        # 2S 更新（略写）
        x_t = DPM_Solver_2S_update(x_t, v, t_k, t_{k+1})

    # 得到残差估计 x0_hat，从而还原 x_i
    x0_hat = x_t
    x_i = unnormalize(x0_hat) + x_prev
    return x_i
```

### 6. 数据规格

- 数据集名称：ERA5（通过 WeatherBench2 WB2 提供）。
- 时间范围：
  - 训练：1979–2018。
  - 验证：2019。
  - 测试：2020（与 WB2 评估保持一致）。
- 空间分辨率与网格：
  - 原生 $0.25^\circ$ ERA5，经极点移除后网格为 $720 \times 1440$ 像素（纬度 × 经度）。
- 时间分辨率：
  - 6 小时时序样本；模型生成 6h 和 24h 预报。

- 变量列表：
  - 5 个地面变量：
    - 2m 温度 T2m；
    - 10m 风 U10, V10；
    - 海平面气压 MSLP；
    - 海表温度 SST。
  - 5 个大气变量（13 个气压层）：
    - 位势高度 Z；
    - 温度 T；
    - 风 U, V；
    - 比湿 Q。
  - 气压层集合：{50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000} hPa。
  - 额外强迫输入：
    - 顶层太阳辐射（top-of-atmosphere solar radiation）；
    - 地表位势高度（surface geopotential）；
    - 陆海掩膜（land-sea mask）。

- 数据处理：
  - 每变量基于训练集计算均值/标准差，进行 z-score 归一化；
  - 所有变量重采样/对齐到 0.25° 统一网格；
  - 采用 HDF5 存储，总数据量约 16 TiB；
  - 在 SWiPe 中：
    - 输入/输出在空间上按窗口切片，每个节点只加载自己负责的窗口；
    - 通过 HDF5 的切片能力，只读局地窗口，减少 I/O；
    - I/O 与流水线预热阶段重叠，不增加训练时间。

---

## 维度二：基础组件知识（Component-Level）

> 说明：以下组件均为 AERIS 论文中核心的结构/并行单元。绑定程度：
> - **强绑定-气象特异**：显式利用地球物理或 ERA5 特性。
> - **弱绑定-通用**：通用深度学习组件。

### 组件 1：像素级非层级 Swin Transformer 主干（AERIS Backbone）

1. 子需求定位：
   - 对应子需求：
     - 在全球 $0.25^\circ$ 分辨率网格上建模多变量三维大气–海洋场的时空依赖；
     - 与扩散框架结合，在像素级别生成细致、光谱一致的状态更新 $x_i - x_{i-1}$。
   - 解决的问题：
     - 传统分层/下采样 Swin 不适合保留像素级细节；
     - 纯 ViT/全局注意在高分辨率上计算和内存成本不可接受；
     - 需要在不牺牲分辨率的前提下实现可扩展的时空注意力。

2. 技术与创新：
   - 关键技术点：
     - 非层级结构：
       - 不进行空间下采样和金字塔式层级，仅使用固定大小窗口（$30\times30$ 或 $60\times60$），保持像素级输出。
     - Shifted Window 注意力：

       $$
       \text{Attention}(Q, K, V) = \operatorname{softmax}\Big(\frac{QK^T}{\sqrt{d_k}}\Big) V
       $$

       - 每层在空间上划分为不重叠窗口，在窗口内做注意力；
       - 每隔一层整体平移窗口，使信息跨窗口传播，等效于全局感受野。
     - pre-RMSNorm + SwiGLU：
       - 用 RMSNorm 替代 LayerNorm，前置于子层；
       - 前馈网络中使用 SwiGLU 代替单一线性层，改善训练稳定性和表现。
     - 2D 旋转位置嵌入（RoPE）：
       - 对 Q/K 应用轴向频率 2D RoPE，替代相对位置偏置，提高空间位置感知。
   - 创新点：
     - 将“像素级 Swin” 与 扩散模型 + SWiPe 大规模并行组合，实现高分辨率地球系统的生成式建模。

3. 上下文与耦合度：
   - 上下文组件：
     - 前接：
       - ERA5 + 强迫变量的标准化与线性嵌入；
       - 2D 正弦位置编码、扩散时间嵌入（Adaptive LayerNorm 控制）。
     - 后接：
       - 解码线性层，将 embedding 投影回物理变量空间，输出 $v_t$ 或状态残差 $x_0$ 估计。
   - 绑定程度：
     - 核心 Swin block：弱绑定-通用；
     - 像素级无下采样+窗口大小选择+变量/通道设计：强绑定-气象特异。

4. 实现细节与伪代码：

```pseudo
# 基于文中描述生成的伪代码（单 Swin 层简化）
function SwinLayer(X):  # X: [B, C, H, W]
    # 1) 分块
    windows = partition_windows(X, window_size)

    # 2) 对每个窗口应用 Transformer block
    for each window in windows:
        # pre-RMSNorm + MHA with 2D RoPE
        Y = RMSNorm(window)
        Q, K, V = linear_QKV(Y)
        Q, K = apply_2D_RoPE(Q, K)
        attn = softmax(Q @ K.T / sqrt(d_k)) @ V
        window = window + attn

        # pre-RMSNorm + SwiGLU FFN
        Z = RMSNorm(window)
        ff = SwiGLU(FFN(Z))
        window = window + ff

    X = merge_windows(windows)
    X = shift_window_if_needed(X)  # 每隔一层平移窗口
    return X
```

---

### 组件 2：SWiPe（Sequence-Window Parallelism）并行框架

1. 子需求定位：
   - 对应子需求：
     - 在 0.25° 、720×1440 网格、数十亿参数 Swin 扩散模型上，实现稳定、高效的训练扩展到数千～万节点：
       - 降低通信开销与激活内存；
       - 避免过度依赖大 batch 的数据并行；
       - 支持小 batch 训练而不牺牲吞吐。
   - 解决问题：
     - 传统 4D 并行（DP+TP+PP+SP）：
       - TP 只能分解模型状态，不能有效切分激活；
       - SP 虽能分解激活，但不能切分参数；
       - 域并行（Domain Parallelism）在非局部操作（如全局 attention 或归一化）下通信代价高；
       - 高分辨率和长序列时，all-to-all 和 halo 交换的开销成为瓶颈。

2. 技术与创新：
   - 关键技术点：
     - Window Parallelism (WP)：
       - 利用 Swin 的非重叠窗口注意力，将窗口作为天然的并行单元：
         - 图像先在空间上划分为 2×2 或 4×4 等象限，再细分为窗口；
         - 每个窗口分配给一个 rank（或 rank 组），不同 rank 上窗口互不重叠，无需 halo。
     - Sequence Parallelism (SP, Ulysses)：
       - 在单节点内沿序列维度（flatten 后的 token 维）切分，通过 all-to-all 在注意力前后重组 token；
       - 结合 WP 后，每 rank 处理的序列长更短，all-to-all 消息大小显著减小：

         $$M = b \times s \times h / (SP \times WP)$$

     - Pipeline Parallelism (PP)：
       - 按层深切分模型，形成多 stage 流水线；
       - 结合 WP，窗口在 stage 之间的传输只需发送 $1/SP$ 的子片，不必在下一 stage 再重分配。
     - 结果：组成层次化并行度：

       $$\text{总并行度} = WP \times PP \times SP$$

       - 例如 40B 模型：$WP=36, PP=20, SP=12$，总并行度为 $36\times20\times12$。
   - 创新点：
     - 将 Swin 的窗口结构形式化为新一维并行（WP），与现有 SP/PP 组合，且不引入额外同步点；
     - 相比域并行无需显式 halo 交换和 token gather，避免复杂通信模式。

3. 上下文与耦合度：
   - 上下文组件：
     - 前接：数据加载与切片（节点只加载自己负责的窗口）。
     - 后接：常规的梯度 allreduce（数据并行），与优化器更新。
   - 绑定程度：
     - 对窗口化 Transformer 通用（弱绑定-通用）；
     - 本文中具体的 2×2 象限划分、720×1440 网格等为 AERIS/ERA5 特化（强绑定-气象特异）。

4. 实现细节与伪代码：

```pseudo
# 基于文中描述生成的伪代码（高层 SWiPe 流程）
# 假设有 WP x PP x SP 个并行单元

for each global image sample:
    # 1) 按空间划分到 WP 组
    windows = split_into_windows(image)
    assign windows to WP ranks in round-robin over (x, y)

    # 2) 在每个节点内部使用 SP
    tokens = flatten_windows_assigned_to_this_rank()
    tokens_sharded = sequence_parallel_shard(tokens, SP_group)

    # 3) 流水线并行执行各 Swin 层
    for stage in pipeline_stages:  # PP stages
        recv_from_prev_stage_if_needed()

        # 注意力前 all-to-all（SP）
        tokens_sharded = alltoall(tokens_sharded)
        tokens_sharded = swin_block(tokens_sharded)
        tokens_sharded = alltoall(tokens_sharded)

        send_fraction_of_tokens_to_next_stage()  # 仅发送 1/SP 的窗口子片

    # 4) 聚合输出
    gather_tokens_from_SP_and_WP()
```

---

### 组件 3：扩散时间嵌入与自适应 LayerNorm

1. 子需求定位：
   - 对应子需求：
     - 将扩散时间步 $t$ 的信息注入到每一层，使模型在不同噪声水平下有不同的特征缩放与偏移，从而稳定大规模扩散训练。
   - 解决问题：
     - 直接使用固定 LayerNorm 或简单时间拼接难以充分表达多噪声层次下的动态；
     - 大规模扩散模型在数十亿参数、0.25° 分辨率下需要更细致的归一化控制。

2. 技术与创新：
   - 关键技术点：
     - 时间嵌入：
       - $t$ 经过共享线性层编码；
       - 然后广播到所有层，每层内有独立线性层，将时间嵌入映射为 $(\alpha, \beta, \gamma)$。
     - 自适应 LayerNorm（参考文中引自 [58, 59]）：
       - 使用 $(\alpha, \beta, \gamma)$ 动态调节均值-方差和输出缩放：

         $$
         \text{AdaLN}(h; t) = \alpha(t) \cdot \frac{h - \mu(h)}{\sigma(h)} + \beta(t) + \gamma(t)
         $$

   - 创新点：
     - 将大模型中的 AdaLN 思想引入到天气扩散模型，提升训练稳定性与表达能力。

3. 上下文与耦合度：
   - 上下文组件：
     - 前接：TrigFlow 扩散时间采样；
     - 后接：所有 Swin block 内的归一化层。
   - 绑定程度：弱绑定-通用（可迁移到任意扩散/生成模型）。

4. 实现细节与伪代码：

```pseudo
# 基于文中描述生成的伪代码
# 预先：time_embed_shared: Linear, time_embed_layer[l]: Linear

function AdaLN(h, t, layer_id):
    e = time_embed_shared(t)              # 全局时间嵌入
    alpha, beta, gamma = split(time_embed_layer[layer_id](e))

    mu = mean(h, dim=channel)
    sigma = std(h, dim=channel)
    h_norm = (h - mu) / (sigma + eps)

    return alpha * h_norm + beta + gamma
```

---

### 组件 4：概率流 ODE 与 DPM-Solver++ 推理器

1. 子需求定位：
   - 对应子需求：
     - 在扩散训练后，将学习到的向量场转换为确定性的 ODE 轨迹，用有限步数生成高质量的未来状态样本和集合。
   - 解决问题：
     - 需要在 6h/24h 步长和 90 天滚动的高分辨率场景中，平衡采样质量与计算成本；
     - 多步求解在 GenCast 中两周后失稳的问题。

2. 技术与创新：
   - 关键技术点：
     - 概率流 ODE：

       $$
       \frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \sigma_d F_\theta(\mathbf{x}_t / \sigma_d, t)
       $$

     - 采用二阶 DPM-Solver++ 2S，使用与训练一致的 log-uniform 时间调度；
     - 在 TrigFlow 框架下加入类似 Langevin 的“churn”噪声注入，提高集合发散与样本多样性。

3. 上下文与耦合度：
   - 上下文组件：
     - 前接：扩散训练得到的 $F_\theta$；
     - 后接：时间滚动与集合成员生成。
   - 绑定程度：弱绑定-通用（适用于扩散模型），在 AERIS 中与气象任务强耦合。

4. 实现细节与伪代码：见维度一 5.2 推理伪代码。

---

以上即为 AERIS 论文中与气象扩散模型和 HPC 并行架构相关的核心模型/组件知识提取，如需，我可以进一步单独拆解 SWiPe 与 ORBIT、Domain Parallelism 等其他方案的对比，或补充 AERIS 在 ENSO/MJO、极端事件诊断上的统计指标与结果结构。