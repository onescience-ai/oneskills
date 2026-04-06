# AI Foundation Models for Weather and Climate — 模型与组件知识提取

> 说明：本文不是单一模型，而是系统性梳理“天气与气候领域基础模型（Foundation Models，FM）”的应用场景、设计准则和实现方案。这里把“天气/气候 FM 家族”视作一个抽象的整体模型范式来提取两大维度知识。

---

## 维度一：整体模型知识（Model-Level）

### 1. 核心任务与需求

- 总体目标：
  - 构建一类**预训练–微调式 AI 基础模型**，在统一的空间–时间–变量表示上，通过自监督预训练，服务于多种下游天气与气候任务：
    - 预报：从分钟级 nowcasting 到日–周尺度数值天气预报，再到季节–年际气候预测（受篇幅/数据限制，论文聚焦“小时–两周以内”的 WeatherFM 范围）。
    - 下游应用：
      - 传统 NWP 后处理与多模式集合调和（MOS、model blending）；
      - 动力和气候模型的下采样/上采样（downscaling / super‑resolution）；
      - 物理过程参数化（深对流、云、湍流等）；
      - 数据同化相关的学习模块；
      - 天气形态检测与追踪（TC/ETC/AR/tornado 等）；
      - 气候科学任务（模式不确定性分析、可预报性来源识别等）；
      - 气象驱动的影响与应用模型（能源、洪水、野火、灾害风险等）。

- 现有痛点：
  - 传统 NWP：昂贵 HPC、复杂方程和参数化、数据/模式双重 bias，且对稀疏观测的利用高度依赖专业 DA 系统。
  - 现有 AI 模式（GraphCast/FourCastNet/Pangu 等）：
    - 多为**单一任务、单一分辨率、监督式训练**，长时滚动易模糊/回归气候态，极端事件谱能量不足；
    - 通常锚定 ERA5 0.25°，难统一 HRRR 等高分辨率区域模式；
    - 对多变量、多数据源（ERA5/MERRA2/CMIP6/观测）缺乏统一自监督框架；
    - 对多种下游任务的可迁移性有限。

- Foundation Model 在本领域的关键需求：
  - **自监督预训练 + 任务轻量解码器**：减少标注/监督依赖，支撑少样本/弱监督下游任务；
  - **多尺度、多物理、多数据源统一表示**：兼容 ERA5、MERRA2、CMIP6、HRRR、雷达与站点观测等；
  - **长时间滚动稳定性与极端事件保真度**：在多步自回归中缓解模糊与谱偏差；
  - **计算效率**：一次大规模预训练 + 多任务小规模微调，替代“每个任务单独训练大模型”的昂贵范式。

### 2. 概念性解决方案：天气/气候 FM 范式

- 抽象结构：
  - 通用形式：

    $$
    g_\phi \circ f_\theta \quad\text{(预训练阶段)}
    $$

    $$
    h_\psi \circ f_\theta \quad\text{(下游任务微调阶段)}
    $$

    - $f_\theta$：大容量编码器（Transformer / Neural Operator / GNN 等），学习跨变量、跨空间、跨时间的通用表示；
    - $g_\phi$：预训练时期的通用解码器，用于自监督任务（重建/预测等）；
    - $h_\psi$：轻量任务解码器，各下游任务独立定义。

- 预训练思路：
  - 使用**大规模自监督**目标在多源气象–气候数据上训练 $f_\theta$：
    - 典型任务：时空掩码重建、未来步预测（forecast as pretext）、变量/层重建、多变量联合重构；
    - 可结合概率/集合目标（如 AtmoRep 的集合统计损失）。
  - 要求预训练分布涵盖下游任务所需变量与物理过程（至少在统计意义上）。

- 下游任务微调：
  - 冻结或部分冻结 $f_\theta$，针对具体任务增添轻量解码头 $h_\psi$：
    - 例：
      - 预报任务 → 回归/生成头；
      - 分类/检测任务（TC/AR 检测）→ 分割/检测头；
      - 气候诊断 → 表征抽取 + 线性探针。

### 3. 设计空间与关键约束

#### 3.1 应用与尺度选择（Design Scope）

- 目标时间–空间窗口：
  - 论文建议当前阶段优先构建 **WeatherFM**：
    - 时间：小时–两周（≤ 10–14 天，中期天气范围内）；
    - 空间：从 ERA5 0.25° 全球到 HRRR ~3 km 区域分辨率；
    - 不直接覆盖微尺度秒级 nowcasting 及季节–多年尺度；
  - 未来可扩展到：
    - Nowcasting FM（融合雷达/站点观测 + DA）；
    - Climate FM（面向 CMIP 级别长序列和情景预测）。

- 物理尺度多样性：
  - 现有 AI 模型多为**单一空间分辨率**（多锚定 ERA5）；
  - 理想 FM 应支持：
    - 多数据集（ERA5/MERRA2/HRRR/CMIP6）共享编码器；
    - 不同区域覆盖（全球 vs 北美局地等）；
    - 多时间步长（1h/3h/6h/...）；
  - 需在复杂性与收益间权衡（“Impact vs Effort”），短期建议锁定 WeatherFM 规模。

#### 3.2 关键设计目标

- Multi-scale：
  - 融合不同网格、时间分辨率与物理过程；
  - 充分利用 ERA5 的长时序和 HRRR 的高分辨局地信息。

- Long-term rollout 稳定：
  - 现有 AI emulator 长时滚动存在模糊和极端退化；
  - 参考 PDE‑Refiner、SFNO、DLWP U‑Net 等在稳定自回归上的经验（频谱空间修正、球谐表示、refinement/diffusion 等）。

- 预训练收益与微调成本：
  - FM 必须**显著降低下游任务训练资源**，否则丧失范式优势。

- 稀疏/观测数据：
  - 当前大部分模型只吃 reanalysis 网格，不能直接 assimilate 稀疏站点数据；
  - 真正长期目标：FM 与 DA 深度耦合，实现“同化+预报+下游任务”一体化。

### 4. 实现要素概览（Implementation Choices）

#### 4.1 数据与表示

- 数据源：
  - Reanalysis：ERA5、MERRA‑2；
  - 预测：NWP 预报场（HRES/GFS/HRRR 等）；
  - 气候：CMIP6/ClimSim 等；
  - 观测：雷达、卫星检索、站点观测（长期目标）。

- 坐标/网格表示：
  - 笛卡尔 lat‑lon 网格（多模型当前做法，迁移 CV 模型方便）；
  - 球面/多面体离散：icosahedral、HEALPix、cubed‑sphere 等，用于避免极区奇点；
  - 文中列举的代表：
    - SFNO：球面 Fourier Neural Operator；
    - HEAL‑Swin：HEALPix + Swin Transformer；
    - ClimFormer：球面 Transformer 等。

#### 4.2 模型骨干候选

- Transformer 路线：
  - 3D Earth‑specific Transformer（如 Pangu‑Weather 的 3DEST）；
  - Swin 类 window attention（非球面 & 球面）；
  - ClimaX 这类多变量多任务 ViT‑style 架构；
  - 关键技术：自注意力、tokenization、位置编码、稀疏/局部注意等。

- Graph / GNN 路线：
  - GraphCast、Graph PDE solver、Encoder–Processor–Decoder 结构等；
  - 优势：天然适配不规则网格与粒子系统；
  - 挑战：多变量配置灵活性、与 FM 多任务需求的适配。

- Neural Operator 路线：
  - FNO / AFNO、FourCastNet、SFNO；
  - CNO：函数–函数映射，减少 aliasing，适合作为算子学习模块；
  - DeepONet 等 operator learning 模型。

- 组合式架构：
  - Transformer + Neural Operator（如 FourCastNet：AFNO + ViT）；
  - Transformer + CNO；
  - 视下游任务和算力做权衡。

#### 4.3 预训练范式与损失

- RMSE/重建类：
  - Masked autoencoding：随机 mask 时空块，重建全场；
  - Forecast‑as‑pretext：给定时刻 $t$，预测 $t+\Delta t$，用 MSE/RMSE；
  - 变量/层重建：mask 整个 pressure level 或变量通道再重建。

- 概率/集合式：
  - AtmoRep：通过集合输出，用统计矩匹配损失代替单一 MSE；

- 对比自监督：
  - 通用 SimCLR‑style，对天气数据可用简单 subsampling 作为唯一增强（AtmoDist 结果表明无需复杂特定变换）；
  - I‑JEPA：joint‑embedding 预测范式，避免手工 augmentation。

- 领域特定 pretext（AtmoDist）：
  - 示例：预测两个状态的时间间隔（temporal separation）、空间间隔、演化关系等。

- Diffusion / 生成式：
  - Dyffusion、SEEDS 等将扩散模型用于概率预报和长时滚动修正；
  - PDE‑Refiner 使用扩散式 refinement 以修正子主模态误差，改善长时间精度。

### 5. 下游任务映射与收益

- 预报/nowcasting：
  - 利用预训练编码器 + 微调解码器，可替代当前单任务 emulator；
  - 通过 diffusion/ensemble 接口做概率预报，改善极端与不确定度刻画。

- NWP 增强：
  - MOS / blending：以 FM 表征作为输入，对多个 NWP 模式输出进行后处理；
  - 参数化：将高分辨模拟/观测映射为低分辨模式的次网格闭合项。

- 数据同化相关：
  - 以 FM 表征空间中的“分析–预报”关系为 DA 模块提供学习先验（当前仅提出方向，未给具体公式）。

- 形态检测与气候诊断：
  - 在 FM 表征上训练 TC/AR/tornado 检测器、teleconnection 识别器等；
  - 用于模式不确定性分析、可预报性来源挖掘、气候影响评估等。

---

## 维度二：基础组件知识（Component-Level）

> 说明：以下把论文中提出或系统化的关键“模块/设计维度”视作可复用组件进行拆解。伪代码均为**基于文中描述生成的高层伪代码**，非论文原始代码。

### 组件 1：基础模型编码–解码框架（Encoder f_θ / Decoder g_φ, h_ψ）

1. 子需求与问题：
   - 需要一个统一的表示空间，使多源、多尺度气象–气候数据在此空间中可共用，并能支撑多类下游任务。

2. 关键设计：
   - 预训练阶段：

     $$
     z = f_\theta(x), \quad \hat{x} = g_\phi(z)\quad\Rightarrow\quad \min\mathcal{L}_{\text{self‑sup}}(\hat{x},x)
     $$

   - 微调阶段：

     $$
     z = f_\theta(x), \quad \hat{y} = h_\psi(z)\quad\Rightarrow\quad \min\mathcal{L}_{\text{task}}(\hat{y},y)
     $$

   - $f_\theta$ 需要具备：
     - 跨变量（T/U/V/Z/Q/...）、跨层、跨空间、跨时间的关系建模能力；
     - 支持变长序列与不同网格；
     - 对观测/再分析/模式输出通用。

3. 上下文&绑定：
   - 前接：统一数据表示（坐标/变量/时间等）；
   - 后接：自监督解码 g_φ + 任务解码 h_ψ；
   - 绑定程度：弱绑定-通用架构，但在气象任务中强依赖物理变量与坐标设计。

4. 训练流程伪代码：

```pseudo
# 基于文中描述生成的高层伪代码
# 预训练阶段
for batch in pretrain_loader:
    x = batch.inputs          # 例如 ERA5/CMIP6 网格
    x_masked, mask_info = apply_pretext_masking(x)  # 时空/变量/层mask

    z = f_θ(x_masked)         # 通用编码器
    x_hat = g_φ(z, mask_info) # 自监督解码器

    L_self = loss_self_supervised(x_hat, x, mask_info)  # MSE/概率/对比等

    L_self.backward()
    optimizer_θφ.step()

# 微调阶段（示意）
for batch in finetune_loader:
    x, y = batch.inputs, batch.labels
    z = f_θ(x)                # 通用编码器（可冻结或半冻结）
    y_hat = h_ψ(z)            # 任务解码器：预报/分类/检测等
    L_task = loss_task(y_hat, y)
    L_task.backward()
    optimizer_ψ.step()
```

---

### 组件 2：数据表示与网格选择（Coordinate & Grid Representation）

1. 子需求：
   - 支撑全球–区域多尺度模拟，同时兼顾：
     - 地球几何一致性（球面 vs 笛卡尔）；
     - 多分辨率（0.25° 全球 vs 3 km 区域）；
     - 高效张量运算与算子学习（Fourier/卷积/注意力）。

2. 关键选项：
   - 笛卡尔 lat‑lon 网格：简化与 2D/3D CV 模型对接，但极区变形严重；
   - 球面多面体：icosahedral、HEALPix、cubed‑sphere，避免极点奇异性；
   - 选择对 operator learning（SFNO/CNO）和 transformer（HEAL‑Swin/ClimFormer）都友好的表示。

3. 实现思路（伪代码框架）：

```pseudo
# 基于文中描述生成的高层伪代码
function build_grid_representation(dataset_config):
    if dataset_config.type == "latlon":
        grid = build_latlon_grid(resolution=dataset_config.res)
    elif dataset_config.type == "icosahedral":
        grid = build_icosahedral_grid(level=dataset_config.level)
    elif dataset_config.type == "healpix":
        grid = build_healpix_grid(Nside=dataset_config.Nside)
    else:
        raise NotImplementedError

    return grid
```

---

### 组件 3：Transformer 相关子模块（Tokenization / Positional Encoding / Sparse Attention）

1. 子需求：
   - 让 Transformer 在超长时空序列上可训练、可扩展，并能编码物理位置/变量/层信息。

2. Tokenization：
   - 按 patch/窗口对 2D/3D 场切块，转换为 token 序列：
     - 变量合并编码（如 Pangu：变量作通道，3D 卷积嵌入）；
     - 变量分离编码（如 ClimaX：按变量单独 patch，再 concat）。

3. 位置/变量编码：
   - 绝对/相对/条件位置编码（空间+时间+层+变量类型）；
   - 可学习位置 embedding，支持对未见压力层的外推。

4. 稀疏注意：
   - 按物理邻近/时间邻近/随机模式稀疏连接，降低 $\mathcal{O}(N^2)$ 成本；
   - 结合窗口注意（Swin）、带状/膨胀注意、BigBird‑式随机稀疏等。

5. 典型自注意力公式（已在原文给出）：

  $$
  \text{Attn}(Q,K,V) = \operatorname{softmax}\Big(\frac{QK^T}{\sqrt{D_k}}\Big) V\tag{2}
  $$

6. 高层伪代码（简化）：

```pseudo
# 基于文中描述生成的高层伪代码
function transformer_block(X, pos_tokens):
    # X: [B, N, D] tokens; pos_tokens: [B, N, D_p]
    H = concat(X, pos_tokens)

    # 多头自注意 + 前馈
    H = MultiHeadSelfAttention(H, sparse_pattern=True)
    H = FeedForward(H)

    return H
```

---

### 组件 4：Graph / Neural Operator 模块（GNN / FNO / SFNO / CNO）

1. 子需求：
   - 面向 PDE/动力系统，直接学习“算子”而非逐点函数：
     - 高效捕捉长程依赖与多尺度结构；
     - 支持网格分辨率变化与不规则几何。

2. 关键家族：
   - FNO / AFNO：Fourier 空间卷积，适合规则网格；
   - SFNO：在球面上推广 FNO（球谐变换），支持稳定自回归大气动力学；
   - CNO：把 CNN 推广到“函数–函数”映射，减轻 aliasing，兼顾连续性与离散计算；
   - GNN：基于图的消息传递算子，适合不规则网格/粒子系统及 mesh PDE 求解。

3. 与 Transformer 的组合方式：
   - 作为 token mixing 模块（FourCastNet：AFNO + ViT）；
   - 作为时间/空间的 operator 层，与自注意力分工协作；
   - 具体选择取决于目标任务（全局大气 vs 区域流场等）和算力预算。

---

### 组件 5：预训练任务与损失模块（Self-supervised & Diffusion Heads）

1. 子需求：
   - 利用无标签或弱标签数据，学习可迁移的时空表示；
   - 兼顾确定性重建和概率/集合特性。

2. 重建/预测类：
   - Masked Reconstruction：空间/时间/变量/层掩码；
   - Forecast as Pretext：预测未来若干步或多步轨迹；
   - 变量/层重建：部分变量/层缺失场景。

3. 对比学习：
   - 通用 SimCLR/InfoNCE；
   - AtmoDist：以时间间隔预测等地球流体动力学任务为 pretext。

4. Diffusion / 生成式：
   - Dyffusion、SEEDS、PDE‑Refiner：通过扩散/去噪过程实现概率预报与长时 refinement；
   - 与主干骨干（Transformer/Operator）共享编码器或作为特殊解码头存在。

5. 高层预训练伪代码示例：

```pseudo
# 基于文中描述生成的高层伪代码
for batch in pretrain_loader:
    x = batch.inputs

    # 选择一种或多种 pretext 任务
    task_type = sample_pretext_task()

    if task_type == "masked_recon":
        x_in, mask_info = mask_spatiotemporal_tokens(x)
        z = f_θ(x_in)
        x_hat = g_φ(z, mask_info)
        L = mse(x_hat[mask_info], x[mask_info])

    elif task_type == "forecast":
        x_t, x_future, Δt = build_forecast_pair(x)
        z = f_θ(x_t)
        x_future_hat = g_φ(z, Δt)
        L = mse(x_future_hat, x_future)

    elif task_type == "contrastive":
        x1, x2 = subsample_tokens(x), subsample_tokens(x)
        z1, z2 = f_θ(x1), f_θ(x2)
        L = contrastive_loss(z1, z2, negatives_in_batch)

    elif task_type == "diffusion":
        noise_level = sample_noise_schedule()
        x_noisy = add_noise(x, noise_level)
        z = f_θ(x_noisy)
        noise_hat = g_φ(z, noise_level)
        L = mse(noise_hat, true_noise)

    L.backward()
    optimizer.step()
```

---

以上把该综述中关于“天气与气候基础模型”的应用、设计和实现要素抽象成一套通用 FM 范式及关键组件，便于与你现有的具体模型卡片（GraphCast、FourCastNet、Pangu、ClimaX 等）对照、拼装和扩展。如果需要，我可以基于这份范式，再帮你做一个“天气FM 设计 checklist/对比表”，用来评估新论文或自研模型是否满足这些标准。