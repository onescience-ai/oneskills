# CorrDiff 模型知识提取（Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling）

信息来源：Mardani et al., “Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling”（本文中内容仅基于原文显式描述，不做主观推断；若原文未说明则标明“未在文中明确给出”）。

---

## 维度一：整体模型与任务知识

### 1. 任务与问题设定

- **任务类型**：
  - 条件生成式的**多变量概率下采样（generative multivariate downscaling）**。
  - 从 25 km 分辨率的全球再分析/预报（ERA5 或类似 FourCastNet、GFS 等）下采样到 2 km 分辨率的区域数值模式（CWA-WRF）场。
  - 同时进行：
    - 多变量（2 m 气温、10 m 风速分量）下采样；
    - **新通道合成**：从未显式给出的水成物信息合成 1 小时最大雷达反射率（radar reflectivity）。

- **输入 / 输出形式**：
  - 区域：覆盖台湾及周边海域的固定区域。
  - 输入粗分辨率场：
    - 记为 $\mathbf{y} \in \mathbb{R}^{c_{\mathrm{in}} \times m \times n}$。
    - 在证明性实验中：
      - 空间网格：$m = n = 36$（插值到 CWA 曲线网格上的 36×36 像素）。
      - 通道数：$c_{\mathrm{in}} = 12$。
      - 通道构成（基于 Table S2）：
        - 单层变量：
          - 2 m 气温（t2m）。
          - 10 m 水平风（u10, v10）。
          - 柱向水汽含量（total column water vapor）。
        - 压力层变量（850 hPa、500 hPa）：
          - 气温（temperature）。
          - 位势高度（geopotential）。
          - 东向风（u）。
          - 北向风（v）。
  - 目标高分辨率场：
    - 记为 $\mathbf{x} \in \mathbb{R}^{c_{\mathrm{out}} \times p \times q}$。
    - 在实验中：
      - 空间网格：$p = q = 448$（2 km Lambert 投影网格）。
      - 通道数：$c_{\mathrm{out}} = 4$。
      - 通道构成：
        - 2 m 气温（t2m）。
        - 10 m 东向风（u10）。
        - 10 m 北向风（v10）。
        - 1 小时最大雷达反射率（radar reflectivity, dBZ）。

- **时空覆盖与样本量**：
  - 时间分辨率：1 小时。
  - 目标 CWA-WRF 数据：2018–2021 年。
    - 原始样本数：37,944 张；清洗（去除 NaN/Inf）后为 33,813 张。
  - 训练 / 测试划分：
    - 2018–2020：训练集（24,154 张）。
    - 2021：测试集；从中随机选取 205 个时间步用于技能评分、谱与分布评估。
    - 部分 2022 年冷锋和 2023 年台风个例用于额外案例研究。

- **下采样比例**：
  - 水平分辨率比约为 12.5 倍（25 km → 2 km）。

- **学习目标**：
  - 近似条件分布 $p(\mathbf{x}\mid \mathbf{y})$，以生成**区域 2 km 分辨率、多变量的概率预报场**。
  - 特别关注：
    - 多变量间物理一致性（风、温度与雷达回波的共址结构）；
    - 机理显著不同的变量（例如雷达反射率）与其他变量的联合统计。

---

### 2. 传统方案的痛点与动机

- **全局/区域数值模式的局限**：
  - 全球 km 级物理模式：
    - 计算成本极高；
    - 模式参数化和调优尚不成熟，系统偏差可能大于成熟的粗分辨率或区域模式；
    - 数据体量庞大，跨机房传输困难，且常不在 GPU 环境附近产生。
  - 区域动力降尺度（WRF 等）：
    - 需要在有限区域内嵌套高分辨率网格并耦合全球模式（GFS 等）；
    - 通过雷达、地面站等同化得到高质量 2 km 分辨率分析场；
    - 计算代价昂贵，限制了用于不确定性量化的集合成员数目。

- **传统统计/机器学习降尺度局限**：
  - 经典统计降尺度：
    - 依赖参数化映射（如分位数映射、广义线性回归等）；
    - 难以捕获复杂非线性和多变量依赖结构。
  - 早期深度学习/卷积网络降尺度：
    - 常用于气候到天气尺度（100 km → 25 km 等）；
    - 多为**确定性映射**，概率结果需额外构造（如集合推断或预测分布参数），对极端和尾部表现有限。

- **GAN 基类生成模型的问题**：
  - 训练不稳定、模式崩溃、难以刻画长尾分布。

- **直接条件扩散 $p(\mathbf{x}\mid \mathbf{y})$ 的困难**：
  - 输入和目标变量之间存在显著的分布差异，尤其是 1 小时最大雷达反射率：
    - 需要高噪声水平和大量反向扩散步数；
    - 导致梯度学习困难、收敛慢、样本质量差和结构不一致。
  - 下采样任务面对：
    - 大尺度偏差修正（如地形相关的静态偏差）；
    - 较大空间位移；
    - 合成新通道（雷达），与输入通道的统计关系复杂。

- **整体动机**：
  - 寻找一种既具备扩散模型稳定性，又能减轻以上分布差异和高维多尺度难度的框架。

---

### 3. CorrDiff 整体解决方案与范式

- **核心思想：两步残差扩散（regression + corrective diffusion）**：
  - 将目标场分解为：

$$
\mathbf{x} = \underbrace{\mathbb{E}[\mathbf{x}\mid \mathbf{y}]}_{:=\boldsymbol{\mu}(\text{regression})} + \underbrace{\big(\mathbf{x}-\mathbb{E}[\mathbf{x}\mid \mathbf{y}]\big)}_{:=\mathbf{r}(\text{generation})}. \tag{1}
$$

  - 第一步：使用 UNet 回归网络学习条件均值

$$
\boldsymbol{\mu} \approx \mathbb{E}[\mathbf{x}\mid \mathbf{y}].
$$

  - 第二步：在残差

$$
\mathbf{r} = \mathbf{x}-\hat{\boldsymbol{\mu}}
$$

    上训练条件扩散模型，学习 $p(\mathbf{r}\mid \mathbf{y})$，从而生成精细随机结构并修正偏差。

- **方差分解与残差的优势**：

  - 若回归近似良好，$\boldsymbol{\mu}\approx\mathbb{E}[\mathbf{x}\mid\mathbf{y}]$，则

$$
\mathbb{E}[\mathbf{r}\mid \mathbf{y}] \approx 0, \quad \operatorname{var}(\mathbf{r}\mid \mathbf{y}) = \operatorname{var}(\mathbf{x}\mid \mathbf{y}).
$$

  - 利用全方差定律：

$$
\operatorname{var}(\mathbf{r}) = \mathbb{E}[\operatorname{var}(\mathbf{r}\mid\mathbf{y})] + \underbrace{\operatorname{var}(\mathbb{E}[\mathbf{r}\mid\mathbf{y}])}_{\approx 0} 
\le \mathbb{E}[\operatorname{var}(\mathbf{x}\mid\mathbf{y})] + \underbrace{\operatorname{var}(\mathbb{E}[\mathbf{x}\mid\mathbf{y}])}_{\ge 0} = \operatorname{var}(\mathbf{x}). \tag{2}
$$

  - 结论：
    - 残差 $\mathbf{r}$ 的总体方差小于原始目标 $\mathbf{x}$，特别是在大尺度方差显著场景（如台风）。
    - 这使得扩散模型所需处理的分布更“窄”、更局地化，有利于训练和采样。

- **物理启发**：
  - 源于流体动力学中的 Reynolds 分解：将物理量拆分为平均与扰动部分；
  - 区分：
    - 可由静态地形、粗分辨率大尺度动力决定的“确定性”部分（由 UNet 回归学习）；
    - 随机对流、降水、雷达回波等更适合作为均值偏差的随机扰动来建模。

- **任务范式**：
  - **回归（fast, deterministic） + 残差生成（probabilistic, stochastic）** 的两步下采样范式；
  - 可概括为：从粗分辨率条件场出发，先给出物理合理的平均场，再用扩散模型补足和修正细节尺度及随机性。

---

### 4. 整体模型架构概览

- **总体结构**：
  - 单一 UNet 架构（带注意力和残差层），在两处复用：
    1. **回归 UNet**：预测条件均值 $\hat{\boldsymbol{\mu}}$；
    2. **扩散 UNet（denoiser）**：作为 EDM 中的 denoiser 网络 $D_{\theta}(\cdot)$，学习残差的 score / 去噪映射。

- **UNet 结构要点（原文给定）**：
  - 编码/解码层数：6 层 encoder + 6 层 decoder。
  - 基础通道数：128。
  - 通道倍增因子：$[1, 2, 2, 2, 2]$（不同分辨率层）。
  - 注意力分辨率：28（在对应空间尺度上加入注意力）。
  - 位置编码：
    - 在空间维度添加 4 通道的正弦位置编码（sinusoidal positional embedding）。
  - 时间 / 噪声编码：
    - 在扩散模型中，用 Fourier-based 时间/噪声嵌入表示扩散时间/噪声尺度 $\sigma$；
    - 在回归网络中不使用时间嵌入（因其不涉及扩散时间维度）。
  - 参数规模：约 8,000 万参数（80M parameters）。

- **扩散过程与 EDM（Elucidated Diffusion Model）**：
  - 采用基于 SDE 的连续时间扩散建模：
    - 正向 SDE（加噪）：

$$
\mathrm{d}\mathbf{x} = \sqrt{2\dot{\sigma}(t)\,\sigma(t)}\,\mathrm{d}\boldsymbol{\omega}(t). \tag{3}
$$

    - 反向 SDE（去噪与采样）：

$$
\mathrm{d}\mathbf{x} = -2\dot{\sigma}(t)\,\sigma(t)\,\nabla_{\mathbf{x}}\log p(\mathbf{x};\sigma(t))\,\mathrm{d}t + \sqrt{2\dot{\sigma}(t)\,\sigma(t)}\,\mathrm{d}\bar{\boldsymbol{\omega}}(t). \tag{4}
$$

  - 噪声日程（noise schedule）：
    - 训练中：$\ln\sigma \sim \mathcal{N}(0, 1.2^2)$ 的对数正态分布；
    - 采样中：从 $\sigma_{\max}=800$ 降到 $\sigma_{\min}=0.002$，使用 18 步二阶 EDM 随机场采样器。

- **score matching / 去噪训练目标**：
  - 记 denoiser 为 $D_{\theta}(\mathbf{x};\sigma)$，设

$$
\nabla_{\mathbf{x}}\log p(\mathbf{x};\sigma) = \frac{D_{\theta}(\mathbf{x};\sigma) - \mathbf{x}}{\sigma^2},
$$

  - 训练目标（无条件情况）为：

$$
\min_{\theta}\;\mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\mathbb{E}_{\sigma\sim p_{\sigma}}\mathbb{E}_{\mathbf{n}\sim\mathcal{N}(0,\sigma^2\mathbf{I})}\big[\|D_{\theta}(\mathbf{x}+\mathbf{n};\sigma) - \mathbf{x}\|^2\big]. \tag{5}
$$

  - 在 CorrDiff 中，将输入从 $\mathbf{x}$ 替换为残差 $\mathbf{r}$ 并加入条件输入 $\mathbf{y}$（见后文组件级说明）。

---

### 5. 数据与训练配置

- **区域与网格**：
  - 区域覆盖：
    - 经度约 $116.371^\circ\mathrm{E}$–$125.568^\circ\mathrm{E}$；
    - 纬度约 $19.5483^\circ\mathrm{N}$–$27.8446^\circ\mathrm{N}$。
  - 目标网格：
    - 2 km Lambert conformal conical 投影，$448\times 448$ 像素。
  - 输入网格：
    - ERA5 原始 ~25 km 全球再分析；
    - 通过 4× 双线性插值到 CWA 曲线网格上的 36×36 像素。

- **输入通道（ERA5，12 通道）**：
  - 单层：
    - total column water vapor；
    - 2 m temperature；
    - 10 m u-wind；
    - 10 m v-wind。
  - 压力层（850 hPa 与 500 hPa）：
    - temperature；
    - geopotential；
    - u-wind；
    - v-wind。

- **输出通道（CWA-WRF，4 通道）**：
  - 2 m temperature；
  - 10 m u-wind；
  - 10 m v-wind；
  - 1 h maximum derived radar reflectivity（雷达反射率）。

- **目标数据来源与预处理**：
  - 目标：CWA WRF-CWA/RWRF 系统，包含雷达同化及地面观测同化。
  - 垂直坐标：从 sigma 层插值至等压层；
  - 存储格式：原始 NetCDF，经预处理后转为 HDFS；
  - 数据清洗：
    - 删除含 Inf / NaN 的样本，导致样本数从 37,944 减至 33,813；
    - 标准化：减去全局均值并除以全局标准差（用于训练）。

- **训练/验证划分**：
  - 训练集：2018–2020 年（24,154 张图像）；
  - 验证/测试集：2021 年全年的小时样本；
  - 额外案例：
    - 2022 年冷锋事件；
    - 2023 年台风 Haikui 等，另含历史 1980–2020 年台风研究（与 JMA best track 对比）。

- **UNet 回归训练配置**：
  - 优化器：Adam；
  - 学习率：$2\times 10^{-4}$；
  - $\beta_1 = 0.9,\ \beta_2 = 0.99$；
  - EMA：系数 $\eta=0.5$；
  - Dropout：0.13；
  - 训练步数：约 2M steps。

- **EDM 残差扩散训练配置**：
  - 噪声分布：$\sigma \sim \mathrm{LogNormal}(0,1.2)$；
  - 训练步数：约 28M steps；
  - 硬件：16 台 DGX 节点，每台 8×H100 GPU；
  - 并行方式：数据并行；
  - 有效 batch size：512；
  - 总训练时间约 7 天（≈21,504 GPU-hours）。

---

### 6. 训练目标与优化指标

- **回归 UNet**：
  - 目标：最小化条件均值预测的均方误差（MSE）：

$$
\mathcal{L}_{\mathrm{reg}} = \mathbb{E}_{(\mathbf{x},\mathbf{y})}\Big[\|\hat{\boldsymbol{\mu}}(\mathbf{y}) - \mathbf{x}\|_2^2\Big].
$$

- **扩散 denoiser（残差）**：
  - 训练在残差 $\mathbf{r} = \mathbf{x} - \hat{\boldsymbol{\mu}}(\mathbf{y})$ 上：
  - 使用 EDM 形式的 score matching 损失（条件版）：

$$
\min_{\theta}\;\mathbb{E}_{(\mathbf{r},\mathbf{y})}\mathbb{E}_{\sigma\sim p_{\sigma}}\mathbb{E}_{\mathbf{n}\sim\mathcal{N}(0,\sigma^2\mathbf{I})}
\big[\|D_{\theta}(\mathbf{r}+\mathbf{n};\sigma;\mathbf{y}) - \mathbf{r}\|^2\big].
$$

- **生成 vs 回归的评分指标差异**：
  - CorrDiff：在评估时关注 CRPS（continuous ranked probability score），更贴近概率分布质量和不确定性刻画；
  - UNet/Random Forest/ERA5：确定性预测，主要用 MAE（与 CRPS 等价于确定性 delta 分布情形）。

---

### 7. 推理与下采样流程

- **输入条件**：某一时刻的 ERA5 条件场 $\mathbf{y}$（12 通道，36×36），经过与训练一致的归一化与插值。

- **步骤 1：均值预测（回归）**：
  1. 将 $\mathbf{y}$（及其空间位置编码）输入回归 UNet；
  2. 得到条件均值预测

$$
\hat{\boldsymbol{\mu}} = f_{\mathrm{UNet\_reg}}(\mathbf{y}).
$$

- **步骤 2：残差扩散生成**：
  1. 为每个样本成员 $k=1,\dots,K$（例如 $K=32$）：
     - 从 $\mathcal{N}(0, \sigma_{\max}^2\mathbf{I})$ 采样初始残差 $\mathbf{r}^{(k)}_0$；
     - 将 $[\mathbf{r}^{(k)}_t, \mathbf{y}, \hat{\boldsymbol{\mu}}]$ 输入 EDM 采样器（UNet denoiser）随扩散时间/噪声水平迭代 18 步；
     - 得到终态残差 $\mathbf{r}^{(k)}_{T}$；
     - 组合得到高分辨率样本：

$$
\mathbf{x}^{(k)} = \hat{\boldsymbol{\mu}} + \mathbf{r}^{(k)}_T.
$$

- **步骤 3：集合统计与后处理**：
  - 计算集合平均：

$$
\bar{\mathbf{x}} = \frac{1}{K}\sum_{k=1}^K\mathbf{x}^{(k)}.
$$

  - 对每个栅格点计算经验 CDF，用于 CRPS；
  - 计算集合标准差与 rank histogram，用于校准诊断。

- **（基于文中描述生成的伪代码，仅为结构化理解）**：

```python
# CorrDiff end-to-end inference pseudocode (based on paper description)

# Inputs: coarse ERA5 field y (12 x 36 x 36), pretrained UNet_reg, UNet_edm
# Output: ensemble of hi-res samples x_samples (K x 4 x 448 x 448)

K = 32                           # ensemble size used in paper

# Step 1: deterministic mean prediction
mu_hat = UNet_reg(y)             # shape: (4, 448, 448)

x_samples = []
for k in range(K):
    # sample initial residual noise at max sigma
    r = sample_gaussian(shape=mu_hat.shape, sigma_max=800.0)
    sigma = sigma_max

    # 18-step EDM stochastic sampler (Karras et al. 2022)
    for step in range(18):
        # compute denoiser output conditioned on y and mu_hat
        denoised_r = UNet_edm(r, sigma=sigma, cond_y=y, cond_mu=mu_hat)

        # update r according to backward SDE/ODE (details follow EDM Alg. 2)
        r = edm_stochastic_step(r, denoised_r, sigma)
        sigma = update_sigma(sigma)  # follow EDM noise schedule down to sigma_min

    # combine residual with mean
    x_k = mu_hat + r
    x_samples.append(x_k)

x_samples = stack(x_samples)     # shape: (K, 4, 448, 448)

# downstream: compute ensemble mean, spread, CRPS, etc.
```

---

### 8. 评估指标与结果概述

- **主要指标**：
  - **MAE（mean absolute error）**：
    - 用于 CorrDiff 集合均值（deterministic 近似）与 UNet、RF、ERA5 的单一路径比较；
  - **CRPS（continuous ranked probability score）**：
    - 对于单个观测标量 $x$ 和预测分布 CDF $F$：

$$
\mathrm{CRPS}(F, x) = \int_{-\infty}^{\infty} \big(F(y) - \mathbb{1}_{\{y \ge x\}}\big)^2\,\mathrm{d}y.
$$

    - 对确定性预测（退化成 Dirac 分布）时，CRPS 等价于 MAE；
    - CorrDiff 使用 32 成员集合估计 $F$。
  - **谱与 PDF**：
    - 空间功率谱（kinetic energy、2 m temperature、radar reflectivity）；
    - 风速、温度、雷达反射率的概率分布对比（尤其尾部分布）。
  - **校准诊断**：
    - 集合 spread 与 ensemble mean RMSE 的关系（理论上校准良好时二者比值 ≈1）；
    - rank histogram 平坦性。

- **总体技能结论（基于文中表格和图）**：
  - 在 205 个 2021 年独立验证时间上：
    - CorrDiff 的 CRPS **优于** UNet、RF 和 ERA5；
    - CorrDiff 的 MAE 略劣于 UNet（因为其关注的是 KL / CRPS 最优，而非 MAE 最优），优于 RF 和 ERA5；
    - 性能排序大致为：CorrDiff（CRPS 最优）、其后是 UNet、再后 RF、最差 ERA5 插值。
  - 谱与 PDF：
    - CorrDiff 显著改善 10 m 风、2 m 温度及雷达反射率的空间谱，使其更接近 WRF 目标；
    - 雷达反射率：UNet 和 RF 无法重现真实的分布尾部，而 CorrDiff 可在 0–43 dBZ 范围内较好匹配；
    - 温度与风速的总体 PDF CorrDiff 与 UNet 接近，但在尺度选择性方差增强上更接近目标。
  - 校准：
    - 当前 CorrDiff 集合仍偏**低分散（under-dispersive）** 或对某些变量过度分散，rank histogram 不完全平坦；
    - 模型不确定性标定仍是未来改进方向。

- **个例分析总结**：
  - 冷锋个例：
    - CorrDiff 能在锋面区域同时锐化 2 m 温度梯度、沿锋/横锋风场及雷达反射率，并保持多变量共址结构；
    - 尖锐程度与目标仍有差距，个例间存在差异。
  - 台风 Haikui 个例：
    - 相比 ERA5 的过宽、过弱台风，CorrDiff 能缩小最大风速半径并提高最大风速，部分修正尺度与强度误差；
    - 仍未完全恢复目标 WRF 的强度和紧致度；
    - 雷达回波展示出高分辨率雨带结构，但仍存在半径偏小、强度偏弱等问题。
  - 大样本台风统计（与 JMA best track 比较）：
    - CorrDiff 倾向于缩小风眼半径并提高最大风速，使统计分布更接近观测；
    - 高频强风事件（≥33 m/s）的概率显著增加，更接近历史记录；
    - 仍存在某些情况下半径过度收缩等偏差。

- **效率比较（与 WRF 动力降尺度）**：
  - 在给定 1 小时预报的前提下：
    - CWA-WRF：在 Fujitsu FX-100 上使用 928 CPU 核进行 13 小时预报，每预报小时约 91.38 秒，能耗约 1,285 kJ / FH；
    - CorrDiff：在单个 NVIDIA H100 GPU 上，每次下采样约 0.18 秒，能耗约 0.126 kJ / FH；
    - 相对比较：CorrDiff 约 **500–650× 更快**，**3–4 个数量级**更节能（具体倍数视比较口径略有不同，原文给出约 652× 速度和 1,310× 能效）。

---

### 9. 创新点、局限与未来方向

- **创新点（相对于已有 ML/统计降尺度工作）**：
  1. **两步残差扩散框架（CorrDiff）**：
     - 先预测条件均值，再在残差上做条件扩散，显式利用方差分解与物理启发的 Reynolds 分解，大幅减轻扩散模型的任务难度。
  2. **多变量联合下采样 + 通道合成**：
     - 同时处理动力（风）、热力（温度）与微物理相关的雷达反射率；
     - 在单一模型中进行多变量联合下采样与新通道合成，而非一变量一模型。
  3. **物理一致性的强约束评估**：
     - 通过冷锋、台风等有组织系统个例，验证多变量下采样的一致性（温度梯度、风场与反射率的共址关系）。
  4. **EDM 在地球系统 km 级下采样中的系统性应用**：
     - 结合 EDM 的物理启发超参数空间与高分辨率大图像（448×448）任务。
  5. **能效与样本效率**：
     - 从 3 年目标数据中有效学习；
     - 在单 GPU 上推理速度和能效显著优于传统区域模式动力降尺度。

- **局限与未解决问题**：
  - 校准不足：
    - 集合 spread 与 RMSE 的匹配尚不理想，rank histogram 显示欠/过分散情况；
  - 极端事件（特别是台风）表示仍有限：
    - 虽有改进，但在最大风速和半径上仍与目标或观测存在差距；
  - 时序一致性：
    - 当前框架仅在单时次条件下采样，未显式建模 km 级时间演化和自回归动力；
  - 区域与气候情景泛化：
    - 模型在台湾区域及特定资料集上训练，对其他区域和未来气候情景的适配需要额外工作。

- **未来工作方向（原文提出）**：
  1. **中期预报下采样**：
     - 在存在显著 lead-time 误差的情形下，同时进行偏差订正与下采样；
     - 结合时间一致性的 km 级动力和数据同化。\
  2. **其他地理区域的下采样**：
     - 面临可靠 km 级训练数据稀缺和计算扩展性问题。
  3. **未来气候情景下的下采样**：
     - 需要考虑人类排放情景、气候敏感性和极端事件变化等；
  4. **子 km 观测合成**：
     - 将 CorrDiff 思路扩展到传感器观测（如密集雷达、雨量计网络），直接在观测空间生成高分辨率样本。

---

## 维度二：基础组件 / 模块级知识

> 说明：本节按组件拆解 CorrDiff 系统。伪代码均为**根据论文文字与公式复原的高保真伪代码**，仅用于结构与逻辑理解，不代表作者给出的真实代码。

### 组件 A：回归 UNet（条件均值预测）

- **子需求 / 子任务**：
  - 学习从粗分辨率 ERA5 条件场 $\mathbf{y}$ 到高分辨率目标场的条件均值 $\boldsymbol{\mu}=\mathbb{E}[\mathbf{x}\mid\mathbf{y}]$；
  - 吸收与地形、背景环流密切相关的“确定性”部分，如：
    - 大尺度风场与温度分布；
    - 受固定地形控制的 2 m 温度梯度等。

- **输入 / 输出**：
  - 输入：
    - $\mathbf{y}\in\mathbb{R}^{12\times 36\times 36}$；
    - 空间正弦位置编码（4 通道）；
  - 输出：
    - $\hat{\boldsymbol{\mu}}\in\mathbb{R}^{4\times 448\times 448}$（与目标分辨率一致）。

- **关键技术点**：
  - 架构：
    - 6 层 encoder + 6 层 decoder 的层次化 UNet；
    - 通道数从 128 基础通道随分辨率倍增因子 [1,2,2,2,2] 变化；
    - 在特定空间尺度（分辨率 28）引入注意力；
    - 使用残差块增强梯度流动。
  - 位置编码：4 通道正弦 embedding，加在输入特征上，以增强地理位置信息。
  - 训练：
    - 损失函数：MSE；
    - 无数据增强；
    - 训练 200 万步，Adam + EMA + dropout。

- **创新点 / 角色**：
  - 相比直接在目标场上做扩散：
    - 回归 UNet 先捕获大尺度、相对确定的物理结构，使后续扩散只需处理残差的小尺度与随机部分；
  - 作为 CorrDiff 的“快速路径”：
    - 在推理时，即使不运行扩散步骤，$\hat{\boldsymbol{\mu}}$ 本身也已具备较高技能（接近或优于 RF 基线）。

- **绑定程度**：
  - 与 CorrDiff 的两步框架强绑定；
  - 但“条件 UNet 回归预测 + 后续随机校正”的思想在其他下采样/生成任务中也具普适性。

- **实现逻辑描述（文字）**：
  - 将 ERA5 多通道场插值并拼接为 12×36×36 的张量；
  - 叠加经归一化的空间位置编码；
  - 经多层卷积+下采样→瓶颈→上采样+skip connection 的标准 UNet 流程；
  - 最终输出 4 通道、448×448 分辨率的连续值场；
  - 在训练中最小化与 WRF 目标场的像素级 MSE。

- **（基于文中描述生成的伪代码）**：

```python
# UNet regression forward pass (pseudocode)

def unet_regression_forward(y):
    # y: (12, 36, 36) normalized ERA5 inputs
    pos = sinusoidal_positional_encoding_2d(y.shape[1:])  # (4, 36, 36)
    inp = concat([y, pos], axis=0)                        # (16, 36, 36)

    # encoder path with 6 levels
    enc_feats = []
    x = inp
    for level in range(6):
        x = residual_block(x, channels=channels_for_level(level))
        if use_attention_at_this_resolution(x):
            x = attention_block(x)
        enc_feats.append(x)
        x = downsample(x)

    # bottleneck
    x = residual_block(x, channels=channels_for_level("bottleneck"))

    # decoder path
    for level in reversed(range(6)):
        x = upsample(x)
        x = concat([x, enc_feats[level]], axis=1)
        x = residual_block(x, channels=channels_for_level(level))

    # final conv to 4 output channels
    mu_hat = final_conv(x, out_channels=4)
    return mu_hat
```

---

### 组件 B：残差 EDM 扩散校正器（Corrective Diffusion）

- **子需求 / 子任务**：
  - 在回归均值 $\hat{\boldsymbol{\mu}}$ 基础上，建模残差 $\mathbf{r}=\mathbf{x}-\hat{\boldsymbol{\mu}}$ 的条件分布 $p(\mathbf{r}\mid\mathbf{y})$；
  - 通过采样生成多成员集合，以：
    - 补足小尺度方差；
    - 合成雷达反射率合理的尾部分布；
    - 提供概率预报与不确定性估计。

- **输入 / 输出**：
  - 训练阶段：
    - 残差样本 $\mathbf{r}$（由当前 UNet 均值预测与目标之差得到）；
    - 噪声扰动 $\mathbf{n} \sim \mathcal{N}(0, \sigma^2\mathbf{I})$；
    - 噪声标准差 $\sigma$（标量或时间嵌入）；
    - 条件输入：ERA5 场 $\mathbf{y}$ 及均值 $\hat{\boldsymbol{\mu}}$；
    - 输出：去噪估计 $D_{\theta}(\mathbf{r}+\mathbf{n};\sigma;\mathbf{y})$ 近似 $\mathbf{r}$。
  - 推理阶段：
    - 初始噪声残差 $\mathbf{r}_0\sim\mathcal{N}(0,\sigma_{\max}^2\mathbf{I})$；
    - 经过 SDE 反向采样得到 $\mathbf{r}_T$；
    - 输出高分辨率样本 $\mathbf{x}=\hat{\boldsymbol{\mu}}+\mathbf{r}_T$。

- **关键技术点**：
  - 采用 EDM（Karras et al. 2022）设计的连续时间扩散过程和二阶随机采样器；
  - 噪声日程：对数正态分布，适应具有大动态范围的数据（尤其雷达反射率）；
  - 条件方式：通过通道拼接 $[\mathbf{r}+\mathbf{n}, \mathbf{y}, \hat{\boldsymbol{\mu}}]$ 并使用噪声嵌入；
  - 训练目标为残差空间中的去噪 MSE，对应 score matching；
  - 采样步数仅 18 步（依托初始回归的均值预测）。

- **创新点 / 角色**：
  - 相对直接在 $\mathbf{x}$ 上训练条件扩散，残差扩散：
    - 显著减小全场方差，尤其大尺度部分；
    - 残差场更加局地化、相关长度缩短，有利于扩散模型聚焦于小尺度与随机物理过程；
  - 对雷达反射率合成尤为关键：
    - UNet 与 RF 在雷达通道的分布与谱严重偏离目标；
    - CorrDiff 在该通道上显著提升谱与概率分布的匹配度。

- **绑定程度**：
  - 与 CorrDiff 框架强绑定；
  - 但“先回归、再对残差做 EDM 式扩散校正”的思想可迁移到其他多尺度物理场建模任务中。

- **实现逻辑描述（文字）**：
  - 在训练循环中，先通过已训练或联合训练的回归 UNet 得到 $\hat{\boldsymbol{\mu}}$，计算残差 $\mathbf{r}$；
  - 对每个 mini-batch：
    - 随机采样噪声规模 $\sigma$ 与加性噪声 $\mathbf{n}$；
    - 构造带噪输入 $\tilde{\mathbf{r}}=\mathbf{r}+\mathbf{n}$；
    - 将 $[\tilde{\mathbf{r}}, \mathbf{y}, \hat{\boldsymbol{\mu}}]$ 及 $\sigma$ 嵌入输入 UNet denoiser；
    - 最小化输出与真实残差 $\mathbf{r}$ 的 L2 损失；
  - 推理时：以 $\sigma_{\max}$ 噪声为初始，按 EDM 反向 SDE 在 18 步中逐步减小噪声并调用 denoiser 更新 $\mathbf{r}$。

- **（基于文中描述生成的伪代码）**：

```python
# Residual EDM training step (pseudocode)

def train_edm_step(batch_x, batch_y):
    # batch_x: target hi-res fields (B, 4, 448, 448)
    # batch_y: coarse inputs       (B, 12, 36, 36)

    # 1) mean prediction from regression UNet (frozen or jointly trained)
    mu_hat = UNet_reg(batch_y)                     # (B, 4, 448, 448)

    # 2) compute residuals
    r = batch_x - mu_hat                          # (B, 4, 448, 448)

    # 3) sample noise level sigma ~ LogNormal(0, 1.2)
    sigma = sample_lognormal(mean=0.0, std=1.2, shape=(B, 1, 1, 1))

    # 4) add Gaussian noise
    noise = sigma * randn_like(r)
    r_noisy = r + noise

    # 5) construct conditioning (upscale y & concat mu_hat if needed)
    cond = build_condition(batch_y, mu_hat)       # e.g., concat and/or encode

    # 6) denoiser forward
    r_pred = UNet_edm(r_noisy, sigma, cond)       # predict clean residual

    # 7) loss and backprop
    loss = mse_loss(r_pred, r)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
```

---

### 组件 C：数据与预处理管线

- **子需求 / 子任务**：
  - 将不同来源、不同网格和垂直坐标的数据对齐到统一训练/推理接口；
  - 进行质量控制与标准化，避免无效样本干扰训练；
  - 提供面向下游模型的高效 I/O（HDFS）。

- **输入 / 输出**：
  - 输入：
    - ERA5 再分析（25 km），含多层、多变量；
    - CWA WRF/RWRF 模式输出（2 km，NetCDF）。
  - 输出：
    - 配对好的 $(\mathbf{y}, \mathbf{x})$ 样本对；
    - 统一分辨率（36×36 和 448×448）、统一通道顺序；
    - 经归一化（零均值、单位方差）。

- **关键技术点**：
  - 垂直插值（sigma → pressure）；
  - 水平插值（ERA5 到 CWA 曲线网格，「4×」双线性插值）；
  - 缺失数据处理：删除含 NaN/Inf 的时间样本；
  - 存储：HDFS 以适配大样本并行读写。

- **创新性**：
  - 更偏工程实践；
  - 关键在于高质量的雷达同化 2 km 目标数据，使得 ML 降尺度可在物理上可信的“真值”附近学习。

- **绑定程度**：
  - 区域与资料源相关，迁移到其他区域/资料集时需重新实施预处理。

- **（基于文中描述生成的伪代码）**：

```python
# Data preprocessing pipeline (pseudocode)

for t in all_times:
    era5_raw  = load_era5(time=t)      # global grid
    wrf_raw   = load_cwa_wrf(time=t)   # 2 km Lambert grid

    # vertical interpolation to pressure levels for WRF (if needed)
    wrf_p = vertical_to_pressure(wrf_raw)

    # horizontal interpolation: ERA5 -> CWA 36x36 grid
    era5_interp = bilinear_interpolate_to_cwa_grid(era5_raw, size=(36, 36))

    # extract required channels and build tensors
    y = build_input_tensor(era5_interp)          # (12, 36, 36)
    x = build_target_tensor(wrf_p)               # (4, 448, 448)

    # quality control
    if has_nan_or_inf(y) or has_nan_or_inf(x):
        continue  # skip sample

    # normalization
    y = (y - y_mean) / y_std
    x = (x - x_mean) / x_std

    save_to_hdfs(y, x, time=t)
```

---

### 组件 D：评估与校准模块

- **子需求 / 子任务**：
  - 面向 CorrDiff 集合输出，评估：
    - 点值误差（MAE）；
    - 概率分布质量（CRPS）；
    - 空间谱与 PDF；
    - 集合校准程度（spread-error、rank histogram）。

- **输入 / 输出**：
  - 输入：
    - 集合样本 $\{\mathbf{x}^{(k)}\}_{k=1}^K$；
    - 目标 WRF 样本 $\mathbf{x}^{\mathrm{obs}}$；
  - 输出：
    - 指标数值表（如 MAE/CRPS for each channel & model）；
    - 谱曲线与 PDF 图；
    - 校准诊断图（spread vs RMSE, rank histogram）。

- **关键技术点**：
  - CRPS 计算：基于经验 CDF 或近似积分；
  - 谱计算：对空间场做 2D FFT → 径向平均得到功率谱；
  - PDF：统计不同变量的直方图 / log-PDF，便于比较尾部；
  - 校准：
    - 计算每个时间、每个格点的 ensemble spread 与 ensemble mean RMSE，统计其比值分布；
    - 通过 rank histogram 诊断欠/过分散。

- **（基于文中描述生成的伪代码）**：

```python
# CRPS computation for one gridpoint (pseudocode)

from scipy.stats import norm


def crps_ensemble(x_obs, ensemble_values):
    # x_obs: scalar observation
    # ensemble_values: array of K samples
    K = len(ensemble_values)
    # empirical CDF F at any y is fraction of ensemble <= y
    # practical implementation uses closed-form formula for finite ensemble
    # Here we sketch a simple numerical approximation

    ys = linspace(min(ensemble_values + [x_obs]) - margin,
                  max(ensemble_values + [x_obs]) + margin, N_grid)
    F = [np.mean(ensemble_values <= y) for y in ys]
    H = [1.0 if y >= x_obs else 0.0 for y in ys]

    integrand = [(F_i - H_i) ** 2 for F_i, H_i in zip(F, H)]
    crps = numerical_integral(ys, integrand)
    return crps
```

---

### 组件 E：案例分析模块（冷锋与台风下采样）

- **子需求 / 子任务**：
  - 冷锋：评估 CorrDiff 是否在锋区生成共址的温度梯度、风场改变与雷达强降水结构；
  - 台风：评估 CorrDiff 是否同时修正台风尺度（最大风速半径）和强度，并生成合理的雨带雷达结构。

- **输入 / 输出**：
  - 输入：
    - 选定个例时间的 ERA5、CorrDiff（集合/成员）、WRF（或 JMA 轨迹）场；
  - 输出：
    - 截面图（如沿锋/横锋风、温度、反射率）；
    - 台风轴对称风速剖面；
    - 分布对比（例如风速 PDF）。

- **关键技术点**：
  - 冷锋分析：
    - 根据沿锋风分量确定锋区位置与冷/暖区；
    - 比较 ERA5 vs CorrDiff vs WRF 的锋区温度梯度与风向转折；
    - 检查雷达反射率是否集中在冷区一侧的锋前/锋后区域。
  - 台风分析：
    - 诊断最大风速半径和最大风速强度（ERA5 / CorrDiff / WRF / JMA）；
    - 计算轴对称风速剖面；
    - 比较风速 PDF 尾部，尤其 ≥33 m/s 区间。

- **（基于文中描述生成的伪代码）**：

```python
# Typhoon axisymmetric profile analysis (pseudocode)


def compute_axisymmetric_profile(wind_speed, center, r_bins):
    # wind_speed: 2D array (448, 448)
    # center: (yc, xc) index of storm center
    # r_bins: array of radial bins in km

    ys, xs = np.indices(wind_speed.shape)
    dy = (ys - center[0]) * grid_spacing_km  # 2 km spacing
    dx = (xs - center[1]) * grid_spacing_km
    r = np.sqrt(dx**2 + dy**2)

    profile = []
    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if np.any(mask):
            profile.append(np.mean(wind_speed[mask]))
        else:
            profile.append(np.nan)

    return np.array(profile)
```

---

## 小结

- CorrDiff 通过“UNet 回归 + 残差 EDM 扩散”的两步框架，在台湾区域实现了从 25 km ERA5 到 2 km CWA-WRF 的多变量概率下采样，并成功合成雷达反射率通道；
- 其在 CRPS、谱和分布上的表现优于传统统计降尺度与确定性深度学习基线，同时大幅降低计算成本；
- 未来改进方向包括：集合校准、极端事件（特别是台风）表示、时间一致性与多区域/气候情景的推广等。