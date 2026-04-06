# Aurora 模型知识提取

> 数据来源：A foundation model for the Earth system（Aurora.md）

---

## 维度一：整体模型知识（Model-Level Knowledge）

### 1. 核心任务与需求

#### 1.1 核心建模任务
- 总体目标：构建一个**地球系统基础模型 Aurora**，在统一框架下对多种地球系统变量进行高效、精确的预测。
- 主要预测任务与场景（通过微调实现）：
  - **全球空气质量与大气成分预测（大气化学）**：
    - 5 天全球空气污染预报，空间分辨率 $0.4^\circ$。
    - 目标：多个化学物种（CO, NO, NO\(_2\), SO\(_2\), O\(_3\)）在各高度层及总柱量（TC），以及颗粒物 PM1, PM\(_{2.5}\), PM\(_{10}\)。
  - **全球海浪预报**：
    - 10 天全球海浪预报，分辨率 $0.25^\circ$。
    - 目标：显著波高 SWH、平均波周期 MWP、平均波向 MWD 等多种波浪变量，区分风浪（WW）、总涌（TS）、一级/二级涌（1/2），以及峰值波周期 PP1D 和 10 m 中性风分量 10UN, 10VN。
  - **全球热带气旋路径预测**：
    - 5 天热带气旋路径预报，使用 $0.25^\circ$ 分辨率 HRES T0 微调模型做推理，不专门针对台风再训练。
  - **高分辨率天气预报**：
    - 10 天全球天气预报，分辨率 $0.1^\circ$（与 IFS HRES 相当）。

- 统一建模对象：
  - 大气–海洋流体及“二阶过程”（大气化学、海浪等）的时空演化；
  - 通过一个**通用 3D 潜在表示**与统一的编码–处理–解码结构支撑不同变量组合与分辨率的预测。

#### 1.2 解决了什么问题
- 传统数值模式的局限：
  - **计算成本极高**：依赖专用超级计算机和大规模工程团队；如 CAMS 在 IFS 上增加化学模块使计算成本约增加 10 倍。
  - **开发和维护复杂**：多年累积的物理模块和参数化方案，使快速改进与维护困难。
  - **近似与参数化误差**：对次网格过程采用近似和经验参数化，限制精度。

- 现有 AI 在地球系统中的空白与不足：
  - 早期神经网络无法**规模化替代完整动力系统**。
  - Pangu-Weather 等突破主要集中在**中期全球天气预报 $0.25^\circ$ 分辨率**，
    - 对海洋动力、波浪建模、大气化学等领域覆盖不足；
    - 极端天气（如热带气旋）预报中，AI 模型相对复杂业务系统和人工分析的优势尚未证明。

- Aurora 试图解决的核心痛点：
  - 提供一个**统一的大规模基础模型**，可通过微调覆盖多种地球系统任务；
  - 在**极低计算成本**下，达到并超越多类复杂数值或业务系统的预测精度；
  - 支持在**多分辨率、多变量、多压力层**下灵活输入输出。

---

### 2. 解决方案

#### 2.1 核心解决思路
- 引入**Aurora**：一个 1.3B 参数量级的地球系统基础模型，架构上分为：
  1. **Encoder**：将异构输入（不同变量、压力层、分辨率）映射到统一的三维潜在表示；
  2. **Processor（Backbone）**：以 3D Swin Transformer U-Net 为核心，在潜在 3D 空间中演化表示；
  3. **Decoder**：将标准 3D 潜在表示解码回所需变量/压力层/分辨率的物理场。
- 训练与使用范式：
  - 第一步：在“超过一百万小时”的多源地球系统数据上进行**预训练**，目标为 6 小时 lead time 的下一步 MAE；
  - 第二步：在特定任务数据集上进行**微调（fine-tuning）**：
    - 先进行短时距（1–2 步）roll-out 微调；
    - 再利用 LoRA + roll-out fine-tuning 在长时间窗口上进一步微调。
  - 预训练和模型规模共同扩展，显著提升下游任务表现。
- 运行方式：
  - 将模型视作一个两步输入的**模拟器 $\Phi$**，通过自回归迭代进行多步预报。

#### 2.2 结构映射（思路到架构）
- “统一多源数据 → 通用潜在表示”
  - 对应组件：**3D Perceiver Encoder**：
    - 将不同压力层、不同比例网格与变量的场作为 $H\times W$ 图像处理；
    - 通过 patch 划分 + 变量特定线性映射 + Perceiver 压缩压力层，得到固定 $L$（如 3）个潜在压力层的 3D 表示；
    - 加入空间、时间、尺度（patch area）Fourier 编码，实现多分辨率兼容。

- “多尺度空间–压力–时间演化”
  - 对应组件：**Multiscale 3D Swin Transformer U-Net Backbone**：
    - 3D U-Net 风格对称上/下采样结构（3 个 encoder stage + 3 个 decoder stage）；
    - 3D Swin Transformer 层在局部窗口内做自注意力，并使用窗口移位传播信息、考虑球面拓扑；
    - 更深的 48 层结构，借助压缩后的潜在表示实现。

- “多任务输出 + 任意变量/压力层/分辨率”
  - 对应组件：**3D Perceiver Decoder**：
    - 将潜在 3D 表示通过 Perceiver 层解聚为任意指定的物理压力层集合；
    - 通过变量特定线性层生成目标变量的 patch，再还原到物理网格。

- “高效长时距微调”
  - 对应组件：
    - **LoRA 适配 + 推前技巧（pushforward trick）+ replay buffer roll-out 训练架构**，以极少额外参数在长序列上微调。

---

### 3. 模型架构概览

#### 3.1 整体结构
- 类型：**Encoder–Processor–Decoder** 架构，带 3D 潜在状态：
  - 输入：
    - 表面变量张量 $S^t \in \mathbb{R}^{V_S \times H \times W}$；
    - 大气变量张量 $A^t \in \mathbb{R}^{V_A \cdot C \times \bar{H} \times W}$；
    - 合并为 $X^t \in \mathbb{R}^{V \times H \times W}$。
  - 处理流程：
    1. **编码器**：
       - 将各变量视为 $H\times W$ 图像；
       - 加入地形、陆海掩膜、土壤类型等静态变量（作为额外 surface 变量）；
       - 图像划分为 $P\times P$ patch，变量特定线性层映射至维度 $D$；
       - 对 surface 与每个压力层：不同变量的嵌入**求和**，并添加压力层或 surface 的编码；
       - 使用 Perceiver 模块，将不定数目的物理压力层 $C$ 压缩为固定的 $L$ 个潜在压力层；
       - 得到 $\frac{H}{P} \times \frac{W}{P} \times L$ 的 3D 嵌入；
       - 加入 patch 位置、patch 面积、绝对时间的 Fourier 编码。
    2. **Backbone（3D Swin Transformer U-Net）**：
       - 对 3D 潜在表示进行多尺度 3D Swin Transformer 处理；
       - 上下采样结构实现粗–细尺度交互；
       - 窗口移位保证跨窗口信息流动并处理球面格点结构；
       - 层归一采用 res-post-norm 形式提高训练稳定性。
    3. **解码器**：
       - 使用 Perceiver 层将潜在压力层重新映射到目标物理压力层集合；
       - 变量特定线性层生成 patch 输出，再还原至目标网格；
       - 可针对不同任务（空气质量、海浪、天气等）输出不同变量集合、分辨率。

- 自回归预测：
  - 模型模拟器为

    $$
    \Phi: (\mathbb{R}^{V\times H \times W})^2 \to \mathbb{R}^{V\times H \times W}, \quad
    \Phi(X^{t-1}, X^{t}) = \hat X^{t+1}.
    $$

  - 多步预测通过：

    $$
    \Phi(X^t, \hat X^{t+1}) = \hat X^{t+2}, \quad
    \Phi(\hat X^{t+k-2}, \hat X^{t+k-1}) = \hat X^{t+k}.
    $$

#### 3.2 训练阶段划分
- **预训练（Pretraining）**：
  - 目标：6 小时 lead time 的下一步 MAE；
  - 训练：150,000 步，32 × A100，batch size=1/GPU，约 2.5 周；
  - 数据：
    - 以大气数据为主，包括 forecasts、analysis、reanalysis、climate simulations 等多源数据；
    - 多数据源、多分辨率、多变量混合。

- **短时距微调（Short-lead-time fine-tuning）**：
  - 在具体任务上微调 1–2 步 roll-out，适应新变量和数据分布。

- **长时距 roll-out 微调（Roll-out fine-tuning）**：
  - 使用 LoRA 适配 backbone 自注意力中的线性层；
  - 使用 pushforward trick 仅在最后一步反传梯度，节省内存；
  - 引入 replay buffer 对 roll-out 状态进行缓存和再利用，提高长序列训练效率。

---

### 4. 创新与未来

#### 4.1 创新点（相对前人工作的改进）

- **统一的地球系统基础模型**：
  - 通过单一 1.3B 参数模型覆盖：空气质量、海浪、热带气旋路径和高分辨率天气四大关键业务领域；
  - 支持任意变量集合与分辨率的输入输出，是“通用 Earth system foundation model”的明确实现。

- **3D Perceiver 编码–解码 + 3D Swin U-Net Backbone 结构**：
  - 编码器：
    - 支持异构数据（不同压力层数、变量、空间分辨率）；
    - Perceiver 压缩物理压力层为固定个数潜在层，实现统一 3D 表示；
    - 添加 patch 面积 Fourier 编码，使模型可在不同分辨率之间共享表示。
  - Backbone：
    - 多尺度 3D Swin Transformer U-Net，具 48 层深度，比之前 3D 气象模型更深；
    - 使用局部窗口注意力 + 窗口移位，模拟数值积分中的局部运算，并兼顾球面拓扑；
    - 无固定位置偏置，可在多分辨率运行。
  - 解码器：
    - 通过 Perceiver 将潜在层解聚为任意目标压力层集合；
    - 可灵活输出多类型变量（气象场、化学场、波浪场等）。

- **预训练数据与模型规模的系统化扩展与标度分析**：
  - 预训练使用多种数据组合（C1–C4），证明：
    - 增加多样化数据（尤其增加 CMIP6 模拟和 GFS 分析）能系统性提高验证指标，特别是极值（极端事件）表现；
  - 模型规模从 113M → 1.3B，发现在固定 GPU 小时下，验证损失满足大致的标度律：

    $$
    \mathcal{L}(N) \propto N^{-0.026},
    $$

    即模型参数每增加 10 倍，验证损失约降低 6%。

- **在多类业务系统上的首次系统超越**：
  - 空气质量：
    - 5 天 $0.4^\circ$ 全球预报中，对 74% 目标优于 CAMS 业务系统，且计算成本小 5 个数量级以上。
  - 海浪：
    - 10 天 $0.25^\circ$ 全球预报中，对 86% 波浪变量优于 IFS HRES-WAM。
  - 热带气旋路径：
    - 在 2022–2023 全球所有热带气旋案例上，Aurora（基于 HRES T0 微调）在四个大洋盆地**全面优于多家业务中心官方预报**，这是首次 ML 模型在 5 天预报上超越完整业务系统。
  - 高分辨率天气：
    - 在 $0.1^\circ$ 分辨率下，对 >92% 目标优于 IFS HRES；
    - 在 WeatherReal-ISD 站点观测上的风速和温度评估中，对所有 lead time（至 10 天）优于 IFS HRES；
    - 是唯一能够准确预测 Ciarán 风暴最大 10 m 风速突增的 AI 模型。

- **缺测/掩码数据的显式支持**：
  - 在 HRES-WAM 海浪数据中，变量在海冰覆盖或陆地区域缺失；
  - Aurora 通过为每个变量追加“存在性通道”（density variables），实现对缺测的显式建模。

- **高效 roll-out 微调机制**：
  - LoRA + pushforward trick + replay buffer 组合，使得在高分辨率长 lead time 任务上微调大型模型成为可能，且参数高效。

#### 4.2 后续研究方向（论文中提到）

- **集合预报（Ensemble forecasting）**：
  - 目前使用单一确定性预测，未来可扩展为集合预报，用于高不确定性和长 lead time 场景。

- **进一步扩大预训练数据与模型规模**：
  - 标度结果表明尚未达到性能上限；
  - 扩充数据多样性和模型规模有望进一步提高微调任务表现。

- **端到端同化 + 直接观测驱动**：
  - 当前 Aurora 仍依赖传统数据同化系统提供初始场；
  - 未来可借鉴 end-to-end weather forecasting，将模型直接应用于观测数据，构建端到端系统。

- **可解释性研究**：
  - 探索模型内部学习到的模式是否可映射到具体物理过程。

- **扩展至更多地球系统任务**：
  - 潜在应用包括：海洋环流、区域/季节预测、植被与物候、洪水与野火、农业生产力、可再生能源产出、海冰范围等。

---

### 5. 实现细节与代码逻辑

#### 5.1 理论公式

##### 5.1.1 状态表示与自回归模拟器

- 大气–地表状态：
  - 观测状态：

    $$
    X^t \in \mathbb{R}^{V \times H \times W},
    $$

    其中 $V$ 为总变量数，$H, W$ 为纬向/经向格点数。

  - 分解为地表与大气部分：

    $$
    X^t = (S^t, A^t),
    $$

    $$
    S^t \in \mathbb{R}^{V_S \times H \times W}, \quad
    A^t \in \mathbb{R}^{V_A \cdot C \times \bar H \times W},
    $$

    其中 $V_S$ 是地表变量数，$V_A$ 是大气变量数，$C$ 是压力层数。

- 模拟器 $\Phi$：

  $$
  \Phi : (\mathbb{R}^{V \times H \times W})^2 \to \mathbb{R}^{V \times H \times W},
  $$

  $$
  \Phi(X^{t-1}, X^{t}) = \hat X^{t+1}.
  $$

- 多步自回归：

  $$
  \Phi(X^{t}, \hat X^{t+1}) = \hat X^{t+2},
  $$

  $$
  \Phi(\hat X^{t+k-2}, \hat X^{t+k-1}) = \hat X^{t+k}.
  $$

##### 5.1.2 训练损失（加权 MAE）

- 训练目标：对每个时间 $t$，最小化预测状态与真值的加权 MAE：

  $$
  \mathcal{L}(\hat X^t, X^t).
  $$

- 将预测与真值分别分解为地表与大气部分：

  $$
  \hat X^t = (\hat S^t, \hat A^t), \quad X^t = (S^t, A^t).
  $$

- 损失写为：

  $$
  \begin{aligned}
  \mathcal{L}(\hat X^t, X^t)
  &= \frac{\gamma}{V_S + V_A} \Bigg[ 
      \alpha \Big( \sum_{k=1}^{V_S} \frac{w_k^S}{H W} \sum_{i=1}^{H} \sum_{j=1}^{W} 
      \big| \hat S^{t}_{k,i,j} - S^{t}_{k,i,j} \big| \Big) \\
  &\quad + \beta \Big( \sum_{k=1}^{V_A} \frac{1}{C H W} \sum_{c=1}^{C} w_{k,c}^A \sum_{i=1}^{H} \sum_{j=1}^{W} 
      \big| \hat A^{t}_{k,c,i,j} - A^{t}_{k,c,i,j} \big| \Big) \Bigg],
  \end{aligned}
  $$

  其中：
  - $w_k^S$：地表变量 $k$ 的权重；
  - $w_{k,c}^A$：大气变量 $k$ 在压力层 $c$ 的权重；
  - $\alpha$：地表损失权重；
  - $\beta$：大气部分损失权重；
  - $\gamma$：数据集特定权重（用于多数据源混合训练时平衡不同数据集贡献）。

##### 5.1.3 预训练标度律（Validation Loss vs Model Size）

- 在 5,000 GPU 小时固定预算下，不同模型规模的验证损失经验上满足：

  $$
  \mathcal{L}(N) \propto N^{-0.026},
  $$

  其中 $N$ 为参数量，意味着**每增加 10 倍参数，验证损失约下降 6%**。

#### 5.2 实现描述（关键训练与推理步骤）

##### 5.2.1 预训练细节

- 优化与调度：
  - 优化器：AdamW；
  - 初始学习率：$5 \times 10^{-4}$，前 1,000 步线性 warm-up，自此后使用半余弦衰减至原来的 1/10；
  - 权重衰减：$5 \times 10^{-6}$；
  - 正则：stochastic depth（drop path）概率 0.2；
  - 精度：bf16 混合精度；
  - 内存：
    - 对 backbone 层做 activation checkpointing；
    - 跨 GPU 对梯度进行 shard。

##### 5.2.2 短时距微调与 roll-out 微调

- 短时距微调：
  - 在具体任务上，先通过 1–2 步 roll-out 对整个网络进行微调，使编码–backbone–解码在新变量/新分辨率上适配；

- roll-out 微调：
  - 使用 LoRA（低秩适配）微调 backbone 中自注意力相关的线性层，其他参数冻结；
  - pushforward trick：在长序列 roll-out 中，仅对最后一步反向传播，前几步只正向传播；
  - replay buffer：
    - 缓存若干起始状态与 roll-out 生成的后续状态；
    - 训练时从 buffer 中采样起点，预测下一步，更新 buffer；
    - 定期从原始数据集中刷新 buffer 中的起点，避免分布漂移。

##### 5.2.3 缺测值处理（海浪任务）

- 对 HRES-WAM 数据中“未定义”的时空位置（陆地或被海冰覆盖区域）：
  - 为每个物理变量增加一个“测量存在性”通道（density variable），指示该位置是否有有效值；
  - 模型在输入阶段同时接收物理值与存在性通道，从而在训练目标中仅对存在位置计算误差，并在预测时输出物理值 + 掩码。

##### 5.2.4 热带气旋路径提取

- 轨迹生成使用一个简单启发式追踪器：
  - 将每个预测时间步上的**最低海平面气压（MSLP）点**作为气旋中心；
  - 在连续时间步中连接这些中心点，形成路径；
  - 评估时与 IBTrACS 提供的真实路径对比计算 MAE。

#### 5.3 伪代码/代码片段

以下伪代码均为**基于文中描述生成的伪代码**。

##### 5.3.1 Aurora 预训练主循环（短 lead time）

```python
# 基于文中描述生成的伪代码

for step in range(num_steps):
    # 1) 从多数据源加载一个 batch
    X_t_minus_1, X_t, X_t_plus_1, meta = multi_source_loader.next_batch()
    # X_*: [V, H, W]

    # 2) 前一时刻与当前时刻作为输入
    inp_prev = X_t_minus_1
    inp_curr = X_t

    # 3) 编码: Perceiver-based encoder -> 3D latent
    z_prev = encoder(inp_prev)   # [H/P, W/P, L, D]
    z_curr = encoder(inp_curr)

    # 4) 组合成模型输入表示 (例如 concat 或差分, 细节略)
    z_in = combine_latents(z_prev, z_curr)

    # 5) Backbone: 3D Swin Transformer U-Net
    z_out = backbone(z_in)

    # 6) 解码到物理空间
    X_hat_t_plus_1 = decoder(z_out, target_meta=meta)

    # 7) 计算加权 MAE 损失
    loss = weighted_mae_loss(X_hat_t_plus_1, X_t_plus_1, meta)

    # 8) 反向传播与优化
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

##### 5.3.2 通用自回归 roll-out（推理阶段）

```python
# 基于文中描述生成的伪代码

def aurora_forecast(X_t_minus_1, X_t, num_steps, task_meta):
    preds = []
    prev = X_t_minus_1
    curr = X_t

    for k in range(num_steps):
        z_prev = encoder(prev)
        z_curr = encoder(curr)
        z_in   = combine_latents(z_prev, z_curr)
        z_out  = backbone(z_in)
        next_state = decoder(z_out, target_meta=task_meta)

        preds.append(next_state)
        prev, curr = curr, next_state

    return preds  # 长度为 num_steps 的预测序列
```

##### 5.3.3 海浪任务缺测掩码处理

```python
# 基于文中描述生成的伪代码

def build_wave_input(raw_wave_vars, raw_mask, raw_meteo_vars):
    # raw_wave_vars: [C_wave, H, W] with NaNs for missing
    # raw_mask: [C_wave, H, W] bool

    # 1) 用 0 或均值填充缺测位置
    wave_values = fill_missing(raw_wave_vars, raw_mask)

    # 2) 构造存在性通道 (density variable)
    density = raw_mask.astype(float)

    # 3) 合并波浪变量与密度通道、气象变量
    inp = concat_channels(wave_values, density, raw_meteo_vars)
    return inp  # 供 encoder 使用
```

##### 5.3.4 热带气旋轨迹追踪器

```python
# 基于文中描述生成的伪代码

def track_tc_from_forecast(mslp_forecasts, lats, lons):
    # mslp_forecasts: [T, H, W]  未来 T 步的海平面气压预报
    track = []
    for t in range(mslp_forecasts.shape[0]):
        p = mslp_forecasts[t]
        i, j = argmin_2d(p)  # 找到最小 MSLP 格点
        track.append((lats[i], lons[j]))
    return track
```

---

### 6. 数据规格

#### 6.1 预训练与总体数据来源

- Aurora 预训练使用多类大气/气候数据集的混合：
  - **ERA5 再分析**（单层与压力层）；
  - **IFS HRES 预报（高分辨率天气预报）**；
  - **IFS 集合预报（ENS）**；
  - **GFS 预报与 GFS T0**；
  - **GEFS 再预报**；
  - **CMIP6 气候模拟（如 CMCC-CM2-VHR4、IFS-HR 等）**；
  - **MERRA-2 大气再分析**；
  - **CAMS 预报、分析与再分析（EAC4）** 等。

- 预训练数据特点：
  - 包含 analysis、reanalysis、forecast、reforecast、climate simulation 五大类；
  - 覆盖不同分辨率、压力层、变量组合与时间跨度。

#### 6.2 下游任务数据与时间范围

- **大气化学与空气质量任务（CAMS）**：
  - 分辨率：
    - CAMS 分析：$0.4^\circ$；
    - CAMS 再分析 EAC4：$0.75^\circ$。
  - 时间范围：
    - 微调：
      - CAMS 分析：2017 年 10 月 – 2022 年 5 月（训练），2022 年 5 月 – 11 月（测试）；
      - CAMS 再分析 EAC4：2003 年 1 月 – 2021 年 12 月纳入微调过程，以弥补分析数据时间有限。
  - 目标变量：
    - 气体：CO, NO, NO\(_2\), SO\(_2\), O\(_3\)，各高度层及总柱量（TC）；
    - 颗粒物：PM1, PM\(_{2.5}\), PM\(_{10}\)。

- **海浪任务（IFS HRES-WAM）**：
  - 变量：
    - 显著波高 SWH；
    - 平均波周期 MWP；
    - 平均波向 MWD；
    - 以上变量针对：风浪（WW）、总涌（TS）、主涌（1）、次涌（2）；
    - 峰值波周期 PP1D；
    - 10 m 中性风分量 10UN, 10VN；
    - 具体变量列表见论文补充 Table C2（原文未完整列出）。
  - 分辨率：
    - HRES-WAM 分析与 HRES T0 统一重网格到 $0.25^\circ$。
  - 时间范围：
    - 微调：2016–2021 年（训练），2022 年（评估）。
  - 特点：
    - 部分变量在陆地与海冰区域缺测（undefined），通过 density 通道处理。

- **热带气旋路径任务**：
  - 模型输入：Aurora 在 HRES T0 上微调的天气模型，分辨率 $0.25^\circ$；
  - 真实轨迹：
    - IBTrACS v4r01 提供全球热带气旋最佳路径数据；
  - 评估时间范围：
    - 2022–2023 年所有全球热带气旋案例。

- **高分辨率天气任务（IFS HRES）**：
  - 分辨率：
    - $0.1^\circ$（TCo1279 Gaussian grid）；
  - 数据：
    - IFS HRES 分析与 HRES T0（零时效预报）；
  - 时间范围：
    - 分析数据：2016–2022 年；
  - 评价：
    - 对 HRES T0 做自评估（即 IFS 对自身零时效预报评估）；
    - 使用 WeatherReal-ISD 站点观测（2024 论文给出的基准），站点数 > 13,000。

#### 6.3 数据处理与存储

- 空间重网格：
  - 将不同来源数据统一到对应任务分辨率（$0.4^\circ$、$0.25^\circ$、$0.1^\circ$ 等）；
  - 对海浪任务，将 HRES-WAM 与 HRES T0 时间对齐并重采样到相同网格。

- 数据存储与加载基础设施：
  - 存储：Azure Blob Storage：
    - 数据与计算资源共址以降低延迟与成本；
    - 数据按合适块进行切分与压缩，减少下载量与并发连接数。
  - 加载：多源数据 pipeline：
    - 每类数据通过 YAML 配置生成相应 BatchGenerator 流；
    - 多个流合并、打乱、按 GPU 分片；
    - 不同数据集根据自身大小/分辨率使用不同 batch size，实现负载均衡。

- 归一化与缺失值：
  - 具体归一化方法与缺失值处理（除海浪缺测掩码外）的细节集中在补充材料中，正文未给出具体公式与参数，此处不做推测。

---

## 维度二：基础组件知识（Component-Level Knowledge）

> 下述组件是根据论文明确描述抽取的核心模块；凡缺乏精确数学定义之处只作文字层面总结，不作额外推断。

### 组件 1：3D Perceiver Encoder（多源 3D 编码器）

#### 1. 子需求定位
- 对应的子需求：
  - 统一处理来自不同数据集、包含不同变量、压力层数和分辨率的输入，将其映射到**固定形状的 3D 潜在表示**，供 backbone 使用。
- 解决的问题：
  - 数值天气/气候产品在变量集合、层结构与网格分辨率上差异巨大，难以用单一网络直接处理；
  - 需要支持静态变量（地形、陆海掩膜、土壤类型）与动态场协同建模。

#### 2. 技术与创新
- 关键技术点：
  - 将所有变量视为 $H\times W$ 图像；
  - 静态变量（orography、land–sea mask、soil-type mask）作为额外 surface 变量叠加；
  - 图像划分为 $P\times P$ patch，经变量特定线性层映射为维度 $D$ 的 embedding；
  - 对 surface 与每个物理压力层：
    - 不同变量的 embedding 求和，形成该层的综合表示；
    - 添加“压力层编码”（pressure-level encoding）或 surface 专用学习向量；
  - 使用 Perceiver 模块通过 cross-attention 将不定数目的物理压力层 $C$ 压缩为固定 $L$（例如 3）个**潜在压力层**；
  - 在 patch 维度上得到 $\frac{H}{P} \times \frac{W}{P} \times L$ 的 3D 表示；
  - 对该 3D 表示添加：
    - patch 位置编码（空间）；
    - patch area 编码（尺度：支持多分辨率）；
    - 绝对时间编码（Fourier expansion，选取合适的波长范围以捕捉相应尺度）。

- 创新点：
  - 通过 Perceiver 将**任意物理压力层集合**映射为固定潜在层数，从而使 backbone 完全与输入压力层数解耦；
  - 通过“patch area encoding”明确注入网格面积/分辨率信息，使同一模型可在 $0.4^\circ$、$0.25^\circ$、$0.1^\circ$ 等不同分辨率下运行。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：多源数据加载与重网格；
  - 后接：3D Swin Transformer U-Net backbone；
- 绑定程度：
  - **强绑定-气象特异**：
    - 组件形式（patch + Perceiver）在视觉/多模态任务中是通用的；
    - 但具体设计紧密围绕“压力层–surface 分解”、“地形与陆海掩膜”等大气/海洋任务特点。

#### 4. 实现描述与伪代码

> 内部 Perceiver cross-attention 的细节在文中由引用给出，此处只描述整体流程。

```python
# 基于文中描述生成的伪代码

def perceiver_encoder(X, static_vars, meta):
    # X: 动态变量 [V_dyn, H, W]
    # static_vars: 静态变量 (orography, land-sea, soil-type) [V_static, H, W]
    # meta: 包含每个变量是否为 surface / 哪个 pressure level 等元信息

    X_all = concat_channels(X, static_vars)

    # 1) patch 划分与变量特定线性嵌入
    patches = []  # 将每个变量和每个 pressure level 分开处理
    for var_id in range(X_all.shape[0]):
        img = X_all[var_id]            # [H, W]
        p_tokens = patchify_and_linear_embed(img, var_id)
        patches.append(p_tokens)       # [N_patches, D]

    # 2) 对 surface 与每个 pressure level: 变量嵌入求和 + 层编码
    level_embeds = aggregate_by_pressure_level(patches, meta)
    # level_embeds: list of [N_patches, D] for each physical level

    # 3) Perceiver: 将物理层数 C 压缩为潜在层数 L
    latent_levels = perceiver_compress(level_embeds, target_L=3)
    # latent_levels: [L, N_patches, D]

    # 4) reshape 为 3D 表示并添加 Fourier 编码
    H_p, W_p = H // P, W // P
    latent_3d = latent_levels.reshape(L, H_p, W_p, D)
    latent_3d = add_fourier_encodings(latent_3d, meta.time, meta.patch_area)

    return latent_3d  # [L, H_p, W_p, D]
```

---

### 组件 2：Multiscale 3D Swin Transformer U-Net Backbone

#### 1. 子需求定位
- 对应的子需求：
  - 在统一 3D 潜在空间中高效地模拟大气/海洋动力及相关二阶过程的时空演化；
  - 同时捕捉多尺度（水平 + 垂直 + 时间）结构，并兼顾计算效率与可扩展性。

- 解决的问题：
  - 传统 Transformer 在大分辨率 3D 网格上的计算成本过高；
  - 需要 U-Net 风格的多尺度表示以模拟从行星尺度环流到中尺度/对流尺度过程。

#### 2. 技术与创新
- 关键技术点：
  - 采用**3D Swin Transformer U-Net**：
    - 下采样阶段（encoder）：逐步在 3D 空间下采样抽取高层特征；
    - 上采样阶段（decoder）：将多尺度特征融合并恢复到原始分辨率；
    - 跨层 skip connection 支持细节恢复。
  - Swin Transformer 层特点：
    - 在 3D 窗口内执行局部自注意力；
    - 每隔一层对窗口进行“移位”，使跨窗口信息传播；
    - 在窗口划分与移位过程中考虑地球球面拓扑。
  - 深度：
    - 48 层 / 3 stages，比先前 3D 气象 Swin/ViT 架构（如 Pangu-Weather 使用的 16 层 2 stage）更深；
    - 深度得益于 encoder 压缩后的低维潜在表示。
  - Normalization：res-post-norm（残差后归一化）以提高深层网络训练稳定性。

- 创新点：
  - 将 3D Swin U-Net 明确用于**地球系统 3D 潜在状态的模拟器**，并与 Perceiver encoder/decoder 紧密结合；
  - 通过潜在层压缩 + Swin 的局部窗口注意力，使深层大模型在高分辨率任务上的训练成为可能。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：3D Perceiver Encoder 输出的 $[L, H/P, W/P, D]$ 潜在表示；
  - 后接：Perceiver Decoder 将 backbone 输出的 3D 表示解码为具体物理变量。

- 绑定程度：
  - **弱绑定-通用**：
    - 3D Swin U-Net 是通用视觉/时空建模架构；
    - 在本工作中通过窗口设置与球面处理适配地球系统，但结构本身可迁移到其他 3D 场景任务。

#### 4. 实现描述与伪代码

> Swin window attention 的细节由原论文给出，此处只描述整体 U-Net 流程。

```python
# 基于文中描述生成的伪代码

class AuroraBackbone3D(nn.Module):
    def __init__(self, depth_config, dim_config, window_size):
        super().__init__()
        # depth_config: 每个 stage 的层数 (e.g. [4, 8, 8, 8, 8, 12])
        # dim_config: 对应的通道数
        # 省略具体初始化

    def forward(self, z_in):
        # z_in: [L, H_p, W_p, D]

        # Encoder stages
        enc_feats = []
        x = z_in
        for stage in self.enc_stages:
            x = stage(x)  # 若干 3D Swin blocks + downsample
            enc_feats.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder stages (with skip connections)
        for stage, skip in zip(self.dec_stages, reversed(enc_feats)):
            x = stage.upsample(x)
            x = concat_channels(x, skip)
            x = stage.blocks(x)  # 若干 3D Swin blocks

        return x  # [L, H_p, W_p, D]
```

---

### 组件 3：3D Perceiver Decoder（多任务 3D 解码器）

#### 1. 子需求定位
- 对应的子需求：
  - 将 backbone 输出的统一 3D 潜在表示，按照下游任务所需的变量、压力层和分辨率，恢复为物理网格预测场。

- 解决的问题：
  - 不同任务需要不同的压力层集合和变量组合；
  - 需要在不同空间分辨率（$0.4^\circ$、$0.25^\circ$、$0.1^\circ$ 等）和 patch 大小下生成物理场。

#### 2. 技术与创新
- 关键技术点：
  - 使用 Perceiver 层将潜在压力层解聚到任意指定的物理压力层集合；
  - 对每个目标变量使用**变量特定线性层**将 3D 潜在表示映射为 patch embedding；
  - 通过 patch 反组装 / 上采样生成目标网格上的物理场；
  - 支持动态配置下游任务输出变量集合（大气化学、波浪、天气等）。

- 创新点：
  - 与 encoder 对称的 Perceiver 结构，使 Aurora 能够灵活适配任意变量/层/分辨率要求，而无需改 backbone。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：AuroraBackbone3D 输出的 $[L, H/P, W/P, D]$ 潜在表示；
  - 后接：任务特定评估（RMSE、ACC 等）。

- 绑定程度：
  - **弱绑定-通用**：
    - 架构是通用的 Perceiver 解码思路；
    - 输出变量定义和分辨率选择则是地球系统特定的。

#### 4. 实现描述与伪代码

```python
# 基于文中描述生成的伪代码

def perceiver_decoder(z_latent, target_meta):
    # z_latent: [L, H_p, W_p, D]
    # target_meta: 指定任务的目标压力层和变量集合

    # 1) 使用 Perceiver 将潜在压力层映射到目标压力层集合
    z_phys_levels = perceiver_expand(z_latent, target_levels=target_meta.levels)
    # [C_target, H_p, W_p, D]

    # 2) 对每个目标变量使用变量特定线性层生成 patch 表示
    outputs = []
    for var in target_meta.variables:
        var_tokens = linear_var_head(z_phys_levels, var)
        # 3) 从 patch 还原到物理网格
        var_field = unpatchify_to_grid(var_tokens, target_meta.grid)
        outputs.append(var_field)

    X_hat = concat_channels(outputs)
    return X_hat
```

---

### 组件 4：LoRA + Roll-out Fine-tuning 机制

#### 1. 子需求定位
- 对应的子需求：
  - 在高分辨率（尤其是 $0.1^\circ$）和长 lead time 任务上，对大模型进行微调，而不引入巨大的额外参数与内存消耗。

- 解决的问题：
  - 直接在全参数上进行长序列 roll-out 微调几乎不可行；
  - 需要在保持预训练表示能力的前提下，低成本适配新任务/新分辨率。

#### 2. 技术与创新
- 关键技术点：
  - 在 backbone 自注意力中的线性层（如 $W_Q, W_K, W_V, W_O$）上应用 LoRA：
    - 将增量参数限制在低秩矩阵，显著减少可训练参数量；
  - 使用 pushforward trick：
    - 在 roll-out 训练时，只对最后一个时间步反向传播；
  - 使用 replay buffer：
    - 存储 roll-out 过程中的中间状态，重复利用以提升计算利用率。

- 创新点：
  - 在地球系统长时序模拟场景中系统化应用 LoRA+replay buffer，使得大模型长时微调可行。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：预训练好的 Aurora 模型权重；
  - 后接：下游任务（高分辨率天气、海浪等）的长 lead time 预测能力提升。

- 绑定程度：
  - **弱绑定-通用**：
    - LoRA 与 replay buffer 是通用的深度学习技巧；
    - 在本工作中专门用于地球系统长序列预测。

#### 4. 实现描述与伪代码

```python
# 基于文中描述生成的伪代码

for epoch in range(num_epochs):
    for _ in range(num_rollouts_per_epoch):
        # 从 replay buffer 采样起点
        X_t_minus_1, X_t = replay_buffer.sample_initial()

        # roll-out 若干步 (仅前向)
        states = [X_t_minus_1, X_t]
        for _ in range(K):  # K 步 roll-out
            X_next = aurora_step(states[-2], states[-1])
            states.append(X_next)

        # 仅对最后一步计算损失和反向传播
        X_pred = states[-1]
        X_true = get_ground_truth_for_last_step()
        loss = weighted_mae_loss(X_pred, X_true)
        loss.backward()
        optimizer.step(); optimizer.zero_grad()

        # 用新预测更新 replay buffer
        replay_buffer.update(states)
```

---

### 组件 5：任务特定头与数据适配（空气质量 / 海浪 / 天气 / 台风）

#### 1. 子需求定位
- 对应的子需求：
  - 在统一 Aurora backbone 上，为不同任务配置合适的输出变量、分辨率和损失权重，并进行相应的微调与评估。

- 解决的问题：
  - 不同业务系统（CAMS、HRES-WAM、IFS HRES 等）的变量集合与评估协议不同；
  - 需要针对每个任务进行合理输入/输出映射与损失加权配置。

#### 2. 技术与创新
- 空气质量任务：
  - 模型六种污染物（CO, NO, NO\(_2\), SO\(_2\), O\(_3\) 及 PM1/2.5/10），兼顾背景值与极大值；
  - 处理分布异质、稀疏以及动态范围大的问题；
  - 在数据有限（CAMS 系统较新）的情况下联合使用 EAC4 再分析增强训练。

- 海浪任务：
  - 将波浪变量与气象变量（HRES T0）对齐、重网格并联合微调；
  - 引入 density 通道处理海冰与陆地区域缺测，扩展 Aurora 对缺失数据的支持。

- 台风路径任务：
  - 在未特定为台风任务微调的 HRES T0 模型基础上，通过简单 MSLP tracker 提取路径；
  - 在 4 个大洋盆地上整体优于官方预报与多个关键模型。

- 高分辨率天气任务：
  - 通过预训练 + LoRA 微调，将模型适配到 $0.1^\circ$ IFS HRES 分析数据；
  - 在 WeatherReal-ISD 站点数据集上评估 10 m 风速与 2 m 温度，优于 IFS HRES。

- 创新点：
  - 展示了“单一基础模型 + 任务特定头 + 轻量微调”在多种高价值业务问题上的可行性与优势。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：共享的 Aurora encoder–backbone–decoder；
  - 后接：各自任务的业务评估和极端事件案例分析（沙尘暴、台风、强风暴等）。

- 绑定程度：
  - **强绑定-气象特异**：
    - 任务头和配置完全围绕特定地球系统预测问题设计。

#### 4. 实现描述与伪代码

> 各任务头主要在变量选择与损失配置层面，与前文伪代码“解码器 + 损失函数”组合即可，此处不再重复。

---

### 组件 6：验证与极端事件评估指标

#### 1. 子需求定位
- 对应的子需求：
  - 为全球、跨任务的预测提供统一的评估指标体系，并特别评估**极端事件**表现。

- 解决的问题：
  - 不同数据集与任务若使用不同评估方式，难以横向比较；
  - 需要关注高影响极端事件（强风、极端温度、极端波高、严重污染等）的表现。

#### 2. 技术与创新
- 关键技术点：
  - 采用纬度加权的 RMSE 与 anomaly correlation coefficient（ACC）；
  - 提出 thresholded RMSE：
    - 仅对超过某一阈值（由 ERA5 训练年均值 ± 标准差在各格点定义）的格点计算 RMSE；
    - 通过改变阈值，得到覆盖不同强度等级现象的 RMSE 曲线。

- 创新点：
  - 将极端事件评估体系系统性引入 AI 基础模型评估流程（如 Storm Ciarán、沙尘暴案例等），并与 IFS HRES/GraphCast/FourCastNet/Pangu-Weather 等模型对比。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：各任务的预测输出及相应真值产品（ERA5、HRES、CAMS、HRES-WAM 等）；
  - 后接：论文中的 scorecard、对比图和案例分析。

- 绑定程度：
  - **弱绑定-通用**：
    - RMSE/ACC 是通用度量；
    - 阈值定义与极端事件分析更贴近地球系统应用。

#### 4. 实现描述（抽象）

> 具体数学公式集中在补充材料中，此处不重复，仅说明：
> - 纬度加权：对每一纬带使用 cos(lat) 进行归一化加权；
> - 异常场：使用 ERA5 训练期气候平均场定义 anomaly；
> - 阈值 RMSE：在 anomaly 或物理量超过给定阈值的格点上计算 RMSE。

---

**说明**：以上所有伪代码均为“基于文中描述生成的伪代码”，仅用于复现论文逻辑结构与数据流，不代表官方实现。