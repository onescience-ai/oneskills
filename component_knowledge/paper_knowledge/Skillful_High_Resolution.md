# Skillful High-Resolution Ensemble Precipitation Forecasting with an Integrated Deep Learning Framework

> 说明：本文件基于论文文字内容进行结构化总结和高层伪代码抽象，不是官方实现代码。

---

## 一、整体模型维度（任务 / 数据 / 结构 / 训练 / 结果）

### 1.1 任务与问题设定

- **场景**：
  - 面向中国及周边区域（0–60°N, 70–140°E）的**高分辨率集合降水预报**；
  - 空间分辨率：从 0.25°（ERA5 / ECMWF IFS / data-driven 模型）提升到 0.05°；
  - 时间分辨率：1 小时降水；预报时效最长可达 5 天（120 小时）。
- **目标**：
  - 给定粗分辨率的大气场（温度、风、湿度、位势高度、MSLP 等），预测高分辨率（0.05°）降水场；
  - 输出为**集合预报**：多成员降水场集合 E，用于刻画小尺度对流降水的不确定性。
- **物理动机**：
  - 借鉴 NWP 中“平均 + 扰动”的分解思想：
    $$X_t = \bar{X}_t + X'_t,$$
    其中 \(\bar{X}_t\) 为网格平均，\(X'_t\) 为未解析的次网格尺度过程；
  - 对降水而言：
    - **中尺度平均降水**（mesoscale mean）可由确定性模型刻画；
    - **对流尺度残差降水**（convective-scale residual）稀疏、随机、强非线性，适合用概率生成模型来表示。

### 1.2 数据与预处理

1. **ERA5 再分析（驱动输入 + 中尺度降水基线）**
   - 源：ECMWF IFS reanalysis；
   - 空间：0.25° 等经纬网，区域 0–60°N, 70–140°E（241×281 点）；
   - 变量：
     - 上空 5 个变量 × 13 个等压层：
       - Geopotential (Z), Temperature (T), U/V wind, Specific humidity (SH)；
       - 压力层：50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa；
     - 地面 5 个变量：
       - 2 m temperature (T2M), 10 m U/V wind (U10, V10), mean sea level pressure (MSLP), total precipitation (TP)。

2. **CMPA 高分辨率降水（目标真值）**
   - 源：CMA CMPA 融合产品；
   - 原分辨率：0.01° × 0.01°，1 小时；覆盖 15–60°N, 70–140°E（4500×7000 网格）；
   - 由地面站 + 雷达 QPE + 卫星降水融合得到；
   - 为降低计算量，通过 `AvgPool2d` 下采样到 0.05° 分辨率（约 900×1400）；
   - 该尺度作为“高分辨率降水真值”。

3. **ECMWF 实时预报（业务推理时的驱动）**
   - 分辨率：0.25°，IFS 实时中期预报；
   - 时效：120 小时（5 天），3 小时步长；
   - 初始时刻：00、12 UTC；
   - 在实况阶段模型用 ERA5 训练，但推理时可直接用 ECMWF 预报作为输入（或将 ERA5 换成任意 data-driven 模型输出，如 Pangu、GraphCast、FuXi）。

4. **降水预处理（TP → dBZ 归一化）**
   - 将 1 小时累积降水 TP (mm) 转为反射率 dBZ（雷达风格）：
     $$dBZ = 10 \log_{10} (200\,TP^{1.6}).$$
   - 进一步做空间归一化：
     $$dBZ^{scale} = \frac{1}{N} \sum_{t=1}^N \max_{i,j} X_{t,i,j},$$
     用历史时序上的空间最大值平均来缩放，使得 dBZ 落入 [0,1] 附近范围。

### 1.3 物理驱动的降水分解

- 高分辨率降水 TP^HR 被表示为：
  $$TP_t^{HR} = \overline{TP}_t + TP'_t,$$
  其中：
  - \(\overline{TP}_t\)：中尺度平均降水（由确定性模型从 ERA5 等粗格点场预测）；
  - \(TP'_t\)：对流尺度残差降水（由概率 diffusion 模型预测）。
- 残差的构造：
  - 将 ERA5 TP 最近邻插值到 0.05° 网格，与 CMPA 高分辨率 TP 对齐；
  - 对缩放后的降水场求差 \(TP'_t = TP_t^{CMPA,scaled} - TP_t^{ERA5,scaled}\)，作为残差真值；
  - 这样：
    - 确定性部分近似承担“ERA5 校正 + 初步细化”；
    - 概率部分承担“次网格随机对流补偿”。

### 1.4 模型整体架构

框架由**确定性+概率**两个子模型组成（参见论文图 2）：

1. **确定性模型（Deterministic mesoscale model）**
   - 主干：3D Swin Transformer；
   - 输入：两个连续时刻的大气状态 \(X_{t-1}, X_t\)：
     - Surface: T2m, U10m, V10m, MSLP（4×241×281）；
     - Upper-air: (T, U, V, SP, Z) at 13 levels（5×13×241×281）；
     - 静态特征：land/sea mask、orography、soil type、纬度/经度；
     - 时间特征：本地时间的 sin/cos、年进度的 sin/cos；
   - 输出：1 小时平均降水 \(\overline{TP}_t\) （0.05° 分辨率）；
   - 核心改进：
     - 非线性 patch embedding；
     - 3D Swin-Transformer 处理 (变量通道 × 高度层 × 空间)；
     - 高质量上采样模块（两种 Upsampler 方案对比）；
     - 加权 MSE+SSIM 损失提升中强降水预测质量。

2. **概率模型（Probabilistic residual model）**
   - 步骤：
     1. VAE 将残差降水 \(TP'_t\) 从 1×900×1400 编码为 latent z ∈ R^{16×90×140}；
     2. 在 latent 空间训练**条件扩散模型（conditional latent diffusion）**：
        - 条件：大气状态 + mean 降水；
        - 主干：DiT（Diffusion Transformer）；
     3. 采样时重复采样 latent z_T 并反向去噪多次，得到多份残差场 TP'_i；
     4. 组合：TP_i^{HR} = \overline{TP}_t + TP'_i，形成集合成员。

- **集合定义**：
  $$E = \{TP_i \mid TP_i = \overline{TP} + TP'_i,\ i = 1..11\},$$
  文中使用 11 个集合成员。

### 1.5 训练设置

1. **确定性模型**
   - 训练数据：ERA5（含派生/静态/时间特征），2001–2019；
   - 损失：weighted MSE + SSIM：
     $$Loss = \lambda_1 (\hat{X}_{ij} - X_{ij})^2 + \lambda_2 (1 - SSIM(\hat{X}_{ij}, X_{ij})), $$
     其中 λ1=0.5, λ2=1.5；
   - 消融实验：
     - baseline：MSE，标准 embedding，无 ST 特征；
     - exp-d1：MSE+SSIM；
     - exp-d2：+ 静态 & 时间特征；
     - exp-d3：+ 非线性 patch embedding；
     - exp-d4：+ 改进 Upsampler（SwinIR 风格图像重建）。

2. **概率模型**
   - 数据期：
     - 训练：2018–2019；
     - 验证：2020；
     - 测试：2021；
   - VAE：基于 LDM 风格的卷积 VAE，带对抗训练与 KL 正则；
   - Diffusion：
     - DDPM 训练目标：\(\mathbb{E}\|\epsilon - \epsilon_\theta(z_t, t, cond)\|^2\)；
     - 采样：DDIM，1000 训练步中，推理采用约 300 采样步；
     - 条件：ERA5 surface+upper-air + deterministic TP 预测（推理时）/ ERA5 TP（训练时）。

### 1.6 结果与结论（定性）

- **CSI（Critical Success Index）指标**：
  - 确定性模型在引入加权 MSE+SSIM、ST 特征、非线性 embedding、改进 upsampler 后，
    - 随降水阈值升高（2/5/10/15/20 mm/h），CSI 提升愈发显著；
    - 对 10–20 mm/h 的中到大雨，提升幅度较大（>30–50%）。
- **集合质量**：
  - Rank histogram 接近平坦：集合系统总体无明显偏差，既不过窄（under-dispersed）也不过宽（over-dispersed）；
  - 集合均值与 CMPA 分布更接近，尤其是强降水带的位置与结构。
- **极端事件案例**：
  - 南方一次强降水个例中：
    - 本框架输出比 ERA5 更接近 CMPA 实况，能捕捉降水带细节和最大值位置；
    - 集合成员展现不同强对流细节，提供更丰富的不确定性刻画。
- **实时系统表现**：
  - 基于 ECMWF 实时预报驱动的 5 天预报系统：
    - 多阈值 CSI 在预报时效上保持较好性能；
    - 体现对 operational setting 的可行性和稳健性。

---

## 二、组件维度：模块拆解与伪代码

### 2.1 数据预处理与降水分解组件

**子需求**：
- 对 ERA5 / CMPA / ECMWF 进行区域裁剪、插值、尺度统一；
- 将降水从 mm 转成雷达风格 dBZ 并归一化；
- 构造 mean + residual 表示，供后续确定性/概率模型使用。

**关键设计**：
1. 区域裁剪与重采样：
   - ERA5 0.25° → 0.05°：使用最近邻插值（保持降水总量结构）；
   - CMPA 0.01° → 0.05°：使用平均池化（AvgPool2d），相当于局地平均；
2. 反射率变换与空间缩放：
   - mm → dBZ：通过 Z–R 关系；
   - 归一化因子 dBZ^{scale} 通过历史最大值的时间均值估计；
3. 残差构造：
   - 在统一 0.05° 网格上：
     $$TP'_t = TP_t^{CMPA,scaled} - TP_t^{ERA5,scaled}.$$

**伪代码框架**：

```python
def preprocess_precip(tp_era5, tp_cmpa):
    # 1) regrid ERA5 to 0.05°
    tp_era5_hr = nearest_neighbor_regrid(tp_era5, target_res=0.05)

    # 2) downsample CMPA from 0.01° to 0.05°
    tp_cmpa_hr = avgpool_downsample(tp_cmpa, factor=5)  # 5x5 blocks

    # 3) mm -> dBZ
    def mm_to_dbz(tp):
        return 10 * torch.log10(200 * (tp.clamp(min=1e-6)) ** 1.6)

    dbz_era5 = mm_to_dbz(tp_era5_hr)
    dbz_cmpa = mm_to_dbz(tp_cmpa_hr)

    # 4) spatial scaling
    scale = spatial_max_time_mean(dbz_cmpa)
    dbz_era5_scaled = dbz_era5 / scale
    dbz_cmpa_scaled = dbz_cmpa / scale

    # 5) residual
    residual = dbz_cmpa_scaled - dbz_era5_scaled
    return dbz_era5_scaled, dbz_cmpa_scaled, residual
```

---

### 2.2 确定性 3D Swin Transformer 降水诊断组件

**子需求**：
- 利用粗分辨率 ERA5 大气状态预测 0.05° 网格上的 mean 降水 \(\overline{TP}_t\)；
- 重点提升中到大雨强度、空间结构与 CSI，减轻模糊问题。

**输入结构**：
- 变量堆叠：
  - Surface：4×241×281（T2m, U10, V10, MSLP）；
  - Upper-air：5×13×241×281（T, U, V, SH, Z）；
  - 时间：\(X_{t-1}, X_t\) 两个时刻；
  - 静态特征：LSM, orography, soil type, lat/lon；
  - 时间特征：time-of-day, day-of-year 的 sin/cos。

**非线性 patch embedding**：
- 每个变量单独卷积 + GELU → MLP + GELU，再在 patch 维度融合：
  - Surface: kernel 4×4；
  - Upper-air: kernel 2×4×4（包含高度维）。
- 相比“单一卷积 + 线性投影”的标准 embedding，更充分利用变量间非线性关系。

**3D Swin Transformer 结构**：
- 输入特征尺寸：C×8×61×71（C 为通道/特征维度，8 为"类高度"维，总计 surface+upper-air）；
- Layer1：3 个 Swin3D block + patch merging → 2C×8×31×36；
- Layer2：9 个 block → 再上采样回 C×8×61×71；
- Layer3：3 个 block；
- Skip connection：
  - Layer1 输出加到 Layer2 输出；
  - Layer1 + Layer2 加到 Layer3 输出。

**Upsampler（两种方案）**：
1. Upsampler1：
   - 3D Conv (kernel 8×1×1) → bilinear upsample（高度/宽度×2）→ 2D Conv + upsample → 恢复到 241×281。
2. Upsampler2（更优）：
   - 3D Conv (8×1×1) + GELU 将高度维压到 1；
   - 模仿 SwinIR 图像重建：若干卷积 + LeakyReLU + pixel-shuffle / 上采样，细化纹理；
   - 更适合恢复细粒度降水结构。

**损失：加权 MSE + SSIM**：

```python
loss = lambda1 * F.mse_loss(y_hat, y_true) + lambda2 * (1 - ssim(y_hat, y_true))
```

**前向伪代码**：

```python
class DeterministicPrecipModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.patch_embed_surface = NonLinearPatchEmbed(...)
        self.patch_embed_upper   = NonLinearPatchEmbed(...)
        self.swin3d_layers = Swin3DBackbone(...)
        self.upsampler = Upsampler2(...)

    def forward(self, X_t_minus_1, X_t, static_feats, time_feats):
        # 1) stack inputs & features
        x = build_input_tensor(X_t_minus_1, X_t, static_feats, time_feats)
        # 2) patch embedding (surface + upper-air)
        z_surf = self.patch_embed_surface(x["surface"])
        z_ua   = self.patch_embed_upper(x["upper_air"])
        z = torch.cat([z_surf, z_ua], dim=1)  # (B, C, H', W') with pseudo-height
        # 3) 3D Swin Transformer
        z = self.swin3d_layers(z)
        # 4) Upsample to 0.05° grid
        tp_mean = self.upsampler(z)  # (B, 1, H_hr, W_hr)
        return tp_mean
```

---

### 2.3 残差降水 VAE 编码组件

**子需求**：
- 将高分辨率残差 TP' (1×900×1400) 压缩到一个更小的 latent 表示 (16×90×140)；
- 在保持关键信息的基础上，为 diffusion 降低计算量。

**VAE 结构要点**：
- Encoder：
  - 4 层 2D Conv + ResNet block：前三层带下采样（stride=2），每层两次残差块 + 下采样；
  - 最后层仅残差，不再下采样；
  - 由于 900×1400 不是 2 的幂，使用插值将中间特征对齐到目标 90×140。
- Decoder：
  - 结构镜像 encoder：上采样 + 残差块；
  - 最终插值到 900×1400。
- 损失：重构 L1/L2 + KL 正则（对齐 N(0, I)）+ adversarial loss（提升视觉质量）。

**伪代码框架**：

```python
class ResidualVAE(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.encoder = Encoder2D(...)
        self.decoder = Decoder2D(...)

    def encode(self, tp_residual):  # (B, 1, 900, 1400)
        mu, logvar = self.encoder(tp_residual)
        z = reparameterize(mu, logvar)  # (B, 16, 90, 140)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)  # (B, 1, 900, 1400)
```

---

### 2.4 条件 latent diffusion（DiT）组件

**子需求**：
- 在 latent 空间中对残差进行概率建模：\(p_\theta(z_{0:T} \mid cond)\)；
- 条件输入应包含：当前大气状态 + mean 降水（训练时用 ERA5 TP，推理时用 deterministic TP）。

**训练目标**：
- DDPM 样式噪声预测损失：
  $$\mathcal{L} = \mathbb{E}_{embedding(x), y, \epsilon, t} \|\epsilon - \epsilon_\theta(z_t, t, cond)\|^2,$$
  其中：
  - x：条件（大气状态 X_t）；
  - y：残差 TP' 的 latent 表示；
  - cond：条件 embedding，来自 x 和 mean TP；
  - \(z_t\)：y 在第 t 步加入噪声后的 latent；
  - \(\epsilon_\theta\)：DiT 主干，预测从 z_t 到 z_{t-1} 的噪声。

**DiT 架构要点**：
- Patch embedding：
  - cond：4×4 patch；
  - latent z_t：2×2 patch；
  - patch 后空间尺寸统一为 45×75；
- 将 cond 和 noisy latent 在 channel 维度拼接，经过 Transformer block；
- 通过时间步嵌入（t embedding）调制网络（类似 DiT / DDPM++）。

**伪代码（训练单步）**：

```python
def train_diffusion_step(vae, dit, tp_residual, atm_state, tp_mean):
    # 1) encode residual to latent
    z0, mu, logvar = vae.encode(tp_residual)

    # 2) build cond from atm_state + tp_mean
    cond = build_cond_embedding(atm_state, tp_mean)

    # 3) sample timestep and noise
    B = z0.size(0)
    t = sample_timesteps(B)
    eps = torch.randn_like(z0)
    z_t = q_sample(z0, eps, t)

    # 4) predict noise
    eps_hat = dit(z_t, t, cond)

    loss = F.mse_loss(eps_hat, eps)
    loss.backward()
    optimizer_dit.step()
```

---

### 2.5 集合采样与后处理组件

**子需求**：
- 从 learned latent diffusion 中多次采样残差场 TP'_i，叠加 deterministic mean 得到集合成员；
- 可以使用 DDIM 等更快的采样器以控制成本。

**集合采样伪代码**：

```python
@torch.no_grad()
def generate_ensemble(vae, dit, det_model, atm_sequence, static_feats, time_feats,
                      n_members=11, n_steps=300):
    # 1) deterministic mean precip
    tp_mean = det_model(...atm_sequence..., static_feats, time_feats)  # (B, 1, H_hr, W_hr)

    # 2) build condition for diffusion
    cond = build_cond_embedding(atm_sequence[-1], tp_mean)

    ensemble = []
    for k in range(n_members):
        # init latent from N(0, I)
        z_t = torch.randn(B, 16, 90, 140, device=tp_mean.device)
        for t in reversed(range(1, n_steps+1)):
            t_batch = torch.full((B,), t, device=tp_mean.device, dtype=torch.long)
            eps_hat = dit(z_t, t_batch, cond)
            z_t = ddim_step(z_t, eps_hat, t_batch)  # 单步更新
        z0 = z_t
        tp_residual = vae.decode(z0)
        tp_member = tp_mean + tp_residual
        ensemble.append(tp_member)

    # stack ensemble members: (B, n_members, 1, H_hr, W_hr)
    return torch.stack(ensemble, dim=1)
```

---

### 2.6 集合评估与可靠性分析组件

**子需求**：
- 评估集合的**可靠性**与**分布合理性**；
- 核心工具：Rank histogram、阈值CSI。

**Rank histogram 计算要点**：
- 对每个格点(i,j)和时间 t：
  1. 收集 N 个集合成员的预测值：\(\{TP_{i,j,t}^{(k)}\}_{k=1}^N\)；
  2. 升序排列；
  3. 找到实况值 TP^{obs}_{i,j,t} 在排序后的区间中的 rank r∈{0..N}；
  4. 统计所有(i,j,t) 的 rank 频数，绘制直方图。
- 形状解释：
  - 平坦：集合系统无偏、离散度合理；
  - U 型：离散度不足（under-dispersed），实况多落在集合外缘；
  - ∩ 型：离散度过大（over-dispersed）。

**CSI（各阈值）**：
- 针对 threshold θ，定义：
  - 观测/预报是否≥θ 的命中/漏报/虚警统计；
- 对 deterministic 模型：直接对单一场计算 CSI；
- 对 ensemble：可对集合均值或某分位（如 90th percentile）计算 CSI，或用概率化指标（Brier score 等）。

---

### 2.7 可迁移建模范式总结

1. **范式：Mean + Residual 的物理分解**
   - 将预报变量拆为“中尺度平均 + 次网格残差”：
     - 平滑、确定性的部分由连续预测模型（Transformers/UNet）负责；
     - 稀疏、随机的对流尺度部分由生成式/扩散模型负责；
   - 该模式可迁移到：
     - 风场（大尺度风 + 阵风残差）、
     - 温度（背景场 + 局地城市热岛残差）等。

2. **范式：Deterministic backbone + Probabilistic head**
   - 先利用一个强大的 deterministic backbone 生成“最佳估计”（mean）；
   - 再在 residual 空间中引入 diffusion / VAE / GAN，提供不确定性与细节纹理；
   - 组合后既有 skillful mean，又有合理 spread。

3. **范式：Latent diffusion for high-res fields**
   - 对于 900×1400 级别的大场，直接在像素上做 diffusion 成本极高；
   - 使用卷积 VAE 压缩至 16×90×140，再做 conditional diffusion 是一种通用技巧；
   - 适用于雷达、卫星、多通道气象图像、海洋场等。

4. **范式：Strong physics-inspired conditioning**
   - 条件不仅包括过去降水，还包括三维大气状态（T, U, V, SH, Z），以及 deterministic TP；
   - 本质相当于“用物理/NWP/大模型生成 coarse guess，再由 diffusion 做 super-resolution + 状态修正”。

5. **范式：集合可靠性与 rank histogram 校准**
   - 将 rank histogram 作为集合校准的首要诊断工具：
     - 若呈 U 型，可在 diffusion 采样或噪声 schedule 上增加 spread；
     - 若呈 ∩ 型，可减少噪声或加强观测约束；
   - 这一范式可落地到任何 ensemble ML 预报系统。

总体而言，这篇工作为“粗分辨率大气场 → 高分辨率集合降水”提供了一套完整且可迁移的框架：
- 通过 deterministic 3D Swin Transformer 提供 mesoscale mean；
- 在 residual latent 空间用 conditional diffusion 刻画 convective-scale 不确定性；
- 结合 rank histogram/CSI 等检验集合质量，为后续基于 Pangu/FourCastNet/GraphCast/FuXi 输出的下游高分辨率降水集合预报提供了现成模版。