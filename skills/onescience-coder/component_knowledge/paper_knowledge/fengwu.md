# FengWu 模型知识提取

信息来源：Chen et al., “FENGWU: Pushing the skillful global medium-range weather forecast beyond 10 days lead”（本文信息严格基于原文，不作额外主观推断；未在文中明确给出的部分标注说明）。

---

## 维度一：整体模型与任务知识

### 1. 任务与问题设定

- **任务类型：全球中期天气预报（medium-range NWP）**
  - 目标：基于当前全球大气状态，预测未来 14 天（56 个 6 小时时间步）的全球大气与地表状态。
  - 预测栅格：$0.25^\circ$ 纬向 × 经向全球格点，37 个气压层。
  - 预测时间分辨率：6 小时一帧（00z, 06z, 12z, 18z）。

- **输入 / 状态表示**
  - 每一时刻大气状态表示为高维张量：
    $$
    X^i \in \mathbb{R}^{C \times W \times H},
    $$
    其中：
    - $C = 189$：包括 5 个多层大气变量 + 4 个地表变量：
      - 大气变量（各 37 层）：
        - 位势高度 $z$（z***）
        - 相对湿度 $r$
        - 经向风分量 $u$
        - 纬向风分量 $v$
        - 气温 $t$
      - 地表变量（单层）：
        - 2 米气温 t2m
        - 10 米经向风 u10
        - 10 米纬向风 v10
        - 海平面气压 msl
    - $W = 721$（纬向格点数），$H = 1440$（经向格点数）。

- **输出 / 预报目标**
  - 单步预测函数（大气模式范式）：
    $$
    \hat{X}^{i+1} = \mathrm{FengWu}(X^i)
    $$
  - 自回归多步预测：
    $$
    \hat{X}^{i+\tau} = \mathrm{FengWu}(\hat{X}^{i+\tau-1}), \quad \tau=1,\dots,56,
    $$
    从而得到未来 14 天 6 小时分辨率的全场预报。

- **评价目标变量（predictands）**
  - 共 880 个“目标”（variables × levels × region/metric 组合，与 GraphCast 保持一致），
  - 论文中特别关注：z500, t850, t2m, u/v 风以及多层相对湿度等。

- **数据与时间段**
  - 数据源：ERA5 再分析（气压层 + 单层变量）。
  - 时段划分（与 GraphCast 一致）：
    - 训练：1979–2015
    - 验证：2016–2017
    - 测试：2018（hindcast）
  - 采样频率：6 小时（T00, T06, T12, T18），而非逐小时。

---

### 2. 传统方案痛点与动机

- **传统中期数值预报局限**
  - 初始场和边界条件存在较大不确定性，且大气物理过程高度非线性，导致误差快速增长；
  - 高分辨率（0.25°、多层）NWP 计算成本巨大，限制了业务上更长可预报提前量的探索；
  - 尽管 ECMWF IFS 等体系自 1980s 起持续提升技能，但在 10–14 天之前的“skillful lead time”仍有限。

- **现有 AI NWP 的不足**
  - 早期 ResNet / RNN 方法多在低分辨率数据上（例如 5.625°），难以直接替代现有业务系统；
  - FourCastNet：利用 ViT + AFNO 实现 0.25° 预报，但仍主要视作“单模态”场（各变量堆叠通道）；
  - PanGu-Weather：通过多时间尺度的 3D Transformer 组合获得优异表现，但在损失设计上仍以人工加权 MSE 为主；
  - GraphCast：使用 GNN + 多步自回归精调，并利用手工设计的变量/层权重 MSE loss；其不足包括：
    - 变量/层权重需要人工调参，成本高且未必最优；
    - 长步长自回归训练对显存和计算需求极高。

- **FengWu 的研究动机**
  1. 将高维大气状态视为**多模态**（按物理变量划分通道），而非简单“单模态通道堆叠”；
  2. 将全变量预测视为**多任务回归问题**，不同变量/层预测难度不同，需要自适应权重；
  3. 在有限显存/算力下有效地改善长时段自回归预测误差累积问题。

---

### 3. 整体解决方案与范式

- **核心思路**
  1. **多模态 + 多任务架构**：
     - 将 189 个变量划分为 6 个“模态”：地表（s）、位势高度（z）、湿度（q）、经向风（u）、纬向风（v）、温度（t）；
     - 为每个模态设计专门的 Transformer 编码器和解码器；
     - 通过跨模态 Transformer 进行特征交互和融合；
     - 输出每个模态所有变量在全场的均值与方差。
  2. **不确定性损失（uncertainty loss）**：
     - 视 FengWu 为高斯概率模型，对每个网格点 $(c,w,h)$ 预测 $\hat{\mu}_{c,w,h}$ 与 $\hat{\sigma}_{c,w,h}$；
     - 使用最大似然/负对数似然作为整体损失，使不同变量/层/经纬点权重由**同方差不确定性**自动调整；
     - 避免 GraphCast 中依赖经验和网格划分的手工权重设计。
  3. **Replay Buffer 长时自回归训练机制**：
     - 在单步预测训练的基础上，引入一个存储若干历史预测场的 buffer；
     - 训练时从“原始数据 + buffer 中的预测场”同时采样，模拟推理阶段自回归输入误差累积；
     - 通过在 CPU 存储中间结果，降低 GPU 显存占用，支持更长 lead 的“近似自回归训练”。

---

### 4. 模型架构与关键公式

#### 4.1 单步预测与自回归

- **整体目标函数**：
  - 理想 14 天预测：
    $$
    \{\hat{X}^{i+1},\hat{X}^{i+2},\dots,\hat{X}^{i+56}\} = \mathrm{FengWu}(X^i)
    $$
  - 实际：由于全场太大，直接学习上式困难，因此改为学习单步预测函数：
    $$
    \hat{X}^{i+1} = \mathrm{FengWu}(X^i)
    $$
  - 多步预测通过自回归方式：
    $$
    \hat{X}^{i+\tau} = \mathrm{FengWu}(\hat{X}^{i+\tau-1}),\;\tau=1,\dots,56.
    $$

#### 4.2 多模态编码–融合–解码

- **模态划分**：
  - 给定 $X^i \in \mathbb{R}^{C\times W\times H}$，按变量类型切分：
    - 地表模态：$X_s^i \in \mathbb{R}^{C_s\times W\times H}$（t2m, u10, v10, msl）；
    - 位势高度模态：$X_z^i \in \mathbb{R}^{C_z\times W\times H}$；
    - 湿度模态：$X_q^i \in \mathbb{R}^{C_q\times W\times H}$；
    - 经向风模态：$X_u^i$；
    - 纬向风模态：$X_v^i$；
    - 温度模态：$X_t^i$。

- **模态编码器**（Transformer-based）：
  - 对每个模态 $m\in\{s,z,q,u,v,t\}$：
    $$
    Z_m = f_{\mathrm{en},m}(X_m\mid\theta_{\mathrm{en},m}),
    $$
    其中 $f_{\mathrm{en},m}$ 为基于 Transformer 的编码器网络，输出模态特征张量 $Z_m$。

- **跨模态融合器（Cross-modal Fuser）**：
  - 将模态特征在通道维度拼接：
    $$
    Z = \mathrm{concat}(Z_s,Z_z,Z_q,Z_u,Z_v,Z_t),
    $$
  - 送入跨模态 Transformer 获得融合表示 $\tilde{Z}$（包含不同变量之间的交互信息）。

- **模态解码器**：
  - 对每个模态 $m$ 使用解码器 $f_{\mathrm{de},m}$ 从 $\tilde{Z}$ 中重建其未来场：
    $$
    \hat{X}_m^{i+1} = f_{\mathrm{de},m}(\tilde{Z}\mid\theta_{\mathrm{de},m}),
    $$
  - 解码器结构与编码器类似，采用 Transformer + 上采样/反投影回到原空间维度。

#### 4.3 不确定性损失（Uncertainty Loss）

- **概率建模**：
  - FengWu 输出未来状态的均值和方差：
    $$
    \hat{\mu}^{i+1},\hat{\sigma}^{i+1} = \mathrm{FengWu}(X^i),
    $$
  - 对每个网格 $(c,w,h)$：
    $$
    x_{c,w,h}^{i+1} \sim \mathcal{N}\big(\hat{\mu}_{c,w,h}^{i+1},\hat{\sigma}_{c,w,h}^{i+1}\big),
    $$
    其中 $c=1,\dots,189$ 表示变量或层，$w,h$ 为纬/经格点索引。

- **负对数似然（简写）**：
  - 单点损失：
    $$
    \ell_{c,w,h} = \frac{1}{2}\log\big((\hat{\sigma}_{c,w,h}^{i+1})^2\big)
      + \frac{\big(x_{c,w,h}^{i+1}-\hat{\mu}_{c,w,h}^{i+1}\big)^2}{2(\hat{\sigma}_{c,w,h}^{i+1})^2},
    $$
  - 总损失为对所有 $(c,w,h)$ 与时间样本求和/平均：
    $$
    \mathcal{L}_{\mathrm{uncertainty}} = \frac{1}{N}\sum_{i,c,w,h} \ell_{c,w,h}.
    $$
  - 含义：
    - 方差大的变量/网格点在误差项中权重相对较小；
    - 模型自动学习各变量、层、地区的“同方差不确定性”，从而自动分配任务权重。

#### 4.4 评价指标公式

- **纬向加权 RMSE**：
  - 对变量/层索引 $c$、lead 时间 $\tau$：
    $$
    \mathrm{RMSE}(c,\tau) = \frac{1}{T} \sum_{i=1}^T
    \sqrt{\frac{1}{W\cdot H}\sum_{w=1}^W\sum_{h=1}^H
    W\cdot\frac{\cos(\alpha_{w,h})}{\sum_{w'=1}^W\cos(\alpha_{w',h})}
    \big(x_{c,w,h}^{i+\tau}-\hat{x}_{c,w,h}^{i+\tau}\big)^2},
    $$
    其中 $\alpha_{w,h}$ 为格点纬度角。

- **纬向加权 ACC（Anomaly Correlation Coefficient）**：
  - 引入气候态场 $C_{c,w,h}^{i+\tau}$（对多年同日的均值）：
    $$
    \mathrm{ACC}(c,\tau) = \frac{1}{T}\sum_{i=1}^T
    \frac{\sum_{w,h}W\cdot\frac{\cos(\alpha_{w,h})}{\sum_{w'}\cos(\alpha_{w',h})}
    (x_{c,w,h}^{i+\tau}-C_{c,w,h}^{i+\tau})(\hat{x}_{c,w,h}^{i+\tau}-C_{c,w,h}^{i+\tau})}
    {\sqrt{A\_o\,A\_p}},
    $$
    其中 $A\_o, A\_p$ 为观测和预测异常平方的同样加权和。
  - 以 $\mathrm{ACC}(z500) > 0.6$ 作为“skillful”标准：
    - FengWu 达到的最远 skillful lead：10.75 天（z500）、11.5 天（t2m）。

---

### 5. 数据与训练细节

- **数据集**
  - ERA5 气压层 + 单层再分析，0.25° 分辨率，37 个气压层（1000–1 hPa）。
  - 预测变量集合：
    - 5 个 3D 大气变量（$z, r, u, v, t$，各 37 层）；
    - 4 个 2D 地表变量（t2m, u10, v10, msl）；
    - 共 189 维通道。

- **训练/验证/测试划分**
  - 训练：1979–2015（6 小时数据）；
  - 验证：2016–2017；
  - 测试：2018，全年度 hindcast，用于与 GraphCast 对比。

- **计算成本**
  - 训练：
    - 框架：PyTorch；
    - 硬件：32× NVIDIA A100；
    - 耗时：17 天；
    - 与 GraphCast（32× TPU v4，21 天）相比，折算后训练时间约为其 47–67%。
  - 推理：
    - 单张 A100 上生成完整 10 天、6 小时步长的预报 < 30 秒；
    - 能耗：约 12 kJ（A100 峰值功率约 0.4 kW），相比估算的 IFS 单成员 26.6 MJ，约低 2000 倍量级。

---

### 6. 技巧与结果（对比 GraphCast）

- **整体量化结果**
  - 在 2018 年 880 个目标上：
    - FengWu 在约 80% 的目标上 RMSE 更低、ACC 更高；
    - 1–5 天提前期，两者表现相当（t2m 略逊），
    - lead 增大后，FengWu 在所有变量上显示更优长期技巧。

- **skillful lead time**
  - 以 $\mathrm{ACC}>0.6$ 为标准：
    - z500：FengWu 可延伸至 10.75 天（首个 AI 模型达到该 lead）；
    - t2m：可至约 11.5 天。

- **Replay Buffer 作用**
  - Ablation 显示：
    - 无 replay buffer 时，随 lead 增大 RMSE 显著劣化；
    - 引入 replay buffer 后，长 lead RMSE 明显降低，表明其在模拟 AR 推理误差累积方面有效。

- **视觉结果**
  - 对特定个例（如 2018-02-11 00:00 初始化）：
    - z500、t850 在 day 3,5,10 的预测场与 ERA5 接近；
    - 随 lead 增加，误差扩散但仍保持合理尺度和大尺度结构。

---

### 7. 创新点与讨论

- **主要创新**
  1. 从“多模态 + 多任务”视角重构全球中期预报问题：
     - 按物理变量划分模态，采用模态专用编码器/解码器与跨模态 Transformer；
  2. 将多任务不确定性损失引入 NWP：
     - 自动学习变量/层/网格的权重，避免人工调参；
  3. 提出 Replay Buffer 自回归训练机制：
     - 在显存有限条件下显式考虑长 lead 自回归误差；
  4. 在 0.25° 分辨率上，在多数指标上超过 GraphCast，并首次将 AI 模型的 skillful lead 延长到 10.75 天以上。

- **公平性与局限**
  - 与 IFS-HRES 的对比：
    - FengWu/GraphCast 使用 ERA5 作为初始场；
    - IFS 使用“当时可用观测”进行实时同化（部分卫星延时观测缺失），因此初始场质量略逊；
    - 因此 AI 与 IFS 的比较对物理模式略不公平，但 AI 模型之间（FengWu vs GraphCast）是公平的。
  - 本文未深入讨论的细节：
    - 精确的 Transformer 层数/头数、Patch 方案等结构超参数；
    - Replay Buffer 容量 N 和采样策略的具体数值；
    - 如需实现，需要自行补充工程细节。

---

## 维度二：基础组件 / 模块级知识

> 说明：以下伪代码均为**基于论文文字描述与公式复原的高保真伪代码**，仅用于理解结构与训练逻辑，并非作者实际实现。

### 组件 A：多模态编码–融合–解码主干

- **子任务**
  - 将高维大气状态 $X^i\in\mathbb{R}^{189\times W\times H}$ 编码为紧凑特征，
  - 在模态间进行信息交互，
  - 解码得到下一时刻 $\hat{X}^{i+1}$ 的均值与方差。

- **数据流**
  1. 切分原始通道到 6 个模态 $X_m^i$；
  2. 各模态编码器 $f_{\mathrm{en},m}$ 提取特征 $Z_m$；
  3. 拼接 $Z$，送入跨模态 Transformer 得到 $\tilde{Z}$；
  4. 各模态解码器 $f_{\mathrm{de},m}$ 输出 $\hat{\mu}_m,\hat{\sigma}_m$；
  5. 聚合为全通道 $\hat{\mu}^{i+1},\hat{\sigma}^{i+1}$，用于采样或取均值作为预测场。

- **伪代码**（结构示意）：

```python
# Pseudocode: FengWu backbone (one-step prediction)

class ModalEncoder(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, d_model)
        self.transformer = TransformerEncoder(d_model=d_model)

    def forward(self, x):
        # x: (B, C_m, W, H)
        tokens = self.patch_embed(x)      # (B, N_tokens, d_model)
        z = self.transformer(tokens)      # (B, N_tokens, d_model)
        return z

class ModalDecoder(nn.Module):
    def __init__(self, d_model, out_channels):
        super().__init__()
        self.transformer = TransformerDecoder(d_model=d_model)
        self.head_mu = nn.Linear(d_model, out_channels)
        self.head_sigma = nn.Linear(d_model, out_channels)

    def forward(self, fused_tokens):
        h = self.transformer(fused_tokens)
        mu_tokens = self.head_mu(h)
        sigma_tokens = F.softplus(self.head_sigma(h))
        mu = tokens_to_grid(mu_tokens)        # (B, C_m, W, H)
        sigma = tokens_to_grid(sigma_tokens)  # (B, C_m, W, H)
        return mu, sigma

class FengWu(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # encoders for 6 modalities
        self.enc_s = ModalEncoder(cfg.C_s, cfg.d_model)
        self.enc_z = ModalEncoder(cfg.C_z, cfg.d_model)
        self.enc_q = ModalEncoder(cfg.C_q, cfg.d_model)
        self.enc_u = ModalEncoder(cfg.C_u, cfg.d_model)
        self.enc_v = ModalEncoder(cfg.C_v, cfg.d_model)
        self.enc_t = ModalEncoder(cfg.C_t, cfg.d_model)

        self.cross_modal = TransformerEncoder(d_model=cfg.d_model)

        # decoders
        self.dec_s = ModalDecoder(cfg.d_model, cfg.C_s)
        self.dec_z = ModalDecoder(cfg.d_model, cfg.C_z)
        self.dec_q = ModalDecoder(cfg.d_model, cfg.C_q)
        self.dec_u = ModalDecoder(cfg.d_model, cfg.C_u)
        self.dec_v = ModalDecoder(cfg.d_model, cfg.C_v)
        self.dec_t = ModalDecoder(cfg.d_model, cfg.C_t)

    def forward(self, X):
        X_s, X_z, X_q, X_u, X_v, X_t = split_modalities(X)

        Z_s = self.enc_s(X_s)
        Z_z = self.enc_z(X_z)
        Z_q = self.enc_q(X_q)
        Z_u = self.enc_u(X_u)
        Z_v = self.enc_v(X_v)
        Z_t = self.enc_t(X_t)

        Z = torch.cat([Z_s, Z_z, Z_q, Z_u, Z_v, Z_t], dim=1)
        Z_fused = self.cross_modal(Z)

        mu_s, sigma_s = self.dec_s(Z_fused)
        mu_z, sigma_z = self.dec_z(Z_fused)
        mu_q, sigma_q = self.dec_q(Z_fused)
        mu_u, sigma_u = self.dec_u(Z_fused)
        mu_v, sigma_v = self.dec_v(Z_fused)
        mu_t, sigma_t = self.dec_t(Z_fused)

        mu = concat_modalities(mu_s, mu_z, mu_q, mu_u, mu_v, mu_t)
        sigma = concat_modalities(sigma_s, sigma_z, sigma_q,
                                  sigma_u, sigma_v, sigma_t)
        return mu, sigma
```

---

### 组件 B：不确定性损失计算模块

- **子任务**
  - 基于预测的 $\hat{\mu},\hat{\sigma}$ 与真实场 $X^{i+1}$ 计算高斯负对数似然，作为训练目标；
  - 自动调节不同变量/层/区域的损失权重。

- **伪代码**：

```python
# Pseudocode: Uncertainty loss

def uncertainty_loss(mu, sigma, target):
    # mu, sigma, target: (B, C, W, H)
    # ensure positive sigma
    sigma = torch.clamp(sigma, min=1e-4)
    var = sigma ** 2

    # negative log-likelihood per grid point
    nll = 0.5 * torch.log(var) + 0.5 * (target - mu) ** 2 / var

    return nll.mean()
```

---

### 组件 C：Replay Buffer 自回归训练机制

- **子任务**
  - 在训练过程中引入“伪自回归”输入：
    - 使用此前训练迭代中生成的预测场作为当前模型的输入之一；
    - 让模型在训练期间就看到“带误差的中间态”，从而学会纠错和稳定长 lead 预报；
  - 控制 GPU 内存：
    - 预测场缓存在 CPU（或磁盘）中，而不维持长序列梯度，节省显存。

- **数据结构**
  - buffer $\mathbf{B} = \{\hat{X}_j^{i+\tau}\}_{j=0}^N$：存储最近 N 个时间步的预测场；
  - 初期：仅推入“一步预测”的结果；
  - 之后：从原始样本和 buffer 混合采样，逐渐覆盖更长 lead 的中间态。

- **伪代码**：

```python
# Pseudocode: Replay buffer training loop

replay_buffer = deque(maxlen=BUFFER_SIZE)

for epoch in range(num_epochs):
    for X_i, X_ip1 in data_loader:  # X_i: state at time i, X_ip1: true at i+1
        # sample mode: original or replay
        if len(replay_buffer) > MIN_BUF and np.random.rand() < p_replay:
            # use a sample from buffer as input
            X_in = random.choice(replay_buffer)  # detached tensor
        else:
            X_in = X_i

        mu, sigma = model(X_in)
        loss = uncertainty_loss(mu, sigma, X_ip1)
        loss.backward()
        optimizer.step(); optimizer.zero_grad()

        # update buffer with current prediction (detach to avoid grads)
        with torch.no_grad():
            X_pred = mu  # or sample from N(mu, sigma)
            replay_buffer.append(X_pred.cpu())
```

---

### 组件 D：自回归推理模块

- **子任务**
  - 给定初始场 $X^i$（如 00z/12z ERA5 分析），生成未来 10–14 天 6 小时间隔预报序列；
  - 与 GraphCast 一样可用于 hindcast 和可能的实时预报。

- **伪代码**：

```python
# Pseudocode: Autoregressive inference

def forecast_autoregressive(model, X_init, n_steps=56):
    preds = []
    X_curr = X_init
    for _ in range(n_steps):
        mu, sigma = model(X_curr)
        X_next = mu  # deterministic forecast
        preds.append(X_next)
        X_curr = X_next
    return torch.stack(preds, dim=1)  # (B, n_steps, C, W, H)
```

---

### 组件 E：评估与对比模块

- **子任务**
  - 按 GraphCast 协议计算 RMSE / ACC；
  - 在 2018 年测试集上对 FengWu 与 GraphCast 做逐变量、逐层、逐 lead 比较。

- **伪代码（RMSE/ACC 轮廓）**：

```python
# Pseudocode: Compute latitude-weighted RMSE & ACC

def lat_weights(latitudes):
    # latitudes: (W, H) or (W,)
    w = np.cos(np.deg2rad(latitudes))
    w = w / w.sum(axis=0, keepdims=True)
    return w

def rmse_weighted(pred, obs, weights):
    diff2 = (pred - obs) ** 2
    w_diff2 = diff2 * weights  # broadcast over C, W, H
    return np.sqrt(w_diff2.mean(axis=(-2, -1)))  # over space

def acc_weighted(pred, obs, clim, weights):
    a = (obs - clim)
    b = (pred - clim)
    num = (weights * a * b).sum(axis=(-2, -1))
    den = np.sqrt((weights * a**2).sum(axis=(-2, -1)) *
                  (weights * b**2).sum(axis=(-2, -1)))
    return (num / den).mean(axis=0)  # average over time
```

---

## 小结

- FengWu 将全球中期 NWP 明确建模为“多模态 + 多任务”的深度学习问题，通过模态专用 Transformer 编解码器与跨模态融合结构，在 0.25°、37 层高维场上实现高效单步预测；
- 通过不确定性损失自动学习变量与层的损失权重，避免了 GraphCast 中高度工程化的手工加权；
- Replay Buffer 机制在显存有限条件下逼近长序列自回归训练，使 FengWu 在 10 天以上提前量显著优于 GraphCast，并首次将 AI 中期预报的 skillful lead（z500, ACC>0.6）推进至约 10.75 天。