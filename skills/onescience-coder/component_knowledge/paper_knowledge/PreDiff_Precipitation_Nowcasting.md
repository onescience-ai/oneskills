# PreDiff：Precipitation Nowcasting with Latent Diffusion Models

> 说明：本文件基于论文文字内容进行结构化总结和高层伪代码抽象，不是官方实现代码。

---

## 一、整体模型维度（任务 / 数据 / 结构 / 训练 / 结果）

### 1.1 任务与问题设定

- **任务类型**：
  - 降水短临预报（precipitation nowcasting），典型时效 0–6 小时；
  - 一般形式：给定过去 L_in 帧雷达/卫星降水观测，预测未来 L_out 帧：
    - 观测：\(y = [y^j]_{j=1}^{L_{in}} \in \mathbb{R}^{L_{in} \times H \times W \times C}\)；
    - 未来：\(x = [x^j]_{j=1}^{L_{out}} \in \mathbb{R}^{L_{out} \times H \times W \times C}\)。
- **概率建模目标**：
  - 建模条件概率分布 \(p(x\mid y)\)，而不是单点预测；
  - 通过生成多个样本刻画不确定性，避免“平均多种可能未来→模糊图像”的问题。
- **核心痛点**：
  1. 深度学习降水预报往往用像素级 L2/L1 损失，得到模糊、缺少细节的预报；
  2. 纯数据驱动 diffusion/生成模型缺乏物理约束，可能产生物理上不合理的结果；
  3. 现有物理先验融合方式多通过**改架构/改损失**，一旦约束变化就要重设架构或重训模型。

### 1.2 PreDiff 总体思路（两阶段框架）

- **阶段 1：训练条件 latent diffusion 模型 PreDiff**
  - 使用 frame-wise VAE 将像素空间压缩到低维 latent 空间；
  - 在 latent 空间上训练条件 diffusion model：
    - 条件：编码后的历史观测 latent \(z_{cond}\)；
    - 目标：未来序列 latent \(z_0\) 的条件分布 \(p_\theta(z_{0:T} \mid z_{cond})\)。
  - UNet 主干替换为 Earthformer-UNet（来自 Earthformer encoder）：更强的时空建模能力，适合雷达/降水数据。

- **阶段 2：知识对齐（Knowledge Alignment, KA）机制**
  - 不改 PreDiff 主体结构、不重训 diffusion，只额外训练一个**知识对齐网络** \(U_\phi\)；
  - 形式化“先验知识/物理约束”：
    $$\mathcal{F}(\hat{x}, y) = \mathcal{F}_0(y) \in \mathbb{R}^d,$$
    例如：
    - N-body MNIST：总能量守恒；
    - SEVIR：预期总降水量/强度统计；
  - 在每个 diffusion 去噪步，将 transition 分布从
    $$p_\theta(z_t \mid z_{t+1}, z_{cond})$$
    调整为
    $$p_{\theta,\phi}(z_t \mid z_{t+1}, y, \mathcal{F}_0) \propto p_\theta(\cdot) \cdot \exp\big(-\lambda_\mathcal{F} \|U_\phi(z_t, t, y) - \mathcal{F}_0(y)\|\big),$$
    抑制预期违反物理约束的中间 latent。

### 1.3 Latent diffusion 形式化

- **VAE 编码器/解码器**：
  - 单帧编码：\(x^j \mapsto z^j = \mathcal{E}(x^j) \in \mathbb{R}^{H_z \times W_z \times C_z}\)；
  - 序列编码：\(x \mapsto z = [z^j]_{j=1}^L\)；
  - 解码：\(z^j \mapsto \hat{x}^j = \mathcal{D}(z^j)\)。

- **条件 latent diffusion**：
  - 条件 latent：\(z_{cond} = \mathcal{E}(y) \in \mathbb{R}^{L_{in} \times H_z \times W_z \times C_z}\)；
  - 未来 latent：\(z_0 = \mathcal{E}(x) \in \mathbb{R}^{L_{out} \times H_z \times W_z \times C_z}\)；
  - 反向过程：
    $$p_\theta(z_{0:T} \mid z_{cond}) = p(z_T) \prod_{t=1}^T p_\theta(z_{t-1} \mid z_t, z_{cond}),$$
    其中 \(z_T \sim \mathcal{N}(0,I)\)。
  - 使用噪声预测参数化：训练 \(\epsilon_\theta(z_t, t, z_{cond})\) 近似注入噪声 \(\epsilon\)。

- **训练目标（CLDM 损失）**：
  $$L_{CLDM} = \mathbb{E}_{(x,y), t, \epsilon}\big[\|\epsilon - \epsilon_\theta(z_t, t, z_{cond})\|_2^2\big].$$

### 1.4 模型骨干：Earthformer-UNet

- 将 LDM 中的 2D UNet 替换为**Earthformer-UNet**：
  - UNet 式 encoder–decoder，带多尺度特征；
  - 使用 self-cuboid attention 作为时空注意力积木，擅长建模大范围时空依赖；
  - 输入：concat(z_cond, noisy z_t) 沿时间维拼接；
  - 输出：对应步的噪声或 z_{t-1} 估计。
- 对比实验表明：
  - 在同一 diffusion 框架中，用 Earthformer-UNet 明显优于普通 UNet（LDM），说明 latent backbone 的时空建模能力非常关键。

### 1.5 知识对齐（KA）思想与收益

- **知识形式**：定义一个约束映射 \(\mathcal{F}(\hat{x}, y)\)，如：
  - N-body MNIST：系统总能量 E(\hat{x})；
  - SEVIR：总降水量、降水强度直方图或其它统计；
  - 要求预报满足 \(\mathcal{F}(\hat{x}, y) \approx \mathcal{F}_0(y)\)。
- **估计违反程度**：
  - 使用 U_\phi 在 latent 空间估计 \(\mathcal{F}(\hat{x}, y)\)：
    $$U_\phi(z_t, t, y) \approx \mathcal{F}(\hat{x}, y);$$
  - 训练损失：
    $$L_U = \|U_\phi(z_t, t, y) - \mathcal{F}(x, y)\|,$$
    其中 x 为真实未来（注意 x 本身也可能不严格满足知识）。
- **推理时偏置采样分布**：
  - 通过调节每步 transition 的均值，抑制高违反样本，提高满足约束的概率；
  - PreDiff-KA 在不重训主模型的前提下，可以灵活替换/增加不同知识对齐网络（对应不同约束）。

### 1.6 数据与结果（定性）

1. **N-body MNIST（合成混沌系统）**：
   - 数据：3 体运动驱动的 MovingMNIST，满足能量守恒；
   - 任务：给定长度 10 上下文，预测 10 步未来 (64×64×1)。
   - 结果：
     - PreDiff 相比 UNet、ConvLSTM、PredRNN、PhyDNet、E3D-LSTM、Rainformer、Earthformer、VideoGPT、LDM 在 MSE/MAE/SSIM/FVD 上均显著领先；
     - PreDiff-KA 在能量误差（E.MSE/E.MAE）上显著改善，且保持图像质量；
     - 展示样例中，PreDiff 能给出清晰、位置准确的数字轨迹，不出现模糊或消失。

2. **SEVIR（真实降水短临，文中只给摘要）**：
   - 数据：SEVIR 雷达降水 nowcasting 基准；
   - KA 约束：预期降水强度（如总量、分布等）；
   - 结果：
     - PreDiff 在感知质量（FVD）上达到 SOTA；
     - PreDiff-KA 能更好对齐业务上重要的强降水统计，提升“可运营实用性”。

---

## 二、组件维度：模块拆解与伪代码

### 2.1 Frame-wise VAE 编码/解码组件

**子需求**：
- 将高维时空降水序列 \(x \in \mathbb{R}^{L\times H\times W\times C}\) 压缩为更小的 latent 空间，以降低 diffusion 成本；
- 支持逐帧编码/解码，以复用成熟图像 VAE 技术。

**关键要点**：
- 编码器 \(\mathcal{E}\)：卷积/残差网络 + downsampling，将 (H,W,C) 压缩为 (H_z,W_z,C_z)；
- 解码器 \(\mathcal{D}\)：对称上采样 + 卷积；
- 训练损失：像素 L2 + adversarial loss（不使用感知损失，因为缺少地球观测预训练网络）。

**伪代码（训练 VAE）**：

```python
class FrameVAE(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.encoder = EncoderCNN(...)
        self.decoder = DecoderCNN(...)
        self.disc   = DiscriminatorCNN(...)

    def encode(self, x):  # x: (B, C, H, W)
        return self.encoder(x)  # (B, C_z, H_z, W_z)

    def decode(self, z):
        return self.decoder(z)  # (B, C, H, W)


def train_vae_step(model: FrameVAE, x):
    z = model.encode(x)
    x_hat = model.decode(z)
    # reconstruction loss
    rec_loss = F.mse_loss(x_hat, x)
    # adversarial loss (GAN-style)
    real_score = model.disc(x)
    fake_score = model.disc(x_hat.detach())
    adv_loss = gan_loss(real_score, fake_score)
    vae_loss = rec_loss + lambda_adv * adv_loss
    vae_loss.backward()
    optimizer_vae.step()
```

在 PreDiff 中，时间维只是批内额外维度：对序列的每一帧独立应用 encode/decode。

---

### 2.2 条件 latent diffusion + Earthformer-UNet 组件

**子需求**：
- 在 VAE latent 空间中实现条件 diffusion：\(p_\theta(z_{0:T}\mid z_{cond})\)；
- 主干网络需具备强的时空建模能力，以捕捉降水组织、移动和变形。

**结构要点**：
- 条件 latent：\(z_{cond} = \mathcal{E}(y)\)；未来 latent：\(z_0 = \mathcal{E}(x)\)；
- 在训练时：
  - 从 \(z_0\) 采样 \(z_t \sim q(z_t \mid z_0)\)；
  - 网络输入：[z_cond, z_t] 沿时间拼接；
  - 网络输出：\(\epsilon_\theta(z_t, t, z_{cond})\)。
- 主干 Earthformer-UNet：
  - 编码器/解码器多尺度结构；
  - 块为 self-cuboid attention，覆盖时间和空间块。

**训练伪代码**：

```python
class PreDiffBackbone(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.earthformer_unet = EarthformerUNet(...)

    def forward(self, z_t, t, z_cond):
        # concat along time axis: (L_cond + L_out, Hz, Wz, Cz)
        z_in = torch.cat([z_cond, z_t], dim=1)
        eps_hat = self.earthformer_unet(z_in, t)
        return eps_hat


def train_prediff_step(vae: FrameVAE, backbone: PreDiffBackbone, x, y):
    # encode sequences
    z_future = encode_sequence(vae, x)      # (B, L_out, Cz, Hz, Wz)
    z_cond   = encode_sequence(vae, y)      # (B, L_in,  Cz, Hz, Wz)

    # sample timestep t and noise
    t = sample_timesteps(B)                 # (B,)
    eps = torch.randn_like(z_future)
    z_t = q_sample(z_future, eps, t)        # forward noising

    # predict noise
    eps_hat = backbone(z_t, t, z_cond)

    loss = F.mse_loss(eps_hat, eps)
    loss.backward()
    optimizer_backbone.step()
```

---

### 2.3 知识对齐网络 U_\phi 与训练

**子需求**：
- 通过一个额外网络 U_\phi 在 latent 空间估计约束量 \(\mathcal{F}(\hat{x}, y)\)；
- 训练 U_\phi 时 **不依赖 diffusion 采样过程**，只需 encode 真实目标 x 并加入噪声；
- 使 U_\phi 在各个噪声级别 t 上都能近似当前 sample 的“物理指标”。

**训练数据构造**：
1. 从数据集中取 (x, y)；
2. 用 VAE 编码器获得 z_0 = E(x)；
3. 用 forward noising q(z_t | z_0) 生成 z_t；
4. 用显式或近似方式计算真实未来的约束值 \(\mathcal{F}(x, y)\)；
5. 最小化 \(L_U = \| U_\phi(z_t, t, y) - \mathcal{F}(x, y) \|\)。

**伪代码（与论文 Alg.1 对应）**：

```python
class KnowledgeAlignmentNet(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 输入 (z_t, t, y_latent)，输出 F-hat in R^d
        self.encoder = SpatiotemporalEncoder(...)
        self.head = nn.Linear(..., d_F)

    def forward(self, z_t, t, y_latent):
        # t 可以通过 embedding 注入
        t_emb = timestep_embedding(t, dim=...)  # (B, d_t)
        h = self.encoder(z_t, y_latent, t_emb)
        F_hat = self.head(h)
        return F_hat


def train_knowledge_alignment_step(vae: FrameVAE, U: KnowledgeAlignmentNet, x, y):
    # 1) encode
    z0 = encode_sequence(vae, x)       # future latent
    z_cond = encode_sequence(vae, y)   # optional condition for U

    # 2) sample t and z_t
    t = sample_timesteps(B)
    eps = torch.randn_like(z0)
    z_t = q_sample(z0, eps, t)

    # 3) compute target constraint
    F_xy = compute_constraint(x, y)    # e.g. total energy, total rainfall

    # 4) train U_phi
    F_hat = U(z_t, t, z_cond)
    loss_U = torch.mean((F_hat - F_xy)**2)
    loss_U.backward()
    optimizer_U.step()
```

---

### 2.4 采样阶段的知识对齐（Guidance）

**子需求**：
- 在不改 diffusion 主模型参数的前提下，利用 U_\phi 对每一步 z_t→z_{t-1} 进行“引导”；
- 抑制会导致高违约束的样本（高 \(\|U_\phi(z_t,t,y) - \mathcal{F}_0(y)\|\)）。

**理论形式**：
- 修正后的 transition：
  $$p_{\theta,\phi}(z_t \mid z_{t+1}, y, \mathcal{F}_0) \propto
    p_\theta(z_t \mid z_{t+1}, z_{cond}) \cdot
    \exp\big(-\lambda_\mathcal{F}\, \|U_\phi(z_t, t, y) - \mathcal{F}_0(y)\|\big).$$
- 按 classifier guidance 理论，可近似为：
  - 在原均值 \(\mu_\theta\) 的基础上施加梯度偏移：
    $$\mu_{guid} \approx \mu_\theta - \lambda_\mathcal{F} \Sigma_\theta \nabla_{z_t}\|U_\phi(z_t, t, y) - \mathcal{F}_0(y)\|.$$

**高层伪代码（采样）**：

```python
@torch.no_grad()
def sample_with_knowledge_alignment(vae, backbone, U, y, F0_y, guidance_scale):
    # encode condition
    z_cond = encode_sequence(vae, y)
    B = z_cond.size(0)

    # init from Gaussian
    z_t = torch.randn(B, L_out, Cz, Hz, Wz, device=y.device)

    for t in reversed(range(1, T+1)):
        t_batch = torch.full((B,), t, device=y.device, dtype=torch.long)

        # 1) predict eps and base mean/var
        eps_hat = backbone(z_t, t_batch, z_cond)
        mu_theta, sigma_theta = compute_p_mean_var(z_t, eps_hat, t_batch)

        # 2) KA guidance
        z_t.requires_grad_(True)
        F_hat = U(z_t, t_batch, z_cond)
        loss_ka = torch.norm(F_hat - F0_y, dim=-1).mean()
        grad = torch.autograd.grad(loss_ka, z_t)[0]
        z_t = z_t.detach()

        mu_guid = mu_theta - guidance_scale * (sigma_theta * grad)

        # 3) sample z_{t-1}
        noise = torch.randn_like(z_t) if t > 1 else 0
        z_t = mu_guid + sigma_theta * noise

    z0 = z_t
    x_hat = decode_sequence(vae, z0)
    return x_hat
```

---

### 2.5 约束示例：N-body 能量守恒 & SEVIR 降水统计

1. **N-body MNIST：能量守恒**
   - 约束：总能量 E = kinetic + potential 近似守恒：
     $$\mathcal{F}(\hat{x}, y) = E(\hat{x}),\quad \mathcal{F}_0(y) = E(y^{L_{in}}).$$
   - compute_constraint(x, y)：
     - 根据已知三体运动方程，从像素中提取坐标/速度（或在数据生成时记录）；
     - 计算每帧能量，并取平均或末帧能量作为 E(x)。
   - 结果：PreDiff-KA 显著减小 E.MSE / E.MAE（预测能量与初始能量差）。

2. **SEVIR：预期降水强度/分布**
   - 约束示例：
     - 区域总降水量：\(\sum_{i,j,t} \hat{x}_{i,j,t}\)；
     - 强降水像素占比；
     - 分区统计（子区域总量）。
   - compute_constraint(x, y)：按业务需求设计统计算子即可；
   - U_\phi 学习从 latent 直接估计这些统计；采样时偏置保留总量/结构。

---

### 2.6 可迁移范式总结

- **范式 1：LDM + 领域 VAE**
  - 先训练 frame-wise VAE 压缩观测，再在 latent 空间上做 diffusion，适合高维时空数据（雷达/卫星）。

- **范式 2：强时空主干替换（Earthformer-UNet）**
  - 在 diffusion 主干中使用专门为地球系统设计的时空 Transformer/UNet，可显著提升生成质量（对比 LDM-UNet）。

- **范式 3：后插拔式知识对齐**
  - 不嵌入到主模型结构/损失，只在采样时用 U_\phi 修改 transition；
  - 新增或修改约束时，只需重训 U_\phi，而无需重训 PreDiff 本体。

- **范式 4：约束在 latent 而非像素空间评估**
  - U_\phi 在 latent 上估计 \(\mathcal{F}(\hat{x}, y)\)，避免每步都 decode 回像素，效率更高；
  - 对于需复杂处理的知识（如物理量、统计量），可在训练 U_\phi 时离线计算目标值，再在采样时只用近似网络。

- **范式 5：Probabilistic nowcasting with controllable physics**
  - Diffusion 负责概率多样性；知识对齐负责“把样本推回物理合理子空间”；
  - 这一框架可向其他地学任务扩展：如风场预报中的能谱结构、整体角动量守恒、降水–水汽闭合关系等。

这些组件和范式为后续“物理可控的概率降水/天气预报”提供了一套可复用模板：
- 将任意降水/雷达/卫星序列数据通过 VAE+LDM 建立基础概率模型；
- 根据具体业务/物理需求，设计 \(\mathcal{F}\) 并训练对应 U_\phi，实现灵活的物理/运营约束引导。