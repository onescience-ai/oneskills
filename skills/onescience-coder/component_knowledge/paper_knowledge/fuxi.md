# FuXi 模型知识提取

信息来源：Chen et al., “FuXi: a cascade machine learning forecasting system for 15-day global weather forecast”（内容严格基于论文，不作主观推断；论文未明确之处标注“未在文中明确给出”或保持空缺）。

---

## 维度一：整体模型与任务知识

### 1. 任务与问题设定

- **目标与范围**
  - 构建 FuXi：一个基于机器学习的级联（cascade）全球天气预报系统。
  - 预测时效：15 天（15 × 24h），时间分辨率为 6 小时，即 60 个时间步。
  - 空间分辨率：全球 $0.25^\circ$（约 31 km），网格大小约为 $721\times 1440$。
  - 采用数据集：39 年 ECMWF ERA5 再分析（1979–2017 训练/验证，2018 测试）。

- **变量与状态空间**
  - 上空 5 个大气变量 × 13 个等压面：
    - 变量：地势高度 Z、温度 T、纬向风 U、经向风 V、相对湿度 R；
    - 等压层：50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa；
    - 共 5 × 13 = 65 个上空变量。
  - 地面 5 个变量：
    - T2M（2 m 气温）、U10、V10、MSL（海平面气压）、TP（6 小时累计降水）；
  - 总计通道数：$C=70$。

- **输入输出形式**
  - FuXi 基础模型（base FuXi）：自回归两步输入，预测下一步：
    - 输入：
      $$
      (\mathbf{X}^{t-1}, \mathbf{X}^{t})\in\mathbb{R}^{2\times C\times H\times W}
      $$
    - 输出：
      $$
      \hat{\mathbf{X}}^{t+1}\in\mathbb{R}^{C\times H\times W}
      $$
    - 其中 $H=721, W=1440, C=70$。
  - 自回归多步预测：以模型输出作为下一步输入，实现 6 小时间隔的长时效滚动。

- **级联（cascade）时间窗划分**
  - 将 0–15 天划分为三个 5 天时间窗（每窗 20 步）：
    1. FuXi-Short：0–5 天（步 1–20）；
    2. FuXi-Medium：5–10 天（步 21–40）；
    3. FuXi-Long：10–15 天（步 41–60）。
  - 级联关系：
    - 使用 FuXi-Short 运行获得第 20 步场，作为 FuXi-Medium 的初值；
    - 使用 FuXi-Medium 第 40 步场作为 FuXi-Long 的初值；
    - 三个模型级联得到完整 15 天预测。

- **对比基线**
  - ECMWF HRES（0.1°，137 层，10 天），HRES-fc0 作为其分析真值；
  - ECMWF EPS/EM（1 控制 + 50 成员，过去为 ~18 km / 91 层），ENS-fc0 为其分析真值；
  - GraphCast、FourCastNet、Pangu-Weather、SwinRDM 等最新 ML 模型。

---

### 2. 传统方案与现有 ML 模型痛点

- **NWP / ECMWF 系统**
  - HRES：0.1°，137 层，10 天高精度预报，但运行一次需数千节点、耗时数小时；
  - EPS：51 成员，历史上分辨率低于 HRES（~18 km / 91 层），受算力限制；
  - 优点：物理一致性、集合预报；
  - 局限：计算代价巨大，集合规模受限，且经验表明 EM 在长时效上技巧高于单一确定性预报。

- **已有 ML 模型共性痛点**
  - 迭代自回归导致长时效误差累积：
    - 多步 AR 导致输出偏离训练数据分布，产生发散或不合理场；
    - Weyn 多步损失、FourCastNet 2 步 fine-tuning、GraphCast curriculum 训练均为缓解此问题，但仍难在所有 lead 上同时最优。
  - 单一模型难以同时兼顾短、中、长 lead：
    - GraphCast：增加 AR 步数，长 lead 技巧提高，但短 lead 表现下降；
    - 直接预测远期 lead 虽精度高，但需要为每个 lead 训练单独模型，代价大；
  - 分辨率：以 WeatherBench 为代表，许多模型工作在 5.625°–1.40625° 的低分辨率，难以支撑业务应用。

- **关键挑战**
  - 15 天时效、0.25° 分辨率、70 多变量的 ML 预报，要：
    - 压制自回归长时效误差累积；
    - 在 deterministic 与 ensemble 指标上接近或达到 ECMWF EM；
    - 同时保持实用的算力与效率。

---

### 3. FuXi 整体方案与范式

- **总体思路**
  1. 设计强大的基础架构：基于 U-Transformer（Swin Transformer V2 + U-Net 结构），高效捕获时空依赖；
  2. 利用 cube embedding 减少时空维度，将 $(2,70,721,1440)$ 压缩到 $(C,180,360)$；
  3. 用两步自回归输入 $(X^{t-1},X^t)$ 提高稳定性；
  4. 在 base 模型上，通过 curriculum autoregressive 训练 + 极小学习率 fine-tuning，针对 0–5d, 5–10d, 10–15d 三个时间窗分别优化；
  5. 用三个模型级联减少长时效误差累积；
  6. 使用 Perlin 噪声 + Monte Carlo Dropout 生成 50 成员 ensemble，评估 CRPS、Spread、SSR 等集合指标。

- **核心组件**
  - Cube embedding：3D 卷积，将 $(T=2, H=721, W=1440)$ 降采到 $(T'=1, H'=180, W'=360)$，输出通道 $C=1536$；
  - U-Transformer：48 个 Swin Transformer V2 block，内部含 Down/Up Block 以及 skip 连接，采用 scaled cosine attention；
  - Down/Up Block：带 GN+SiLU 的残差 CNN 下采样/上采样模块；
  - FC 输出层：将特征映射回 70 通道，再通过双线性插值恢复到 $721\times 1440$；
  - 级联训练与缓存：通过预先缓存 FuXi-Short 在 2012–2017 年的 5 天预测，避免在 fine-tune FuXi-Medium/Long 时进行在线长链推理，节省显存和计算。

---

### 4. 模型架构与关键公式

#### 4.1 Cube embedding

- 输入：
  $$
  \mathbf{X}_{in}\in\mathbb{R}^{2\times C\times H\times W},\quad C=70, H=721, W=1440
  $$
- 3D 卷积核与步长：
  - kernel: $2\times 4\times 4$；
  - stride: $2\times 4\times 4$；
  - 输出通道：$C_{emb}=1536$。
- 输出：
  $$
  \mathbf{Z}\in\mathbb{R}^{C_{emb}\times 180\times 360}
  $$
- 归一化：LayerNorm 作用于 channel 维度以稳定训练。

#### 4.2 U-Transformer（Swin Transformer V2 + U-Net）

- 由 48 个 Swin Transformer V2 block 组成，注意力为 scaled cosine attention：
  $$
  \text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
  = \big(\cos(\mathbf{Q},\mathbf{K})/\tau + \mathbf{B}\big)\mathbf{V}
  $$
  - $\tau$：每层每个头独立的可学习标量；
  - $\mathbf{B}$：相对位置偏置；
  - cosine 归一化使注意力值更稳定。

- U 型结构：
  - Down Block：
    - 2D conv（kernel=3, stride=2）→ 降采样到 $C_{emb}\times 90\times 180$；
    - 残差块：两层 3×3 conv + GN + SiLU；
  - Up Block：
    - transpose conv（kernel=2, stride=2）→ 上采样回 $C_{emb}\times 180\times 360$；
    - 残差块与 Down Block 相同；
  - skip 连接：将 Down Block 输出与 Transformer block 输出 concat 后输入 Up Block。

#### 4.3 输出层与插值

- U-Transformer 输出：$\mathbf{H}\in\mathbb{R}^{C_{emb}\times 180\times 360}$；
- FC 层：线性映射 $C_{emb}\to 70$，得到：
  $$
  \tilde{\mathbf{X}}^{t+1}\in\mathbb{R}^{70\times 180\times 360}
  $$
- 双线性插值：恢复到 ERA5 原始网格：
  $$
  \hat{\mathbf{X}}^{t+1}=\mathrm{Interp}_{bilinear}(\tilde{\mathbf{X}}^{t+1}, H=721,W=1440)
  $$

#### 4.4 训练目标

- **预训练：单步 L1 损失**

  $$
  L_1 = \frac{1}{C\,H\,W} \sum_{c=1}^C \sum_{i=1}^H \sum_{j=1}^W a_i
  \left|\hat{X}^{t+1}_{c,i,j} - X^{t+1}_{c,i,j}\right|
  $$

  - $a_i$：纬向权重，随纬度增大而减小；
  - 损失在全变量和全栅格上平均。

- **fine-tune：多步自回归 L1（具体公式未展开）**
  - 通过 curriculum 训练，将自回归步数逐渐从 2 增加到 12；
  - 各步预测与对应真值之间的纬向加权 L1 损失求和。

---

### 5. 训练数据与超参数

- **数据集划分**
  - Train：1979–2015，样本数：
    $$
    54020 = 365\times 4\times 37
    $$
  - Val：2016–2017，2920 样本；
  - Test：2018，1460 样本（365×4）。

- **预训练设置**
  - 框架：PyTorch；
  - 硬件：8 × NVIDIA A100；
  - 迭代：40,000 step；
  - batch size：每 GPU 1；
  - 优化器：AdamW，$\beta_1=0.9,\beta_2=0.95$，weight decay=0.1；
  - 初始学习率：$2.5\times10^{-4}$；
  - DropPath 比例：0.2；
  - 混合精度：bfloat16；
  - 并行：FSDP（Fully-Sharded Data Parallel）；
  - 记忆优化：gradient checkpointing；
  - 训练时间：约 30 小时（8×A100）。

- **fine-tuning 设置**
  - FuXi-Short / Medium / Long：逐个在 base 模型上 fine-tune；
  - 学习率：常数 $1\times10^{-7}$；
  - 每个模型 fine-tune 时间：约 2 天（8×A100）；
  - 自回归步数在 fine-tune 中按照 curriculum 策略递增。

- **缓存策略**
  - 为 fine-tune FuXi-Medium，需要 FuXi-Short 第 20 步输出作为输入：
    - 若在线前向会导致显存和计算开销过大；
    - 实际做法：预先用 FuXi-Short 在 2012–2017 年跑出 0–5 天预测，并缓存到硬盘；
  - 同理，为 FuXi-Long 预先缓存 FuXi-Medium 第 40 步输出。

---

### 6. 评价指标与对比

#### 6.1 RMSE 与 ACC

- 采用 WeatherBench / GraphCast 常用的纬向加权 RMSE / ACC 定义，对每个变量 $c$、lead 步长 $\tau$：

  $$
  \mathrm{RMSE}(c,\tau)= \frac{1}{|D|} \sum_{t_0\in D}
  \sqrt{\frac{1}{HW}\sum_{i=1}^H\sum_{j=1}^W a_i\big(\hat{X}_{cij}^{t_0+\tau}-X_{cij}^{t_0+\tau}\big)^2}
  $$

  $$
  \mathrm{ACC}(c,\tau)= \frac{1}{|D|} \sum_{t_0\in D}
  \frac{\sum_{i,j} a_i (\hat{X}_{cij}^{t_0+\tau}-M_{cij}^{t_0+\tau})
                    (X_{cij}^{t_0+\tau}-M_{cij}^{t_0+\tau})}
       {\sqrt{\sum_{i,j} a_i (\hat{X}_{cij}^{t_0+\tau}-M_{cij}^{t_0+\tau})^2
               \sum_{i,j} a_i (X_{cij}^{t_0+\tau}-M_{cij}^{t_0+\tau})^2}}
  $$

  - $M$：1993–2016 年 ERA5 气候态均值；
  - ACC > 0.6 通常视为“skillful”。

- **归一化差值**（比较 FuXi 与 ECMWF EM、HRES）：
  - 归一化 RMSE 差：
    $$
    \Delta_{\mathrm{RMSE}} = \frac{\mathrm{RMSE}_A-\mathrm{RMSE}_B}{\mathrm{RMSE}_B}
    $$
  - 归一化 ACC 差：
    $$
    \Delta_{\mathrm{ACC}} = \frac{\mathrm{ACC}_A-\mathrm{ACC}_B}{1-\mathrm{ACC}_B}
    $$
  - 当 $\Delta_{\mathrm{RMSE}}<0$ 且 $\Delta_{\mathrm{ACC}}>0$ 时，模型 A 优于基线 B。

#### 6.2 集合指标：CRPS, Spread, SSR

- **CRPS**

  $$
  \mathrm{CRPS}=\int_{-\infty}^{\infty}\big[ F(\hat{X}_{cij}^{t_0+\tau})
  - \mathcal{H}(X_{cij}^{t_0+\tau}\le z)\big]\,dz
  $$

  - 实现：使用 `xskillscore` 包；
  - 对 ensemble，假设成员服从高斯分布，由 ensemble mean 与方差近似计算 CRPS；
  - 对 deterministic 预测时，CRPS 退化为 MAE。

- **Spread 与 SSR**

  - 集合 spread：
    $$
    \mathrm{Spread}(c,\tau)= \frac{1}{|D|}\sum_{t_0\in D}
    \sqrt{\frac{1}{HW}\sum_{i=1}^H\sum_{j=1}^W a_i\,\mathrm{var}(\tilde{X}_{cij}^{t_0+\tau})}
    $$
    - $\mathrm{var}$ 为 ensemble 维方差。

  - SSR（spread-skill ratio）：
    - 定义（论文未给出明确公式，遵循常规定义：SSR = Spread / RMSE(EM)）；
    - 理想 SSR ≈ 1：集合 spread 与 ensemble mean 的 RMSE 匹配；
    - SSR < 1：集合偏窄（under-dispersive）；SSR > 1：集合偏宽（over-dispersive）。

#### 6.3 主要对比结论

- 相对 HRES：
  - 对 8 个关键变量（MSL, T2M, U10, V10, Z500, T500, U500, V500）：
    - FuXi 与 GraphCast 在 7 天内均显著优于 HRES；
    - 7 天之后，FuXi 超过 GraphCast，并持续优于 HRES；
  - ACC > 0.6 的 skillful 时效：
    - Z500：从 HRES 的 9.25 天提升到 10.5 天；
    - T2M：从 10 天提升到 14.5 天。

- 相对 ECMWF EM：
  - 在 0–9 天范围内：FuXi deterministic 在多数变量与层次上 ACC 更高、RMSE 更低；
  - 9 天之后：FuXi 略差于 ECMWF EM，但整体 15 天内表现“可比”；
  - 统计上：在 240 组合（变量×层×lead）中，FuXi 的 ACC 更高占 67.92%，RMSE 更低占 53.75%。

- 相对 FuXi EM（FuXi ensemble mean）：
  - 短 lead（<3 天）：FuXi EM 稍逊于 deterministic FuXi；
  - 超 3 天后：FuXi EM 优于 deterministic（与 Pangu-Weather、FourCastNet 现象一致）。

- 空间误差分布：
  - Z500、T2M 在 5/10/15 天的 RMSE 空间分布显示：
    - 三个系统（FuXi、HRES、EM）在空间模式上类似，高纬误差大、热带较小，陆地误差大于海洋；
    - HRES-FuXi 的 RMSE 差图中大部分区域为红色（HRES 更差）；
    - EM-FuXi 的差图以白色为主（两者接近）。

- 集合性能：
  - CRPS：FuXi ensemble 在 Z500, T850, MSL, T2M 上，0–9 天内略好于 ECMWF ensemble，之后略差；
  - SSR：
    - FuXi ensemble 在早期 lead（Z500, T850, MSL）SSR>1，表现为 over-dispersive；
    - 随 lead 增大 SSR 逐渐下降至 <1，表现为 under-dispersive；
    - ECMWF ensemble 的 SSR 接近 1（T2M 除外），FuXi 在 T2M 上也始终 under-dispersive；
  - Spread：
    - ECMWF ensemble spread 随 lead 单调增加；
    - FuXi ensemble spread 先随 lead 增长，在 9 天之后开始减小（Perlin 噪声为流场无关扰动，长期积分后衰减）。

---

### 7. 创新点与局限

- **创新点**
  1. 提出级联 ML 架构：以三个预训练 FuXi 模型（Short/Medium/Long）级联，针对不同 lead 时间窗分别最优化，缓解单模型在全 lead 范围内难以兼顾的问题；
  2. 基于 U-Transformer（Swin Transformer V2 + U-Net）的大规模高分辨率架构：通过 cube embedding 与 U 型结构，使在 $0.25^\circ$ / 70 通道 / 双时间步输入下仍可训练；
  3. curriculum autoregressive 训练：在 fine-tune 过程中逐步增加自回归步数（2→12），提高长 lead 稳定性；
  4. 高效训练/推理工程实践：
     - FSDP + bf16 + gradient checkpointing；
     - 预缓存上一个模型输出供下一个模型 fine-tune 使用；
  5. 基于 Perlin 噪声 + Monte Carlo Dropout 的 ML 集合预报方案：
     - 同时对初值和模型物理（参数）施加扰动，构建 50 成员 ensemble；
  6. 结果层面：
     - 在 15 天 0.25° 分辨率下，使 deterministic FuXi 在 0–9 天整体优于 ECMWF EM，15 天内表现可比；
     - 显著延长 Z500 和 T2M 的 skillful lead（ACC>0.6）。

- **局限与未来工作**
  - 流场无关扰动（Perlin 噪声）长期积分会衰减，导致中长期 spread 偏小（under-dispersive），需发展 flow-dependent 扰动策略；
  - 仍依赖 NWP 生成的分析场作为初值，不是完全端到端；
  - 子季节（14–28 天）预报仍是“predictability desert”，FuXi 尚未覆盖；
  - 未来计划：
    - 扩展 cascade 架构到 14–28 天，研究 MJO、土壤湿度、积雪、平流层–对流层相互作用、海温等对可预报性的贡献；
    - 发展数据驱动的数据同化方法，直接从观测生成初始场，构建真正端到端 ML 预报系统。

---

## 维度二：基础组件 / 模块级知识

> 说明：以下所有代码片段均为“基于论文描述的高保真伪代码”，用于阐明结构与算法逻辑，不对应作者实际实现。

### 组件 A：Cube Embedding 模块

- **职责**
  - 将两帧高维 $0.25^\circ$ 全球多变量场 $(X^{t-1},X^t)$ 压缩为较小的时空体积，以降低后续 Transformer 计算成本。

- **输入 / 输出**
  - 输入：$X\in\mathbb{R}^{B\times 2\times C\times H\times W}$；
  - 输出：$Z\in\mathbb{R}^{B\times C_{emb}\times 180\times 360}$。

- **伪代码（CubeEmbedding）**

```python
# Pseudocode: Cube Embedding for FuXi (based on paper description)

class CubeEmbedding(nn.Module):
    def __init__(self, in_channels=70, emb_channels=1536):
        super().__init__()
        # (T=2, C, H, W) -> (C_emb, T/2, H/4, W/4) = (C_emb, 1, 180, 360)
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,  # applied channel-wise in (time,lat,lon)
            out_channels=emb_channels,
            kernel_size=(2, 4, 4),
            stride=(2, 4, 4)
        )
        self.ln = nn.LayerNorm(emb_channels)

    def forward(self, x):
        # x: (B, 2, C, H, W)
        # merge variable dimension into channels for 3D conv
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)  # paper does not detail exact arrangement
        # NOTE: precise tensor layout not fully specified in paper
        z = self.conv3d(x.unsqueeze(2))  # placeholder; real impl reshapes differently
        # z: (B, C_emb, T', H', W') -> squeeze time dim
        z = z.squeeze(2)  # (B, C_emb, 180, 360)
        # layernorm over channel dim
        z = z.permute(0, 2, 3, 1)           # (B, 180, 360, C_emb)
        z = self.ln(z)
        z = z.permute(0, 3, 1, 2)           # (B, C_emb, 180, 360)
        return z
```

> 注：论文只给出 kernel/stride 与输出大小，实际张量布局及是否先对时间/变量重排未完全展开，上述为保持形状一致的伪实现。

---

### 组件 B：U-Transformer（Swin Transformer V2 + U-Net）

- **职责**
  - 在压缩后的 $(C_{emb},180,360)$ 时空网格上捕获多尺度相关性；
  - 利用 Swin Transformer V2 的 scaled cosine attention 与 U-Net 的下/上采样结构。

- **伪代码（注意：块内部为简化表示）**

```python
# Pseudocode: Scaled cosine attention block (Swin Transformer V2 style)

class ScaledCosineAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # tau is learnable, per head and per layer in paper
        self.tau = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.rel_pos_bias = RelativePositionBias(num_heads)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)  # each: (B, N, H, d)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # cosine similarity
        attn = torch.einsum("bnhd,bmhd->bhnm", q, k)  # (B, H, N, N)
        attn = attn / self.tau + self.rel_pos_bias(attn)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v)
        out = out.reshape(B, N, C)
        return self.proj(out)
```

```python
# Pseudocode: U-Transformer main body

class UTransformer(nn.Module):
    def __init__(self, emb_channels=1536, depth=48):
        super().__init__()
        self.swin_blocks = nn.ModuleList([
            SwinV2Block(emb_channels) for _ in range(depth)
        ])
        self.down = DownBlock(emb_channels)  # -> (C, 90, 180)
        self.up = UpBlock(emb_channels)      # -> (C, 180, 360)

    def forward(self, z):
        # z: (B, C_emb, 180, 360)
        x = z
        for blk in self.swin_blocks[:24]:
            x = blk(x)
        x_down = self.down(x)             # (B, C_emb, 90, 180)
        x = x_down
        for blk in self.swin_blocks[24:]:
            x = blk(x)
        x_up = self.up(x, skip=x_down)    # skip-concat inside UpBlock
        return x_up  # (B, C_emb, 180, 360)
```

> 注：实际论文并未拆分多少层在下采样前后，这里仅为结构示意。

---

### 组件 C：FuXi 基础模型（单步预测）

- **职责**
  - 给定 $(X^{t-1}, X^t)$，预测 $X^{t+1}$；
  - 作为 FuXi-Short/Medium/Long 的基础架构。

- **伪代码**

```python
# Pseudocode: Base FuXi model (single-step)

class FuXiBase(nn.Module):
    def __init__(self, in_channels=70, emb_channels=1536):
        super().__init__()
        self.embed = CubeEmbedding(in_channels, emb_channels)
        self.u_transformer = UTransformer(emb_channels)
        self.fc = nn.Conv2d(
            in_channels=emb_channels,
            out_channels=in_channels,
            kernel_size=1
        )

    def forward(self, x_tminus1, x_t):
        # x_t*: (B, C, H, W)
        inp = torch.stack([x_tminus1, x_t], dim=1)   # (B, 2, C, H, W)
        z = self.embed(inp)                         # (B, C_emb, 180, 360)
        h = self.u_transformer(z)                   # (B, C_emb, 180, 360)
        y_low = self.fc(h)                          # (B, C, 180, 360)
        y = F.interpolate(y_low, size=(721, 1440), mode="bilinear")
        return y                                    # (B, C, 721, 1440)
```

---

### 组件 D：预训练与多步 fine-tuning（FuXi-Short/Medium/Long）

- **职责**
  - 预训练：学习单步映射 $(X^{t-1},X^t)\to X^{t+1}$；
  - fine-tune：采用 curriculum autoregressive 训练，在指定时间窗内最小化多步 L1 损失。

- **预训练伪代码**

```python
# Pseudocode: Pre-training (single-step L1)

model = FuXiBase()
optimizer = AdamW(model.parameters(), lr=2.5e-4,
                  betas=(0.9, 0.95), weight_decay=0.1)
scheduler = CosineLRScheduler(optimizer, T_max=40000)  # conceptually

for it, (X_tm1, X_t, X_tp1) in enumerate(train_loader):
    pred = model(X_tm1, X_t)
    loss = lat_weighted_L1(pred, X_tp1, lat_weights)
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
    scheduler.step()
```

- **FuXi-Short fine-tune（0–5d，多步 AR）**

```python
# Pseudocode: Autoregressive curriculum fine-tuning (FuXi-Short)

model.load_state_dict(pretrained_state)
optimizer = AdamW(model.parameters(), lr=1e-7)

for n_steps in curriculum_steps:  # e.g., 2, 4, 6, ..., 12
    for epoch in range(epochs_per_stage):
        for X_seq in train_loader_short:  # sequence length >= n_steps+2
            # X_seq: (B, T_seq, C, H, W)
            X_tm1 = X_seq[:, 0]
            X_t   = X_seq[:, 1]
            loss = 0.0
            for k in range(n_steps):
                X_tp1 = model(X_tm1, X_t)
                target = X_seq[:, k + 2]  # ground truth at t+1
                loss += lat_weighted_L1(X_tp1, target, lat_weights)
                X_tm1, X_t = X_t, X_tp1.detach()  # teacher forcing optional
            loss /= n_steps
            loss.backward()
            optimizer.step(); optimizer.zero_grad()
```

- **FuXi-Medium / FuXi-Long fine-tune**
  - 类似 FuXi-Short，但训练序列起始于 5–10d / 10–15d 时间窗；
  - 输入初值来自缓存的 FuXi-Short / FuXi-Medium 输出，而非在线推理。

---

### 组件 E：级联推理（15 天）

- **职责**
  - 将 FuXi-Short / Medium / Long 串联，实现从 $t_0$ 开始的 15 天、6 小时步长的全球预报。

- **伪代码**

```python
# Pseudocode: Cascaded 15-day forecast

def forecast_15d(fuxi_short, fuxi_med, fuxi_long, X_tm1, X_t):
    preds = []

    # 0–5 days: 20 steps with FuXi-Short
    for _ in range(20):
        X_tp1 = fuxi_short(X_tm1, X_t)
        preds.append(X_tp1)
        X_tm1, X_t = X_t, X_tp1

    # 5–10 days: 20 steps with FuXi-Medium
    for _ in range(20):
        X_tp1 = fuxi_med(X_tm1, X_t)
        preds.append(X_tp1)
        X_tm1, X_t = X_t, X_tp1

    # 10–15 days: 20 steps with FuXi-Long
    for _ in range(20):
        X_tp1 = fuxi_long(X_tm1, X_t)
        preds.append(X_tp1)
        X_tm1, X_t = X_t, X_tp1

    # preds: list of 60 tensors (B, C, H, W)
    return torch.stack(preds, dim=1)
```

---

### 组件 F：FuXi 集合预报（Perlin 噪声 + MC Dropout）

- **职责**
  - 通过扰动初始场与模型参数，生成 50 成员集合；
  - 计算 ensemble mean / spread / CRPS / SSR 等指标。

- **初始场扰动**
  - 对每个成员 $e=1..49$，叠加 4 个 octave 的 Perlin 噪声，缩放因子 0.5；
  - 噪声周期数：
    - channel 维：1；
    - 纬度 / 经度：6；
  - 伪公式：
    $$
    X_e^{t_0} = X^{t_0} + \alpha\,\mathrm{PerlinNoise}(c\_period=1, lat\_period=6, lon\_period=6)
    $$
    - $\alpha$ 为缩放系数（论文中为 0.5）。

- **模型参数扰动（MC Dropout）**
  - 在 U-Transformer / FC 等层加入 Dropout 层，fine-tune 后在推理阶段保持 `train()` 模式，dropout rate=0.2；
  - 通过随机失活神经元模拟参数不确定性。

- **伪代码**

```python
# Pseudocode: FuXi ensemble generation

class FuXiEnsembleWrapper(nn.Module):
    def __init__(self, short, med, long, dropout_p=0.2):
        super().__init__()
        self.short = apply_mc_dropout(short, p=dropout_p)
        self.med = apply_mc_dropout(med, p=dropout_p)
        self.long = apply_mc_dropout(long, p=dropout_p)

    def forward_member(self, X_tm1, X_t):
        return forecast_15d(self.short, self.med, self.long, X_tm1, X_t)


def generate_perlin_perturbation(shape, seed):
    # uses external perlin-numpy implementation (paper reference)
    np.random.seed(seed)
    noise = perlin_noise_3d(
        shape=shape, octaves=4, scale=0.5,
        periods=(1, 6, 6)
    )
    return torch.from_numpy(noise).float()


def ensemble_forecast(fuxi_ens, X_tm1, X_t, n_members=50):
    members = []
    for e in range(n_members):
        if e == 0:
            X0_tm1, X0_t = X_tm1, X_t  # control
        else:
            eps_tm1 = generate_perlin_perturbation(X_tm1.shape, seed=e)
            eps_t = generate_perlin_perturbation(X_t.shape, seed=100 + e)
            X0_tm1 = X_tm1 + eps_tm1.to(X_tm1.device)
            X0_t = X_t + eps_t.to(X_t.device)
        Y_e = fuxi_ens.forward_member(X0_tm1, X0_t)  # (B, 60, C, H, W)
        members.append(Y_e)
    return torch.stack(members, dim=0)  # (E, B, 60, C, H, W)
```

---

### 组件 G：评估与可视化逻辑

- **职责**
  - 计算 deterministic 与 ensemble 的 RMSE / ACC / CRPS / Spread / SSR；
  - 生成 Z500/T2M 空间 RMSE 与差值图，支持对 HRES/EM 的空间比较。

- **伪代码轮廓**

```python
# Pseudocode: Deterministic metrics

def evaluate_deterministic(pred_seq, true_seq, clim, lat_weights):
    # pred_seq,true_seq: (N_cases, T, C, H, W)
    metrics = {}
    for c in vars_of_interest:
        for tau in lead_steps:
            P = pred_seq[:, tau, c]
            T_true = true_seq[:, tau, c]
            M = clim[c, tau]
            rmse = compute_rmse(P, T_true, lat_weights)
            acc = compute_acc(P, T_true, M, lat_weights)
            metrics[(c, tau)] = {"rmse": rmse, "acc": acc}
    return metrics

# Pseudocode: Ensemble metrics using xskillscore

def evaluate_ensemble(members, true_seq):
    # members: (E, N_cases, T, C, H, W)
    ens_mean = members.mean(axis=0)
    spread = compute_spread(members, lat_weights)
    rmse_em = compute_rmse(ens_mean, true_seq, lat_weights)
    ssr = spread / rmse_em
    crps = xskillscore.crps_gaussian(
        truth=true_seq, mu=ens_mean, sig=members.var(axis=0)**0.5
    )
    return {"spread": spread, "ssr": ssr, "crps": crps}
```

---

## 小结

- FuXi 通过 U-Transformer + cube embedding 架构，在 0.25°、70 变量、6 小时时间分辨率的 ERA5 数据上，构建了一个能够进行 15 天全球预报的 ML 系统；
- 通过对 0–5d、5–10d、10–15d 三个时间窗分别 fine-tune 并级联，有效缓解了自回归长时效误差累积，使得 FuXi 在 deterministic 与 ensemble 指标上均能在 0–9 天内优于 ECMWF EM，并在 15 天范围内表现可比；
- 结合 Perlin 噪声与 MC Dropout 的 ensemble 方案，使 FuXi 能在有限算力下生成 50 成员集合，支持 CRPS、Spread、SSR 等概率指标评估；
- 论文指出 FuXi 仍依赖 NWP 的分析初值，未来方向包括子季节（14–28 天）扩展与数据驱动同化，从而构建真正端到端、系统性无偏且高效的 ML 天气预报系统。