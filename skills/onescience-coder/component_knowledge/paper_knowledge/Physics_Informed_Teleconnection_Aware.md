# TelePiT：Physics-Informed Teleconnection-Aware Transformer for Global S2S Forecasting

> 说明：本文件基于论文文字内容进行结构化总结和高层伪代码抽象，不是官方实现代码。

---

## 一、整体模型维度（任务 / 数据 / 结构 / 训练 / 结果）

### 1.1 任务与问题设定

- **任务类型**：全球次季节到季节（S2S）预测。
  - 预测对象：多变量、大气三维场（多气压层）+ 近地面变量。
  - 时间尺度：超越天气可预报极限（约 2 周），目标是 3–4 周、5–6 周的平均状态。
- **输入/输出形式**：
  - 输入：某一初始日的全球大气状态
    $$\mathbf{X}_{t_1} \in \mathbb{R}^{C \times H \times W}$$
    - $H, W$：纬度 × 经度网格（例如 121×240，对应 1.5° 网格）。
    - $C$：变量数（多层多变量）。
  - 输出：两个未来多周窗口的空间场均值：
    $$
    \mathcal{F}_\Theta(\mathbf{X}_{t_1}) \to (\hat{\mathbf{Y}}_{t_{15}:t_{28}}, \hat{\mathbf{Y}}_{t_{29}:t_{42}}),
    $$
    其中：
    - $\hat{\mathbf{Y}}_{t_{15}:t_{28}} \in \mathbb{R}^{C \times H \times W}$：第 3–4 周平均；
    - $\hat{\mathbf{Y}}_{t_{29}:t_{42}} \in \mathbb{R}^{C \times H \times W}$：第 5–6 周平均。
- **挑战特征**：
  - 时间尺度跨越天气混沌极限，需要捕捉较慢的气候信号、模态和边界过程；
  - 多尺度动力过程：行星波、准定常波、对流尺度等耦合；
  - 远程相关（teleconnection）：如 NAO、ENSO、MJO 等远程模态对区域气候有显著影响。

### 1.2 数据与实验设置

- **数据集**：ERA5 + ChaosBench 处理方案。
  - 空间分辨率：1.5° × 1.5°，网格大小约 121 × 240；
  - 时间：1979–2018；
    - 训练集：1979–2016；
    - 验证集：2017；
    - 测试集：2018。
- **变量构成**：共 63 个变量：
  - 6 个气压层三维场，每个在 10 个气压层上：
    - Geopotential (Z)、Specific humidity (q)、Temperature (T)、
      Zonal wind (u)、Meridional wind (v)、Vertical velocity (w)；
    - 压力层：10, 50, 100, 200, 300, 500, 700, 850, 925, 1000 hPa。
  - 3 个单层近地面变量：
    - 2 m temperature (T2m)、10 m u-wind、10 m v-wind。
- **任务设置**：
  - 给定初始日大气状态（全变量），预测接下来第 3–4 周、第 5–6 周窗口的平均场；
  - 采用 climatology 作为气候基准，用于计算 anomaly 与 ACC。

### 1.3 模型整体结构（TelePiT）

TelePiT 由三大关键模块组成：

1. **Spherical Harmonic Embedding（球谐嵌入）**：
   - 将经纬格点上的多变量场嵌入到一个沿纬向的一维 token 序列；
   - 使用球谐启发的正余弦位置编码，显式编码纬度和经度的周期结构；
   - 对经度方向做 zonal 平均，得到每一纬带的代表性向量，体现大尺度纬向相关性。

2. **Multi-Scale Physics-Informed Neural ODE（多尺度物理约束 Neural ODE）**：
   - 在纬度方向上做可学习的多尺度分解（类似小波/多尺度分解），得到 L+1 个频带；
   - 对每个频带在 latent 空间中演化一个物理约束 ODE：
     - 含**扩散项**、**平流项**、**外强迫项**和**神经网络校正项**；
     - 在隐空间中重现大气动力方程结构（advection–diffusion–forcing + NN correction）。

3. **Teleconnection-Aware Transformer（遥相关感知 Transformer）**：
   - 对每个频带的纬向 token 序列进行自注意力建模；
   - 通过显式学习的遥相关模式向量，构造“teleconnection query”，给 self-attention logits 加偏置；
   - 使注意力机制更关注目前活跃的全球遥相关模式（如 NAO、MJO 等）。

此外，TelePiT 还包括：
- **Cross-Scale Interaction / Fusion**：多尺度 Transformer 输出之间的自上而下融合；
- **Forecasting Head**：将融合后的纬向表示映射回 (C, H, W) 空间场，输出两个多周窗口的均值。

### 1.4 训练与指标

- **损失函数**：
  - 目标是最小化两段时间窗口的加权 MSE：
    $$
    \mathcal{L} = \frac{1}{2 C H W} \left(\|\hat{\mathbf{Y}}_{15:28}^{(1)} - \mathbf{Y}_{15:28}^{(1)}\|_2^2
    + \|\hat{\mathbf{Y}}_{29:42}^{(2)} - \mathbf{Y}_{29:42}^{(2)}\|_2^2\right).
    $$
  - 训练时对每个样本（初始日）同时预测两个时间窗口。
- **评价指标**：
  1. 加权 RMSE（↓）：考虑纬向 cos(φ) 权重，匹配球面几何；
  2. ACC（↑）：对 anomaly 的空间相关系数，衡量模式一致性；
  3. Spectral Divergence / SpecDiv：基于频谱的距离，比较预测与真值在频域上的功率谱差异，反映物理结构是否真实。

### 1.5 与基线模型对比（定性总结）

- **数据驱动基线**：
  - FuXi-S2S（ERA5 上的 encoder–decoder 架构，用于降水等 S2S 任务）；
  - CirT（几何感知 Transformer，建模球面结构）；
  - 其他 ChaosBench 中的强基线（如 ClimaX、FourCastNet 类）。
- **数值模式基线**：
  - 传统 S2S NWP 系统（如 ECMWF、NCEP 的 S2S 产品）。

**实验结论（论文主张）**：
- TelePiT 在所有预报时效（weeks 3–4, 5–6）和多个变量上：
  - RMSE 更低、ACC 更高，尤其在周 5–6 长时效上优势更显著；
  - 在 Spectral Divergence 上更接近真实大气的能谱分布（例如保持大尺度行星波能量）；
  - 对关键大尺度模态（如 NAO/MJO）表现出更好的模式捕捉与位相预报能力。
- 物理约束 ODE + 遥相关感知注意力的组合，是优于“单纯 geometry-aware Transformer”的关键。

---

## 二、组件维度：可复用建模范式与伪代码

### 2.1 球谐启发的 Spherical Harmonic Embedding 组件

**子需求**：
- 将全球 C×H×W 多变量场映射到一维纬向 token 序列，保留球面几何和纬向结构；
- 对经度方向做物理上合理的压缩（zonal average），突出大尺度环流特征；
- 引入球谐启发的正余弦位置编码，编码纬度和经度的周期性。

**关键要点**：
- 输入：X ∈ R^{C×H×W}；网格 G_{h,w} = (θ_h, φ_w)；
- Zonal 平均：
  - 对每个纬度 i：
    $$\mathbf{u}_i = \frac{1}{W} \sum_{j=1}^W X_{:, i, j} \in \mathbb{R}^{C}.$$
- 线性投影到 D_emb 维隐空间：
  $$\mathbf{h}_i = W \mathbf{u}_i + b \in \mathbb{R}^{D_{emb}}.$$
- 位置编码：
  - 纬度编码 E_lat(i,·)：多频率 sin/cos((k+1) θ_i)；
  - 经度编码 E_lon(j,·)：多频率 sin/cos((q+1) φ_j)；
  - 对经度再做平均得到 p^{lon}；再与 p_i^{lat} 拼接成 p_i ∈ R^{D_emb}。
- 最终 token 表示：
  $$\mathbf{z}_i = W_{proj}(\mathbf{h}_i + \mathbf{p}_i) + b_{proj},$$
  堆叠成 Z ∈ R^{H×D_emb}。

**伪代码框架**：

```python
class SphericalHarmonicEmbedding(nn.Module):
    def __init__(self, C, H, W, d_lat, d_lon, d_emb):
        super().__init__()
        assert d_lat + d_lon == d_emb
        self.W = nn.Linear(C, d_emb)
        # learnable harmonic encodings
        self.E_lat = nn.Parameter(torch.randn(H, d_lat))
        self.E_lon = nn.Parameter(torch.randn(W, d_lon))
        self.proj = nn.Linear(d_emb, d_emb)

    def forward(self, X, theta, phi):
        # X: (B, C, H, W)
        B, C, H, W = X.shape
        # 1) zonal average over longitude
        u = X.mean(dim=-1)  # (B, C, H)
        u = u.permute(0, 2, 1)  # (B, H, C)

        # 2) project to latent
        h = self.W(u)  # (B, H, d_emb)

        # 3) harmonic positional encoding
        p_lat = self.E_lat  # (H, d_lat)
        p_lon = self.E_lon.mean(dim=0, keepdim=True).repeat(H, 1)  # (H, d_lon)
        p = torch.cat([p_lat, p_lon], dim=-1)  # (H, d_emb)
        p = p.unsqueeze(0).expand(B, -1, -1)

        # 4) final embedding
        z = self.proj(h + p)  # (B, H, d_emb)
        return z  # token sequence along latitude
```

> 该组件适合任何“全球球面场 → 纬向 token 序列”的任务，可与任意 Transformer/ODE 组合。

---

### 2.2 Learnable Multi-Scale Decomposition 组件

**子需求**：
- 在纬向 token 序列上构建多尺度频带表示，区分低频行星波与高频扰动；
- 采用可学习的“类小波”分解，而非固定基函数；
- 为后续物理 ODE 和 Transformer 提供按频带分组的输入。

**关键公式**：
- 初始尺度：A_0 = Z ∈ R^{H×D_emb}；
- 对 ℓ = 1..L：
  $$[A_\ell, D_\ell] = split(MLP_\ell(A_{\ell-1}), \text{dim}=-1),$$
  其中：
  - MLP_ℓ: R^{D_emb} → R^{2 D_emb}，带 GELU；
  - split 沿通道维切分成近似 (A_ℓ) 和细节 (D_ℓ)。
- 频带集合：
  - X^{(0)} = Ψ A_L：最低频；
  - X^{(ℓ)} = D_{L+1-ℓ}（从低到高频重新排序）。

**伪代码框架**：

```python
class MultiScaleDecomposition(nn.Module):
    def __init__(self, d_emb, L):
        super().__init__()
        self.L = L
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_emb, 2 * d_emb),
                nn.GELU(),
                nn.Linear(2 * d_emb, 2 * d_emb)
            )
            for _ in range(L)
        ])
        self.proj_low = nn.Linear(d_emb, d_emb)  # Ψ

    def forward(self, Z):
        # Z: (B, H, d_emb)
        A = Z
        As, Ds = [], []
        for ell in range(self.L):
            out = self.mlps[ell](A)  # (B, H, 2*d_emb)
            A, D = out.chunk(2, dim=-1)
            As.append(A)
            Ds.append(D)
        A_L = As[-1]
        X_bands = []
        X_bands.append(self.proj_low(A_L))  # X^(0)
        for ell in range(self.L):
            X_bands.append(Ds[self.L - 1 - ell])
        # X_bands: list of (B, H, d_emb), len = L+1
        return X_bands
```

---

### 2.3 Physics-Informed Neural ODE 组件

**子需求**：
- 在 latent 空间中表达大气的经向平流–扩散–外强迫动力学；
- 保持物理结构（diffusion/advection/forcing），同时用 NN 校正复杂非线性/未解析过程；
- 按频带分别演化，体现多尺度动力差异。

**关键公式**：对每个频带 X^{(ℓ)} = [x_1, …, x_H]，x_i ∈ R^{D_emb}：

- ODE：
  $$\frac{d \mathbf{x}_i}{dt} = \gamma \cdot \tanh(\mathbf{R}_i),$$
  其中：
  $$
  \mathbf{R}_i = \underbrace{\nu \odot (\mathbf{x}_{i+1} - 2\mathbf{x}_i + \mathbf{x}_{i-1})}_{\text{diffusion}}
  + \underbrace{\mu \odot \frac{\mathbf{x}_{i+1} - \mathbf{x}_{i-1}}{2}}_{\text{advection}}
  + \underbrace{f}_{\text{forcing}}
  + \underbrace{\alpha \cdot MLP(\mathbf{x}_i)}_{\text{neural correction}}.
  $$
- 边界条件：x_0 = x_{H+1} = 0（极区 zero padding）；
- 数值积分（Euler）：
  $$\mathbf{x}_i(t+\Delta t) = \mathbf{x}_i(t) + \Delta t \cdot \frac{d \mathbf{x}_i}{dt}.$$

**伪代码框架**：

```python
class PhysicsInformedLatentODE(nn.Module):
    def __init__(self, d_emb, gamma=1e-2, dt=1.0):
        super().__init__()
        self.nu   = nn.Parameter(torch.randn(d_emb))   # diffusion
        self.mu   = nn.Parameter(torch.randn(d_emb))   # advection
        self.f    = nn.Parameter(torch.zeros(d_emb))   # forcing
        self.alpha = nn.Parameter(torch.tensor(1.0))   # NN weight
        self.gamma = gamma
        self.dt    = dt
        self.mlp   = nn.Sequential(
            nn.Linear(d_emb, 2 * d_emb),
            nn.GELU(),
            nn.Linear(2 * d_emb, d_emb)
        )

    def step(self, X):
        # X: (B, H, d_emb)
        B, H, D = X.shape
        # pad for poles
        x_pad = torch.zeros(B, H + 2, D, device=X.device, dtype=X.dtype)
        x_pad[:, 1:-1, :] = X
        x_ip1 = x_pad[:, 2:, :]
        x_i   = x_pad[:, 1:-1, :]
        x_im1 = x_pad[:, :-2, :]

        diffusion = self.nu * (x_ip1 - 2 * x_i + x_im1)
        advection = self.mu * (x_ip1 - x_im1) / 2.0
        forcing   = self.f.view(1, 1, -1)
        nn_corr   = self.alpha * self.mlp(x_i)

        R = diffusion + advection + forcing + nn_corr
        dxdt = self.gamma * torch.tanh(R)
        X_next = X + self.dt * dxdt
        return X_next

    def forward(self, X, T=1):
        # integrate for T steps in latent time
        for _ in range(T):
            X = self.step(X)
        return X
```

- 在完整 TelePiT 中，对每个频带 X^{(ℓ)} 分别应用该 ODE，得到演化后的 \tilde{X}^{(ℓ)}。

---

### 2.4 Teleconnection-Aware Self-Attention 组件

**子需求**：
- 显式编码全球遥相关模式（teleconnections），使注意力机制更关注与当前气候模态相关的纬带；
- 提供一个“全局气候状态 → teleconnection 向量 → 注意力偏置”的链路。

**关键步骤**：
1. 对每个频带的纬向序列 \tilde{X}^{(ℓ)} ∈ R^{H×D_emb}：
   - 计算 Q, K, V：
     $$Q = \tilde{X}^{(ℓ)} W^Q,\quad K = \tilde{X}^{(ℓ)} W^K,\quad V = \tilde{X}^{(ℓ)} W^V.$$
2. 计算全局大气状态向量：
   $$\bar{x} = \frac{1}{H} \sum_{i=1}^H \tilde{x}_i.$$
3. 通过可学习的“遥相关模式基”生成 teleconnection 向量：
   - teleconnection basis：P_j ∈ R^{D_emb}, j = 1..n_p；
   - 权重：ω = softmax( \bar{x} W^p ) ∈ R^{n_p}；
   - 全局 teleconnection 向量：
     $$c = \sum_{j=1}^{n_p} ω_j P_j.$$
4. 将 c 投影到 query 空间得到 teleconnection query：
   $$q^{tel} = c W^Q \in \mathbb{R}^{d_k}.$$
5. 对所有纬带 key 计算 teleconnection 打分：
   $$b_j = \frac{1}{\sqrt{d_k}} (q^{tel} \cdot K_j).$$
6. 将 b_j 作为 bias 加入常规 self-attention logits：
   $$\tilde{A}_{ij} = \frac{1}{\sqrt{d_k}} (Q_i \cdot K_j) + \lambda b_j,$$
   其中 λ 控制定量影响。

**伪代码框架**：

```python
class TeleconnectionAwareAttention(nn.Module):
    def __init__(self, d_emb, n_heads, n_patterns, lambda_tc=1.0):
        super().__init__()
        self.n_heads = n_heads
        self.dk = d_emb // n_heads
        self.lambda_tc = lambda_tc
        self.W_q = nn.Linear(d_emb, d_emb)
        self.W_k = nn.Linear(d_emb, d_emb)
        self.W_v = nn.Linear(d_emb, d_emb)
        self.W_p = nn.Linear(d_emb, n_patterns)
        self.P   = nn.Parameter(torch.randn(n_patterns, d_emb))

    def forward(self, X):
        # X: (B, H, d_emb)
        B, H, D = X.shape
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # reshape to multi-head form
        def split_heads(t):
            return t.view(B, H, self.n_heads, self.dk).transpose(1, 2)  # (B, nH, H, dk)
        Qh, Kh, Vh = map(split_heads, (Q, K, V))

        # global climate state
        x_bar = X.mean(dim=1)  # (B, d_emb)
        omega = torch.softmax(self.W_p(x_bar), dim=-1)  # (B, n_patterns)
        c = omega @ self.P  # (B, d_emb)
        q_tel = self.W_q(c).view(B, self.n_heads, 1, self.dk)  # (B, nH, 1, dk)

        # teleconnection bias
        # Kh: (B, nH, H, dk)
        b = (q_tel * Kh).sum(dim=-1) / math.sqrt(self.dk)  # (B, nH, H)

        # standard attention logits
        logits = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(self.dk)  # (B, nH, H, H)
        logits = logits + self.lambda_tc * b.unsqueeze(-2)         # add bias on keys

        attn = torch.softmax(logits, dim=-1)
        out = attn @ Vh  # (B, nH, H, dk)
        out = out.transpose(1, 2).contiguous().view(B, H, D)
        return out
```

> 该组件可以与任意“纬向 token 序列”模型结合，用于显式编码 teleconnection 模式。

---

### 2.5 多尺度跨频带交互与 Forecast Head 组件

**子需求**：
- 让不同频带之间信息交互（例如低频行星波调制高频天气扰动）；
- 将各频带的 teleconnection-aware 表达融合，得到统一的纬向表示；
- 映射回 (C, H, W) 空间场，并输出两个时间窗口的平均值。

**跨尺度交互**：
- 对每个频带 ℓ：
  $$\hat{X}^{(ℓ)} = TA\text{-}Transformer_ℓ(\tilde{X}^{(ℓ)}).$$
- 自上而下（低频→高频）融合：
  $$\hat{X}^{(ℓ+1)} \leftarrow \text{LayerNorm}( MLP([\hat{X}^{(ℓ)}, \hat{X}^{(ℓ+1)}]) ).$$
- 最终融合：
  $$Z_{final} = \frac{1}{L+1} \sum_{ℓ=0}^L \hat{X}^{(ℓ)}.$$

**Forecast Head**：
- 对每个纬度 i：
  - 取 z_i^{final} ∈ R^{D_emb}；
  - 通过两层 MLP 映射到 2 C W 维向量：包含两个时间窗口、所有变量、所有经度：
    $$\mathbf{y}_i = W^{(2)} GELU(W^{(1)} z_i^{final} + b^{(1)}) + b^{(2)},\quad \mathbf{y}_i \in \mathbb{R}^{2 C W}.$$
- 重塑并堆叠所有纬度得到：
  - \hat{Y}_{15:28}^{(1)}, \hat{Y}_{29:42}^{(2)} ∈ R^{C×H×W}。

**伪代码框架（汇总端到端前向）**：

```python
class TelePiT(nn.Module):
    def __init__(self, C, H, W, d_emb, L, n_heads, n_patterns):
        super().__init__()
        self.embed = SphericalHarmonicEmbedding(C, H, W, d_lat=..., d_lon=..., d_emb=d_emb)
        self.msdec = MultiScaleDecomposition(d_emb, L=L)
        self.odes  = nn.ModuleList([PhysicsInformedLatentODE(d_emb) for _ in range(L+1)])
        self.ta_blocks = nn.ModuleList([
            nn.Sequential(
                TeleconnectionAwareAttention(d_emb, n_heads, n_patterns),
                nn.LayerNorm(d_emb),
                nn.Sequential(
                    nn.Linear(d_emb, 4*d_emb), nn.GELU(), nn.Linear(4*d_emb, d_emb)
                ),
                nn.LayerNorm(d_emb)
            )
            for _ in range(L+1)
        ])
        self.cross_scale_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*d_emb, 2*d_emb), nn.GELU(), nn.Linear(2*d_emb, d_emb)
            )
            for _ in range(L)
        ])
        self.head1 = nn.Linear(d_emb, 2 * C * W)

    def forward(self, X, theta, phi):
        # 1) embedding
        Z = self.embed(X, theta, phi)  # (B, H, d_emb)

        # 2) multi-scale decomposition
        X_bands = self.msdec(Z)  # list of (B, H, d_emb), len L+1

        # 3) physics-informed ODE evolution
        evolved = []
        for band, ode in zip(X_bands, self.odes):
            evolved.append(ode(band))

        # 4) teleconnection-aware transformer per band
        hats = []
        for Xl, block in zip(evolved, self.ta_blocks):
            Yl = block(Xl)  # 包含 attn + FFN
            hats.append(Yl)

        # 5) cross-scale fusion (low -> high)
        for ell in range(len(hats) - 1):
            concat = torch.cat([hats[ell], hats[ell+1]], dim=-1)
            fused  = self.cross_scale_mlp[ell](concat)
            hats[ell+1] = nn.LayerNorm(hats[ell+1].size(-1))(fused)

        Z_final = sum(hats) / len(hats)  # (B, H, d_emb)

        # 6) forecasting head
        B, H, D = Z_final.shape
        Y = self.head1(Z_final)  # (B, H, 2*C*W)
        Y = Y.view(B, H, 2, C, W)  # two windows
        Y = Y.permute(0, 2, 3, 1, 4)  # (B, 2, C, H, W)
        Y_34, Y_56 = Y[:, 0], Y[:, 1]
        return Y_34, Y_56
```

---

### 2.6 经验启示与可迁移范式

1. **范式：球面几何 + 纬向 token 序列**
   - 将全球网格场压缩为“沿纬度的一维序列 + 球谐位置编码”，是在保持几何先验的前提下，显著减少 token 数量的一种有效方式；
   - 适用于全球 S2S/季节气候/年际预测等大量使用纬向结构的任务。

2. **范式：多尺度频带 + 物理 ODE 组合**
   - 将 latent 表达按频带分解，再对每个频带施加结构化 ODE（扩散 + 平流 + 强迫 + NN 校正）：
     - 低频：行星波、QBO、PDO 等长期模态；
     - 高频：天气尺度扰动；
   - 该设计可迁移到其他多尺度物理系统（海洋环流、海气耦合等）。

3. **范式：显式 Teleconnection 建模**
   - 通过“全局状态 → teleconnection basis 加权 → 注意力偏置”这一路径，
     - 使模型能在不同气候态（如 ENSO 位相、MJO 活跃度）下改变其注意力模式；
   - 同类思想可用于：
     - ENSO-aware、MJO-aware 的区域降水预测；
     - 将观测/再分析中提取的 EOF/PC 模式作为 P_j 初始化。

4. **范式：物理项 + NN correction 的混合动力学**
   - 在 ODE 中保留最重要的物理算子（advection, diffusion, forcing），
     - 用少量参数捕捉可解释结构；
     - 再用 NN 校正复杂、未解析过程；
   - 在长时效预测中，这有助于保持物理一致性与能谱合理性（尤其对 Spectral Divergence 指标）。

5. **范式：S2S 任务的 week-mean 输出设计**
   - 直接预测周平均场（而非逐日/逐小时序列），
     - 与用户需求（农业、水资源、能源）更匹配；
     - 减少高频噪声的影响；
   - 该设计可借鉴到任意“事件窗口平均”预测任务（如极端事件频数、月平均水文量）。

以上组件与范式，可与现有的 FuXi-S2S、CirT、ClimaX、GraphCast 等模型输出相结合，构建：
- 以 TelePiT 式 Physics-Informed + Teleconnection-Aware 模块为“后处理头”；
- 或直接在“foundation model 输出 + ERA5”之上训练新一代 S2S/季节预测网络。