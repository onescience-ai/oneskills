# GenCast 模型知识提取

信息来源：Price et al., “Probabilistic weather forecasting with machine learning”（Nature, 2024）（下文内容仅基于论文文本与图示，不做主观扩展；未明确之处保留为“未在文中明确给出”或仅作结构性描述）。

---

## 维度一：整体模型与任务知识

### 1. 任务与问题设定

- **总体目标**
  - 构建 GenCast：一个高分辨率、全概率（probabilistic）的中期全球天气预报系统。
  - 与当前最强业务集合预报 ECMWF ENS 对比，在整体技巧和极端事件、热带气旋路径、风电发电预测等方面显著超越或持平，同时具有更高计算效率。

- **预报范围与分辨率**
  - 时间：15 天预报，时间步长 12 小时，共 $T=30$ 步；
  - 空间：全球 $0.25^\circ$ 等经纬网格；
  - 变量：
    - 6 个地面变量 + 6 个大气变量 × 13 个等压层（共 80+ 变量；具体列表在 Extended Data Table 1 中，文本只给出数量级）。

- **状态表示**
  - 将某时刻天气状态记为 $\mathbf{X}^t$：
    - $\mathbf{X}^t \in \mathbb{R}^{C\times H\times W}$，$C\approx 80$（6 surface + 6 atm×13 levels），$H,W$ 为 0.25° 栅格；
  - GenCast 条件分布：
    $$
    P(\mathbf{X}^{t+1}\mid \mathbf{X}^t, \mathbf{X}^{t-1})
    $$
  - 整个 15 天轨迹的条件分解：
    $$
    P(\mathbf{X}^{1:T}\mid \mathbf{X}^0,\mathbf{X}^{-1})
    = \prod_{t=0}^{T-1} P(\mathbf{X}^{t+1}\mid \mathbf{X}^t, \mathbf{X}^{t-1})
    $$

- **数据与初始化**
  - 训练数据：ERA5 再分析（称为 “analysis”）1979–2018 共 40 年；
  - 评估：2019 年；
  - 初始化时刻：
    - 遵循 GraphCast 协议：用 ERA5 在 06 UTC 与 18 UTC 的分析作为初值（仅有 3h look-ahead，避免对 ML 模型不公平优势）。

- **对比基线**
  - ECMWF ENS：
    - 集合成员数：50；
    - 原生 0.2° 分辨率，重网格到 0.25° 以便与 GenCast 对比；
    - 公开 TIGGE 数据只提供地面变量和 8 个对流层等压面的大气变量 → 评估仅限这些变量/层。
  - GenCast-Perturbed：
    - 使用与 GenCast 相同架构的“确定性单步模型”（预测 12 小时均值场），通过对初始场加高斯过程噪声生成 ensemble，作为 ML ablation 基线。

---

### 2. 现有方法痛点与 GenCast 的定位

- **传统 NWP 集合（以 ENS 为代表）**
  - 优点：
    - ensemble 成员是“清晰的轨迹”（sharp, spectrally realistic）；
    - 在格点 marginals 上技巧高且校准良好（spread/skill ≈ 1, rank histogram 近似平）；
    - 能捕捉大尺度联合结构（如气旋路径、风电场空间关联）。
  - 局限：
    - 运行缓慢，工程复杂；
    - 仍有显著误差，极端事件技巧有限；
    - 需要大型超算和长期工程投入。

- **既有 MLWP（GraphCast、FourCastNet 等）局限**
  - 多数是**确定性**模型：
    - 优化目标通常是 MSE 或 L2，与“预报均值”对应；
    - 结果是预报场趋近“集合均值”而非单个轨迹 → 模糊、缺细节，尤其是长 lead；
  - 简单用“初值扰动 + 确定性模型”做 ensemble：
    - 成员之间高度相关，轨迹仍然偏“平均”，频谱能量在高波数显著不足；
    - 没有真正建模未来轨迹的概率分布；
  - 总体上：在**全概率**层面，MLWP 尚未挑战 ENS 这类业务 NWP 集合系统。

- **GenCast 的目标**
  1. 生成**真实、清晰的样本轨迹**，单个成员频谱与 ERA5 接近，区别于“模糊场”；
  2. 在格点 marginals 上，CRPS skill 大规模超越 ENS，且校准良好；
  3. 在联合结构（气旋路径、风电生产等空间相关应用）上，与 ENS 相比提供更高技巧；
  4. 大幅提升计算效率：在单个 Cloud TPU v5 上 8 分钟即可生成一个 15 天样本，ensemble 可并行生成。

---

### 3. 核心方法与生成流程

- **条件扩散模型（conditional diffusion）框架**
  - 目标：从噪声出发，通过多步“去噪/细化”，生成 $\mathbf{X}^{t+1}$ 的单个样本；
  - 每个时间步 $t+1$ 的采样过程：
    1. 初始化候选状态：
       $$
       \mathbf{Z}_0^{t+1} \sim \mathcal{N}(0, I)
       $$
    2. 在 $n=1..N$ 次迭代中，利用神经网络 denoiser/refiner $r_\theta$ 进行细化（伪公式）：
       $$
       \mathbf{Z}_n^{t+1} = \mathrm{Refine}(\mathbf{Z}_{n-1}^{t+1}, \mathbf{X}^t, \mathbf{X}^{t-1})
       $$
    3. 最终预测：
       $$
       \hat{\mathbf{X}}^{t+1} = \mathbf{X}^t + \mathbf{Z}_N^{t+1}
       $$
    4. 对 $t=0..T-1$ 自回归滚动，得到 $\mathbf{X}^{1:T}$ 一条轨迹样本；
    5. 重复采样不同的噪声序列 $\{\mathbf{Z}_0^{1:T}\}$，得到 ensemble 成员。

- **网络结构（denoiser $r_\theta$）**
  1. **Encoder**：
     - 将待预测的噪声态 $\mathbf{Z}_n^{t+1}$ 及条件 $(\mathbf{X}^t, \mathbf{X}^{t-1})$，从经纬网格投影到多次 refine 的 icosahedral 网格上；
     - 网格为六次 refined icosahedral mesh，用于更均匀覆盖球面；
  2. **Processor**：
     - 图 Transformer（graph transformer）；
     - 每个网格节点在图上对其 $k$-hop 邻域执行注意力；
  3. **Decoder**：
     - 将在 icosahedral mesh 上处理后的特征映射回经纬网格，输出去噪目标态（同维度的 $\mathbf{Z}_n^{t+1}$ 或修正量）。

- **训练**
  - 训练数据：ERA5 analysis（1979–2018）；
  - 损失：标准扩散模型中用于训练 denoiser 的 loss，论文仅说明“通过添加人工噪声并训练网络去噪”，具体形式在 Methods 中给出；
  - 自回归时间步：
    - 训练时同样是条件在 $(\mathbf{X}^t, \mathbf{X}^{t-1})$ 上学习 $P(\mathbf{X}^{t+1}\mid \mathbf{X}^t, \mathbf{X}^{t-1})$。

---

### 4. 评价指标与对比结论

> 评价均使用“各自最佳估计分析场”作为真值：ENS 对比 HRES-fc0 / ENS-fc0，GenCast 对比 ERA5 analysis。

#### 4.1 CRPS（ensemble skill）

- CRPS 衡量格点 marginal 分布与真值的匹配程度，是集合概率预报的标准指标；
- 结果：
  - 在 1320 组（变量 × 层次 × lead）中，GenCast 在 97.2% 的组合上 CRPS 显著优于 ENS（$P<0.05$）；
  - 对 lead > 36h 的目标中，有 99.6% 的组合上 GenCast 技巧更高；
  - 改善最显著的变量/层：
    - 多数地面变量（T2m, 10m wind, MSL），以及高层温度 / 比湿（specific humidity）等；
    - CRPS skill 提升约 10%–30%（即 CRPS 降低 10%–30%）；
  - GenCast-Perturbed 也在 82% 的组合上优于 ENS，但仍在 99% 的组合上劣于 GenCast 本体。

#### 4.2 Ensemble-mean RMSE

- 虽然不含不确定性信息，仍是常用 deterministic 指标；
- 结果：GenCast ensemble mean 的 RMSE 与 ENS 相当或更好：
  - 在 96% 的目标上 RMSE 不差于 ENS；
  - 在 78% 的目标上 RMSE 显著更小（$P<0.05$）。

#### 4.3 校准（spread/skill, rank histogram）

- Spread/skill ratio：
  - 理想值为 1：集合 spread 与 ensemble mean 的 RMSE 匹配；
  - GenCast：
    - 多数变量上稍小于但接近 1，表明整体轻微 under-dispersive 但仍校准良好；
  - ENS：同样接近 1，代表现有业务系统已有良好校准；
  - GenCast-Perturbed：
    - spread/skill 明显小于 1，显著 U 形 rank histogram → 持续 under-dispersive（过于自信）。

- Rank histograms：
  - GenCast：整体较为平坦，表明真实值在 ensemble 成员中的秩分布接近均匀；
  - ENS：同样接近平坦；
  - GenCast-Perturbed：明显 U 形，佐证其 under-dispersion。

#### 4.4 局地极端事件（local surface extremes）

- 事件类型：
  - 高温：T2m 超过气候分布的 99, 99.9, 99.99 分位数；
  - 大风：10 m 风速超过同样高分位数；
  - 极端低温和极端低 MSL：分别低于 1%, 0.1%, 0.01 分位数。

- 度量：
  - Brier skill score（BSS）：评估二元极端事件的概率预报；
  - 结果：
    - GenCast 在上述各类型事件上，在绝大多数 lead 上 BSS 显著优于 ENS；
    - 少数情况下（如 >7 天的 10m 风速 99.99 分位、某些低 MSL 分位）改进不显著。

- Relative Economic Value（REV）：
  - 用于评估在一系列 cost/loss 比下的决策价值；
  - Climatology REV = 0，完美预报 REV = 1；
  - 结果（以 2m T 与 10m 风速 99.99 分位为例）：
    - 在 1, 5, 7 天 lead、所有 cost/loss 范围内，只要某个模型优于气候基线，则 GenCast 的 REV 一致高于 ENS，尤其在低 cost/loss 区域（对极端事件更敏感）表现更强。

#### 4.5 轨迹与场景层面的表现

- 热带气旋路径（以台风 Hagibis 为例）：
  - 1 日 lead：GenCast ensemble 成员轨迹集中、清晰，路径与 ERA5 接近；
  - 7 日 lead：轨迹 spread 较大，反映不确定性；Lead 逐渐缩短，spread 缩小，表明模型能合理收敛不确定性；
  - Typhoon Hagibis 在 2019 年 GenCast ensemble mean 位置误差分布中位于第 55 百分位，代表中等难度个例。

- 频谱特性（sharpness vs blur）
  - 对比 1 日与 15 日 lead 的样本：
    - GenCast 单个样本的空间功率谱与 ERA5 非常接近（保留高波数能量），说明成员是“真实轨迹”；
    - GenCast ensemble mean 以及 GenCast-Perturbed 的成员谱在高波数明显能量不足，场景模糊，更接近“均值场”。

---

### 5. 创新点与局限

- **创新点**
  1. 首个在整体 probabilistic 技巧上显著超越 ENS 的纯 MLWP 模型：在绝大多数变量/层/lead 组合上 CRPS 与 ensemble-mean RMSE 均优于 ENS；
  2. 基于条件扩散的轨迹级建模：
     - 直接建模 $P(\mathbf{X}^{t+1}\mid \mathbf{X}^t, \mathbf{X}^{t-1})$ 的复杂分布，生成 sharp 的单成员轨迹，而非均值场；
  3. 结合球面几何的 encoder–graph transformer–decoder 架构：
     - 使用六次 refined icosahedral mesh + graph transformer，有效捕获球面上局部与长程依赖；
  4. 全面概率验证：
     - 同时在 CRPS、spread/skill、rank histogram、Brier score、REV 等多维度评估模型概率质量与决策价值；
  5. 计算效率：
     - 单个 Cloud TPU v5 上 8 分钟即可生成 15 天全变量轨迹，ensemble 成员可完全并行，明显高于 NWP 集合效率。

- **局限与讨论**
  - 评估仍基于 ERA5/HRES-fc0/ENS-fc0 分析，不直接包含观测误差与同化不确定性；
  - Precipitation：由于对 ERA5 降水质量信心有限，主文中未重点报告 TP 结果，仅在补充材料中讨论；
  - 与 NWP 一样，GenCast 目前依赖分析初值，并非端到端同化系统；
  - 文中未深入讨论长期积分（>15 天）、气候态漂移或物理一致性（守恒律等）。

---

## 维度二：基础组件 / 模块级知识

> 下列代码为“基于论文描述的高保真伪代码”，用于表达结构与流程；不等同于作者开源实现。

### 组件 A：条件扩散时间步采样器

- **子任务**
  - 条件在 $(\mathbf{X}^t, \mathbf{X}^{t-1})$ 上，从噪声生成下一步样本 $\hat{\mathbf{X}}^{t+1}$；
  - 多次迭代去噪形成 candidate state $\mathbf{Z}_N^{t+1}$，再残差加到 $\mathbf{X}^t$。

- **伪代码**

```python
# Pseudocode: one-step conditional diffusion sampler (structure only)

def sample_step(x_t, x_tm1, denoiser, N=K):
    """Sample X^{t+1} ~ P(X^{t+1} | X^t, X^{t-1}).
    x_t, x_tm1: (C, H, W)
    denoiser: r_theta that refines noisy state given conditioning.
    """
    z = sample_gaussian_like(x_t)   # Z_0^{t+1} ~ N(0, I)
    for n in range(N):
        # r_theta operates on (z, x_t, x_tm1); details in components below
        delta = denoiser(z, x_t, x_tm1)
        # update rule schematic; real schedule (step-size, noise) from Methods
        z = z + delta
    x_tp1 = x_t + z   # residual connection as in Fig. 1
    return x_tp1
```

---

### 组件 B：GenCast 主循环（自回归轨迹生成）

- **子任务**
  - 从 $(\mathbf{X}^{-1}, \mathbf{X}^0)$ 出发，自回归生成长度 $T=30$ 的轨迹样本；
  - 重复不同噪声，生成 ensemble 成员。

- **伪代码**

```python
# Pseudocode: autoregressive trajectory sampler

def sample_trajectory(x_minus1, x_0, denoiser, T=30):
    traj = []
    x_tm1, x_t = x_minus1, x_0
    for t in range(T):
        x_tp1 = sample_step(x_t, x_tm1, denoiser)
        traj.append(x_tp1)
        x_tm1, x_t = x_t, x_tp1
    # traj: list of T states X^{1:T}
    return torch.stack(traj, dim=0)


def sample_ensemble(x_minus1, x_0, denoiser, T=30, E=50):
    members = []
    for e in range(E):
        # different noise is sampled inside sample_step
        traj_e = sample_trajectory(x_minus1, x_0, denoiser, T)
        members.append(traj_e)
    return torch.stack(members, dim=0)  # (E, T, C, H, W)
```

---

### 组件 C：Encoder – 网格映射到六次 refined icosahedral mesh

- **子任务**
  - 将经纬网格上的状态 $(\mathbf{Z}_n^{t+1}, \mathbf{X}^t, \mathbf{X}^{t-1})$ 编码到球面 icosahedral 网格；
  - 为 graph transformer 提供节点特征。

- **伪代码**

```python
# Pseudocode: Encoder (grid -> icosahedral mesh)

class SphericalEncoder(nn.Module):
    def __init__(self, mesh, in_channels, hidden_dim):
        super().__init__()
        self.mesh = mesh  # precomputed ico mesh with vertex -> (lat, lon)
        self.lin = nn.Linear(in_channels, hidden_dim)

    def forward(self, z, x_t, x_tm1):
        # z, x_t, x_tm1: (C, H, W)
        # stack channels: (3*C, H, W)
        stacked = torch.cat([z, x_t, x_tm1], dim=0)
        # interpolate from lat-lon grid to each mesh vertex
        feats_grid = stacked.unsqueeze(0)        # (1, 3C, H, W)
        feats_mesh = bilinear_sample_on_mesh(self.mesh, feats_grid)
        # feats_mesh: (V, 3C)
        h = self.lin(feats_mesh)
        return h  # (V, hidden_dim)
```

> 注：论文只说明 encoder 进行网格到 mesh 的映射，具体插值方法未详细给出；伪代码采用双线性采样表示。

---

### 组件 D：Processor – Graph Transformer on icosahedral mesh

- **子任务**
  - 在 icosahedral mesh 上，用图 Transformer 做 $k$-hop 邻域注意力，建模局部与长程依赖；
  - 多层堆叠更新节点特征。

- **伪代码**

```python
# Pseudocode: Graph Transformer block

class GraphTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadGraphAttention(hidden_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h, edges):
        # h: (V, D), edges: graph structure (k-hop neighborhood)
        h2 = self.attn(self.norm1(h), edges)
        h = h + h2
        h2 = self.ff(self.norm2(h))
        h = h + h2
        return h


class MeshProcessor(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphTransformerBlock(hidden_dim, num_heads=8)
            for _ in range(num_layers)
        ])

    def forward(self, h, edges):
        for layer in self.layers:
            h = layer(h, edges)
        return h  # (V, D)
```

> 注：论文仅说明 processor 是在 mesh 上的 graph transformer，节点对 k-hop 邻域做注意力，未给出具体层数、头数等超参数。

---

### 组件 E：Decoder – 从 mesh 回到经纬网格并形成残差

- **子任务**
  - 将 Processor 输出的 mesh 特征解码回经纬网格残差场 $\Delta z$，用于更新 $\mathbf{Z}_n^{t+1}$。

- **伪代码**

```python
# Pseudocode: Decoder (mesh -> grid)

class SphericalDecoder(nn.Module):
    def __init__(self, mesh, hidden_dim, out_channels):
        super().__init__()
        self.mesh = mesh
        self.lin = nn.Linear(hidden_dim, out_channels)

    def forward(self, h):
        # h: (V, D)
        res_mesh = self.lin(h)  # (V, C)
        # interpolate from mesh vertices back to lat-lon grid
        res_grid = interpolate_from_mesh_to_grid(self.mesh, res_mesh)
        # res_grid: (C, H, W)
        return res_grid
```

---

### 组件 F：Denoiser 网络 r_θ 整体结构

- **子任务**
  - 组合 Encoder → Processor → Decoder，对给定的噪声态与条件场产生去噪残差；

- **伪代码**

```python
# Pseudocode: Denoiser r_theta

class GenCastDenoiser(nn.Module):
    def __init__(self, mesh, in_channels, hidden_dim, num_layers):
        super().__init__()
        self.encoder = SphericalEncoder(mesh, in_channels=3 * in_channels,
                                        hidden_dim=hidden_dim)
        self.processor = MeshProcessor(hidden_dim, num_layers)
        self.decoder = SphericalDecoder(mesh, hidden_dim, out_channels=in_channels)

    def forward(self, z, x_t, x_tm1):
        # z, x_t, x_tm1: (C, H, W)
        h0 = self.encoder(z, x_t, x_tm1)
        h = self.processor(h0, self.mesh.edges)
        res = self.decoder(h)
        return res  # residual update for z
```

> 注：实际实现中还应包含时间步编码、噪声水平编码等扩散模型常见输入，这些在主文中未明确列出，此处不作主观补充。

---

### 组件 G：概率评价与极端事件决策

- **子任务**
  - 对 ensemble 成员计算 CRPS、spread/skill、rank histogram、Brier score、REV 等指标。

- **伪代码轮廓**

```python
# Pseudocode: CRPS & spread/skill (outline)

import xskillscore as xs

# members: (E, T, C, H, W), truth: (T, C, H, W)

def compute_crps(members, truth):
    ens_mean = members.mean(axis=0)
    ens_var = members.var(axis=0)
    # assume Gaussian as in paper; per grid cell CRPS
    crps = xs.crps_gaussian(truth, mu=ens_mean, sig=np.sqrt(ens_var))
    return crps


def compute_spread_skill(members, truth, lat_weights):
    ens_mean = members.mean(axis=0)
    spread = lat_weighted_spread(members, lat_weights)
    rmse = lat_weighted_rmse(ens_mean, truth, lat_weights)
    ssr = spread / rmse
    return spread, ssr

# Pseudocode: Brier score & REV for extreme event E (e.g., T2m > q99.99)

def brier_score(probs, events):
    # probs: forecast probability of event
    # events: 0/1 truth
    return np.mean((probs - events)**2)


def relative_economic_value(hits, false_alarms, misses, correct_negatives,
                            cost_loss_ratio):
    # Following standard definition; paper refers to canonical formula
    # and uses climatology / perfect-forecast baselines.
    # Details in Supplement A.5.6 (not repeated here).
    pass
```

---

## 小结

- GenCast 通过条件扩散 + 球面 graph transformer 的架构，将 ML 天气预报从单一确定性“均值场”拓展为高分辨率、高质量的概率预报体系，在 CRPS、ensemble-mean RMSE、极端事件 BSS 与 REV 等指标上大幅超越 ECMWF ENS；
- 其成员轨迹在空间频谱上与 ERA5 非常接近，保证了“sharpness”，同时通过 spread/skill 与 rank histogram 验证在校准上与 ENS 同级或略优；
- 与基于初始扰动的确定性 ensemble（如 GenCast-Perturbed）相比，GenCast 显示了扩散生成式建模在捕获真实不确定性和维持轨迹清晰度方面的显著优势，为下一代业务级 ML 概率预报系统提供了可行范式。