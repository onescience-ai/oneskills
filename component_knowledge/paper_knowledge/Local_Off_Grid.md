# Local Off-Grid：多模态 Transformer/LGNN 的离网站点局地预报框架

> 说明：本文件基于论文文字内容进行结构化总结和高层伪代码抽象，不是官方实现代码。

---

## 一、整体模型维度（任务 / 数据 / 结构 / 训练 / 结果）

### 1.1 任务与问题设定

- **场景**：
  - 目标是对近地面局地天气（风、气温、露点）做精确预报，地点是**不在规则格点上的离网站点**（MADIS 气象站）。
  - 应用：野火管理、可再生能源调度等对局地天气极敏感的场景。
- **现有问题**：
  - NWP / ML 大模型（如 ERA5 目标场、HRRR、FourCastNet、GraphCast、Pangu 等）在**规则格点**上输出；
  - 将格点场线性插值到站点：
    - 存在显著系统偏差（尤其是 10 m 风，ERA5 在内陆系统性高估、过于光滑）；
    - 难以捕捉地形、建筑、植被等引起的局地近地层效应。
- **任务形式**：
  - 传统“纯站点时间序列预报”形式：
    $$\mathbf{w}(t + l\Delta t) = F\big(\mathbf{w}(t - b\Delta t : t)\big),$$
    其中 $\mathbf{w} = [w_0, ..., w_n]$ 为所有站点观测；$b\Delta t$ 为 back hours。
  - 本文改写为“误差订正 / 下采样”形式：
    $$\mathbf{w}(t + l \Delta t) = F\big(\mathbf{w}(t - b\Delta t : t), \mathbf{g}(t - b\Delta t : t + l \Delta t)\big),$$
    其中 $\mathbf{g}$ 为 ERA5/HRRR 等格点预报（覆盖回溯到 lead time 这一段）。
  - 即：在给定大尺度格点预报 $\mathbf{g}$ 的前提下，用站点历史观测 $\mathbf{w}$ 去订正/下采样到离网位置。
- **预测变量**：
  - 10 m 风向量 $(u, v)$；
  - 2 m 气温（T2m）；
  - 2 m 露点温度（Td2m）。
- **lead time**：
  - $l \in \{1, 2, 4, 8, 12, 18, 24, 36, 48\}$ 小时
  - 每个 lead time 训练一个独立模型（one model per lead time）。

### 1.2 数据与实验设置

- **空间范围**：美国东北部（9 个州，358 个 MADIS 站）。
- **时间范围**：2019–2023 共 5 年，1 小时间隔：
  - 训练：2019–2021；
  - 验证：2022；
  - 测试：2023。
- **MADIS（离网站点观测）**：
  - 变量：10 m 风速、10 m 风向（再转为 u/v）、2 m 温度、2 m 露点；
  - 仅保留质量标记为“Screened/Verified”的数据；
  - 站点筛选：5 年内有效数据比例 ≥ 90%；
  - 对少量缺失做前向填充；
  - 共 358 站点，约 8760 时刻/年。
- **ERA5（再分析，粗网格）**：
  - 分辨率：0.25° × 0.25°（~31 km）；
  - 变量：10 m u, 10 m v, 2 m T, 2 m Td；
  - 时间分辨率：1 小时，2019–2023，裁剪到研究区。
- **HRRR（高分辨率 NWP）**：
  - 分辨率：约 3 km；
  - 变量：10 m u, 10 m v, 2 m T, 2 m Td；
  - 提取分析场（HRRR-A）和实验中的预报场（HRRR-F）：
    - HRRR-F：常规每小时更新，提供到 18 h；个别循环（00, 06, 12, 18 UTC）到 48 h，但为一致性只用到 18 h；
    - 对于 lead time > 18 h，仅能使用 HRRR-F 的 0–18 h 段作为未来输入，其后的区间不再有 HRRR-F；
  - HRRR-A 被视为“高质量格点真值”，对比 ERA5。
- **样本统计**：
  - 训练样本：约 26,280 条（8760×3 年）；
  - 验证、测试各 ~8760 条。

### 1.3 模型族与整体结构

论文比较了三类局地预报模型（均以站点为输出）：

1. **多模态 Transformer（主角）**：
   - 结构：encoder-only Transformer（类 ViT）；
   - token：每个站点对应一个 token；
   - station token 输入 = [站点历史观测时间序列, 最近格点的 ERA5/HRRR 历史+未来时间序列]，经 MLP embedder；
   - 全连接自注意力学习站点间空间相关性；
   - MLP head 逐站输出未来目标变量（特定 lead time）。

2. **MPNN / GNN（强基线）**：
   - 构造一个**异质图**：
     - 站点节点：MADIS；
     - 格点节点：ERA5 或 HRRR 网格；
   - 站点间边：基于 Delaunay 三角剖分（保持稀疏且方向均衡）；
   - 站点 → 站点：双向消息传递；
   - 格点 → 站点：单向消息（global→local），每个站点连到最近的 o 个格点节点（实验中 o=8）；
   - 三阶段 MPNN：global→local, local↔local, global→local；
   - 最后通过 MLP decoder 在站点上输出预报。

3. **MLP（无空间结构基线）**：
   - 与 MPNN 结构相同的“encode-decode”，但去掉图结构和消息传递；
   - 只在每个站点单点上做前向（但参数在站点间共享）；
   - 格点信息作为直接拼接特征输入，而非图上的邻接节点。

**非 ML / 插值基线**：
- 插值 ERA5/HRRR：最近邻将格点值映射到站点位置；
- Persistence：假定未来 = 现在（观测向前平移 lead time）。

### 1.4 训练与损失

- 所有模型：
  - 目标：同时预测 10 m u, 10 m v, 2 m T, 2 m Td 于各站点。
  - 损失：标准 MSE / RMSE（按站点+时间平均）。
  - 优化器：Adam（lr=1e-4, weight decay=1e-4）；
  - batch size=128，训练 200 epochs，基于验证集选择最佳模型。
- lead time：
  - 每个 lead time 独立训练一个模型（1, 2, 4, 8, 12, 18, 24, 36, 48 h）。

### 1.5 结果总结（定性）

- **Transformer 是最优局地预报模型**：
  - 使用 HRRR 分析（HRRR-A）作为格点输入时，Transformer 在所有变量、所有 lead time 上均优于 MLP/GNN 以及非 ML 基线：
    - 10 m 风：平均向量误差 0.48 m/s，比次优 ML（MLP+HRRR-A）低 22%，比 persistence 低 49%，比 HRRR-A 插值低 80%。
    - 2 m 温度：平均 RMSE 相比 MLP+HRRR-A 下降 41%，比 persistence 低 73%，比 HRRR-A 插值低 25%。
    - 2 m 露点：平均 RMSE 比 MLP+HRRR-A 低 36%，比 persistence 低 59%，比 HRRR-A 插值低 58%。

- **格点天气数据对所有 ML 模型都有帮助**：
  - 对 Transformer：加入 HRRR-A 后，相比仅用站点数据：
    - 风向量误差降低 ~33%；
    - 温度 RMSE 降低 ~56%；
    - 露点 RMSE 降低 ~49%。

- **格点数据质量显著影响 ML 模型表现**：
  - 插值：HRRR-A < ERA5 < HRRR-F（误差由小到大）；
  - 对应 Transformer：以 HRRR-A 作为输入的 Transformer 性能最佳，以 ERA5 次之，以 HRRR-F 再次之；
  - lead time > 18h 时，基于 HRRR-F 的模型误差明显上升（输入 HRRR-F 已无未来信息）。

- **风预报收益最大**：
  - 相比 HRRR-A 插值，Transformer+HRRR-A+MADIS：
    - 风误差下降 80%；
    - 露点 RMSE 下降 58%；
    - 温度 RMSE 下降 25%。
  - 说明：局地风场最受地形/粗糙度等局地因素影响，而格点场对这些效应刻画最差。

- **空间泛化与站点差异**：
  - 几乎所有站点 Transformer 都优于 HRRR-F 插值，表明模型在空间上泛化良好；
  - 海岸开放环境的某些站点，模型收益较小——那里的格点场已较为接近真实局地风场。

- **动态注意力带来的优势**：
  - Transformer 的自注意力可以**按当前天气状态动态选择相关站点**，比固定邻接结构（GNN）/无邻接（MLP）更灵活；
  - 消融表（表 A5）中也表明，在固定网络结构和外部节点数目下，Transformer 的误差显著低于 GNN/MLP。

---

## 二、组件维度：可复用建模范式

### 2.1 多模态数据整理与样本构造组件

**子需求**：
- 在统一时间轴上对接：站点观测 (MADIS) + 粗网格 ERA5 + 高分辨率 HRRR；
- 构造包含 back hours 和 lead time 的输入窗口；
- 生成适合 Transformer 或 GNN 的输入结构（token 或图）。

**关键要点**：
- 对 MADIS 站点：
  - 筛选高质量观测；
  - 时间对齐到整点，形成 `(time, station, variables)`；
  - 对少量缺失用前向填充；
  - 计算 5 年有效数据占比并筛站点。
- 对 ERA5 / HRRR：
  - 裁剪到东北 US；
  - 提取 4 个表面变量（10 m u/v, 2 m T/Td）；
  - 形成 `(time, grid, variables)`。
- 样本构造：
  - 固定 back hours（如 48 h），固定 lead time（每模型一个）；
  - 对每个时间 t，构造：
    - MADIS 序列：`w(t - b:t)`；
    - 格点序列：`g(t - b:t + l)`（若 HRRR-F，超过 18h 部分截断）。

**伪代码框架**：

```python
# 构造站点和格点索引
stations = load_madis_stations(region="NE_US", min_coverage=0.9)
madis_ts = load_madis_timeseries(stations, vars=["ws", "wd", "T2m", "Td2m"])
madis_ts = quality_filter_and_ffill(madis_ts)

era5_grid = load_era5_grid(region="NE_US", vars=["u10", "v10", "T2m", "Td2m"])
hrrr_grid = load_hrrr_grid(region="NE_US", vars=["u10", "v10", "T2m", "Td2m"])

# 时间对齐与切分
train_range = ("2019-01-01", "2021-12-31")
val_range   = ("2022-01-01", "2022-12-31")
test_range  = ("2023-01-01", "2023-12-31")

madis_train, madis_val, madis_test = split_by_time(madis_ts, train_range, val_range, test_range)
era5_train,  era5_val,  era5_test  = split_by_time(era5_grid,  train_range, val_range, test_range)
hrrr_train,  hrrr_val,  hrrr_test  = split_by_time(hrrr_grid,  train_range, val_range, test_range)

# 生成样本窗口
B_BACK = 48  # back hours

samples = []
for t in valid_times(train_range, lead=l):
    w_hist = madis_train[t-B_BACK:t]           # (B_BACK+1, N_station, d_w)
    g_seq  = era5_train[t-B_BACK:t+l]         # 或 hrrr_train
    y_tgt  = madis_train[t+l]                 # 目标站点观测
    samples.append((w_hist, g_seq, y_tgt))

train_loader = DataLoader(samples, batch_size=128, shuffle=True)
```

---

### 2.2 多模态 Transformer 局地预报组件

**子需求**：
- 处理不规则站点集合（off-grid），利用 station-to-station 空间关系；
- 融合站点历史观测与就近格点的历史/未来时间序列；
- 对每个 lead time 直接 one-step 输出，不按时间滚动自回归。

**结构要点**：
- token 粒度：每个站点一个 token；
- 输入特征：
  - 展平的 MADIS 时间序列（back hours × 变量数）；
  - 展平的格点时间序列（back hours+lead time × 变量数）；
  - 可拼接坐标（lat, lon）等静态特征；
- MLP embedder 将上述特征映射到 d_model；
- 加上位置/站点嵌入；
- 若干层自注意力编码器；
- 站点级 MLP head 输出 (u, v, T2m, Td2m)。

**高层伪代码**：

```python
class StationEmbedder(nn.Module):
    def __init__(self, w_dim, g_dim, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(w_dim + g_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, w_hist, g_seq):
        # w_hist: (B, N, T_w, C_w)
        # g_seq:  (B, N, T_g, C_g)  # 最近格点插值到站点
        B, N, T_w, C_w = w_hist.shape
        _, _, T_g, C_g = g_seq.shape
        w_flat = w_hist.reshape(B, N, T_w * C_w)
        g_flat = g_seq.reshape(B, N, T_g * C_g)
        x = torch.cat([w_flat, g_flat], dim=-1)
        return self.mlp(x)  # (B, N, d_model)


class LocalOffGridTransformer(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, num_vars_out):
        super().__init__()
        self.embedder = StationEmbedder(...)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_vars_out)  # 输出 4 个变量
        )
        self.station_pos = nn.Parameter(torch.randn(1, N_max, d_model))  # 若 N 固定

    def forward(self, w_hist, g_seq, station_mask=None):
        # w_hist, g_seq: (B, N, ...)
        x = self.embedder(w_hist, g_seq)  # (B, N, d_model)
        x = x + self.station_pos[:, :x.size(1), :]
        x_enc = self.encoder(x, src_key_padding_mask=station_mask)
        y_hat = self.head(x_enc)  # (B, N, 4)
        return y_hat
```

训练时，每个 lead time 训练一个这样的模型，loss 为所有站点、所有样本上的 MSE 之和。

---

### 2.3 异质图 MPNN 局地预报组件

**子需求**：
- 在图结构上表达站点—站点及站点—格点的关系；
- 支持 multi-modal：MADIS（点）+ ERA5/HRRR（网格）两类节点；
- 通过消息传递实现 global→local、local↔local、多步聚合。

**图结构设计**：
- 站点节点集 $V_{st}$、格点节点集 $V_{grid}$；
- 站点间边：Delaunay triangulation（站点经纬度），双向；
- 格点到站点边：每站点连到最近 o 个格点节点（单向 grid→station）。

**Encode 阶段**：
- 对站点节点 $i$：
  $$f_i \leftarrow \alpha(w_i(t-b:t), p_i)$$
- 对格点节点 $r$：
  $$h_r \leftarrow \psi(g_r(t-b:t+l), p_r)$$

**Process 阶段（多层）**：
1. 站点 ← 站点：
   $$
   \mu_{ij} \leftarrow \beta(f_i, f_j, w_i - w_j, p_i - p_j),\\
   f_i \leftarrow f_i + \gamma\Big(f_i, \frac{1}{|\mathcal{N}(i)|} \sum_{j\in\mathcal{N}(i)} \mu_{ij}\Big)
   $$
2. 站点 ← 格点：
   $$
   \nu_{ir} \leftarrow \chi(h_r, f_i, p_i - p_r),\\
   f_i \leftarrow f_i + \omega\Big(f_i, h_r, \frac{1}{|\mathcal{M}(i)|} \sum_{r\in\mathcal{M}(i)} \nu_{ir}\Big)
   $$

**Decode 阶段**：
- $$w_i(t+l) = \phi(f_i)$$。

**整体伪代码**：

```python
class HeteroStationGridMPNN(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.enc_station = MLP(...)
        self.enc_grid    = MLP(...)
        self.beta  = MLP(...)
        self.gamma = MLP(...)
        self.chi   = MLP(...)
        self.omega = MLP(...)
        self.dec   = MLP(...)

    def encode(self, w_hist, g_seq, p_st, p_gr):
        f = self.enc_station(torch.cat([w_hist_flat(w_hist), p_st], dim=-1))
        h = self.enc_grid(torch.cat([g_seq_flat(g_seq), p_gr], dim=-1))
        return f, h

    def process(self, f, h, station_edges, grid2station_edges):
        # station_edges: list of (i, j)
        # grid2station_edges: list of (i, r)
        # 1) station <- station
        msgs_ss = defaultdict(list)
        for i, j in station_edges:
            mu_ij = self.beta(f[i], f[j], ..., ...)
            msgs_ss[i].append(mu_ij)
        for i in msgs_ss:
            agg = torch.mean(torch.stack(msgs_ss[i]), dim=0)
            f[i] = f[i] + self.gamma(f[i], agg)

        # 2) station <- grid
        msgs_gs = defaultdict(list)
        for i, r in grid2station_edges:
            nu_ir = self.chi(h[r], f[i], ...)
            msgs_gs[i].append(nu_ir)
        for i in msgs_gs:
            agg = torch.mean(torch.stack(msgs_gs[i]), dim=0)
            f[i] = f[i] + self.omega(f[i], h_agg_for_i(i, h), agg)
        return f

    def decode(self, f):
        return self.dec(f)  # (N_station, 4)
```

> 注：论文中还探索了多层 message passing 步数、不同邻接结构、连接 ERA5 节点数量等消融，发现：
> - 增加 message passing 次数收益有限；
> - fully-connected GNN 有时略好，但代价是复杂度上升；
> - 连接过多 ERA5 节点易“噪声过载”，反而影响性能；
> - 总体上 GNN 性能仍落后于 Transformer。

---

### 2.4 基线与对比组件

**子需求**：
- 为评估多模态 Transformer / GNN 的价值，构建一套完整基线；
- 区分“纯站点时间序列”vs“站点+格点”vs“仅格点插值”。

**基线集合**：
1. **插值基线**：
   - ERA5/HRRR-A/HRRR-F 最近邻插值到站点：
     - $$\hat{w}_i(t+l) = g_{\text{nearest}(i)}(t+l)$$。
2. **Persistence**：
   - $$\hat{w}_i(t+l) = w_i(t)$$。
3. **MLP 基线**：
   - 与 MPNN 结构相同，但去掉图；
   - 输入 = [站点时间序列, 最近格点时间序列]，逐站点预测；
   - 无站点间交互。

**效果**：
- 插值基线揭示 ERA5/HRRR 的格点偏差；
- Persistence 对短 lead time 气温有一定能力，但随着时间急剧恶化；
- MLP 显示仅时间序列建模 + 局地格点信息也能明显优于简单基线，但仍不如 Transformer。

---

### 2.5 误差分析与可解释性组件

**子需求**：
- 理解 transformer 模型如何利用 HRRR/ERA5 与站点数据；
- 分析空间误差分布与站点环境之间的关系；
- 衡量不同变量（风、温度、露点）收益差异。

**关键分析手段**：
- 变量级误差曲线：风向量误差（m/s）、T/Td RMSE 随 lead time 的变化；
- 站点级误差地图：
  - 每站点、多个 lead time 平均误差；
  - 对比：HRRR-F 插值 vs Transformer(仅 MADIS) vs Transformer(MADIS+HRRR-F)。
- 时间序列案例：显示 Transformer 如何在某些站点：
  - 当 HRRR-F 质量好时，紧跟 HRRR-F 但做细调；
  - 当 HRRR-F 明显偏差时，更多依赖 MADIS 历史信息；
  - 随 lead time 增长，对 HRRR-F 的依赖度增加但仍优于 HRRR-F 插值。
- 环境分析：
  - 对比误差最小/最大站点附近的卫星图：
    - 被建筑/树木遮蔽的站点，局地风受地表粗糙度强烈影响，格点模型表现差，多模态模型收益大；
    - 开阔海岸站点，HRRR/ERA5 已接近真实，提升有限。

---

### 2.6 经验启示与可迁移范式

- **范式 1：将“预测”改写为“订正”**：
  - 利用已有的大尺度预报（NWP 或 ML foundation model），将本地模型设计为 **F(站点历史, 大尺度预报)**，做 bias correction / downscaling，而不是从零预测；
- **范式 2：直接输入站点数据而非只依赖资料同化结果**：
  - 即使 HRRR 已同化站点数据，格点预报中的局地信息仍被平滑；
  - 若目标是“站点级精度”，必须显式把站点观测作为输入特征之一；
- **范式 3：Transformer 适合 off-grid 站点集合**：
  - 自注意力可以自然处理不规则 station 集合作为 token 序列；
  - 动态空间注意力在局地预报任务中优于固定邻接的 GNN；
- **范式 4：多模态融合与分层建模**：
  - MADIS（局地、噪声大）+ ERA5/HRRR（大尺度、偏差但平滑）提供互补信息；
  - 类似思路可扩展到：站点 + 雷达 + 卫星 + foundation model 输出等。
- **范式 5：区域可扩展性与复杂度权衡**：
  - 在东北 US 范围内，Transformer 可以“全站点同时”处理；
  - 更大区域可采用分区/分块或稀疏注意力以控制 $O(N^2)$ 复杂度；
  - GNN 在大规模稀疏图上的复杂度更友好，但本任务中表现略逊。

这些组件和范式可以直接迁移到你后续的“站点级订正 + 大模型格点预报”设计中：
- 将 GraphCast/NeuralGCM/FourCastNet 输出视为 $\mathbf{g}$；
- 将本地站点、雷达、观测视为 $\mathbf{w}$；
- 在“Local Off-Grid”这一框架上，替换数据源即可构建新的局地订正模块。