# GNADET：地理空间神经平流–扩散方程 + 图 Transformer 的地表气温预测框架

> 说明：本文件基于论文文字内容进行结构化总结和高层伪代码抽象，不是官方实现代码。

---

## 一、整体模型维度（任务 / 数据 / 结构 / 训练 / 结果）

### 1.1 任务与问题设定

- **目标变量**：全球 2 m 地表气温 T2M（6 小时分辨率，多步提前预报）。
- **输入变量（共 11 个）**：
  - 动态（9 项）：T2M, T850, RH1000, Q1000, TP, TISR, Z500, U10, V10。
  - 静态（2 项）：Oro（地形高程）, LSM（土地-海洋掩膜）。
- **时间范围**：
  - 训练：2006-01–2015-12
  - 验证：2016 全年
  - 测试：2017-01–2018-12
- **空间分辨率**：5.625° 规则全球格点。
- **任务类型**：多步前向预报，提前时间 $h \in \{6, 12, 18, 24, 72, 144\}$ 小时。
- **核心难点**：
  - 既要高精度，又要物理一致性（守恒、合理的热量输运）。
  - 纯数据驱动模型常违背物理、外推能力弱；传统 ML 难以建模复杂的时空关系。

### 1.2 数据处理与样本构造

- **变量合并**：将 11 个变量（含静态场）合并成一个 3D 数组 `(time, grid, channels)`。
- **图结构构建**：
  - 每个格点是一个节点 $v_i$。
  - 采用 Moore 邻域（3×3 stencil）：每个节点与其 8 个相邻格点相连。
  - 得到二值邻接矩阵 $A_{mask}$，只允许局部连接。
- **划分与标准化**：
  - 按时间连续划分 train/val/test，避免泄露。
  - 标准化参数（mean/std）仅由训练集计算，应用到 val/test。
- **滑动窗口监督样本**：
  - 对每个时间序列使用 sliding window 构造 `(lags, horizon)` 对：
    - 输入：多步历史观测（lags）。
    - 输出：对应未来多步 T2M 序列（horizon）。
- **气候基线**：
  - 由未标准化的训练集 T2M 计算逐格点的多年平均，作为 ACC 的气候场基线。

### 1.3 物理建模：平流–扩散方程与 PI-NODE

- **物理控制方程**：
  - 平流通量：$J_{adv} = \vec{u} T$。
  - 扩散通量：$J_{diff} = - D \nabla T$。
  - 对应的 PDE：
    $$
    \frac{\partial T}{\partial t}
      = \nabla \cdot (D \nabla T)
      - \nabla \cdot (\vec{u} T)
      = DiffusionTerm - AdvectionTerm.
    $$
- **连续时间表述（Neural ODE）**：
  - 将全局格点温度场在时间 $t$ 的状态记为 $H(t)$（向量，长度 = 节点数）。
  - 核心 ODE：
    $$
    \frac{dH(t)}{dt} = F_{phys}(H(t), \mathcal{G}) + F_{uncertainty}(H(t), \mathcal{G}; \theta_f)
    $$
  - $F_{phys}$：在图上的离散平流–扩散算子（图拉普拉斯近似 $\nabla, \nabla^2$）。
  - $F_{uncertainty}$：由 Graph Transformer 学习的“残差/不确定性”动力学。

### 1.4 GNADET 总体架构

1. **Encoder**：
   - 输入：多时刻、多变量场（含静态变量），按节点堆叠。
   - 输出：初始潜在状态 $H(t_0)$。
2. **图结构**：
   - 使用 Moore 邻域构造图 $\mathcal{G}=(\mathcal{V}, \mathcal{E})$，并提供 $A_{mask}$。
3. **PI-NODE 模块**：
   - 内部由 ODE solver 积分：
     $$H(t_1) = H(t_0) + \int_{t_0}^{t_1} \left[ F_{phys}(H, \mathcal{G}) + F_{uncertainty}(H, \mathcal{G}) \right] dt.$$
   - 每一步积分调用：
     - 物理项 $F_{phys}$：基于学习的图拉普拉斯算子 $L_{diff}, L_{adv}$。
     - 不确定性项 $F_{uncertainty}$：Graph Transformer 计算的校正向量。
4. **Decoder**：
   - 将 $H(t_1)$ 映射回物理空间的 T2M 预报场（可多 lead time，一次积分到目标时刻）。

### 1.5 物理项：动态图拉普拉斯（learnable edge weights）

- 离散形式：
  $$
  F_{phys}(H, \mathcal{G}) = - (L_{diff} H + L_{adv} H).
  $$
- 学习的邻接矩阵：
  - 通过节点嵌入 $W_1, W_2$ 和遮罩矩阵 $A_{mask}$ 生成：
    $$
    A_{diff} = A_{mask} \odot ReLU(\tanh(\alpha W_1 W_1^T)),\\
    A_{adv}  = A_{mask} \odot ReLU(\tanh(\alpha (W_1 W_2^T - W_2 W_1^T))).
    $$
  - $A_{diff}$ 对称（扩散：双向相互作用），$A_{adv}$ 非对称（平流：有向输运）。
  - 图拉普拉斯 $L = D - A^T$，$D_{ii} = \sum_j A_{ij}$。
- 好处：
  - 让扩散/平流强度在空间上可变，贴近实际地区差异；
  - 同时用 ReLU + tanh 限制权重为非负且有界，提升物理合理性与数值稳定性。

### 1.6 不确定性项：Graph Transformer

- 作用：建模显式物理无法解释的复杂残差，如：
  - 次网格陆气相互作用、复杂地形局地效应；
  - 遥感观测噪声；
  - 物理方程简化带来的系统偏差等。
- 表达式：
  $$
  F_{uncertainty}(H(t), \mathcal{G}; \theta_f) = GraphTransformer(H(t), \mathcal{G}).
  $$
- Graph Transformer 层：
  - 多头自注意力，在图节点上运算：
    - 对每个节点，从 $H(t)$ 线性投影得到 $Q, K, V$；
    - 使用邻接/边信息限定注意力范围（局部或稀疏全局）；
    - 聚合得到更新后的节点特征作为残差校正项。
  - 多层堆叠后输出与 $H(t)$ 维度一致的向量场。

### 1.7 损失函数与训练策略

- **监督损失**：T2M 预测与观测的 MSE：
  - 使用 R-Drop，需要对同一样本做两次前向：得到 $P_1, P_2$。
  - 监督部分：
    $$
    L_{supervised} = \frac{1}{2} (MSE(P_1, y) + MSE(P_2, y)).
    $$
- **R-Drop 一致性损失**：
  - 对 $P_1, P_2$ 计算双向 KL 散度：$L_{consistency}(P_1, P_2)$。
- **总损失**：
  $$
  L_{total} = L_{supervised} + \beta \cdot L_{consistency}(P_1, P_2).
  $$
- **优化**：
  - Adam + 学习率调度（验证损失 plateau 时降低 LR）。
  - Dropout + R-Drop 增强泛化和鲁棒性。

### 1.8 超参数搜索（贝叶斯优化）

- 使用 Optuna/Bayesian Optimization：
  - 超参空间包括：
    - Graph Transformer 层数、头数、hidden dim；
    - Encoder/Decoder 容量；
    - R-Drop 中 $\beta$；
    - ODE solver 步长/容差等。
  - 目标函数：验证集损失 $f(x)$。
  - GP 或其它 surrogate 模型 + acquisition function $\alpha(x|D_n)$：
    $$
    x_{n+1} = \arg\max_x \alpha(x | D_n).
    $$

### 1.9 评价指标与基线模型

- **指标**：
  - 纬向加权 RMSE：
    $$
    RMSE = \frac{1}{T} \sum_t \sqrt{\frac{1}{S} \sum_i w_i (\hat{y}_{t,i} - y_{t,i})^2},\\
    w_i = \frac{\cos(\phi_i)}{\overline{\cos(\phi)}}.
    $$
  - ACC（异常相关系数）：对去气候平均后的异常场做加权相关。
- **对比模型**：
  - Persistence（持续性）；
  - 标准 Neural ODE（无物理约束）；
  - ClimaX（Transformer Foundation Model，CMIP6 预训练）；
  - FourCastNet（FNO + ViT，高分辨率全球预报）；
  - ClimODE（物理启发的 Neural ODE，神经流速度驱动）。

### 1.10 核心结果（定性总结）

- **全球多步预报**：
  - GNADET 在所有 lead time 上 RMSE/ACC 都优于 Persistence 和普通 NODE。
  - 相比 ClimaX、FCN、ClimODE：
    - 6 h：RMSE 0.86 K、ACC 0.99，为所有模型中最优（或并列最优）。
    - 长时（72/144 h）：RMSE 2.52/3.01，ACC 0.91/0.86，明显优于 ClimaX、FCN 和 ClimODE。
  - 误差随 lead time 单调增加，曲线平滑，无异常震荡，显示良好稳定性。
- **区域预报（澳洲、南美、北美）**：
  - 几乎所有区域与提前时间上，GNADET RMSE 最低或接近最低（个别点位略输于 ClimaX / ClimODE）。
- **损失分解与消融**：
  - 去掉不确定性模块（仅物理）：平均 RMSE 增加约 14.5%，ACC 略降。
  - 去掉物理项（仅 Graph Transformer）：RMSE 增加约 9.4%。
  - 完整模型显著优于两种残缺版本，证明“物理 + 不确定性”深度耦合的必要性。

---

## 二、组件维度：可复用建模范式

### 2.1 数据处理与图构建组件

**子需求**：
- 将规则格点气象数据转换为适合图神经/图 Transformer 的结构；
- 同时保留静态因子（地形、陆海）与时间序列特征；
- 防止训练-测试信息泄露。

**关键技术要点**：
- 多变量合并为 `(time, node, channels)`；
- Moore 邻域图构造，生成稀疏邻接矩阵及掩膜 $A_{mask}$；
- 仅用训练集统计量做标准化；
- 滑动窗口构造 `(lags, horizon)` 监督样本；
- 预计算 T2M 气候平均场，用于 ACC。

**高层伪代码示意**：

```python
# 构造节点和 Moore 邻域边
nodes = build_global_grid(resolution=5.625)  # 每个格点一个 node
A_mask = build_moore_adjacency(nodes)       # (N, N) 二值邻接矩阵

# 合并多变量数据
X_raw = load_era5_weatherbench(variables=[
    "T2M", "T850", "RH1000", "Q1000", "TP",
    "TISR", "Z500", "U10", "V10", "Oro", "LSM"
])  # shape: (T, H, W, C)

X_merged = reshape_to_nodes(X_raw, nodes)   # (T, N, C)

# 按时间划分
train, val, test = split_by_time(X_merged, [
    ("2006-01", "2015-12"),
    ("2016-01", "2016-12"),
    ("2017-01", "2018-12")
])

# 仅用训练集计算标准化参数
mean, std = compute_mean_std(train)
train = (train - mean) / std
val   = (val   - mean) / std
test  = (test  - mean) / std

# 计算气候平均场(未标准化的 T2M)
climatology_T2M = compute_climatology_T2M(raw_train_T2M)

# 使用滑动窗口生成 (lags, horizon) 样本
train_loader = build_dataloader(train, lags=L, horizon=H, batch_size=B)
val_loader   = build_dataloader(val,   lags=L, horizon=H, batch_size=B)
```

---

### 2.2 物理项组件：图上的平流–扩散算子

**子需求**：
- 在图上近似连续 PDE 中的 $\nabla, \nabla^2$ 操作；
- 允许扩散和平流强度在空间上非均匀；
- 保持物理约束与数值稳定性。

**关键技术要点**：
- 使用图拉普拉斯算子 $L = D - A^T$；
- 通过节点嵌入 $W_1, W_2$ 学习 $A_{diff}, A_{adv}$；
- 使用 $A_{mask}$ 强制局部连接；
- ReLU + tanh 控制权重范围，符合“非负扩散系数”等物理直觉。

**高层伪代码示意**：

```python
class PhysicalAdvectionDiffusion(nn.Module):
    def __init__(self, num_nodes, embed_dim, alpha, A_mask):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.W2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.alpha = alpha
        self.A_mask = A_mask  # (N, N) 0/1

    def build_adj_matrices(self):
        A_diff = self.A_mask * F.relu(torch.tanh(
            self.alpha * (self.W1 @ self.W1.T)
        ))
        A_adv = self.A_mask * F.relu(torch.tanh(
            self.alpha * (self.W1 @ self.W2.T - self.W2 @ self.W1.T)
        ))
        return A_diff, A_adv

    def laplacian(self, A):
        # L = D - A^T
        degree = A.sum(dim=1)              # (N,)
        D = torch.diag(degree)
        L = D - A.T
        return L

    def forward(self, H):
        # H: (batch, N) 或 (batch, N, d)（这里只示意标量场）
        A_diff, A_adv = self.build_adj_matrices()
        L_diff = self.laplacian(A_diff)
        L_adv  = self.laplacian(A_adv)
        # 物理项: - (L_diff H + L_adv H)
        phys_term = -(H @ L_diff.T + H @ L_adv.T)
        return phys_term
```

---

### 2.3 不确定性组件：Graph Transformer 残差动力学

**子需求**：
- 对物理项未能解释的局地/次网格过程做数据驱动的“纠偏”；
- 保证与物理项在 ODE 中可加和，不破坏整体稳定性；
- 能利用图结构（Moore 邻域）挖掘空间相关性。

**关键技术要点**：
- Graph Transformer 层堆叠，对节点特征做多头自注意力；
- 注意力范围由图邻接控制；
- 输出与 $H(t)$ 同维度，用作 ODE 右端的加性残差项。

**高层伪代码示意**：

```python
class UncertaintyGraphTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, A_mask):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, A_mask)
            for _ in range(num_layers)
        ])

    def forward(self, H):
        # H: (batch, N, hidden_dim)
        x = H
        for layer in self.layers:
            x = layer(x)  # 残差 + LayerNorm 可包含在 layer 内
        return x  # 作为 F_uncertainty(H)
```

> 注：`GraphTransformerLayer` 内部实现包括基于邻接稀疏注意力的 Q/K/V 计算，此处不展开细节。

---

### 2.4 PI-NODE 时间演化与 ODE Solver 组件

**子需求**：
- 在连续时间上对状态 $H(t)$ 积分，跨越多步提前时间；
- 将物理项与不确定性项统一在同一 ODE 框架下；
- 避免强自回归导致的误差快速累积。

**关键技术要点**：
- 使用常微分方程求解器（如 `odeint`）对
  $$dH/dt = F_{phys}(H) + F_{uncertainty}(H)$$
  积分；
- ODE right-hand-side 封装两个子模块；
- 解算器步长/容差可调，兼顾精度与效率。

**高层伪代码示意**：

```python
class GNADetDynamics(nn.Module):
    def __init__(self, phys_module, unc_module):
        super().__init__()
        self.phys = phys_module
        self.unc  = unc_module

    def forward(self, t, H):
        # H: (batch, N, hidden_dim)
        phys_term = self.phys(H_mean_or_scalar(H))   # 按需要选取/汇聚
        unc_term  = self.unc(H)
        # 将 phys_term broadcast 回 hidden_dim 维度
        phys_term = phys_term.unsqueeze(-1).expand_as(H)
        dHdt = phys_term + unc_term
        return dHdt


def integrate_gnadet(dynamics, H0, t0, t1, solver="dopri5"):
    # 使用神经 ODE 库接口，此处仅示意
    t_span = torch.tensor([t0, t1])
    H1 = odeint(dynamics, H0, t_span, method=solver)[-1]
    return H1
```

> 注：实际实现中，物理项可能直接在 hidden 空间中定义，这里用 `H_mean_or_scalar` 仅作概念说明。

---

### 2.5 Encoder / Decoder 与端到端训练组件

**子需求**：
- 将多变量、多时刻观测编码成初始潜状态 $H(t_0)$；
- 将预测潜状态 $H(t_1)$ 解码为物理场（T2M）；
- 支持 R-Drop 训练与多 lead time 输出。

**关键技术要点**：
- Encoder：1D/2D 卷积 + MLP / 小型 Transformer over time，将 `(lags, vars)` 压缩到节点 hidden 表示；
- Decoder：简单 MLP，将节点 hidden 映射为标量 T2M，或多 lead time 通道；
- 端到端多任务：可以同时预测多个 lead time，或逐个积分多次。

**高层伪代码示意**：

```python
class GNADetModel(nn.Module):
    def __init__(self, encoder, decoder, dynamics):
        super().__init__()
        self.encoder  = encoder
        self.decoder  = decoder
        self.dynamics = dynamics

    def forward_once(self, x_hist, t0, t_targets):
        # x_hist: (batch, lags, N, C)
        H0 = self.encoder(x_hist)  # (batch, N, hidden_dim)
        preds = []
        for t1 in t_targets:
            H1 = integrate_gnadet(self.dynamics, H0, t0, t1)
            y_hat = self.decoder(H1)  # (batch, N)
            preds.append(y_hat)
        return torch.stack(preds, dim=1)  # (batch, T_lead, N)

    def forward(self, x_hist, t0, t_targets):
        # R-Drop: 两次前向，使用不同 dropout mask
        p1 = self.forward_once(x_hist, t0, t_targets)
        p2 = self.forward_once(x_hist, t0, t_targets)
        return p1, p2


def train_step(model, batch, optimizer, beta, climatology=None):
    x_hist, y_true = batch  # y_true: (batch, T_lead, N)
    p1, p2 = model(x_hist, t0=0.0, t_targets=[6,12,18,24,72,144])

    loss_sup = 0.5 * (mse(p1, y_true) + mse(p2, y_true))
    loss_cons = kl_div_bi(p1, p2)
    loss = loss_sup + beta * loss_cons

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach()
```

---

### 2.6 评估与误差分析组件

**子需求**：
- 计算全球/区域 RMSE 与 ACC；
- 进行空间误差聚类分析，揭示模型在哪些区域更可靠；
- 分解物理项与不确定性项对单步预报的贡献。

**关键技术要点**：
- 纬向权重 $w_i = \cos(\phi_i)/\overline{\cos(\phi)}$；
- ACC 对异常场（减气候平均）计算；
- K-means 对单步 RMSE 地图做聚类（低/中/高误差簇）；
- 单步预报分解：分别可视化
  - 物理 tendency，
  - 不确定性 tendency，
  - 叠加后的总 tendency。

**高层伪代码示意**：

```python
def compute_rmse_acc(preds, targets, lat, climatology):
    # preds, targets: (T, N)
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()

    rmse = 0
    num_t = preds.shape[0]
    for t in range(num_t):
        diff = preds[t] - targets[t]
        rmse_t = np.sqrt((weights * diff**2).mean())
        rmse += rmse_t
    rmse /= num_t

    # ACC（异常场）
    anom_p = preds - climatology
    anom_t = targets - climatology
    num = (weights * anom_p * anom_t).sum()
    den = np.sqrt((weights * anom_p**2).sum() * (weights * anom_t**2).sum())
    acc = num / den
    return rmse, acc


def cluster_spatial_rmse(rmse_map, K=3):
    # rmse_map: (N,) 单步预测 RMSE
    clusters = kmeans(rmse_map.reshape(-1, 1), K)
    return clusters.labels_  # 标记低/中/高误差区域
```

---

### 2.7 经验启示与可迁移范式

- **连续时间 + 物理 + 不确定性**：
  - 将显式 PDE 作为 ODE 右端的一部分，剩余复杂过程交给图神经/Transformer 学习，是一种通用的“物理–数据深度耦合”范式。
- **图上的 PDE 离散化**：
  - 使用 Moore 邻域图 + 学习的边权构建图拉普拉斯，可推广到其他标量/矢量场（如降水、风场等）。
- **局地不确定性建模**：
  - Graph Transformer 适合捕捉复杂空间相关的不确定性，比简单 GCN/GAT 更灵活。
- **稳定性优先的设计**：
  - 使用 PI-NODE 而非简单自回归步进，误差随 lead time 单调且平滑，更接近物理系统行为。
- **面向未来的扩展方向（论文中提出）**：
  - 更高分辨率数据；
  - 多尺度/全局注意力以捕捉长程依赖；
  - 将更多物理过程（辐射、潜热等）显式纳入 PDE 部分；
  - 允许扩散/平流系数随状态 $H(t)$ 动态变化，以适应极端事件环境。
