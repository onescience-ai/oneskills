# GraphCast：结构化模型知识（基于论文内容整理）

> 本文档基于论文《GraphCast: Learning skillful medium-range global weather forecasting》的公开内容进行结构化总结，不加入主观推断；若某信息论文未明确给出，则标明未在文中明确给出。所有伪代码均为**基于论文描述的高保真伪代码**，用于帮助实现与复现。

---

## 维度一：整体模型与任务知识

### 1. 任务设定与预测目标

- **任务类型**：
  - 全球中期（最多 10 天）天气预报。
  - 单次运行产生 0.25° 全球格点、6 小时间隔的多变量未来轨迹。
- **时间与空间分辨率**：
  - 水平：$0.25^\circ$ 纬向–经向网格，尺寸 $721 \times 1440$（约 28km 赤道分辨率）。
  - 垂直：
    - 大气变量在 37 个等压面上：1–1000 hPa，具体层列表见 Table 1/2。
    - 还包含 5 个地面单层变量。
  - 时间步长：$\Delta t = 6\,\mathrm{h}$。
  - 最大预报时长：$T=40$ 步，对应 10 天。
- **输入与输出变量**（ERA5/ECMWF 参数，参见 Table 2）：
  - 大气变量（在 37 层上）：
    - 温度 $t$
    - 水平风 $u, v$
    - 垂直速度 $w$
    - 位势高度 $z$
    - 比湿 $q$
  - 地面变量（单层）：
    - 2 米温度 2t
    - 10 米风分量 10u, 10v
    - 海平面气压 msl
    - 总降水量 tp（6 小时累计）
  - 作为输入但不预测的强迫/静态变量：
    - 顶层太阳入射辐射 tiser（1 小时积分）
    - 地表位势（地形高程）
    - 陆海掩膜 lsm
    - 纬度、经度
    - “钟表”特征：当地时刻（sin/cos）、年进度（sin/cos）
- **输入历史长度**：
  - 使用最近两个时刻的状态 $(X^{t-1}, X^t)$ 作为输入，预测 $X^{t+1}$。
- **输出**：
  - 下一时刻完整格点场 $\hat X^{t+1}$，包含上述 11 个“可预测”变量的 227 个变量–层组合。
  - 采用残差形式：$\hat X^{t+1}=X^t+\hat Y^t$，其中 $\hat Y^t$ 是预测残差。

### 2. 数据与训练/验证划分

- **数据源**：
  - 训练与评估基于 ECMWF 再分析 ERA5（0.25°、1 小时原始分辨率），时间范围 1979–2022 的子集。
  - 训练集中的 ERA5 数据被下采样为 6 小时步长（00z, 06z, 12z, 18z），tp 在对应 6 小时时段累计。
- **训练/验证/测试时间划分**：
  - 开发阶段：
    - 训练：1979–2015
    - 验证：2016–2017
    - 测试：2018–2021
  - 正式测试阶段：
    - 多个版本 GraphCast 分别训练到 2017/2018/2019/2020 结束，用于探究“训练数据新鲜度”对 2021 年性能的影响。
- **HRES/HRES-fc0 用途**：
  - HRES（IFS 的 0.1° 确定性业务模式）：作为对比基准。
  - HRES-fc0：从 HRES 预报初始化场构造的“step-0 预报”数据集，作为评估 HRES skill 的地面真值，避免 HRES 在 0 步有非零误差。
- **再分析与同化窗口对齐**：
  - ERA5 同化窗口：两个 ±9h/−3h 窗口（中心在 00z 与 12z），等价于两个 +3h/−9h 窗口（中心在 06z, 18z）。
  - HRES 同化窗口：每天 4 个 ±3h 窗口（00z, 06z, 12z, 18z）。
  - 为保证 GraphCast 与 HRES 具有相同“未来观测 lookahead”，GraphCast 只在 06z、18z 初始化并每 12 小时评估（目标 ERA5 也具有 +3h lookahead），HRES 则在匹配的同化窗口下评估。

### 3. 模型整体架构与计算范式

- **总体范式**：
  - 将 0.25° 纬经格点上的多变量场，映射到一个基于等二十面体 refined multi-mesh 的图结构上，通过 GNN 执行多层消息传递，再解码回原始格点。
  - 单步预测器，按式
    $$\hat X^{t+1} = \text{GraphCast}(X^t, X^{t-1}) = X^t + \hat Y^t$$
    可自回归滚动生成完整 10 日轨迹。
- **Encode–Process–Decode GNN 架构**：
  1. **Encoder（Grid→Multi-mesh）**：
     - 结点类型：
       - Grid nodes：0.25° 格点，带 2 个时间步的状态、强迫、静态特征。
       - Mesh nodes：多层 refined icosahedral 网格上的结点（R=6，40962 结点）。
     - 边类型：
       - Grid2Mesh：从每个格点到距离阈值内的 mesh 结点的有向边（约 1.6M 条）。
       - Mesh edges：多尺度 multi-mesh 边（0–6 级 refinement，总 327660 条，双向）。
     - Encoder 步骤：对所有结点/边特征用 MLP 嵌入；执行一轮 Grid2Mesh GNN message passing，将格点信息汇聚到 mesh 上。
  2. **Processor（Multi-mesh GNN）**：
     - 在 multi-mesh 上堆叠 16 层 Interaction Network 风格 GNN（边更新→结点更新），层间参数不共享。
     - 利用 coarse-level 边实现大尺度长程信息传播，fine-level 边描述局地相互作用。
  3. **Decoder（Multi-mesh→Grid）**：
     - Mesh2Grid：单层 GNN，将 mesh 表示通过三角面邻接（每个格点连到其三角面的 3 个 mesh 结点）回投到格点。
     - Output MLP：对每个格点结点向量输出 227 维残差 $\hat y_{i}$，通过反标准差缩放得到物理量残差，再加到 $X^t$ 得到 $\hat X^{t+1}$。
- **参数规模与效率**：
  - 总参数量 36.7M，所有 MLP 隐层宽度 512，激活为 swish，LayerNorm（输出层除外）。
  - 在单个 Cloud TPU v4 上，可在 1 分钟内生成 10 天 0.25° 全局预报（6 小时步长）。

### 4. 训练目标与自回归训练

- **损失函数**：
  - 以 12 步（3 天）自回归滚动轨迹为单位，最小化加权 MSE：
    $$
    \mathcal L_\text{MSE} = \frac{1}{|D_\text{batch}|}\sum_{d_0}\frac{1}{T_\text{train}}\sum_{\tau=1}^{T_\text{train}}\frac{1}{|G|}\sum_{i\in G}\sum_{j\in J} s_j w_j a_i (\hat x^{d_0+\tau}_{i,j}-x^{d_0+\tau}_{i,j})^2
    $$
    - $s_j$：变量–层时间差 $(X^{t+1}-X^t)$ 的逆方差，用于单位化尺度。
    - $w_j$：变量–层损失权重（大气层级按压力权重，地面变量为调参结果）。
    - $a_i$：格点面积权重（按纬度归一）。
- **自回归训练**：
  - 采用 BPTT，在训练时使用模型预测 $\hat X^{t+1}$ 作为下一步输入，计算所有步的损失并反向传播。
  - 最终采用 $T_\text{train}=12$ 步的自回归 horizon（对应 3 天）。
- **训练日程（curriculum）**：
  1. Phase 1：1000 step，单步预测，学习率线性上升到 1e−3。
  2. Phase 2：299k step，单步预测，学习率余弦衰减到 0。
  3. Phase 3：11k step，多步自回归训练，autoregressive 步数从 2 增至 12，每 1000 step+1，学习率固定 3e−7。
- **优化与工程细节**：
  - 优化器：AdamW，$\beta_1=0.9, \beta_2=0.95$，权重衰减 0.1。
  - 梯度裁剪：范数上限 32。
  - 批大小：32（通过 32 TPU 设备做 data parallel，每设备 1 个样本）。
  - 精度：激活与训练图使用 bfloat16，评估指标用 float32。
  - 内存优化：梯度检查点 + 多设备并行。完整训练约 4 周（32× TPU v4）。

### 5. 评估方法与指标

- **评估分割与初始化时次**：
  - 主结果：2018 年测试（GraphCast 训练至 2017），初始化时刻 06z/18z，lead time 每 12 小时至 10 天。
  - HRES：
    - 06z/18z 预报只到 3.75 天；3.5 天后切换为 00z/12z 初始化的 10 天预报（图中用虚线标记切换）。
- **Skill 指标**：
  - RMSE（地球表面面积加权）：
    $$
    \text{RMSE}(j,\tau)=\frac{1}{|D_\text{eval}|}\sum_{d_0}\sqrt{\frac{1}{|G|}\sum_{i\in G} a_i (\hat x^{d_0+\tau}_{i,j}-x^{d_0+\tau}_{i,j})^2}
    $$
  - ACC（Anomaly Correlation Coefficient）：
    - 基于与气候平均场（按日历日）之差计算的相关系数（详见原文式 (29)）。
  - Skill score 定义：
    - RMSE skill score：$\frac{\text{RMSE}_A-\text{RMSE}_B}{\text{RMSE}_B}$（负值代表 A 优于 B）。
    - ACC skill score：$\frac{\text{ACC}_A-\text{ACC}_B}{1-\text{ACC}_B}$。
- **统计显著性**：
  - 对逐 lead time、逐变量–层的 RMSE 差值时间序列，构造 AR(2) 过程并用配对 t 检验，修正自相关后得到有效样本数 $n_\text{eff}$ 与 p 值。

### 6. 整体结果与对比

- **对 HRES 的整体 skill**：
  - 在 1380 个“变量–层–lead time”组合中，GraphCast 在 90.3% 目标上 RMSE 优于 HRES，在 89.9% 目标上显著优于 HRES（p≤0.05）。
  - 若剔除 50 hPa，则在剩余 1280 目标中 96.9% 显著优于 HRES；再剔除 100 hPa 后，在 1180 目标中 99.7% 显著优于 HRES。
  - HRES 优势主要集中在平流层高层（权重极小），与训练损失层权重设置一致。
- **代表性变量 z500**：
  - 在 RMSE、RMSE skill score 与 ACC 上，GraphCast 在 10 天内均优于 HRES，z500 RMSE skill score 提升约 7–14%。
- **模糊化与不确定性表达**：
  - 增大自回归步数会促使输出更平滑（long-lead blur），表现为更偏向空间模糊的场以表达不确定性。
  - 对 HRES 与 GraphCast 施加“最优模糊滤波”（RMSE 最小化），结果 GraphCast 在 88% 目标上仍优于 HRES，说明优势不只是“更模糊”。
- **与 Pangu-Weather 对比**：
  - 在 Pangu-Weather 报告的 252 个目标中，GraphCast 在 99.2% 目标上 RMSE 更优，仅在 z500 的前两步（6h, 12h）略逊（约 1.7% 更大 RMSE）。

### 7. 严重天气事件与应用结果

- **热带气旋路径**：
  - 追踪算法：基于 ECMWF 公开的气旋追踪协议，对 z、10m 风、层结风场与 msl 进行检测；HRES 基准轨迹来自 TIGGE 数据库；真值来自 IBTrACS。
  - 评估方式：仅统计 GraphCast 与 HRES 都检测到的气旋，使用轨迹误差的中位数及配对差异。
  - 结果：2018–2021 年，GraphCast 在 18 小时到 4.75 天 lead time 上具有更低的中位路径误差；误差差异在统计意义上显著。
- **大气河流（Atmospheric Rivers）**：
  - 通过预测的 $u, v, q$ 计算垂直积分水汽输送 ivt，并在北美西海岸冷季（10–4 月）评估。
  - GraphCast 对 ivt 的 RMSE 相比 HRES 提升约 10–25%（短 lead 更高，长 lead 略低），在未专门针对 AR 训练的条件下表现更优。
- **极端高温/低温**：
  - 以 2m 温度在地点–时刻–月份上的气候分布上 2% 分位作为极端阈值（也测试 0.5%、5% 等）。
  - 在夏季、陆地、南北半球区域，对 2t 极端事件做精确率–召回率曲线评估：
    - Lead=5d,10d 时，GraphCast PR 曲线整体优于 HRES，表明其更适合长时段极端高温预报。
    - Lead=12h 时，HRES 精度略优，这与 2t 在短期 skill score 接近相符。

### 8. 训练数据新鲜度与再训练

- 训练 4 个版本：GraphCast-<2018, <2019, <2020, <2021，在 2021 年上评估。
- 结果：
  - 随着训练数据终止年向 2021 靠近，z500 等变量的 skill score 持续改善。
  - 说明通过定期用最新 ERA5 再训练，可捕捉 ENSO、气候趋势等低频变化，提升后续年份预报 skill。

### 9. 结论与局限

- **结论**：
  - GraphCast 在 0.25°/10 天中期预报上，与 ECMWF HRES 相比有显著整体优势（特别是在对流层主层），并在热带气旋、AR、极端温度等关键应用上表现优良。
  - 通过高效的 multi-mesh GNN 与自回归训练，将 MLWP 推到了与最强 NWP 系统一致甚至更优的水平。
- **局限性**：
  - 当前版本为**确定性**模型：
    - 不输出明确的概率分布或集合成员；不确定性主要通过“模糊化”体现。
    - 长 lead 时的空间平滑可能限制部分应用（例如极端局地降水）。
  - 分辨率受 ERA5 与工程约束限制：0.25°、37 层、6 小时步长；IFS/HRES 在 0.1°、137 层、1 小时步长上运行。
  - 论文指出，面向不确定性建模（如集合/概率预测）的扩展是下一步关键工作。

---

## 维度二：组件与基础结构知识（含高保真伪代码）

> 下述伪代码均为**基于论文描述的高保真伪代码**，用于帮助实现 GraphCast 或同类模型；若具体实现细节仅在补充材料中出现或未给出，则在伪代码中保持抽象或标明未在文中明确给出。

### 1. 数据与状态表示

#### 1.1 天气状态张量结构

- 设：
  - 空间网格 $G$: 0.25° 格点集合，尺寸 $(H=721, W=1440)$，索引 $i=(h,w)$。
  - 压力层集合 $P$：37 层。
  - 变量集合 $J$：
    - 大气变量：$z, t, u, v, w, q$（在 $P$ 上定义）。
    - 地面变量：2t, 10u, 10v, msl, tp（单层）。
- 单个时间步状态 $X^t$：
  - 可表示为形状 `[H, W, C]` 的张量，其中 $C = 5 + 6\times 37 = 227$。

```python
# 伪代码：ERA5 单步状态组织
class WeatherState:
    # shape: [H, W, C]
    grid_values: Tensor

    # 可选：分解访问
    def at_surface(self):
        return self.grid_values[..., surface_channel_indices]

    def at_level(self, level_idx):
        return self.grid_values[..., level_channel_slice(level_idx)]
```

#### 1.2 输入特征打包（含强迫与静态特征）

```python
# 伪代码：构造 GraphCast 输入特征（单时间步对）
def build_input_features(X_t_minus_1: WeatherState,
                         X_t: WeatherState,
                         forcing_t_minus_1: ForcingFields,
                         forcing_t: ForcingFields,
                         forcing_t_plus_1: ForcingFields,
                         static_fields: StaticFields) -> Tensor:
    """返回 shape [H, W, F] 的特征张量。

    其中 F ≈ 474，包含：
    - 2 个时间步的 227 变量 (X^{t-1}, X^t)
    - 3 个时间点的 forcing（如 tiser、time-of-day、year-progress）
    - 静态特征（地形、陆海掩膜、经纬度编码）
    """
    # 1. 归一化（按变量/层统计的均值方差）
    x_tm1 = normalize_physical(X_t_minus_1.grid_values)
    x_t   = normalize_physical(X_t.grid_values)

    f_tm1 = normalize_forcing(forcing_t_minus_1.to_tensor())
    f_t   = normalize_forcing(forcing_t.to_tensor())
    f_tp1 = normalize_forcing(forcing_t_plus_1.to_tensor())

    s     = encode_static(static_fields)

    # 2. 拼接通道
    features = concat_channels([
        x_tm1, x_t,
        f_tm1, f_t, f_tp1,
        s,
    ])  # [H, W, F]
    return features
```

### 2. Multi-mesh 图构造与特征

#### 2.1 等二十面体多级 mesh 与 multi-mesh

```python
# 伪代码：构造 refined icosahedral multi-mesh
def build_icosahedral_multimesh(R: int = 6) -> MultiMesh:
    """构造从 M^0 到 M^R 的多级等二十面体网格，并合并为 multi-mesh。
    - M^0: 基础 12 结点 + 20 三角面
    - 每一级 r: 每个三角面分裂为 4 个，新增结点并投影到单位球面
    - multi-mesh 结点集 = M^R 所有结点
    - multi-mesh 边集 = 所有层级的边集合并
    """
    meshes = [refine_icosahedron_level0()]
    for r in range(1, R + 1):
        meshes.append(refine_mesh(meshes[-1]))  # 拆分三角面, add nodes & edges

    # 结点：使用最高分辨率 M^R 的所有结点
    nodes = meshes[-1].nodes  # [N_mesh, 3] on unit sphere

    # 边：合并所有层级的 edges（双向）
    edges = []
    for m in meshes:
        edges.extend(to_bidirectional(m.edges))

    return MultiMesh(nodes=nodes, edges=edges, level_meshes=meshes)
```

- 每个 mesh 结点的初始特征：
  - $(\cos \text{lat}, \sin \text{lon}, \cos \text{lon})$ 等简单几何编码。

#### 2.2 Grid↔Mesh 映射边

- **Grid2Mesh**：
  - 对每个格点，找到所有距离小于等于某阈值（约 0.6 × finest edge length）的 mesh 结点，建立从 grid→mesh 的边。
- **Mesh2Grid**：
  - 对每个格点，在 $M^6$ 上找到包含该点的三角面，将格点连到该三角面 3 个顶点，形成 mesh→grid 的 3 条有向边。
- **边特征**：
  - 使用球面 3D 坐标计算：
    - 边长度
    - 在接收点局部坐标系下的方向向量差 $(\Delta x, \Delta y, \Delta z)$。

```python
# 伪代码：构造 Grid2Mesh, Mesh2Grid 边

def build_grid_mesh_edges(grid_locs: Tensor,  # [H, W, 3] on sphere
                           mesh: MultiMesh,
                           radius_factor: float = 0.6) -> Tuple[Edges, Edges]:
    # 估计 finest-level 邻接边长
    base_edge_len = estimate_finest_edge_length(mesh)
    radius = radius_factor * base_edge_len

    grid2mesh_edges = []
    mesh2grid_edges = []

    # 1. Grid2Mesh: 基于球面距离邻居
    for i, grid_xyz in enumerate(grid_locs.reshape(-1, 3)):
        nbr_mesh_ids = find_mesh_nodes_within_radius(mesh.nodes, grid_xyz, radius)
        for j in nbr_mesh_ids:
            feat = edge_geo_features(sender=grid_xyz, receiver=mesh.nodes[j])
            grid2mesh_edges.append((i, j, feat))

    # 2. Mesh2Grid: 利用三角面包含关系
    tri_faces = mesh.level_meshes[-1].faces  # [F, 3]
    face_spatial_index = build_face_spatial_index(tri_faces, mesh.nodes)

    for i, grid_xyz in enumerate(grid_locs.reshape(-1, 3)):
        face_id = find_containing_face(face_spatial_index, grid_xyz)
        v_ids = tri_faces[face_id]  # 3 mesh nodes
        for j in v_ids:
            feat = edge_geo_features(sender=mesh.nodes[j], receiver=grid_xyz)
            mesh2grid_edges.append((j, i, feat))

    return grid2mesh_edges, mesh2grid_edges
```

### 3. GraphCast 前向结构伪代码

#### 3.1 高层接口：单步与多步预测

```python
# 伪代码：GraphCast 单步
class GraphCastModel:
    def __init__(self, graph_meta: GraphMeta, params: Params):
        self.meta = graph_meta  # 包含 multi-mesh, edges, normalization 统计等
        self.params = params    # 所有 MLP/GNN 权重

    def one_step(self, X_tm1: WeatherState, X_t: WeatherState,
                 forcing_tm1: ForcingFields,
                 forcing_t: ForcingFields,
                 forcing_tp1: ForcingFields,
                 static_fields: StaticFields) -> WeatherState:
        # 1. 构造格点特征
        grid_feat = build_input_features(
            X_t_minus_1=X_tm1,
            X_t=X_t,
            forcing_t_minus_1=forcing_tm1,
            forcing_t=forcing_t,
            forcing_t_plus_1=forcing_tp1,
            static_fields=static_fields,
        )  # [H, W, F]

        # 2. 调用 encode–process–decode GNN
        delta_norm = graphcast_core_forward(
            grid_feat,            # 归一化空间上的特征
            self.meta,            # 包含 grid2mesh, mesh, mesh2grid
            self.params,
        )  # [H, W, C], 单位为“标准差单位”的残差

        # 3. 反标准化残差
        delta_physical = denormalize_time_diff(delta_norm)  # 乘以每变量-层 std

        # 4. 残差加到当前状态得到预测
        X_tp1 = WeatherState(grid_values=X_t.grid_values + delta_physical)
        return X_tp1

    def rollout(self, X_hist: List[WeatherState],
                forcing_seq: List[ForcingFields],
                static_fields: StaticFields,
                T: int) -> List[WeatherState]:
        """给定 X^{t-1}, X^t 与之后的 forcing，生成长度 T 的轨迹。
        forcing_seq 对应 [t-1, t, t+1, ..., t+T]
        """
        states = []
        X_tm1, X_t = X_hist[-2], X_hist[-1]
        for k in range(T):
            f_tm1 = forcing_seq[k]
            f_t   = forcing_seq[k + 1]
            f_tp1 = forcing_seq[k + 2]

            X_tp1 = self.one_step(X_tm1, X_t, f_tm1, f_t, f_tp1, static_fields)
            states.append(X_tp1)

            X_tm1, X_t = X_t, X_tp1
        return states
```

#### 3.2 Encode：嵌入与 Grid2Mesh GNN

```python
# 伪代码：GraphCast core 前向

def graphcast_core_forward(grid_feat: Tensor,  # [H, W, F]
                            meta: GraphMeta,
                            params: Params) -> Tensor:
    # 0. 展平格点为结点
    Vg_features = grid_feat.reshape(-1, grid_feat.shape[-1])  # [N_grid, F]

    # 1. 嵌入各类结点/边特征
    Vg = mlp_embed_Vg(Vg_features, params.Vg_embed)          # [N_grid, D]
    Vm = mlp_embed_Vm(meta.mesh_nodes_features, params.Vm_embed)  # [N_mesh, D]

    Eg2m = mlp_embed_Eg2m(meta.grid2mesh_edge_features,
                          params.Eg2m_embed)  # [E_g2m, D_e]
    Em   = mlp_embed_Em(meta.mesh_edge_features,
                        params.Em_embed)      # [E_m, D_e]
    Em2g = mlp_embed_Em2g(meta.mesh2grid_edge_features,
                          params.Em2g_embed)  # [E_m2g, D_e]

    # 2. Grid2Mesh 单层 GNN（Interaction Network）
    Vg, Vm, Eg2m = grid2mesh_gnn_step(
        Vg, Vm, Eg2m, meta.grid2mesh_edges, params.grid2mesh_gnn)

    # 3. Multi-mesh Processor：16 层 Mesh GNN
    for l in range(16):
        Vm, Em = mesh_gnn_layer(
            Vm, Em, meta.mesh_edges, params.mesh_layers[l])

    # 4. Mesh2Grid 单层 GNN，将信息回投到格点
    Vg, Em2g = mesh2grid_gnn_step(
        Vg, Vm, Em2g, meta.mesh2grid_edges, params.mesh2grid_gnn)

    # 5. 输出残差预测（归一化单位）
    delta_norm = mlp_output(Vg, params.output_mlp)  # [N_grid, C]

    return delta_norm.reshape(grid_feat.shape[0], grid_feat.shape[1], -1)
```

```python
# 伪代码：Grid2Mesh GNN 单层

def grid2mesh_gnn_step(Vg, Vm, Eg2m, edges_g2m, p):
    # edges_g2m: list of (grid_id, mesh_id)

    # 1. Edge 更新
    for e_idx, (g_id, m_id) in enumerate(edges_g2m):
        Eg2m[e_idx] = mlp(
            concat([Eg2m[e_idx], Vg[g_id], Vm[m_id]]),
            p.edge_mlp,
        )

    # 2. Mesh 结点聚合（接收 Eg2m）
    msg_sum_to_mesh = zeros_like(Vm)
    for e_idx, (g_id, m_id) in enumerate(edges_g2m):
        msg_sum_to_mesh[m_id] += Eg2m[e_idx]

    Vm_update = zeros_like(Vm)
    for m_id in range(len(Vm)):
        Vm_update[m_id] = mlp(
            concat([Vm[m_id], msg_sum_to_mesh[m_id]]),
            p.node_mesh_mlp,
        )

    # 3. Grid 结点独立更新（无聚合）
    Vg_update = mlp(Vg, p.node_grid_mlp)

    # 4. 残差连接
    Vg = Vg + Vg_update
    Vm = Vm + Vm_update
    Eg2m = Eg2m  # 若需要残差可再加一次

    return Vg, Vm, Eg2m
```

```python
# 伪代码：Multi-mesh Processor 层

def mesh_gnn_layer(Vm, Em, mesh_edges, p):
    # mesh_edges: list of (sender_id, receiver_id)

    # 1. Edge 更新
    for e_idx, (s_id, r_id) in enumerate(mesh_edges):
        Em[e_idx] = mlp(
            concat([Em[e_idx], Vm[s_id], Vm[r_id]]),
            p.edge_mlp,
        )

    # 2. Node 聚合
    msg_sum_to_node = zeros_like(Vm)
    for e_idx, (s_id, r_id) in enumerate(mesh_edges):
        msg_sum_to_node[r_id] += Em[e_idx]

    Vm_update = zeros_like(Vm)
    for m_id in range(len(Vm)):
        Vm_update[m_id] = mlp(
            concat([Vm[m_id], msg_sum_to_node[m_id]]),
            p.node_mlp,
        )

    # 3. 残差
    Vm = Vm + Vm_update
    Em = Em  # 如需额外残差更新可添加
    return Vm, Em
```

```python
# 伪代码：Mesh2Grid GNN 单层

def mesh2grid_gnn_step(Vg, Vm, Em2g, edges_m2g, p):
    # edges_m2g: list of (mesh_id, grid_id)

    # 1. Edge 更新
    for e_idx, (m_id, g_id) in enumerate(edges_m2g):
        Em2g[e_idx] = mlp(
            concat([Em2g[e_idx], Vm[m_id], Vg[g_id]]),
            p.edge_mlp,
        )

    # 2. Grid 结点聚合
    msg_sum_to_grid = zeros_like(Vg)
    for e_idx, (m_id, g_id) in enumerate(edges_m2g):
        msg_sum_to_grid[g_id] += Em2g[e_idx]

    Vg_update = zeros_like(Vg)
    for g_id in range(len(Vg)):
        Vg_update[g_id] = mlp(
            concat([Vg[g_id], msg_sum_to_grid[g_id]]),
            p.node_mlp,
        )

    # 3. 残差
    Vg = Vg + Vg_update
    return Vg, Em2g
```

### 4. 训练循环与自回归损失

```python
# 伪代码：GraphCast 训练 step（autoregressive, 多步）

def train_step(batch: Batch,  # 含多条 12+ 步 ERA5 轨迹
               model: GraphCastModel,
               optimizer: Optimizer,
               T_train: int = 12):
    # batch 初始化时间集合 {d0}

    def loss_fn(params):
        total_loss = 0.0
        count = 0
        for sample in batch:
            X_seq = sample.X_seq   # [t-1, t, ..., t+T_train]
            F_seq = sample.F_seq   # 对应 forcing
            static = sample.static

            X_tm1 = X_seq[0]
            X_t   = X_seq[1]

            for tau in range(1, T_train + 1):
                f_tm1 = F_seq[tau - 1]
                f_t   = F_seq[tau]
                f_tp1 = F_seq[tau + 1]

                # 单步预测
                X_pred = model.one_step(
                    X_tm1, X_t,
                    f_tm1, f_t, f_tp1,
                    static_fields=static,
                )

                # 目标为 ERA5 的 X^{t+tau}
                X_target = X_seq[tau + 1]

                # 计算加权 MSE（按变量–层–格点–面积–权重）
                total_loss += weighted_mse_over_grid_and_channels(
                    X_pred.grid_values,
                    X_target.grid_values,
                    weights_sj=model.meta.s_j,
                    weights_wj=model.meta.w_j,
                    cell_area=model.meta.cell_area,
                )
                count += 1

                # 自回归：将预测反馈为下一步输入
                X_tm1, X_t = X_t, X_pred

        return total_loss / max(count, 1)

    grads = grad(loss_fn)(model.params)
    updates, new_opt_state = optimizer.update(grads, model.opt_state)
    model.params = apply_updates(model.params, updates)
    model.opt_state = new_opt_state
```

### 5. 评估与统计检验伪代码（简化）

```python
# 伪代码：按 WeatherBench 协议计算 RMSE/ACC

def compute_rmse(model, eval_dataset, variable, level, tau):
    errors_sq_sum = 0.0
    area_sum = 0.0
    for sample in eval_dataset:  # 遍历 d0
        X_hist, F_seq, static = sample.init_inputs
        forecast_seq = model.rollout(X_hist, F_seq, static, T=tau)
        X_pred = forecast_seq[-1]
        X_true = sample.get_target(variable, level, tau)

        diff = X_pred.var_level(variable, level) - X_true
        errors_sq_sum += (diff**2 * sample.area_weights).sum()
        area_sum += sample.area_weights.sum()

    rmse = sqrt(errors_sq_sum / area_sum / len(eval_dataset))
    return rmse
```

（更完整的 ACC、skill score、t 检验与 AR(2) 修正过程在论文补充材料中有严格定义，本文不再重复全部推导。）

---

以上即 GraphCast 论文在你统一模版下的结构化知识与高保真伪代码抽取；如需，我可以继续基于 knowledge 目录内各模型（如 FourCastNet/FuXi/GenCast/GraphCast）做横向比较表或统一 API 设计。