# NeuralGCM：结构化模型知识（基于论文内容整理）

> 本文档基于论文《Neural general circulation models for weather and climate》的公开内容进行结构化总结，不加入主观推断；若某信息论文未明确给出，则标明“未在文中明确给出”。所有伪代码均为**基于论文描述的高保真伪代码**，用于帮助实现与复现。

---

## 维度一：整体模型与任务知识

### 1. 任务设定与总体目标

- **总体目标**：
  - 构建一个可同时胜任**天气预报（1–15 天）**与**气候模拟（多年到数十年）**的神经网络–物理混合 GCM（NeuralGCM），在精度上与最优传统 GCM 与顶级 ML 模型相当或更优，同时具备：
    - 稳定的长时间积分能力（无严重气候漂移或数值发散）。
    - 支持**集合预报**并给出校准良好的不确定性估计。
    - 显著低于传统 GCM 的计算成本。
- **对比基准**：
  - 传统物理模式：ECMWF-HRES（确定性）、ECMWF-ENS（集合）、CMIP6 AMIP 试验、全球云分辨模式 X-SHiELD 等。
  - 机器学习模式：GraphCast、Pangu 等 deterministic MLWP。

### 2. 空间–时间分辨率与变量

- **分辨率**：
  - 训练并评估了一系列水平分辨率：
    - 2.8°、1.4°、0.7° 网格（对应约 280 km / 140 km / 70 km 量级）。
  - 垂直层数与具体变量集合在文中方法部分与补充表格中给出，此处未完全列出（**细节未在当前摘录中完全给出**）。
- **时间步长与预报范围**：
  - 训练时：最大 rollout 长度为 5 天，基于 6 小时步长递推（在训练课程中从 6 小时逐步增加至 5 天）。
  - 评估：
    - 中期天气预报：1–15 天 lead time。
    - 气候模拟：多年到数十年的长期积分（在给定 SST/海冰条件下）。

### 3. 数据与训练/验证设置

- **数据集**：
  - ERA5 再分析：约 40 年历史（具体年份段参见原文），用于：
    - 训练 NeuralGCM 的天气轨迹（最长 5 天）。
    - 初始化中期预报评估（例如 2020 年数据）。
    - 气候验证（例如 1981–2014 温度趋势、1980–2019 年度平均等）。
- **训练目标（天气模式）**：
  - 使用 ERA5 作为“真值”，在 rollout 长度从 6 小时到 5 天的序列上，最小化 NeuralGCM 预测轨迹与 ERA5 轨迹之间的差异（具体损失形式在 Methods/补充材料中详细定义，当前摘录中未给出完整公式）。
  - 使用**端到端反向传播（BPTT）**，在整个隐式–显式 ODE 时间积分图上反向传播梯度，从而将**物理核与学习物理模块的耦合效应**也纳入训练。
- **训练策略**：
  - 采用**curriculum rollout schedule**：
    - 先对短时（6h）预测进行训练，以获得稳定的一步预报。
    - 之后逐步增加 rollout 长度（多步 BPTT），直至 5 天；在训练早期更长的 rollout 会导致不稳定或误差爆炸，因此必须渐进式增加。
  - 训练了**确定性**与**随机（集合）**版本，二者训练协议有所不同（随机版本需学习条件概率分布或噪声注入机制，具体损失未在摘录中详细给出）。

### 4. 模型整体架构

- **核心思想**：
  - 将传统 GCM 分解为：
    1. **可微分动力核（differentiable dynamical core）**：
       - 数值求解离散形式的大尺度动力学与热力学方程（含重力、科氏力等）；
       - 负责**平流与大尺度动力过程**（resolved dynamics）。
    2. **学习物理模块（learned physics module）**：
       - 以垂直柱（column）为单位，用神经网络表示未解析物理过程（云、辐射、降水、次网格湍流等）对局地变量的“物理趋势”（tendencies）。
  - 两者通过统一的 ODE 形式耦合，并通过隐式–显式时间积分器推进：
    $$
    \frac{dx}{dt} = f_\text{dyn}(x, F_t) + f_\text{phys}(x, F_t, z_t) ,
    $$
    其中 $x$ 为状态向量，$F_t$ 为外强迫（如辐射、SST），$z_t$ 为潜在随机噪声（仅在随机模型中）。

- **结构如 Fig. 1 所示**：
  1. 外部输入（再分析场、强迫、随机噪声等）经编码器组成内部模型状态 $x_t$；
  2. $x_t$ 输入动力核与学习物理模块：
     - 动力核给出大尺度动力学趋势；
     - 学习物理模块在每一垂直柱上给出物理过程趋势；
  3. 两类趋势通过 IMEX ODE 积分器合成，生成下一步状态 $x_{t+1}$；
  4. $x_{t+1}$ 既可作为下一步输入继续积分，也可经解码器得到预报变量场。

### 5. 中期天气预报评估

- **评估设置**：
  - 按 WeatherBench2 协议：
    - 所有模型（NeuralGCM、GraphCast、Pangu、ECMWF 模型）输出统一重采样至 1.5° 网格。
    - 时间：使用 2020 年全年 732 次初始化（00z 与 12z），2020 年为所有 ML 模型的 held-out 年份。
  - 地面真值：
    - 对 ML 模型（NeuralGCM、GraphCast、Pangu）：ERA5。
    - 对 ECMWF-HRES/ENS：ECMWF 操作分析（HRES step-0），避免因 ERA5 与 HRES 偏差导致 HRES 在 step-0 就有误差。

- **指标**：
  1. **RMSE（Root Mean Square Error）**：
     - 用于 deterministic 与 ensemble mean 轨迹，与 WeatherBench 指标一致；
  2. **RMSB（Root Mean Square Bias）**：
     - 用于刻画长期偏差（例如多年模拟中的系统性偏冷/偏暖）；
  3. **CRPS（连续秩概率评分）**：
     - 针对集合预报的适当概率评分，考察完整边缘分布的质量；
  4. **Spread–skill ratio（SSR）**：
     - 集合预报“集合散度”与“误差”的比值，近似为 1 表示集合校准良好。

- **中期预报主结果**（基于当前摘录）：
  - **确定性模式（NeuralGCM-0.7°）**：
    - 在 1–3 天短 lead 上，RMSE 与 GraphCast 等最佳 ML 模型相当或略优，具体变量略有差异：
      - 对 z500、t850 等 headline 变量，NeuralGCM-0.7° 与 GraphCast 领先；
    - 在更长 lead（>3 天）上，deterministic 模型 RMSE 迅速增大（混沌导致轨迹发散），但 NeuralGCM 在 RMSB 上表现更好（长期偏差更小），尤其是 700 hPa 比湿在热带的偏差显著低于其他方法。
  - **集合模式（NeuralGCM-ENS，1.4°）**：
    - 在 ensemble-mean RMSE、RMSB 与 CRPS 上，NeuralGCM-ENS 在几乎所有变量、lead time 与垂直层上均优于 ECMWF-ENS：
      - CRPS 相对 ECMWF-ENS 更低，表示整体集合分布更接近真值；
      - 空间分布上 RMSE/CRPS 的 skill pattern 与 ECMWF-ENS 类似，但数值上更优；
    - Spread–skill ratio 接近 1，与 ECMWF-ENS 类似，说明集合分布的离散度与误差匹配较好，预报校准良好。

- **物理一致性（案例分析）**：
  - 选取 2020 年 Hurricane Laura、AR 与 ITCZ 等现象的预报案例：
    - ML 模型（GraphCast、Pangu）在 5–10 天时呈现明显“模糊化”现象，NeuralGCM-0.7° 虽分辨率更粗，仍给出相对更清晰、物理一致的结构；
    - NeuralGCM-ENS 与 ECMWF-ENS 的 ensemble mean 在长 lead 时也较平滑，但**单个集合成员**保持清晰结构，与 ERA5 与 ECMWF-ENS 类似。

- **谱分析（blurriness）**：
  - 通过能谱分析（球谐域），NeuralGCM-0.7° 的谱更加接近 ERA5，虽然仍略逊于 ECMWF 物理模式；
  - GraphCast 等 deterministic 模型在长 lead 时谱快速失真，表现为小尺度能量衰减（模糊化加重）；
  - 提高 NeuralGCM 分辨率（例如从 2.8° 提升到 0.7°）可以显著改善谱特性，表明进一步提升分辨率具有潜力。

- **地转风平衡**：
  - Pangu 被指出在长 lead 时严重偏离地转/年龄转风垂直结构；
  - GraphCast 也出现随 lead time 增长而恶化的地转平衡误差；
  - NeuralGCM 对比 ERA5 显示：
    - 地转与年龄转风垂直结构更为接近 ERA5；
    - 在 5 天后，结构误差不再明显恶化；
    - ECMWF-HRES 仍然略优于 NeuralGCM，但后者已明显好于纯 ML 模型。

### 6. 长期气候模拟与 AMIP 实验

- **总体设置**：
  - 使用 2.8° 与 1.4° 的 deterministic NeuralGCM 进行气候模拟。
  - 外部强迫：
    - 历史 SST（海表温度）与海冰浓度作为边界条件（AMIP 风格）。
  - 初始条件：
    - 多个起始时间（例如每 10 天一次），产生多条长期模拟，考察稳定性与气候统计量。

- **稳定性与漂移**：
  - 之前 hybrid 模型在长时间积分中易出现数值不稳定或气候漂移；
  - NeuralGCM 通过端到端训练与合适超参数选择，可以：
    - 在多年甚至数十年时间尺度上保持稳定（未见严重 blow-up）；
    - 主要气候指标（如全球平均温度）在长期内保持合理水平。

- **气候指标与 CMIP6 AMIP 对比**：
  - 指标示例：
    - 年平均全球 2m 温度（1981–2014、1980–2019 等时段）；
    - 850 hPa 温度 RMS 偏差（RMSB）；
    - 热带（20°S–20°N）温度趋势垂直剖面；
  - 结果要点：
    - NeuralGCM-2.8° 在 22 组 AMIP-like 模拟上的 850 hPa 温度 RMSB 与 22 个 CMIP6 AMIP 模式处于同一量级（箱线图）；
    - 若对 CMIP6 AMIP 做简单 bias correction（去除全球均温偏差），两者 RMSB 分布更为接近；
    - 热带温度趋势垂直剖面方面，NeuralGCM 与 ERA5/RAOBCORE/CMIP6 一致性较好（详细数值在图中给出）。

- **涌现现象（emergent phenomena）**：
  - 在 1.4° 分辨率的气候模拟中：
    - 能产生具有真实频率与轨迹分布的热带气旋；
    - 降水–蒸发差值（P−E）在中高纬的空间模式与 ERA5 接近；
    - 在热带，极端 P−E 事件被低估，但总体分布仍较合理。

- **水收支与可诊断性**：
  - 与 GraphCast 等 ML 模型直接预测降水不同，NeuralGCM 仍显式区分：
    - 动力核主导的平流/大尺度过程；
    - 学习物理模块产生的局地源汇（如凝结、辐射加热等）。
  - 因此可明确诊断水收支（例如 P−E），提高可解释性。

- **外推性与限制**：
  - 论文指出：
    - NeuralGCM 可在历史/相近气候条件下进行数十年稳定模拟；
    - 但**无法可靠外推到“显著不同”的未来气候状态**（例如极端强增温情景），这是当前版本的重要限制。

### 7. 泛化能力（时间外测试）

- 与 GraphCast 的对比实验：
  - 将两者都训练至 2017 年，评估 2018–2022 多年上的预报性能；
  - GraphCast 随“离训练期越远”性能出现下降趋势；
  - NeuralGCM 未表现出明显随时间退化的趋势，长年的测试性能更平稳；
  - 还训练了仅使用 2000 年前数据的 NeuralGCM-2.8°，在 21 年未见数据上评估，仍保持较好技能（细节见补充图）。

- 解释：
  - 由于学习物理模块在空间上为**局地垂直柱网络**，且动力核保持物理方程约束，NeuralGCM 在统计分布变化（例如气候变率）上的泛化能力优于 purely data-driven、全局大卷积/Transformer 模型。

### 8. 总结与局限

- **优势**：
  - 在 1–10 天 deterministic 预报上与 GraphCast/Pangu 等顶级 ML 模型相当；
  - 在 1–15 天 ensemble 预报上整体优于 ECMWF-ENS（CRPS、RMSB、RMSE 等）；
  - 能进行多年–数十年气候模拟，涌现出合理的热带气旋统计与温度趋势，并与 CMIP6 AMIP 水平接近；
  - 计算成本相对传统 GCM 有数量级级的下降；
  - 保持良好的物理一致性（地转风、谱特性、水收支等）。
- **局限**：
  - 目前版本仍依赖 ERA5/当前气候统计，对远离当前气候状态的外推能力有限；
  - 具体随机模型（NeuralGCM-ENS）的噪声注入机制与损失形式细节在 Methods，当前摘录未给出；
  - 分辨率仍明显低于云分辨模式，对极端局地事件（强对流降水等）的模拟能力有限。

---

## 维度二：组件与基础结构知识（含高保真伪代码）

> 下述伪代码均为**基于论文描述的高保真伪代码**，用于帮助实现 NeuralGCM 或类似混合 GCM；若某实现细节仅在补充材料中给出而当前摘录未包含，则在伪代码中保持抽象或标明“未在文中明确给出”。

### 1. 状态与输入输出表示

#### 1.1 模型状态向量

- 设：
  - 空间网格：水平 $(i,j)$，分辨率 $\Delta \lambda, \Delta \phi$ 对应 2.8°/1.4°/0.7°；
  - 垂直层：压力层或 sigma 层索引 $k$；
  - 物理变量集合：例如温度 $T$、水平风 $u,v$、比湿 $q$、位势高度等（精确列表见 Methods）。
- 将状态记为：
  $$
  x_t = \text{flatten\_state}(T_t, u_t, v_t, q_t, \ldots) \in \mathbb R^{N_x}
  $$
  或编程上保留多维数组结构 `[nlev, ny, nx, nch]`。

```python
# 伪代码：GCM 状态封装
class GCMState:
    # shape: [nlev, ny, nx, nvar]
    fields: Tensor

    def copy(self):
        return GCMState(fields=self.fields.copy())
```

#### 1.2 外部强迫与随机噪声

- 外部强迫 $F_t$：
  - 包括顶层辐射、地表通量、SST/海冰、地形等；
  - 在气候模拟中，SST/海冰由历史观测或情景给定。
- 随机噪声 $z_t$（用于 NeuralGCM-ENS）：
  - 代表次网格随机性或初值/参数不确定性；
  - 可在每个时间步、每个柱或每个网格点上采样高斯噪声。

```python
class Forcing:
    # 包含各种时变/时不变强迫，具体字段略
    fields: Tensor  # [nlev, ny, nx, nf]

class NoiseField:
    noise: Tensor  # [nlev, ny, nx, nz_dim]
```

### 2. 动力核与学习物理模块

#### 2.1 可微分动力核（dynamical core）

- 任务：在给定 $x_t$ 与强迫 $F_t$ 下，计算大尺度动力学趋势：
  $$
  \dot x_\text{dyn} = f_\text{dyn}(x_t, F_t)
  $$
  - 包含：平流、重力波、科氏力等；
  - 采用经典数值格式实现（有限差分/谱方法等），但在框架中保持可微分。

```python
# 伪代码：动力核接口（细节为占位，不展开数值格式）

def dynamical_core_tendency(x: GCMState, forcing: Forcing) -> GCMState:
    """计算解析方程的 resolved 动力学趋势 f_dyn(x, F)。"""
    # 下面所有函数代表数值算子，具体实现依赖 GCM 格式
    advec = compute_advection(x)
    pres  = compute_pressure_gradient(x)
    cori  = compute_coriolis(x)
    grav  = compute_gravity_waves(x)
    # ... 其他 resolved 过程

    tendency = advec + pres + cori + grav
    return tendency
```

#### 2.2 学习物理模块（column-wise NN）

- 对每个**垂直柱**（固定 $(i,j)$，沿 $k$）独立应用神经网络，生成物理趋势：
  $$
  \dot x_\text{phys}(i,j,:) = \mathcal N_\theta\big(x_t(i,j,:), F_t(i,j,:), z_t(i,j,:)\big)
  $$
  - 物理内容包括：云参数化、凝结/蒸发、辐射加热、次网格湍流等；
  - NN 通常为深度前馈网络或轻量 RNN/Transformer，具体结构详见 Supplementary Fig. 1（当前摘录未给出细节）。

```python
# 伪代码：单柱神经网络物理参数化

def physics_nn_column(col_state: Tensor,
                      col_forcing: Tensor,
                      col_noise: Optional[Tensor],
                      params) -> Tensor:
    """对单个垂直柱 (levels, nvar+nf+noise) 输出物理趋势 (levels, nvar)。"""
    # 拼接状态、强迫与噪声
    x = concat([col_state, col_forcing] + ([col_noise] if col_noise is not None else []), axis=-1)
    # 多层 MLP 或其他结构
    h = x
    for layer in params.layers:
        h = swish(linear(h, layer.W, layer.b))
    # 输出物理 tendency
    tendency = linear(h, params.out_W, params.out_b)
    return tendency


def physics_module(x: GCMState, forcing: Forcing,
                   noise: Optional[NoiseField], params) -> GCMState:
    fields = x.fields
    forc   = forcing.fields
    nz     = noise.noise if noise is not None else None

    tend_phys = zeros_like(fields)
    # 遍历水平网格柱
    for j in range(fields.shape[1]):  # ny
        for i in range(fields.shape[2]):  # nx
            col_state   = fields[:, j, i, :]
            col_forcing = forc[:, j, i, :]
            col_noise   = nz[:, j, i, :] if nz is not None else None

            col_tend = physics_nn_column(col_state, col_forcing, col_noise, params)
            tend_phys[:, j, i, :] = col_tend

    return GCMState(fields=tend_phys)
```

### 3. 隐式–显式 ODE 时间积分（IMEX）

- 论文提到使用隐式–显式（implicit–explicit）求解器，将刚性项与非刚性项拆分；
- 精确公式与系数在 Methods/补充材料中给出，此处给出抽象伪代码：

```python
# 伪代码：单时间步 IMEX 更新（高保真但抽象）

def step_imex(x_t: GCMState,
              forcing_t: Forcing,
              noise_t: Optional[NoiseField],
              dt: float,
              params_dyn,
              params_phys) -> GCMState:
    # 显式部分：例如非刚性动力学与物理参数化
    k_dyn_exp  = dynamical_core_tendency(x_t, forcing_t)    # 视情况拆分刚性/非刚性
    k_phys     = physics_module(x_t, forcing_t, noise_t, params_phys)

    # 隐式部分：例如重力波、垂直扩散等刚性项（此处抽象表示）
    # 在实际实现中，需解线性/非线性方程: x_{t+1} = x_t + dt * f_implicit(x_{t+1})
    # 这里使用占位 solve_implicit 表达：

    x_pred = x_t.copy()
    x_pred.fields = x_pred.fields + dt * (k_dyn_exp.fields + k_phys.fields)

    x_tp1 = solve_implicit_part(x_pred, forcing_t, dt, params_dyn)
    # solve_implicit_part 的具体线性代数/迭代方法未在摘录中给出

    return x_tp1
```

### 4. NeuralGCM 前向与集合生成

#### 4.1 单步前向与 rollout

```python
# 伪代码：NeuralGCM 单步 deterministic 前向

class NeuralGCM:
    def __init__(self, params_dyn, params_phys, dt: float):
        self.params_dyn = params_dyn
        self.params_phys = params_phys
        self.dt = dt

    def one_step(self, x_t: GCMState,
                 forcing_t: Forcing,
                 noise_t: Optional[NoiseField] = None) -> GCMState:
        return step_imex(x_t, forcing_t, noise_t,
                         self.dt, self.params_dyn, self.params_phys)

    def rollout(self, x0: GCMState,
                forcing_seq: List[Forcing],
                noise_seq: Optional[List[NoiseField]] = None) -> List[GCMState]:
        T = len(forcing_seq) - 1
        states = [x0]
        x = x0
        for t in range(T):
            nt = noise_seq[t] if noise_seq is not None else None
            x = self.one_step(x, forcing_seq[t], nt)
            states.append(x)
        return states
```

#### 4.2 集合预报（NeuralGCM-ENS）

```python
# 伪代码：集合生成与 CRPS/SSR 评估（框架级）

def generate_ensemble(model: NeuralGCM,
                      x0: GCMState,
                      forcing_seq: List[Forcing],
                      n_members: int) -> List[List[GCMState]]:
    ensembles = []
    for m in range(n_members):
        # 对每个成员采样独立噪声序列
        noise_seq = [sample_noise_field(x0) for _ in range(len(forcing_seq) - 1)]
        traj = model.rollout(x0, forcing_seq, noise_seq)
        ensembles.append(traj)
    return ensembles


def compute_crps(ensembles, truth_seq, variable, level):
    """按 WeatherBench/ENS 协议计算 CRPS（具体积分略）。"""
    # 1. 对每个网格点/时间，收集 ensemble 成员的该变量–层值
    # 2. 构造经验 CDF，与真值的 Heaviside CDF 做积分差
    # 3. 空间与时间平均得到 CRPS
    pass


def compute_spread_skill_ratio(ensembles, truth_seq, variable, level):
    # 1. spread: ensemble 成员对 ensemble mean 的均方差
    # 2. skill: ensemble mean 与真值的均方误差
    # 3. SSR = sqrt(spread) / sqrt(skill)
    pass
```

### 5. 端到端训练循环（多步 BPTT）

> 注意：以下伪代码仅展示 deterministic NeuralGCM 的训练框架，随机版本在损失函数与噪声处理上有所不同，具体见原文 Methods；损失的精确形式（例如是否使用 CRPS/对数似然）在当前摘录中未完全给出。

```python
# 伪代码：多步 rollout 端到端训练（在线训练）

def train_step(batch_trajectories,  # 若干 ERA5 轨迹样本
               model: NeuralGCM,
               optimizer: Optimizer,
               rollout_steps: int):  # 当前 curriculum 对应的步数

    def loss_fn(params_dyn, params_phys):
        total_loss = 0.0
        n_total = 0

        for sample in batch_trajectories:
            # sample 提供: x_init, forcing_seq, truth_seq
            x0 = sample.x_init  # 对应 ERA5 某时刻
            forcing_seq = sample.forcing_seq  # 长度 >= rollout_steps+1
            truth_seq = sample.truth_seq      # 对应目标 ERA5 序列

            x = x0
            for t in range(rollout_steps):
                # 无噪声：deterministic 版本
                x = step_imex(x, forcing_seq[t], None,
                              model.dt, params_dyn, params_phys)

                x_true = truth_seq[t + 1]
                # 计算多变量、多层、多格点的加权 MSE / 其他误差
                total_loss += weighted_mse(x.fields, x_true.fields)
                n_total += 1

        return total_loss / max(1, n_total)

    grads = grad(loss_fn)(model.params_dyn, model.params_phys)
    updates, new_opt_state = optimizer.update(grads, model.opt_state)
    model.params_dyn, model.params_phys = apply_updates(
        (model.params_dyn, model.params_phys), updates)
    model.opt_state = new_opt_state
```

- **训练调度**：
  - 在实际实现中，`rollout_steps` 由 1 逐渐增至对应 5 天（例如 20 步），学习率 schedule 与正则化策略详见 Supplementary Table 4 / G 节；
  - 在早期 rollout 较短时，模型可以先学习稳定的一步动力学与物理参数化；随着 rollout 增长，梯度在更长时间范围内对不稳定性与偏差进行惩罚。

### 6. 气候模拟与 AMIP 运行伪代码

```python
# 伪代码：AMIP 风格气候模拟

def run_amip_simulation(model: NeuralGCM,
                        init_state: GCMState,
                        sst_ice_forcings: List[Forcing],
                        n_years: int) -> List[GCMState]:
    """给定历史 SST/海冰强迫，模拟大气多年演化。
    - sst_ice_forcings: 按固定 dt 时间步的外强迫序列
    - 假设 dt 与强迫时间分辨率一致
    """
    x = init_state
    history = []
    for t, F_t in enumerate(sst_ice_forcings):
        x = model.one_step(x, F_t, noise_t=None)  # 气候模拟可选 deterministic
        history.append(x)

        # 可周期输出年度平均等诊断量
    return history


def compute_climate_metrics(history, metrics_spec):
    # 如全球年平均 2m 温度、850 hPa 温度偏差、热带温度趋势、P-E 统计等
    pass
```

---

以上为 NeuralGCM 论文在统一模版下的结构化知识与高保真伪代码抽取；如需，我可以基于 knowledge 目录中 FourCastNet/FuXi/GraphCast/GenCast/NeuralGCM 等进一步整理“deterministic vs ensemble vs climate-capable” 的对比矩阵，或抽象出统一的 API 设计与实现草图。