# 维度一：整体模型知识（Model-Level Knowledge）

## 1. 核心任务与需求

### 1.1 核心建模任务
- 构建一个 **物理约束的连续时间神经 PDE 模型 ClimODE**，用于：
  - 全球与区域尺度的中短时天气预报（6–36 小时 lead time）；
  - 月平均气候态预测。
- 关键思想：
  - 将天气视作 **守恒的平流（advection）系统**，由连续时间偏微分方程控制；
  - 通过 **神经 ODE** 形式学习流速场 \(\mathbf{v}(\mathbf{x}, t)\)，在保证守恒的前提下演化多变量天气场；
  - 增加 **高斯发射（emission）头** 建模源项/耗散及不确定性，实现概率预报。

### 1.2 解决了什么问题
- 传统深度学习天气模型的缺陷：
  - 多为 **离散时间自回归“跳步”**，破坏连续时间守恒（质量守恒）特性，长时段会累积偏差；
  - 多为完全数据驱动黑箱，缺乏物理结构（如平流方程）的显式约束；
  - 通常不输出/不建模预测不确定性；
  - 模型规模巨大（如 Pangu 256M、ClimaX 107M 等），训练与部署成本高。
- ClimODE 试图解决：
  - 通过 **基于连续时间平流 PDE 的神经 ODE**，在结构层面保证数值守恒和更稳定的长时演化；
  - 将物理知识（连续性方程、平流、压缩项）显式融入神经网络结构；
  - 通过高斯 emission 头引入 **不确定性估计** 与源项（开闭系统扩展）；
  - 在 **仅 2.8M 参数** 下达到或超过现有数据驱动方法的全球与区域预报性能。


## 2. 解决方案

### 2.1 怎么解决的（核心思路）
- 从统计力学/连续性方程出发：
  - 一般形式的连续性方程：
    $$
    \underbrace{\frac{\mathrm{d}u}{\mathrm{d}t}}_{\text{time evolution } \dot u}
    + \underbrace{\overbrace{\mathbf{v}\cdot\nabla u}^{\text{transport}} + \overbrace{u\,\nabla\cdot\mathbf{v}}^{\text{compression}}}_{\text{advection}}
    = \underbrace{s}_{\text{sources}}.
    $$
  - 对于每个气象量 \(u_k(\mathbf{x}, t)\)，在无源闭合系统中采用：
    $$
    \dot u_k(\mathbf{x}, t)
    = -\underbrace{\mathbf{v}_k(\mathbf{x}, t)\cdot\nabla u_k(\mathbf{x}, t)}_{\text{transport}}
      -\underbrace{u_k(\mathbf{x}, t)\,\nabla\cdot\mathbf{v}_k(\mathbf{x}, t)}_{\text{compression}}.
    $$
- 用神经网络 **显式地学习流速场的时间导数**：
  - 采用二阶结构：
    $$
    \dot{\mathbf{v}}_k(\mathbf{x}, t)
    = f_\theta\bigl(\mathbf{u}(t), \nabla\mathbf{u}(t), \mathbf{v}(t), \psi\bigr),
    $$
    其中 \(f_\theta\) 是结合卷积与注意力的混合网络；\(\psi\) 是时空嵌入。
- 将 PDE 通过 **方法行（method of lines）+ 二阶变一阶** 转化为一大组耦合的一阶 ODE：
  - 空间离散成网格 \(H\times W\)，对每个网格点和每个量 \(k\) 建立 ODE；
  - 联立 \(\mathbf{u}(t)\) 和 \(\mathbf{v}(t)\) 成系统，用常规数值 ODE 求解（如 Runge–Kutta）。
- 加入 **高斯 emission 头**，补充源项和不确定性：
  - 建模观测/输出为：
    $$
    u_k^{\text{obs}}(\mathbf{x}, t)
      \sim \mathcal{N}\bigl(u_k(\mathbf{x}, t)+\mu_k(\mathbf{x}, t),\, \sigma_k^2(\mathbf{x}, t)\bigr),
    $$
    其中 \(\mu_k\) 为系统外的增减（如昼夜温度变化等源项），\(\sigma_k^2\) 为预测不确定性。
- 通过 **负对数似然 + 先验正则** 优化参数，实现联合拟合均值轨迹与不确定性。

### 2.2 结构映射（思路 → 模块）
- 守恒平流建模 → `神经平流 PDE/ODE 系统`：
  - 方程 (2)、(5) 对应的 ODE 集成模块；
  - 数值求解器（Runge–Kutta 等）实现连续时间演化。
- 流速场学习 → `流速网络 f_θ`：
  - 卷积网络 \(f_{\text{conv}}\)：建模局地传输；
  - 注意力卷积网络 \(f_{\text{att}}\)：全局长程依赖；
  - 二者线性组合 (6)，\(f_\theta = f_{\text{conv}} + \gamma f_{\text{att}}\)。
- 时空结构/季节性 → `时空嵌入 ψ(x,t)`：
  - 正弦/余弦时间编码（每日周期、年周期）；
  - 位置编码（纬度/经度正余弦 + 球面坐标）；
  - 以及静态场（陆海掩膜 lsm、地形 oro）。
- 系统闭合与初值 → `初始速度场推断模块`：
  - 利用连续性方程约束求解 \(\mathbf{v}_k(t_0)\) 的反问题（(10)）；
  - 使用带 RBF 核的高斯先验保证空域平滑。
- 源项与不确定性 → `高斯发射网络 g_k` + `损失函数`：
  - 输出 \(\mu_k, \sigma_k\)；
  - 最大似然 + \(\sigma\) 先验约束（(11)、(12)）。


## 3. 模型架构概览

- 整体框架：
  - **状态变量**： \(\mathbf{u}(\mathbf{x}, t) \in \mathbb{R}^{K}\)，\(K=5\) 个量（t2m, t, z, u10, v10）。
  - **流速变量**： \(\mathbf{v}_k(\mathbf{x}, t) \in \mathbb{R}^2\)，对每个量 \(k\) 有二维平面速度（纬向 + 经向）。
  - **时空特征**： \(\psi(\mathbf{x}, t)\)（时间周期 + 位置 + 静态场）。
  - **动力学方程**：
    $$
    \dot u_k = -\mathbf{v}_k\cdot\nabla u_k - u_k\,\nabla\cdot\mathbf{v}_k,\\
    \dot{\mathbf{v}}_k = f_\theta(\mathbf{u}, \nabla\mathbf{u}, \mathbf{v}, \psi).
    $$
  - **数值求解**：
    - 将上述 PDE 在空间上离散，对 \(t\) 上用 Runge–Kutta 等 ODE 求解器积分得到未来时间的 \(\mathbf{u}(t), \mathbf{v}(t)\)。
  - **观测/输出层**：
    - 给定 \(\mathbf{u}(t)\)，通过发射网络 \(g_k\) 输出 \(\mu_k, \sigma_k\)，形成观测分布：\(\mathcal{N}(u_k+\mu_k,\sigma_k^2)\)。

- 流速网络结构：
  - **局部卷积分支** \(f_{\text{conv}}\)：
    - ResNet 结构，3×3 卷积层堆叠，深度 \(L\) 对应可感受的邻域大小；
    - 输入通道：\([\mathbf{u}, \nabla \mathbf{u}, \mathbf{v}, \psi]\)。
  - **注意力分支** \(f_{\text{att}}\)：
    - CNN 产生 K/Q/V，利用点积注意力实现全球信息聚合；
    - 输出与卷积分支同型，二者加权求和（权重 \(\gamma\) 可学习）。

- 时空嵌入结构：
  - 时间编码：
    $$
    \psi(t) = \bigl\{\sin 2\pi t,\, \cos 2\pi t,\, \sin\tfrac{2\pi t}{365},\, \cos\tfrac{2\pi t}{365}\bigr\}.
    $$
  - 位置编码（纬度 \(h\)、经度 \(w\)）：
    $$
    \psi(\mathbf{x}) = [\{\sin, \cos\}\times\{h, w\},\, \sin h\cos w,\, \sin h\sin w].
    $$
  - 联合编码：
    $$
    \psi(\mathbf{x}, t) = [\psi(t),\, \psi(\mathbf{x}),\, \psi(t)\times\psi(\mathbf{x}),\, \psi(c)],
    $$
    其中 \(\psi(c) = [\psi(h), \psi(w), \mathrm{lsm}, \mathrm{oro}]\)。

- 参数规模与对比（来自表 1）：
  - ClimODE：2.8M 参数，value-preserving ✓，显式周期/季节性 ✓，不确定性 ✓，连续时间 ✓；
  - 相比：
    - GraphCast：37M；ClimaX：107M；Pangu-Weather：256M；FourCastNet：未列具体数值；
    - 多数为非守恒、非连续时间、无不确定性估计。


## 4. 创新与未来

### 4.1 创新点
- **物理启发的连续时间神经平流 PDE 模型**：
  - 直接从连续性方程推导，保证 **值守恒**（闭合系统时满足 \(\int u_k(\mathbf{x}, t)\,\mathrm{d}\mathbf{x} = \mathrm{const}\)），避免长时预报崩塌；
  - 采用二阶流速（学习 \(\dot{\mathbf{v}}\)），增强神经 ODE 表达能力和稳定性。

- **局部卷积 + 全局注意力** 的混合流速网络：
  - 在同一 \(f_\theta\) 中融合局地平流/扩散效应与远距关联（如跨洋 teleconnection）；
  - 用 CNN 实现 K/Q/V，适配规则网格。

- **显式日周期/年周期 + 球面位置编码**：
  - 通过正弦/余弦时间与球面坐标，将季节性和昼夜周期编码入动力学方程中；
  - 同时输入 lsm/orography 等静态场，提高地表相关变量（t2m、u10、v10）的可预测性。

- **初始速度的物理解算 + 高斯先验**：
  - 使用连续性方程约束，将速度估计转化为一个带 RBF 先验的惩罚最小二乘问题，而非引入额外编码器；
  - 保证速度场的空间平滑性与物理合理性。

- **联合建模源项与不确定性**：
  - 通过高斯 emission 头，同时表示：
    - 源项/耗散（均值偏差 \(\mu_k\)）；
    - Aleatoric + epistemic 综合方差 \(\sigma_k^2\)；
  - 提供 **预测不确定性估计** 和 CRPS 评估能力。

- **高效训练与 SOTA 结果**：
  - 仅 2.8M 参数，可在单块 GPU 上从头训练；
  - 鉴于只有 5 个变量、较低分辨率（5.625°），在全球/区域 RMSE、ACC、CRPS 等指标上超过 ClimaX、FourCastNet 等神经基线，在多数指标上接近或略逊于 IFS。

### 4.2 后续研究方向（文中隐含或可见的局限）
- 模型仍落后于 IFS：
  - 在某些全量物理变量上，IFS 仍是“金标准”；
  - ClimODE 当前仅使用 5 个变量，未充分利用 ERA5 更丰富变量集。
- 分辨率与变量扩展：
  - 目前基于 WeatherBench 5.625° 数据，未来可探索更高分辨率与更多变量下的可扩展性；
- 更复杂的源项建模：
  - 当前 emission 模型只提供高斯偏差与方差，未来可探索更复杂的物理源项（如辐射、相变等）建模方式。


## 5. 实现细节与代码逻辑

### 5.1 核心方程与损失公式

1. 一般连续性/平流方程（有源）
\[
\dot u + \mathbf{v}\cdot\nabla u + u\,\nabla\cdot\mathbf{v} = s.
\]

2. 对每个量 \(u_k\) 的闭合系统平流方程（无源情形，方程 (2)）
\[
\dot u_k(\mathbf{x}, t)
= -\mathbf{v}_k(\mathbf{x}, t)\cdot\nabla u_k(\mathbf{x}, t)
  - u_k(\mathbf{x}, t)\,\nabla\cdot\mathbf{v}_k(\mathbf{x}, t).
\]

3. 值守恒约束（方程 (3)）：
\[
\int u_k(\mathbf{x}, t)\,\mathrm{d}\mathbf{x} = \text{const},\quad\forall t, k.
\]

4. 流速二阶方程（方程 (4)）：
\[
\dot{\mathbf{v}}_k(\mathbf{x}, t)
= f_\theta\bigl(\mathbf{u}(t), \nabla\mathbf{u}(t), \mathbf{v}(t), \psi\bigr).
\]

5. ODE 系统形式（方程 (5)）：
\[
\begin{bmatrix}
\mathbf{u}(t)\\
\mathbf{v}(t)
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{u}(t_0)\\
\mathbf{v}(t_0)
\end{bmatrix}
+
\int_{t_0}^t
\begin{bmatrix}
\{-\nabla\cdot(u_k(\tau)\,\mathbf{v}_k(\tau))\}_k\\[3pt]
\{f_\theta(\mathbf{u}(\tau), \nabla\mathbf{u}(\tau), \mathbf{v}(\tau), \psi)_k\}_k
\end{bmatrix}
\mathrm{d}\tau.
\]

6. 局部卷积 + 注意力的流速网络（方程 (6)）：
\[
 f_\theta(\mathbf{u}, \nabla\mathbf{u}, \mathbf{v}, \psi)
 = f_{\text{conv}}(\cdot) + \gamma f_{\text{att}}(\cdot).
\]

7. 时间与空间嵌入（方程 (7)–(9））：
\[
\psi(t) = \{\sin 2\pi t, \cos 2\pi t, \sin(2\pi t/365), \cos(2\pi t/365)\},
\]
\[
\psi(\mathbf{x}) = [\{\sin, \cos\}\times\{h, w\}, \sin h\cos w, \sin h\sin w],
\]
\[
\psi(\mathbf{x}, t) = [\psi(t), \psi(\mathbf{x}), \psi(t)\times\psi(\mathbf{x}), \psi(c)],
\]
\[
\psi(c) = [\psi(h), \psi(w), \mathrm{lsm}, \mathrm{oro}].
\]

8. 初始速度推断（方程 (10)）：
\[
\hat{\mathbf{v}}_k(t)
= \arg\min_{\mathbf{v}_k(t)}
\Bigl\|\tilde u_k(t)
+ \mathbf{v}_k(t)\cdot\tilde\nabla u_k(t)
+ u_k(t)\,\tilde\nabla\cdot\mathbf{v}_k(\mathbf{x}, t)\Bigr\|_2^2
+ \alpha\,\bigl\|\mathbf{v}_k(t)\bigr\|_{\mathbf{K}},
\]
其中 \(\tilde\nabla\) 和 \(\tilde u\) 为数值导数与时间差分近似，\(\|\mathbf{v}\|_{\mathbf{K}}\) 为带 RBF 核 \(\mathbf{K}\) 的高斯先验正则项。

9. 高斯发射/不确定性建模（方程 (11)）：
\[
 u_k^{\text{obs}}(\mathbf{x}, t)
 \sim \mathcal{N}\bigl(u_k(\mathbf{x}, t)+\mu_k(\mathbf{x}, t),\, \sigma_k^2(\mathbf{x}, t)\bigr),\\
 \mu_k(\mathbf{x}, t), \sigma_k(\mathbf{x}, t) = g_k(\mathbf{u}(\mathbf{x}, t), \psi).
\]

10. 训练损失（负对数似然 + 方差先验，方程 (12)）：
\[
\mathcal{L}(\theta; \mathcal{D})
= -\frac{1}{N K H W}\sum_{i=1}^N
\Bigl[
\log \mathcal{N}\bigl(\mathbf{y}_i\mid \mathbf{u}(t_i)+\boldsymbol{\mu}(t_i),\, \mathrm{diag}\,\boldsymbol{\sigma}^2(t_i)\bigr)
+ \log \mathcal{N}_+\bigl(\boldsymbol{\sigma}(t_i)\mid \mathbf{0}, \lambda_\sigma^2 I\bigr)
\Bigr],
\]
其中 \(\mathcal{N}_+\) 为仅支持正值的高斯先验（约束方差不爆炸），\(\lambda_\sigma\) 为超参数，在训练中通过余弦退火衰减其正则作用，最终近似最大似然。

11. 评估指标 RMSE 与 ACC（方程 (13)）：
\[
\mathrm{RMSE} = \frac{1}{N}\sum_t\sqrt{\frac{1}{HW}\sum_h\sum_w \alpha(h)\bigl(y_{thw}-u_{thw}\bigr)^2},
\]
\[
\mathrm{ACC} = \frac{\sum_{t,h,w}\alpha(h)\tilde y_{thw}\tilde u_{thw}}
{\sqrt{\sum_{t,h,w}\alpha(h)\tilde y_{thw}^2\,\sum_{t,h,w}\alpha(h)\tilde u_{thw}^2}},
\]
其中 \(\alpha(h) = \cos h / (\frac{1}{H}\sum_{h'}\cos h')\)，\(\tilde y = y - C\)、\(\tilde u = u - C\)、\(C = \frac{1}{N}\sum_t y_{thw}\)。

### 5.2 关键实现逻辑（文字）

1. PDE → ODE（方法行 + 二阶转一阶）：
- 将物理域 \(\Omega = [-90^\circ, 90^\circ]\times[-180^\circ, 180^\circ]\) 离散为 \(H\times W\) 网格；
- 对每个网格点/每个变量建立状态 \(u_{k, i}(t)\)、速度 \(\mathbf{v}_{k, i}(t)\) 的一阶 ODE；
- 数值力学中的 \(\nabla, \nabla\cdot\) 通过有限差分近似；
- 使用标准 ODE 求解器（如 Runge–Kutta）在时间上积分，兼容自动微分与 adjoint 方法。

2. 流速网络混合局部与全局效应：
- \(f_{\text{conv}}\)：堆叠 3×3 卷积的 ResNet，感受野随深度 \(L\) 增大，对应局部邻域的影响范围；
- \(f_{\text{att}}\)：通过 CNN 生成 K/Q/V，在全球网格上计算点积注意力，实现长程依赖（如跨大洋的遥相关）；
- 通过可学习标量 \(\gamma\) 平衡两者贡献。

3. 初始速度估计：
- 在时间 \(t_0\) 及其之前的观测/分析场 \(u(t \le t_0)\) 可用于近似时间导数 \(\tilde{\dot u}(t_0)\)；
- 联合空间导数 \(\tilde\nabla u\)、\(\tilde\nabla\cdot\mathbf{v}\) 构造残差项并做惩罚最小二乘，附加 RBF 核正则提升平滑性。

4. 不确定性估计与正则：
- emission 网络 \(g_k\) 从 \(\mathbf{u}(\mathbf{x}, t), \psi(\mathbf{x}, t)\) 输出均值偏差 \(\mu_k\) 与标准差 \(\sigma_k\)；
- 对数似然中包含对 \(\sigma_k\) 的先验，避免训练早期通过膨胀方差“逃避”拟合任务；
- 逐渐衰减先验影响，最终逼近最大似然估计。

5. 端到端训练：
- 输入：初始状态 \(\mathbf{u}(t_0)\)、根据 (10) 推断的 \(\mathbf{v}(t_0)\)、时空嵌入 \(\psi\)；
- 前向：ODE 求解器积分到各个观测时间 \(t_i\)，得到 \(\mathbf{u}(t_i), \mathbf{v}(t_i)\)；
- 输出：通过发射头 \(g_k\) 得到观测分布参数，并计算 NLL + \(\sigma\) 先验；
- 反向：自动微分对 ODE 解及网络参数求梯度，更新 \(f_\theta\) 与 \(g_k\) 等参数。

### 5.3 训练/推理伪代码（基于文中描述生成的伪代码）

```pseudo
# 基于文中描述生成的伪代码（非论文原文代码）

# ----- 核心模块 -----

function velocity_network(u, grad_u, v, psi):
    # u, grad_u, v, psi: [B, C, H, W]
    v_conv = f_conv(u, grad_u, v, psi)   # 局部卷积分支
    v_att  = f_att(u, grad_u, v, psi)    # 全局注意力分支
    return v_conv + gamma * v_att

function dyn_rhs(t, state, psi):
    # state = (u, v)
    u, v = state
    grad_u    = spatial_gradient(u)
    div_v     = spatial_divergence(v)
    # du/dt from continuity
    du_dt = - dot(v, grad_u) - u * div_v
    # dv/dt from neural network
    dv_dt = velocity_network(u, grad_u, v, psi)
    return (du_dt, dv_dt)

# ----- 初始速度推断 -----

function infer_initial_velocity(u_hist, psi, alpha, K_rbf):
    # u_hist: past states up to t0
    u_t0 = last(u_hist)
    du_dt_approx = temporal_derivative(u_hist)  # ~ tilde{u_dot}
    grad_u = spatial_gradient(u_t0)

    # solve penalized least squares for v0
    v0 = argmin_v || du_dt_approx
                   + v . grad_u
                   + u_t0 * div(v) ||_2^2
         + alpha * quadratic_form(v, K_rbf)
    return v0

# ----- 训练前向 -----

for batch in data_loader:
    # batch: (times t_i, observations y_i)
    u0 = y_at_time(t0)
    psi = build_spatiotemporal_embedding(grid, t0)
    v0  = infer_initial_velocity(past_y, psi, alpha, K_rbf)

    # ODE solve from t0 to all t_i
    state0 = (u0, v0)
    states = ode_solve(dyn_rhs, state0, times=t_i, args=(psi,))

    losses = []
    for i, t in enumerate(t_i):
        u_t, v_t = states[i]
        mu_t, sigma_t = emission_head(u_t, psi_at_time(t))
        loss_i = gaussian_nll(y_i, u_t + mu_t, sigma_t)
               + sigma_prior_logprob(sigma_t, lambda_sigma)
        losses.append(loss_i)

    loss = mean(losses)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ----- 推理：给定 u(t0)，预测未来 u(t0 + Δt) -----

function forecast(u0, t0, lead_times):
    psi0 = build_spatiotemporal_embedding(grid, t0)
    v0   = infer_initial_velocity(past_y, psi0, alpha, K_rbf)
    state0 = (u0, v0)
    states = ode_solve(dyn_rhs, state0, times=t0 + lead_times, args=(psi0,))

    forecasts = []
    for (u_t, v_t), t in zip(states, t0 + lead_times):
        mu_t, sigma_t = emission_head(u_t, psi_at_time(t))
        # 取均值作为点预测，也可采样获得不确定性
        u_mean = u_t + mu_t
        forecasts.append((u_mean, sigma_t))
    return forecasts
```


## 6. 数据规格

### 6.1 数据集名称
- ERA5 重分析数据（通过 WeatherBench 预处理版本）。

### 6.2 时间范围
- 数据集配置（全球 5.625° 分辨率，6 小时间隔）：
  - 训练：2006–2015 共 10 年；
  - 验证：2016 年；
  - 测试：2017–2018 年两年；
  - 任务：
    - 6–36 小时时效的全球、区域天气预报；
    - 月平均气候态预测（与 FourCastNet 比较）。

### 6.3 变量列表
- 选取 ERA5 中的 5 个关键变量（\(K = 5\)）：
  1. 2m 地面温度：t2m（ground temperature）；
  2. 大气温度：t（atmospheric temperature）；
  3. 位势高度：z（geopotential）；
  4. 10m 地面风：u10（zonal）、v10（meridional）。
- 说明：
  - z 与 t 为中期数值预报中标准的验证量；
  - t2m、u10、v10 直接关系人类活动（地面气温与近地面风）。

### 6.4 数据处理
- 分辨率与时间步长：
  - 使用 WeatherBench 提供的 **5.625°** 规则经纬度网格；
  - 时间步长 **6 小时**。
- 归一化：
  - 所有变量通过 min–max 归一化缩放到 \([0, 1]\) 区间；
- 损失与评估中的纬度加权：
  - 使用 \(\alpha(h) = \cos h / (\frac{1}{H}\sum_{h'}\cos h')\) 作纬度权重，用于 RMSE 和 ACC；
- 评估指标：
  - 反归一化后计算 **纬度加权 RMSE 和 ACC**；
  - 概率评估中还计算 CRPS，用于对比 ClimODE 与 FourCastNet 的月平均预测等。


# 维度二：基础组件知识（Component-Level Knowledge）

## 组件 1：神经平流 PDE / ODE 系统（Continuous-time Neural Advection PDE / ODE）

1. 子需求定位
- 对应子问题：
  - 如何在模型结构上显式体现大气演化的平流与守恒特性，并在连续时间上进行预测，而不是离散的自回归跳步；
- 技术难点：
  - 直接在球面上求解偏微分方程成本高；
  - 需要在保持物理结构前提下，将 PDE 转为可用 ODE 求解器和自动微分训练的形式。

2. 技术与创新
- 关键技术点：
  - 采用连续性方程 \(\dot u_k = -\mathbf{v}_k\cdot\nabla u_k - u_k\nabla\cdot\mathbf{v}_k\) 作为动力学核心；
  - 用方法行（空间离散）+ 二阶转一阶，将 PDE 转化为一大组耦合 ODE；
  - 用标准 ODE 求解器（如 Runge–Kutta）在时间上数值积分；
  - 综合使用神经 ODE 训练技术（自动微分/伴随法）进行端到端学习。
- 创新点：
  - 在全球天气预测上首次系统构建 **完整的神经连续时间平流 PDE**，并证明其可在小参数量下达 SOTA 水平。

3. 上下文与耦合度
- 上下文组件：
  - 前接：初始状态和初始速度估计；
  - 后接：高斯发射头（输出预测和不确定性）。
- 绑定程度：
  - 强绑定-气象特异：
    - 方程形式和守恒假设直接源自大气物理平流/连续性方程。

4. 实现描述与伪代码
- 见“维度一 5.3”中 `dyn_rhs` 与 `ode_solve` 的伪代码；
- 求解器可以是 Runge–Kutta 等常用 ODE 方法，适配自动微分。


## 组件 2：流速网络 f_θ（卷积 + 注意力混合）

1. 子需求定位
- 子问题：
  - 给定当前全场状态 \(\mathbf{u}(t)\)、梯度 \(\nabla\mathbf{u}(t)\)、速度 \(\mathbf{v}(t)\) 和时空嵌入 \(\psi\)，如何预测流速的时间导数 \(\dot{\mathbf{v}}\)，同时兼顾局地输运与远程关联？
- 技术难点：
  - PDE 自身局地（仅依赖同一点的状态与梯度），但大气中存在显著长程 teleconnection；
  - 需要在计算可承受的前提下处理全球相关性。

2. 技术与创新
- 关键技术点：
  - 局部卷积分支 \(f_{\text{conv}}\)：
    - ResNet + 3×3 卷积，聚合近邻信息，模拟局地输运与扩散；
  - 注意力分支 \(f_{\text{att}}\)：
    - 利用 CNN 生成 K/Q/V，在整个地球网格上通过 dot-product attention 建模远程关联；
  - 使用可学习权重 \(\gamma\) 进行线性组合，允许网络自动平衡局部/全局贡献。
- 创新点：
  - 在物理启发的神经 PDE 中使用混合 CNN+Attention 的流速表示，既保留 PDE 的局地结构，又捕捉跨区域影响。

3. 上下文与耦合度
- 上下文组件：
  - 仅出现在 \(\dot{\mathbf{v}}\) 的定义中；
  - 与时空嵌入、状态/梯度特征强耦合。
- 绑定程度：
  - 强绑定-气象特异：
    - 虽然 CNN+Attention 是通用技术，但此处的输入/输出结构和感受野设计是为全球大气场专门配置。

4. 伪代码
- 见“维度一 5.3”中 `velocity_network` 函数。


## 组件 3：时空嵌入 ψ(x,t)

1. 子需求定位
- 子问题：
  - 如何让动力学显式感知 **季节性（年周期）与日周期**，以及球面位置（纬度/经度）和静态下垫面信息？
- 技术难点：
  - 时间与空间均具周期性和球面几何结构，直接用笛卡尔坐标不利于网络学习；

2. 技术与创新
- 关键技术点：
  - 时间编码：以 \(\sin,\cos\) 形式注入每日和年度周期；
  - 位置编码：结合平面正弦/余弦与球面坐标 \((\sin h\cos w,\,\sin h\sin w)\)；
  - 联合编码：时间×空间，捕捉不同地点上季节/昼夜变化模式；
  - 静态场：添加 latitude/longitude 映射、陆海掩膜 lsm、地形 oro。
- 创新点：
  - 将 ViT 风格的位置编码与球面几何和气象静态场结合，构成专门面向大气的时空特征通道。

3. 上下文与耦合度
- 上下文组件：
  - 作为输入通道注入流速网络 \(f_\theta\) 与发射头 \(g_k\)；
- 绑定程度：
  - 强绑定-气象特异：
    - 编码形式基于地球自转/公转和球面几何。

4. 实现伪代码
- 见“维度一 5.3”中 `build_spatiotemporal_embedding`（概念性）。


## 组件 4：初始速度推断（Initial Velocity Inference）

1. 子需求定位
- 子问题：
  - 数值积分神经 ODE 需要初始状态 \(\mathbf{u}(t_0), \mathbf{v}(t_0)\)，其中速度场 \(\mathbf{v}(t_0)\) 通常不可直接观测，如何从时间序列数据中推断？
- 技术难点：
  - 一般神经 ODE 需要引入额外编码器来估计速度；
  - 希望借助连续性方程本身的物理约束避免额外复杂结构。

2. 技术与创新
- 关键技术点：
  - 利用连续性方程中的恒等式 \(\dot{u} + \nabla\cdot(u\mathbf{v})=0\)；
  - 将 \(\tilde{\dot u}\)、\(\tilde\nabla u\) 和 \(u\tilde\nabla\cdot\mathbf{v}\) 组成残差，对 \(\mathbf{v}\) 做惩罚最小二乘拟合；
  - 采用高斯 RBF 核先验 \(\mathcal{N}(\mathrm{vec}\,\mathbf{v}_k \mid 0, \mathbf{K})\)，使速度场在空间上平滑。
- 创新点：
  - 通过物理等式直接解反问题，避免单独训练复杂的“速度编码器”。

3. 上下文与耦合度
- 上下文组件：
  - 用于构造 ODE 初值；
- 绑定程度：
  - 强绑定-气象特异：
    - 依赖于连续性方程及大气场的空间相关性结构。

4. 伪代码
- 见“维度一 5.3”中 `infer_initial_velocity`。


## 组件 5：高斯发射网络 g_k（源项 + 不确定性）

1. 子需求定位
- 子问题：
  - 基于守恒平流 ODE 系统输出的是“闭合系统”状态 \(u_k(\mathbf{x}, t)\)，如何建模：
    - 非守恒源/汇（如昼夜辐射、物相变化导致的能量变化）；
    - 观测/模型不确定性？

2. 技术与创新
- 关键技术点：
  - 定义观测模型：
    $$
    u_k^{\text{obs}} \sim \mathcal{N}(u_k + \mu_k, \sigma_k^2),
    $$
    由发射网络 \(g_k(\mathbf{u}, \psi)\) 输出 \(\mu_k, \sigma_k\)；
  - 将 \(\sigma_k^2\) 解释为预测不确定性（总的 aleatoric + epistemic），并通过 NLL 最优化；
  - 用高斯先验正则 \(\mathcal{N}_+(\sigma_k)\) 控制方差尺度。
- 创新点：
  - 在物理守恒基础上，用独立发射头来补充非守恒源项与不确定性，分离“动力学演化”与“观测/源项”两部分建模。

3. 上下文与耦合度
- 上下文组件：
  - 前接：ODE 集成得到的 \(\mathbf{u}(t)\)；
  - 后接：损失函数与评估指标（CRPS 等）。
- 绑定程度：
  - 强绑定-气象特异：
    - 源项和不确定性的物理含义直接来自气候/天气系统。

4. 伪代码
- 已在“维度一 5.3”前向与推理流程中体现（`emission_head`）。


## 组件 6：损失函数与方差先验

1. 子需求定位
- 子问题：
  - 如何在训练时同时拟合均值和不确定性，而不让网络通过“膨胀方差”逃避拟合任务？

2. 技术与创新
- 关键技术点：
  - 使用高斯 NLL 作为主损失；
  - 对 \(\sigma\) 添加高斯先验 \(\mathcal{N}_+(\sigma\mid 0, \lambda_\sigma^2I)\)，限制其幅度；
  - 通过 **余弦退火** 衰减 \(\lambda_\sigma^{-1}\)，逐渐解除正则，使训练后期接近最大似然估计。

3. 上下文与耦合度
- 上下文组件：
  - 与发射网络和 ODE 模块共同决定训练目标；
- 绑定程度：
  - 弱绑定-通用（方法本身）+ 强绑定-气象（量纲与变量选择）。

4. 理论公式
- 见“维度一 5.1”中方程 (12)。


## 组件 7：评价模块（RMSE、ACC、CRPS 等）

1. 子需求定位
- 子问题：
  - 如何在全球和区域尺度上公平比较 ClimODE 与其它 DL 模型和 IFS？

2. 技术与创新
- 关键技术点：
  - 使用纬度加权 RMSE 与 ACC，遵循 WeatherBench (Rasp et al., 2020) 标准；
  - 在 CRPS 上对比 ClimODE 与 FourCastNet 的月尺度预测能力；
  - 区域评估：北美、南美、澳大利亚等区域上进行 RMSE 表，量化区域性能（表 2）。

3. 上下文与耦合度
- 上下文组件：
  - 作为评价与消融分析模块，不回传梯度；
- 绑定程度：
  - 强绑定-气象特异：
    - 所选变量、区域和加权方式为气象社区标准实践。

4. 实现描述
- 公式见“维度一 5.1”的 RMSE/ACC 定义；
- CRPS 实现未在片段内展开，依赖于标准集合预报评分公式。
