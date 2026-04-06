# 维度一：整体模型知识（Model-Level Knowledge）

## 1. 核心任务与需求

### 1.1 核心建模任务
- 任务类型：
  - 针对 **热带气旋（TC）强度** 的中期预报（最长 5 天），构建一个生成式的、轻量级的改进模块。
- 目标：
  - 结合 **FuXi/FuXi‑2.0 深度学习全球中期预报** 的优良路径预测能力，与 **高分辨率 WRF 物理模式** 的强度与结构模拟能力；
  - 在保持 FuXi 路径技巧的前提下，显著提高 TC 强度（10 m 最大持续风速等）预报精度，并重建更真实的台风结构；
  - 同时将计算成本降至远低于高分辨率 WRF 的量级，实现秒级实时业务应用、支持大规模集合预报。

### 1.2 主要问题与痛点
- 传统 NWP：
  - HWRF、HAFS、CMA‑TYM 等高分辨率 TC 业务模式需要长时间积分（小时级），计算成本极高；
  - 初始场不确定性、物理参数化与分辨率限制使强对流、台风强度等中尺度过程仍有明显偏差；
  - 即使继续提高分辨率，其可预报性收益已接近理论上限，边际效益下降。
- 现有 AI 全球中期模式（FuXi、Pangu‑Weather、GraphCast 等）：
  - 在 **路径预报** 上已显著优于 NWP，但在 **强度** 上往往低估极端事件：
    - 训练多基于 ERA5，ERA5 本身就系统性低估 TC 强度；
    - 模型为优化全球均方误差与整体技巧，极值被“平滑化”。
- 现有 Pangu+WRF/GraphCast+GEM 等 **AI+物理下垫缩放框架**：
  - 通过高分辨率 WRF/GEM 改善强度与结构，但仍需大量数值积分，时效和算力开销难以满足实时大规模业务。

### 1.3 FuXi‑TC 的目标
- 利用 **扩散概率生成模型（DDPM）**：
  - 从 FuXi/FuXi‑2.0 生成的台风相关多变量场出发，学习映射到 WRF 生成的高质量强度/结构场；
  - 作为 FuXi 的“后处理/精细化”模块，专注局地 TC 区域的风场与结构修正；
  - 保持 FuXi 的大尺度与路径信息不变，仅对局地强度和细节进行生成式重构。
- 效果：
  - 在 2024 年 4–10 月的 21 个西北太平洋台风个例上，5 日强度预报 RMSE 明显低于 FuXi 和 ERA5，与 ECMWF HRES 可比；
  - 单次预报在 A100 上耗时约 2 秒，比 32 核 CPU 上 83 分钟的 WRF 快 3 个数量级以上。


## 2. 解决方案与结构概要

### 2.1 整体思路
1. **构建高质量“教师”数据集**：
   - 使用 FuXi‑2.0（AI 中期预报）输出 + ERA5 土壤与 SST 为背景场，驱动 WRF 区域高分辨率模拟：
     - 通过谱向倾向（spectral nudging）约束 WRF 的大尺度场与 FuXi 保持一致；
     - 同时让 WRF 自由发展出更精细的中小尺度 TC 结构与强度；
   - 得到在 **FuXi 统计特征空间中**、但具有更真实 TC 强度和结构的“教师”场，用作 FuXi‑TC 的训练标签。

2. **构建 FuXi‑TC 扩散生成模型**：
   - 输入：FuXi/FuXi‑2.0 对台风相关区域的多层多变量预报场（地面 + 若干标准压层）及时间步；
   - 目标输出：对应时刻 WRF 预报的高质量风场、温度场等（当前文章重点为 10 m 风、近地强度结构）；
   - 模型形式：条件 DDPM，学习在 FuXi 预报条件下 WRF 目标场的条件分布 \(p(y\mid x)\)。

3. **作为轻量级后处理模块部署**：
   - 仅对 FuXi 已预测的 TC 区域做区域子域推理，无需重新训练或修改 FuXi 全局模式；
   - 一次 DDPM 推理仅需秒级时间，可用于业务实时更新与集合生成。

### 2.2 概念性架构
- 上下游关系：
  - 上游：
    - FuXi‑2.0：提供全球预报背景场与边界条件；
    - WRF：以 FuXi 场 + spectral nudging 生成“教师”高质量 TC 强度数据集；
    - CMA Best Track：评价路径与强度技巧、选取样本。
  - FuXi‑TC：
    - 训练阶段：\(x =\) FuXi 相关变量场，\(y =\) 对应时刻 WRF 输出 → 训练条件扩散模型；
    - 推理阶段：对给定 FuXi 场做快速条件生成，得到 FuXi‑TC 修正后的 TC 强度与结构场。

- 模块划分（从功能角度）：
  1. 数据准备模块：提取 TC 个例轨迹、裁剪区域子域、配对 FuXi–WRF 场；
  2. 条件 DDPM 网络：U‑Net/Transformer‑like 神经网络（论文片段未给具体结构，细节记为“未明确说明”）；
  3. 评估与可视化模块：对比 FuXi, ERA5, WRF, HRES 和 FuXi‑TC 的 RMSE、谱、空间结构。


## 3. 关键技术点与优势

### 3.1 生成式改进而非重新训练全球模式
- FuXi‑TC 设计为 **后置 refinement 模块**：
  - 不需要重训 FuXi/GraphCast/Pangu 等全球大模型；
  - 不改变 FuXi 的路径和大尺度结构，仅局地增强强度和结构；
  - 可以平滑对接现有 AI 业务流程。

### 3.2 物理引导的教师数据构造
- 使用 FuXi‑2.0 驱动 WRF，并通过 spectral nudging 约束大尺度：
  - 保留 FuXi 的路径与环境场优势；
  - WRF 在 0.25°、56 层垂直网格上，用 SHTM 同类物理方案（微物理、对流、边界层等）生成真实感更强的 TC 细节；
  - 由于 WRF 被强制贴合 FuXi 大尺度，FuXi 与 WRF 间的“域差距”更小，利于学习映射关系。

### 3.3 扩散模型的适配
- 利用 DDPM 强项：
  - 擅长学习复杂多尺度分布，可在保持大尺度结构的前提下生成细致的局地结构（如台风眼墙、雨带）；
  - 可以自然生成不确定性（通过多次采样形成集合），虽文中主要展示点估计与 RMSE 对比。

### 3.4 计算效率
- 对比：
  - WRF：32 核 Intel Xeon 8369B，单次预报约 83 分钟；
  - FuXi‑TC：单卡 A100，单次预报约 2 秒；
- 意义：
  - 适合实时业务系统与灾害应急；
  - 容易构造大规模集合，用于风险评估与不确定性分析。


## 4. 数学/损失与评估

### 4.1 训练目标（概念性）
- 虽原文片段未给出具体扩散损失公式，这里给出典型 DDPM 形式（标注为“基于通用 DDPM 形式的推断”）：
- 前向加噪：
  \[
  q(\mathbf{x}_t\mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I} ),
  \]
  其中 \(\mathbf{x}_0\) 是 WRF 目标场，\(\mathbf{x}_t\) 为加噪后的中间状态，\(\bar\alpha_t\) 为预设噪声调度；
- 反向模型：
  \[
  p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t, \mathbf{c}) = \mathcal{N}\bigl(\boldsymbol\mu_\theta(\mathbf{x}_t, t, \mathbf{c}), \boldsymbol\Sigma_\theta(\mathbf{x}_t, t, \mathbf{c})\bigr),
  \]
  其中 \(\mathbf{c}\) 为条件（FuXi 预报场）；
- 训练损失（噪声预测版）：
  \[
  \mathcal{L}_{\text{DDPM}} = \mathbb{E}_{\mathbf{x}_0, t, \boldsymbol\epsilon}\bigl[\,\| \boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon, t, \mathbf{c})\|_2^2\bigr].
  \]
- 注意：上述为典型公式，论文片段未显式给出，具体实现细节应以原文为准。

### 4.2 评价指标：RMSE
- 对 10 m 风速（WS10M）的 RMSE 随预报时效的曲线：
  - 对比对象：ERA5、FuXi、FuXi‑TC、WRF、ECMWF HRES；
  - FuXi‑TC RMSE 明显低于 FuXi 和 ERA5，与 HRES 接近，WRF 最优（教师数据）。

- 形式上，可写为（概念）：
  \[
  \mathrm{RMSE}(l) = \sqrt{ \frac{1}{N}\sum_{n=1}^N (y_{n,l}^{\text{pred}} - y_{n,l}^{\text{true}})^2 },
  \]
  其中 \(l\) 为 lead time，\(y\) 为 TC 中心最大持续风或区域内某种加权平均强度。


## 5. 数据与实验配置

### 5.1 观测/评估数据：CMA Best Track + IBTrACS
- 使用 NOAA 维护的 IBTrACS 档案，并选取其下 **中国气象局（CMA）最佳路径数据集**：
  - 提供 6 小时（登陆前 3 小时）分辨率的 TC 轨迹：中心位置、最大风速、最小中心气压、强度等级等；
  - 覆盖西北太平洋与南海所有 TC；
- 用途：
  - 进行路径与强度客观评分；
  - 确定样本时间窗与区域裁剪范围。

### 5.2 FuXi‑2.0 数据
- FuXi‑2.0 概述：
  - 以 ERA5 为训练与驱动数据，分辨率 0.25°，137 层，0.01 hPa 顶；
  - 训练期 2012–2017；
  - 支持小时级输出（从 6 小时输入生成 1 小时步长），并显式建模海洋反馈（海表热通量等）。

- 本研究使用变量：
  - 地面：MSLP, T2M, D2M, SP, U10M, V10M；
  - 13 个标准压层（50–1000 hPa）的上空变量：Z, T, U, V, RH；
  - RH 由 Q, P, T 通过热力关系推导，与 WRF 变量表一致。

- 时间范围：
  - 2019–2024 年 4–10 月（西北太平洋主台风季）。

### 5.3 WRF 配置
- 版本：WRF v4.3；
- 区域与网格：
  - 经纬度长方形：100°E–160°E, 0°–50°N；
  - 水平分辨率 0.25°，格点数约 242×202；
  - 垂直：Eta 坐标，顶部 50 hPa，56 层。

- 物理参数化方案（跟随上海台风模式 SHTM）：
  - Thompson 微物理；
  - Multi‑scale Kain‑Fritsch 对流；
  - RRTMG 长波辐射；
  - Noah 陆面模式；
  - Yonsei University 行星边界层方案。

- 谱向倾向（spectral nudging）：
  - 仅对大尺度（超过给定波长阈值）特征施加约束；
  - 约束变量：U, V, 虚温（virtual temperature）；
  - 不约束比湿 Q，以避免削弱 TC 强度；
  - 仅在 850 hPa 以上施加，近地边界层不做 nudging，以保留边界层物理过程与复杂地形影响；
  - 采用较弱松弛系数，既保持与 FuXi 大尺度保持一致，又允许 WRF 发展自身的中小尺度结构；
  - 从积分开始到结束全程激活，保证大尺度场持续贴合 FuXi。

- 提取给 AI 的 WRF 变量：
  - 地面：T2M, U10M, V10M, WS10M, MSLP, 总降水 TP；
  - 高空：Z700, 以及 200/300/500/700/850 hPa 的 T, Q, U, V；
  - 这些变量作为目标/条件，增强垂直结构表达。

### 5.4 训练/测试划分
- WRF 输出时段：2019–2023 & 2024；
  - 2019–2023：用于 FuXi‑TC 训练；
  - 2024：用于独立测试（21 个 TC）。

- 预报初始化时间：
  - 每日 0000 与 1200 UTC，积分 120 h（5 天）。


# 维度二：基础组件知识（Component-Level Knowledge）

## 组件 1：FuXi‑2.0 → WRF 教师数据生成模块

1. 子需求与作用
- 解决子问题：
  - ERA5 作为训练标签会系统低估 TC 强度，与 FuXi 在空间分布上存在“域差距”；
  - 需要一套 **既符合 FuXi 大尺度结构，又有更真实强度/结构** 的高质量数据作为监督信号。

2. 技术要点
- 使用 FuXi‑2.0 + ERA5 SST/土壤作为 WRF 初始与边界条件；
- 使用 spectral nudging 使 WRF 大尺度与 FuXi 对齐，小尺度由 WRF 自行发展；
- 产出 0.25°、56 层的连续时空场，作为 FuXi‑TC 的目标数据集。

3. 耦合度
- 与 FuXi‑TC 高度耦合，是训练阶段的“教师模型”；
- 但从概念上可替换为“任意物理模式 + spectral nudging”。


## 组件 2：FuXi‑TC 条件扩散生成模型

1. 子需求定位
- 子问题：
  - 给定 FuXi 预报场 \(x\)，如何生成逼近“WRF 真实强度场” \(y\) 的高分辨率局地 TC 结构？
  - 希望在保留 FuXi 轨迹/大尺度的前提下，增强强度与眼墙/雨带结构。

2. 技术与创新（高层描述）
- 使用条件 DDPM：
  - 条件 \(\mathbf{c}=x\) 由 FuXi 多变量场提供；
  - 模型学习 \(p(y\mid x)\)，可以近似建模强度与结构的不确定性；
  - 推理阶段可通过多次采样形成集合（文中侧重点预报与 RMSE 评估）。

3. 耦合度
- 强绑定-气象特异：
  - 条件输入、损失和下游评价完全针对 TC 强度问题设计；
- 但扩散框架本身为通用生成式建模方式。

4. 伪代码（概念性，基于通用 DDPM 形式）
```pseudo
# 训练阶段
for (x_fuxi, y_wrf) in training_pairs:
    # x_fuxi: FuXi forecast fields (condition)
    # y_wrf:  WRF target fields
    t ~ Uniform({1,...,T})
    eps ~ N(0, I)
    x_t = sqrt(alpha_bar[t]) * y_wrf + sqrt(1 - alpha_bar[t]) * eps

    eps_pred = eps_theta(x_t, t, cond=x_fuxi)
    loss = mse(eps_pred, eps)
    optimize(loss)

# 推理阶段（给定 FuXi 预报）
function refine_fuxi(x_fuxi):
    x_T ~ N(0, I)
    for t in reversed(1..T):
        eps_pred = eps_theta(x_T, t, cond=x_fuxi)
        x_T = denoise_step(x_T, eps_pred, t)
    y_hat = x_T          # FuXi-TC refined field
    return y_hat
```


## 组件 3：谱向倾向（Spectral Nudging）机制

1. 子需求定位
- 子问题：
  - 如何在 WRF 中同时保持 FuXi 提供的大尺度引导（路径和环境场），又允许 WRF 自由发展中小尺度 TC 结构？

2. 技术细节（定性）
- 在谱空间对超过某个水平波长的模式施加弱约束：
  - 对应变量：U, V, 虚温；
  - 垂直范围：850 hPa 以上（不扰动边界层）；
- 采用高度依赖权重：越接近地面，nudging 权重越小；
- 使用较弱松弛时间，保证大尺度贴合但不压制中小尺度。

3. 耦合度
- 与 FuXi‑2.0 输出和 WRF 强绑定，是教师数据构造的关键；
- 不直接出现在 FuXi‑TC 模型结构内，但通过数据决定其学习目标。


## 组件 4：数据抽取与区域裁剪

1. 子需求定位
- 子问题：
  - FuXi/WRF/ERA5 皆为全球或大区域场，如何提取围绕 TC 区域的子域用于训练与推理？

2. 技术要点
- 借助 CMA Best Track 提供的 TC 中心与强度，确定每个样本的中心位置与半径；
- 在 FuXi 与 WRF 场中裁剪固定大小的区域（文中未给具体大小，记为“未明确说明”）；
- 形成配对样本 (FuXi patch, WRF patch)。

3. 耦合度
- 与 FuXi‑TC 强耦合：决定输入/输出张量尺寸；
- 与上游轨迹数据（CMA/IBTrACS）强耦合。


## 组件 5：评估与诊断模块（RMSE、结构、谱）

1. 子需求
- 检验 FuXi‑TC 相比 FuXi / ERA5 / WRF / HRES 在强度与结构上的改进。

2. 技术点
- 强度 RMSE：
  - 对 21 个 TC、0–5 天 lead time 计算 WS10M RMSE，画时间序列曲线；
- 个例分析（如 2024 Typhoon Bebinca）：
  - 比较 FuXi‑TC/FuXi/WRF/ERA5 的 10 m 风场空间分布、台风眼位置；
  - 计算从风眼向外的径向风速剖面；
  - 计算 10 m 风速 PDF 的对数分布；
  - 计算动能谱与 2 m 温度谱，对比结构尺度分布。

3. 结果特征（定性）
- FuXi‑TC 的径向剖面峰值接近 WRF 高值；
- 风速 PDF、能谱与温度谱曲线比 FuXi/ERA5 更接近 WRF，显示其成功纠正 FuXi 的强度低估与能量分布缺陷。


## 组件 6：计算成本与业务部署

1. 子需求
- 在业务化中，需要比较 FuXi‑TC 与 WRF 的算力与时效，验证其实用性。

2. 结果
- 表 1：
  - WRF：32 × Xeon 8369B，单次预报 ~83 分钟；
  - FuXi‑TC：1 × A100，单次预报 ~2 秒；
- 意义：
  - FuXi‑TC 可作为 WRF 下缩放的替代方案，在业务中实现高频刷新或大规模集合；
  - 特别适合极端事件应急预报场景。

---

> 注：本文档基于提供的摘要与方法片段整理，关于 FuXi‑TC 网络内部结构、具体扩散参数与损失实现的细节，原文未完全展开，已在相应位置标注“未明确说明”或给出通用形式以供理解。