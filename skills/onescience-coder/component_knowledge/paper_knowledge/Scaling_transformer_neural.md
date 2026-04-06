# Stormer：可扩展 Transformer 的中期数值天气预报模型

---

## 一、整体模型知识（Model-level）

### 1.1 问题背景与任务

- **任务类型**：全球中期数值天气预报（medium-range, 1–10+ 天），基于 ERA5 再分析数据，学习从初始场到未来多变量三维大气场的映射。
- **输入/输出形式**：
  - 初始条件：\(X_0 \in \mathbb{R}^{V \times H \times W}\)
  - 目标预报：\(X_T \in \mathbb{R}^{V \times H \times W}\)，其中 \(T\) 为目标预报时效（如 1–10 天）。
  - \(V\)：变量通道数（近地面 + 多等压面变量），\(H \times W\)：全球经纬网格。
- **数据来源与分辨率**：
  - 数据集：WeatherBench 2 上的 ERA5（ECMWF）再分析
  - 时空分辨率：6 小时、1.40625°（128×256 网格）
  - 时间范围：1979–2018 训练，2019 验证，2020 测试。
- **物理难点**：
  - 大气是典型的混沌系统，长时效直接一跳预报非常困难，误差随时间快速放大。
  - 多变量（温度、风、位势高度、比湿等）跨 13 个等压面，变量间存在复杂物理关系（如风场与位势高度梯度的地转平衡）。
  - 传统 NWP 依赖显式方程与参数化，计算昂贵且不随数据量自然提升。

### 1.2 Stormer 的总体思路

- **目标**：在尽量保持标准 Transformer/ViT 结构简单性的前提下，通过“训练策略 + 损失设计 + 轻量结构改动”，实现与 Pangu-Weather、GraphCast 等高度定制架构相当甚至更优的中期预报能力，并具备良好的缩放（scaling）性质。
- **核心设计三要素**：
  1. **随机化动力学预报目标（Randomized Dynamics Forecasting）**：
     - 不直接预报 \(X_T\)，而是预报在随机时间间隔 \(\delta t\) 上的“天气增量” \(\Delta_{\delta t} = X_{\delta t} - X_0\)。
     - 训练时随机采样 \(\delta t \in \{6,12,24\}\) 小时，统一使用一个模型同时学习不同时间尺度的动力学。
  2. **天气特异嵌入（Weather-specific Embedding）**：
     - 将多变量三维大气场拆分为“变量 token + 变量聚合”两步，利用跨变量注意力建模变量间非线性交互，比标准 ViT patch embedding 更适合气象数据。
  3. **气压加权损失（Pressure-weighted Loss）**：
     - 在多变量多层损失中，按等压面气压作为大气密度 proxy，对近地面更重要的变量给出更高权重，并叠加常见的纬度权重 \(L(i)\)。

### 1.3 训练范式：随机化增量迭代预报

#### 1.3.1 迭代动力学预报

- 采用 **迭代预报（iterative forecasting）** 而非一次性直达：
  - 模型预测 \(\Delta_{\delta t} = X_{\delta t} - X_0\)，再通过
    \[
      X_{\delta t} = X_0 + \Delta_{\delta t}
    \]
  - 长时效预报通过多步滚动实现（多次 roll-out）。
- 与直接预报 \(X_T\) 相比：
  - 增量往往更平滑、集中于动力变化，易于学习。
  - 适配多种时间间隔的组合，提高长时效表现。

#### 1.3.2 随机时间间隔目标

- 损失函数：
  \[
    \mathcal{L}(\theta) = \mathbb{E}_{\delta t \sim P(\delta t),\,(X_0, X_{\delta t}) \sim \mathcal{D}} \Big[\, \| f_\theta(X_0, \delta t) - \Delta_{\delta t}\|_2^2 \Big] \tag{1}
  \]
- 时间间隔分布：\(P(\delta t) = \mathcal{U}\{6, 12, 24\}\) 小时。
- 物理含义：
  - 6/12 小时：有助于学习昼夜周期等短期振荡（diurnal cycle）。
  - 24 小时：弱化日变化影响，聚焦于天气尺度（synoptic scale）动力过程，对 7+ 天中期预报尤其关键。
- 优势：
  - 数据增强：同一时间段可构造多种 \(\delta t\) 组合。
  - 统一模型：单一模型覆盖多个步长，减少多模型训练成本（对比 Pangu 为每个步长训练一个模型）。

#### 1.3.3 气压加权 + 纬度加权损失

- 完整单步损失：
  \[
    \mathcal{L}(\theta) = \mathbb{E}\Big[ \frac{1}{VHW} \sum_{v=1}^V \sum_{i=1}^H \sum_{j=1}^W w(v)\, L(i) \big(\hat{\Delta}_{\delta t}^{vij} - \Delta_{\delta t}^{vij}\big)^2 \Big] \tag{2}
  \]
  - \(w(v)\)：变量层权重，按变量所在气压层密度设计，近地面权重更高。
  - \(L(i)\)：纬度权重，补偿球面网格在高纬变密的问题（常用 \(\cos \varphi\) 或其变体）。
- 物理目的：
  - 强化近地面温度、风、海平面气压等对实际天气影响更大的变量。
  - 保证全球积分误差度量更接近物理意义上的能量/质量加权。

#### 1.3.4 多步微调（Multi-step finetuning）

- 为缓解迭代滚动中的误差累积，采用三阶段训练：
  1. 阶段 1：单步损失（式 (2)）。
  2. 阶段 2：在阶段 1 权重基础上，用 \(K=4\) 步多步损失：
     \[
       \mathcal{L}(\theta) = \mathbb{E}\Big[ \frac{1}{K V H W} \sum_{k=1}^K \sum_{v,i,j} w(v) L(i) (\hat{\Delta}_{k\delta t}^{vij} - \Delta_{k\delta t}^{vij})^2 \Big] \tag{3}
     \]
  3. 阶段 3：进一步用 \(K=8\) 微调。
- 多步微调时，所有 \(K\) 步使用同一个随机采样的 \(\delta t\)，避免不同步长带来尺度差异导致训练不稳定。

### 1.4 推理（Inference）与多路径组合

- 训练后，Stormer 可在任一训练过的 \(\delta t \in \{6,12,24\}\) 下进行预报。
- 对于目标时效 \(T\)，通过不同 \(\delta t\) 组合实现多条滚动路径，例如：
  - 3 天：6h×12 次，12h×6 次，或 24h×3 次。
- **两类推理策略**：
  1. **同质组合（Homogeneous）**：
     - 只用同一 \(\delta t\) 的组合，如 \([6,6,6,6]\)、\([12,12]\)、\([24]\) 等。
     - 对于给定 \(T\)，枚举少数几种组合，计算平均预测。
  2. **Best m in n**：
     - 从所有可能（或随机采样）的异质时间间隔组合中生成 \(n\) 条路径，在验证集上评估误差，选出最优 \(m\) 条用于测试。
     - 最终预测为这 m 条路径输出的平均。
- 与 NWP 集合预报类比：
  - 不同滚动路径类似“不同扰动初值/不同模型配置”，平均相当于 Monte Carlo 整合，缓解中长期混沌敏感性。

### 1.5 与现有方法的关系与对比

- **对比直接/连续/迭代预报**：
  - 直接预报：\(\hat{X}_T = f_\theta(X_0)\)，每个 T 需单独模型，且大 T 难以收敛。
  - 连续预报：\(\hat{X}_T = f_\theta(X_0, T)\)，单模型覆盖多时效，但仍一次性映射到大 T。
  - 迭代预报：\(\hat{X}_{\delta t} = f_\theta(X_0)\)，多次滚动，存在误差累积。
  - **Stormer**：采用“随机化迭代动力学”介于连续/迭代之间，通过 conditioning on \(\delta t\) 统一建模不同时间尺度动力学。

- **对比 Pangu-Weather / GraphCast / ClimaX 等**：
  - Pangu：3D Earth-specific Transformer，针对球面做大量结构定制，为每个步长训练单独模型。
  - GraphCast：基于图神经网络，多网格 message passing，结构复杂。
  - ClimaX：通用气候基座模型，采用连续 lead-time 输入与天气特异嵌入，但未使用随机化增量目标 + adaLN 组合。
  - Stormer：以标准 ViT 为基础，最少结构改动（天气嵌入 + adaLN），主要依靠训练目标和损失设计取得 SOTA 表现。

### 1.6 数据与变量配置

- 空间分辨率：1.40625°（128×256 网格）
- 时间分辨率：6 小时
- 变量：
  - 地表：T2m、U10、V10、MSLP
  - 大气 13 个等压面：{50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000} hPa
    - 位势高度 Z
    - 温度 T
    - 风场 U/V 分量
    - 比湿 Q

### 1.7 模型规模与缩放性质

- 主模型（大号 Stormer）：
  - 24 层 Transformer block
  - 隐层维度 1024
  - patch size = 2（与 ViT-L 类似，但 patch 更小以获得更细粒度空间表示）
- 其它实验：
  - 为了更快试验与消融，使用 patch size = 4 的同构模型。
- 缩放实验表明：
  - 随着模型容量（层数/维度）与训练 token 数增加，Stormer 的误差稳定下降，呈现良好的 scaling law 行为。

### 1.8 性能与案例

- 在 WeatherBench 2 上：
  - 1–7 天短中期预报与当前 SOTA（Pangu、GraphCast 等）相当或略优。
  - 7 天以上时效显著优于现有方法。
  - 训练数据分辨率更低、GPU 小时数量级更少，性价比高。
- 极端事件案例：
  - 2020 年 12 月 31 日影响阿拉斯加的创纪录北太平洋低压系统案例，Stormer 在 5 天提前量上准确预报其位置与强度。

---

## 二、基础组件知识（Component-level）

### 2.1 天气特异嵌入（Weather-specific Embedding）

#### 2.1.1 变量 token 化（Variable Tokenization）

- 输入：\(X_0 \in \mathbb{R}^{V \times H \times W}\)
- 步骤：
  1. 对每个变量通道 v 独立应用线性 embedding，将 \(H \times W\) 网格划分为 \(p \times p\) patch：
     - 得到形状 \((H/p) \times (W/p) \times D\) 的 token 序列（D 为隐藏维度）。
  2. 将所有 V 个变量的 token 串联：
     \[
       \text{Tokenized} \in \mathbb{R}^{(H/p) \times (W/p) \times V \times D}
     \]
- 作用：
  - 在空间上进行分块嵌入，同时保持变量维度显式保留，便于后续在变量维上做 cross-attention 聚合。

#### 2.1.2 变量聚合（Variable Aggregation）

- 使用单层 cross-attention，在变量维度上聚合：
  - Query：可学习的 query 向量（或小集合），不随变量变化。
  - Key/Value：来自不同变量的 token 表示。
- 结果：
  - 将变量维度聚合掉，输出形状为 \((H/p) \times (W/p) \times D\) 的空间 token 序列。
- 优势：
  - 序列长度减少约 V 倍，大幅降低后续 Transformer 的计算成本。
  - 通过注意力建模变量间非线性物理联系（如风与位势高度梯度、湿度/温度关系）。
- 消融结论：
  - 在 1–10 天各预报时效上优于标准 ViT patch embedding。

### 2.2 Stormer Transformer Block 与 adaLN 条件化

#### 2.2.1 标准 Transformer 基块

- 结构：
  - Multi-head Self-Attention (MSA)
  - 前馈网络 (FFN)
  - 残差连接 + LayerNorm
- 对输入 token 序列执行全局自注意力，捕捉大尺度空间相关性。

#### 2.2.2 自适应 LayerNorm（Adaptive LayerNorm, adaLN）

- 目标：
  - 用时间间隔 \(\delta t\) 条件化每一层，而非只在网络入口处加一次 embedding。
- 做法：
  - 用一层 MLP 将 \(\delta t\) 的 embedding 映射为每层的 \(\gamma, \beta\)：
    \[
      (\gamma, \beta) = \text{MLP}(\text{Embed}(\delta t))
    \]
  - 在每个 LayerNorm 中使用该 \(\gamma, \beta\) 作为缩放与平移：
    \[
      \text{adaLN}(h) = \gamma \odot \frac{h - \mu(h)}{\sigma(h)} + \beta
    \]
- 对比：
  - ClimaX：只在第一层前做一次时间 embedding 的加法。
  - Stormer：在每个 block 都用 adaLN 进行条件化，放大 \(\delta t\) 的影响。
- 实验：
  - 在相同架构下，adaLN 明显优于简单的 additive lead-time embedding。

### 2.3 随机增量目标与多步训练细节

#### 2.3.1 单步随机增量训练

- 每个 batch：
  1. 从数据集中采样 \((X_0, X_{\delta t})\)。
  2. 随机采样 \(\delta t \in \{6,12,24\}\)。
  3. 计算 \(\Delta_{\delta t} = X_{\delta t} - X_0\)。
  4. 前向计算 \(\hat{\Delta}_{\delta t} = f_\theta(X_0, \delta t)\)。
  5. 用气压+纬度加权 MSE（式 (2)）计算损失。

#### 2.3.2 多步滚动微调

- 设定多步长度 K：
  - 阶段 2：K=4
  - 阶段 3：K=8
- 多步训练伪流程：
  1. 采样 \(\delta t\)，初始化 \(X^{(0)} = X_0\)。
  2. 对 \(k=1..K\)：
     - 用 \(X^{(k-1)}\) 做输入，预测 \(\hat{\Delta}_{\delta t}^{(k)}\)。
     - 得到 \(\hat{X}_{k\delta t} = X^{(k-1)} + \hat{\Delta}_{\delta t}^{(k)}\)。
     - 计算与真值 \(X_{k\delta t}\) 的增量误差 \(\hat{\Delta}_{k\delta t}, \Delta_{k\delta t}\)。
     - 更新 \(X^{(k)} = \hat{X}_{k\delta t}\)，供下一步使用。
  3. 将所有步的损失平均（式 (3)）。

### 2.4 推理策略组件

#### 2.4.1 同质组合策略（Homogeneous）

- 给定目标时效 \(T\)：
  - 枚举所有只包含单一 \(\delta t\) 的组合（如 T=24h → [6,6,6,6]，[12,12]，[24]）。
  - 对每种组合执行多次 roll-out，得到一组预测。
  - 对所有组合的预测取平均作为最终结果。
- 特点：
  - 组合数量少，推理成本低。

#### 2.4.2 Best m in n 策略

- 预处理阶段：
  - 为每个目标 T 随机/系统性生成 n 种（可能异质的）\(\delta t\) 序列，使其求和为 T。
  - 在验证集上评估每种组合对应预测的验证误差。
  - 选出误差最小的 m 条组合。
- 测试阶段：
  - 对这 m 条组合执行 roll-out，取平均为最终预测。
- 特点：
  - 兼顾效率与表达力，可探索更丰富的时间组合结构，对长时效尤为有利。

### 2.5 损失与权重函数设计

#### 2.5.1 纬度权重 \(L(i)\)

- 目的：
  - 校正经纬网格在高纬更密集、面积更小的问题，使每格权重与其实际地表面积近似成正比。
- 常用形式：
  - \(L(i) \propto \cos \varphi_i\)，其中 \(\varphi_i\) 为第 i 行对应的纬度。
  - 论文未给出精确公式 → 记为“未明确说明”的具体数值形式，只说明遵循标准做法。

#### 2.5.2 气压权重 \(w(v)\)

- 依据 GraphCast 提出的方法：
  - 等压面变量按所在气压层加权，近地面层（如 1000/925/850 hPa）权重更高。
  - 高空变量权重较低。
- 物理意义：
  - 近地面变量更直接关联地表天气与社会影响（降水、风、温度）。

### 2.6 计算配置与实现

- 框架：标准 Vision Transformer (ViT) 变体
  - 24 blocks, hidden size 1024
  - patch size 2 或 4
- 优化器、学习率策略、batch size 等具体数值在本文摘录部分未详述 → 记为“未明确说明”。

---

## 三、小结与迁移思路

- **本质定位**：Stormer 更像是“训练目标与损失工程”的成功案例，在不大幅改动 Transformer 架构的前提下，通过：
  - 随机化动力学增量目标（多 \(\delta t\) 统一建模）、
  - 气压/纬度物理加权、
  - 多步微调抑制误差累积、
  - 天气特异嵌入 + adaLN 条件化，
  实现对复杂大气动力学的高效拟合。

- **可能的迁移方向（论文未展开，属推测需验证）**：
  - 将随机化增量 + adaLN 思路用于其他地球系统变量（如海洋、陆面水文）中期预报。
  - 将 weather-specific embedding 与 GraphCast/ClimaX 等其他结构结合，进一步统一大气–海洋–陆面多模态预报框架。

- **与已有知识库的关系**：
  - 与 AIFS、GraphCast、ClimaX 一样，Stormer 属于“全球、数据驱动、中期预报 Transformer 家族”；本文件可作为该家族中“以标准 ViT 为基、训练策略驱动型”的代表。