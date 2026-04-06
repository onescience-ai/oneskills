# FourCastNet 模型知识提取

信息来源：Pathak et al., “FourCastNet: A global data-driven high-resolution weather model using Adaptive Fourier Neural Operators”（内容严格基于原文，不作主观推断；缺失处标明“未在文中明确给出”）。

---

## 维度一：整体模型与任务知识

### 1. 任务与问题设定

- **任务类型：全球短–中期数据驱动天气预测**
  - 目标：基于 ERA5 再分析构建纯数据驱动模型 FourCastNet，在 $0.25^\circ$ 分辨率下对一组关键大气变量进行短到中期（约 1 周甚至更长）全球预报。
  - 时间分辨率：6 小时一帧（从 ERA5 每小时数据子采样得到 00,06,12,18 UTC）。
  - 空间分辨率：$0.25^\circ$ 纬向 × 经向；等效约 $30\,\text{km} \times 30\,\text{km}$（赤道附近）；栅格大小 $720\times 1440$。

- **状态与变量集合**
  - 预测变量（prognostic variables），共 20 个通道，覆盖面层、多个高度层及积分变量（见表 1）：
    - Surface：
      - U10：10m 经向风（zonal wind）
      - V10：10m 纬向风（meridional wind）
      - T2m：2m 气温
      - sp：surface pressure
      - mslp：mean sea-level pressure
    - 1000 hPa：
      - U, V, Z
    - 850 hPa：
      - T, U, V, Z, RH
    - 500 hPa：
      - T, U, V, Z, RH
    - 50 hPa：
      - Z
    - Integrated：
      - TCWV（total column water vapor）
  - 诊断变量（diagnostic）：
    - TP：total precipitation（单位为水深，6 小时累计降水），由单独的 AFNO 模型从 backbone 预测的 20 个变量诊断得到。

- **输入 / 输出形式**
  - 将上述 20 个变量在给定时间 $k\Delta t$ 的场记为：
    $$
    \mathbf{X}(k) \in \mathbb{R}^{H\times W\times C},\quad H=721,\;W=1440,\;C=20
    $$
    （文中给出为 $721\times 1440\times 20$，略与 720 有出入，整体为全球网格）。
  - 单步预测：
    $$
    \mathbf{X}(k+1) = \mathcal{F}_{\text{AFNO}}(\mathbf{X}(k))
    $$
  - 两步精调阶段：
    - 首先预测 $\mathbf{X}(k+1)$，再以其为输入预测 $\mathbf{X}(k+2)$，对两步误差共同反向传播。
  - 多步自回归推理：
    $$
    \mathbf{X}_{\text{pred}}(k+i) = \mathcal{F}_{\text{AFNO}}\big(\mathbf{X}_{\text{pred}}(k+i-1)\big),\quad i=1,\dots,\tau
    $$
    - 典型实验中允许自由运行 16 步（96 小时）或更长时间（至 7–10 天）。

- **关注变量与应用场景**
  - 重点任务变量：
    1. 10m 近地面风速与风矢量（U10,V10）—风能规划、极端风灾害；
    2. 6 小时累计降水 TP —极端降水、暴雨、飓风降水、气候评估；
    3. TCWV —大气河流、长距离水汽输送；
    4. Z500, T850, T2m 等大尺度变量 —与 IFS、DLWP等进行技巧对比；
  - 场景示例：
    - 超强台风 Mangkhut（2018-09-08 起始，96 小时 lead）；
    - 飓风 Michael 的生成与登陆过程；
    - 北加州“Pineapple Express”型大气河事件（2018-04-04 开始）；
    - 北美风能资源与陆地风速预报；
    - 全球极端事件（极端风、降水）的概率评估与集合预报。

- **数据集与时间划分**
  - 数据源：ERA5 再分析（0.25°，37 层），1979–now；
  - 训练/验证/测试：
    - Train：1979–2015
    - Validation：2016–2017
    - Test（out-of-sample）：2018 及之后
  - 6 小时子采样：每天 4 个时间（00, 06, 12, 18 UTC）。
  - 样本数：
    - 训练：54,020 条样本；
    - 验证：2,920 条；
    - 测试：2018 全年（按变量与 lead 设定不同 $N_f, D$）。

---

### 2. 传统方案及现有 DL 模型的痛点

- **传统 NWP（以 ECMWF IFS 为代表）**
  - 优点：物理一致性好，变量丰富（>150 变量、>50 层），多模式、多集合；
  - 局限：
    - 计算成本极高：高分辨率、高层数 + 数据同化 + 集合预报 → 需大型 CPU 集群；
    - 分辨率提高带来近 $2^{3.5}$ 的成本增长；
    - 对降水等参数化过程存在系统性偏差；
    - 大集合（>50 成员）几乎不可负担。

- **既有 DL 天气模型痛点**
  - 多数在低分辨率训练：
    - 典型分辨率：$5.625^\circ$（WeatherBench）、$2^\circ$（DLWP），对应 $32\times64$ 或类似 coarse grid；
    - 丢失 500 km 以下小尺度结构，无法解析飓风、锋区、狭窄大气河等；
  - 变量类型有限，多为 Z500、厚度、T2m 等大尺度变量；
  - 对近地面风速、降水等高难度、高分辨率变量几乎无尝试或效果有限；
  - 卷积网络在超高分辨率下内存成本和感受野深度急剧膨胀，例如：
    - WeatherBench 19 层 ResNet 若直接移植到 $720\times1440$，batch=1 需约 83GB 显存（FourCastNet AFNO 约 10GB）。

- **动机总结**
  - 需要：
    - 在 $0.25^\circ$ 分辨率上对近地面风、降水等复杂小尺度变量进行全球预测；
    - 在保持或接近 IFS 技巧前提下，大幅降低计算成本（速度提高 4–5 个数量级）；
    - 支持快速生成上千成员的集合预报；
    - 与 IFS、DLWP 等进行公平对比，证明数据驱动模型的潜力。

---

### 3. 整体解决方案与范式

- **四大核心贡献（作者总结）**
  1. 在一周提前量内以前所未有的精度预测近地面风和降水；
  2. 分辨率为现有 DL 全球天气模型的 8 倍（0.25° vs 2°），可解析飓风、大气河等极端事件；
  3. 在 RMSE/ACC 指标上与 IFS 在 3 天内相当，1 周内接近；
  4. 极高推理速度和极低能耗，使上千成员大集合预报成为现实。

- **整体建模范式**
  1. **AFNO + ViT 主干（backbone）**：
     - 将 ERA5 20 个变量的 2D 场映射为 patch tokens；
     - 使用多层 AFNO Transformer 进行空间 token mixing + 通道 mixing；
     - 解码出下一时间步 20 个变量全场；
  2. **两阶段训练（pre-training + two-step fine-tuning）**：
     - 预训练：学习一步映射 $\mathbf{X}(k)\to\mathbf{X}(k+1)$；
     - 微调：在预训练基础上学习两步链式预测 $\mathbf{X}(k+1), \mathbf{X}(k+2)$ 并联合优化；
  3. **诊断降水模型**：
     - 单独的 AFNO 模型，以 backbone 预测的 20 变量为输入，输出 6 小时累计 TP；
     - 通过 log-transform 减少偏斜分布；
  4. **自回归多步推理**：
     - 利用训练好的 backbone + TP 诊断模型，自回归生成最长约一周（甚至更长）的多变量预报；
  5. **大规模集合预报**：
     - 使用高斯噪声扰动 ERA5 初始场生成 $E$ 个初始状态；
     - 将 ensemble 维度视作 batch 维，GPU 上批量推理生成上百至上千成员集合；
  6. **评估与对比**：
     - 与 IFS（TIGGE 中的 2018 年 6 小时预报）在 Z500、T2m、U10、V10、TP、T850 等变量上比较 ACC/RMSE；
     - 与 DLWP 在 2° 分辨率下比较 Z500, T2m 技巧（将 FCN 输出降采样）。

---

### 4. 模型架构与关键公式

#### 4.1 AFNO 主干结构（Fourier Token Mixing + ViT）

- **Patch embedding 与 token 序列**
  - 对输入场 $\mathbf{X} \in \mathbb{R}^{H\times W\times C}$：
    - 划分为 $h\times w$ patch 网格，每个 patch 大小 $p\times p$（如 $p=8$）；
    - 每个 patch 展平并经线性映射为 $d$ 维 token；
    - 加上位置编码，得到 $h\times w$ 个 token，张量形状 $(h,w,d)$ 或 $(N,d)$，$N=h\times w$。

- **单个 AFNO 层空间 mixing（Fourier 域）**

  给定输入 token 张量：
  $$
  \boldsymbol{X} \in \mathbb{R}^{h\times w\times d}
  $$

  1. 2D 离散 Fourier 变换：
     $$
     z_{m,n} = [\mathrm{DFT}(X)]_{m,n}, \quad m=1..h,\;n=1..w
     $$
  2. 频域 MLP + Soft-thresholding（稀疏性）：
     $$
     \tilde{z}_{m,n} = S_{\lambda}(\mathrm{MLP}(z_{m,n}))
     $$
     其中：
     $$
     S_{\lambda}(x) = \mathrm{sign}(x)\,\max(|x|-\lambda, 0)
     $$
     - $\lambda$：稀疏阈值；
     - MLP：2 层、block-diagonal 权重，并在所有 patch 上共享。
  3. 逆 DFT + 残差：
     $$
     y_{m,n} = [\mathrm{IDFT}(\tilde{Z})]_{m,n} + X_{m,n}
     $$

- **通道 mixing**
  - 在空间 mixing 后的 token 上，对每个 token 进行 MLP 通道混合（与 ViT 类似），包含非线性与残差。

- **多层堆叠 + 解码**
  - 重复 AFNO block + MLP block 共 $L$ 层（文中 depth=12，AFNO blocks=8）；
  - 最后通过线性解码器映射回 patch 空间，再重组到原始栅格 $H\times W$ 得到下一时刻 20 变量场。

#### 4.2 训练目标与步骤

- **记号**
  - ERA5 真值：$\mathbf{X}_{\text{true}}(k)$；
  - 模型输入：$\bar{\mathbf{X}}(k)$（可能经过标准化/预处理的输入）；
  - 模型预测：$\mathbf{X}(k+1)$（省略上标 pred）。

- **预训练阶段**
  - 目标：学习一步映射：
    $$
    \mathcal{F}_{\text{AFNO}}: \bar{\mathbf{X}}(k) \mapsto \mathbf{X}(k+1)
    $$
  - 损失：对 20 个变量的逐像素 L2/MSE（文中未给出精确公式，推断为标准像素均方误差）。
  - 学习率调度：
    - cosine schedule，起始 lr $\ell_1 = 5\times10^{-4}$；
    - epochs = 80。

- **两步 fine-tuning 阶段**
  - 使用预训练权重初始化；
  - 前向：
    1. $\mathbf{X}(k) \to \hat{\mathbf{X}}(k+1)$；
    2. 以 $\hat{\mathbf{X}}(k+1)$ 为输入预测 $\hat{\mathbf{X}}(k+2)$；
  - 损失：
    $$
    \mathcal{L} = \mathrm{MSE}(\hat{\mathbf{X}}(k+1), \mathbf{X}_{\text{true}}(k+1))
                 + \mathrm{MSE}(\hat{\mathbf{X}}(k+2), \mathbf{X}_{\text{true}}(k+2))
    $$
  - 学习率：cosine schedule，lr $\ell_2 = 1\times10^{-4}$，epoch=50。

- **降水诊断模型训练**
  - 输入：backbone 输出的 20 变量场序列；
  - 输出：对应时刻的 6 小时累计 TP；
  - 模型结构：
    - 与 backbone 相同 AFNO 主干；
    - 末端增加 2D 卷积层（periodic padding）+ ReLU 确保非负；
  - 预处理：
    $$
    \tilde{TP} = \log\left(1 + \frac{TP}{\epsilon}\right), \quad \epsilon = 10^{-5}
    $$
  - 损失：对 log-transformed TP 的 MSE；
  - 学习率：cosine schedule，lr $\ell_3 = 2.5\times10^{-4}$，epoch=25。

#### 4.3 评价指标（ACC 与 RMSE）

- **ACC（纬向加权异常相关系数）**

  设对变量 $v$ 在预测步 $l$ 的预测与真值分别为 $\mathbf{X}_{\text{pred}}, \mathbf{X}_{\text{true}}$，并减去长期平均（climatology 或长期均值）后的异常为 $\tilde{\mathbf{X}}_{\text{pred/true}}$：

  $$
  \mathrm{ACC}(v,l) =
  \frac{\sum_{m,n} L(m)\, \tilde{\mathbf{X}}_{\text{pred}}(l)[v,m,n]\,\tilde{\mathbf{X}}_{\text{true}}(l)[v,m,n]}
  {\sqrt{\sum_{m,n} L(m)(\tilde{\mathbf{X}}_{\text{pred}}(l)[v,m,n])^2
          \sum_{m,n} L(m)(\tilde{\mathbf{X}}_{\text{true}}(l)[v,m,n])^2}}
  $$

  其中纬向权重：
  $$
  L(m) = \frac{\cos(\mathrm{lat}(m))}{\frac{1}{N_{lat}}\sum_{j=1}^{N_{lat}}\cos(\mathrm{lat}(j))}
  $$

- **RMSE（纬向加权均方根误差）**

  $$
  \mathrm{RMSE}(v,l) =
  \sqrt{\frac{1}{NM}\sum_{m=1}^{M}\sum_{n=1}^{N} L(m)\big(\mathbf{X}_{\text{pred}}(l)[v,m,n]
                                                     -\mathbf{X}_{\text{true}}(l)[v,m,n]\big)^2}
  $$

- **极值评估（Relative Quantile Error, RQE）**

  - 对每个时间步 $l$，以及若干上分位数 $q\in Q$（从 90% 至 99.99%）：
    $$
    \mathrm{RQE}(l) = \sum_{q\in Q}\frac{\mathbf{X}_{\text{pred}}^q(l) - \mathbf{X}_{\text{true}}^q(l)}{\mathbf{X}_{\text{true}}^q(l)}
    $$
  - 若 RQE 为负，说明模型系统性低估极端值；
  - 文中发现：
    - 对 U10，FourCastNet 与 IFS 的 RQE 接近，均略为负（轻微低估极端风）；
    - 对 TP，FourCastNet 约低估 35%，IFS 约低估 15%。

- **陆地/海洋分区 ACC**

  - 使用 land-sea mask $\Phi_{land}^{m,n}, \Phi_{sea}^{m,n}$：
    $$
    \Phi_{sea}^{m,n} = 1 - \Phi_{land}^{m,n}
    $$
  - 定义陆地/海洋 ACC：
    $$
    \mathrm{ACC}_{land/sea}(v,l) = \frac{\sum_{m,n}\Phi_{land/sea}^{m,n}L(m)
    \tilde{X}_{\text{pred}}(l)[v,m,n]\tilde{X}_{\text{true}}(l)[v,m,n]}
    {\sqrt{\sum_{m,n}\Phi_{land/sea}^{m,n}L(m)(\tilde{X}_{\text{pred}})^2
           \sum_{m,n}\Phi_{land/sea}^{m,n}L(m)(\tilde{X}_{\text{true}})^2}}
    $$
  - 结论：FourCastNet 在陆地与海洋上的 U10/V10 ACC 非常接近，对陆地风能应用具有重要意义。

---

### 5. 训练数据与超参数

- **数据预处理**
  - 原始 ERA5 为 Gaussian grid → 通过 Copernicus CDS API 差值到规则经纬网格（Euclidean grid）；
  - 所有 20 变量均表示为 $721\times1440$ 2D 场；
  - 标准化：对每变量计算均值/方差并标准化（文中未列公式，但推断如此）。

- **训练与验证设置**
  - 训练集：1979–2015，54020 样本；
  - 验证集：2016–2017，2920 样本；
  - 测试集：2018+，用于与 IFS 对比。

- **AFNO 超参数（见表 3）**
  - Global batch size：64；
  - Patch size $p\times p$：$8\times8$；
  - 深度：12；
  - AFNO blocks：8；
  - embedding dim：768；
  - MLP ratio：4；
  - 稀疏阈值 $\lambda=10^{-2}$；
  - 激活：GELU；
  - Dropout：0；
  - 学习率：
    - pre-train：$5\times10^{-4}$；
    - fine-tune：$1\times10^{-4}$；
    - TP 模型：$2.5\times10^{-4}$；
    - 调度：cosine schedule。

- **计算成本**
  - 训练：
    - 硬件：64× NVIDIA A100（Perlmutter, Selene 等集群）；
    - 时间：端到端训练约 16 小时 wall-clock；
  - 推理（24h 100-member ensemble）：
    - 在 Perlmutter 上单节点（4×A100）批量推理，batch=25；
    - 100 成员 24h 预报耗时约 7 秒（node-seconds），能耗约 8 kJ（节点峰值功率约 1kW）。

---

### 6. 技巧与结果摘要

- **总体 ACC/RMSE 与 IFS 对比（2018 年全年, 多个初始场）**
  - 变量：U10, V10, TP, T2m, Z500, T850 等：
    - 在 1–2 天 lead：FourCastNet 通常在 ACC 和/或 RMSE 上优于 IFS，尤其是对 U10、TP、T2m；
    - 在 3 天内：技巧接近 IFS，在某些变量上略逊；
    - 在 7 天内：整体仍与 IFS 接近，稍有落后；
  - 对所有 backbone 变量（U/V/T/Z 各层、RH、TCWV、sp/mslp）：
    - 多数变量的 ACC 在 5–10 天内仍保持 > 0.6；
    - RH（r500, r850）相对更难预测，ACC 下降较快。

- **降水技巧**
  - 首次在全球尺度、0.25° 分辨率下，用 DL 模型诊断出与 IFS 竞争的 6 小时 TP；
  - 可解析极端降水事件（大气河、温带气旋）的小尺度结构；
  - 在短 lead（<48h）下，对 TP 的 ACC/RMSE 优于 IFS；
  - 极值方面仍存在低估问题（RQE ~ -35%），高于 IFS（-15%）。

- **极端事件案例**
  1. **Super Typhoon Mangkhut**（2018-09-08 00:00 初始化，96h lead）：
     - 能较好捕捉台风生成、路径与强度演变；
     - 同时跟踪大西洋上三个飓风 Florence, Issac, Helene；
  2. **Hurricane Michael（2018-10-07 起始）**：
     - 使用 100 成员集合，利用 MSLP 最小值追踪台风眼；
     - 预测眼压急剧下降（快速增强）趋势，与 ERA5 接近，但在 36–48h 最大下跌幅度上有所低估；
     - Surface wind, 850hPa wind, MSLP 场可重现从热带低压→5 级飓风→登陆全过程；
  3. **Atmospheric River（Pineapple Express）**：
     - 利用 TCWV 变量预测 2018-04 的一次大气河事件；
     - FourCastNet 在 36h、72h lead 上准确刻画水汽带及其在北加州的登陆位置；
  4. **陆地近地面风速**：
     - 在北美大陆上对 10m 风速的 18,36,54,72h 预报与 ERA5 高度吻合；
     - 陆地 ACC 与海洋 ACC 相近，且在复杂地形区域也能保持较好技巧。

- **集合预报技巧提升**
  - 通过对初始场加高斯噪声 $\mathbf{X}^{(e)}(k) = \hat{\mathbf{X}}_{true}(k) + \sigma \xi$（$\sigma=0.3$），构造 100 成员集合；
  - 比较 control（未扰动）与 ensemble mean 的 ACC/RMSE：
    - 对 U10：在 >70h lead 上，ensemble mean 明显优于 control；
    - 对 Z500：在 >100h lead 上，ensemble mean 也有显著提升；
    - 短 lead（<48h）时，ensemble mean 稍微平滑了小尺度结构，技巧略低于 control。

- **与 DLWP（Weyn et al., 2020）对比**
  - 将 FourCastNet 和 IFS 输出 8× 降采样至 2° 分辨率：
    - Z500, T2m 的 ACC/RMSE 明显优于 DLWP；
  - 更重要的是：FourCastNet 在原生 0.25° 下能解析 DLWP 完全无法捕捉的小尺度现象（如飓风）。

---

### 7. 创新点与局限

- **创新点**
  1. **AFNO + ViT 在 NWP 场景的首次大规模应用**：
     - 以 Fourier 域 token mixing 替代自注意力，复杂度 $O(N\log N)$，在 $720\times1440$ 高分辨率下仍可训练；
  2. **全球 0.25° 分辨率下的多变量联合预测**：
     - 包含近地面风、降水、TCWV 等高度小尺度变量；
  3. **独立的降水诊断 AFNO 模型**：
     - 利用 backbone 输出，通过 log-transform + ReLU 处理稀疏、偏斜的降水分布；
  4. **两步 fine-tuning 技术**：
     - 在避免长序列 AR 训练高显存开销的前提下，显式优化 2-step 预测；
  5. **极高推理速度与能效**：
     - 在 30km 或假想 18km 分辨率下，100 成员 24h 预报的节点时延与能耗远低于 IFS（约 4–5 个数量级）；
  6. **大集合预报与极端事件分析**：
     - 能在秒级时间内生成上百至上千成员集合，用于概率预报与不确定性评估。

- **局限与讨论**
  - 物理一致性：
    - 当前 FourCastNet 不显式约束物理守恒（质量、能量、动量等），与 IFS 有本质不同；
    - 存在预测场在长期积分中的稳定性与物理合理性问题（虽本文主要评估一周内）。
  - 变量/层覆盖：
    - FourCastNet 仅预测 20 个变量、5 个高度层 + 1 个积分变量，远少于 IFS 的 >150 个变量、>50 层；
  - 降水极端：
    - 对 TP 极端值仍有较大低估（RQE ~ -35%）；
  - 数据同化与实时业务：
    - 当前版本缺乏同化组件，无法直接用于实时预报；
    - 未来可与 Ensemble Kalman Filter、物理 NWP 混合，或引入 physics-informed FNO 扩展。
  - 气候变暖与泛化性：
    - 训练集（1979–2015）与测试集（2016–2020）处于气候变暖过程，但未来更极端气候下泛化需进一步验证。

---

## 维度二：基础组件 / 模块级知识

> 说明：以下伪代码基于论文公开描述与公式“高保真”复原，仅为结构与逻辑参考，不代表作者实际实现代码。

### 组件 A：AFNO 主干（Backbone）网络

- **子任务**
  - 从 ERA5 20 变量 $\mathbf{X}(k)$，在 0.25°、$720\times1440$ 网格上预测下一时间步 $\mathbf{X}(k+1)$；
  - 支持后续自回归推理与降水诊断。

- **输入 / 输出**
  - 输入：
    - $X \in \mathbb{R}^{B\times C\times H\times W}$，$C=20$；
  - 输出：
    - $Y \in \mathbb{R}^{B\times C\times H\times W}$，为下一个 6 小时时间步的 20 变量。

- **伪代码：AFNO Layer**

```python
# Pseudocode: Single AFNO layer (spatial mixing)

class AFNO2D(nn.Module):
    def __init__(self, d_model, sparsity_thresh=1e-2):
        super().__init__()
        self.d_model = d_model
        self.sparsity_thresh = sparsity_thresh
        # block-diagonal MLP in Fourier domain
        self.mlp = FourierBlockMLP(d_model)

    def soft_threshold(self, x, lamb):
        return torch.sign(x) * torch.clamp(torch.abs(x) - lamb, min=0.0)

    def forward(self, x):
        # x: (B, H_p, W_p, d_model)
        X_fft = torch.fft.rfft2(x, norm="ortho")        # Fourier transform
        Z = self.mlp(X_fft)                              # shared MLP per frequency
        Z = self.soft_threshold(Z, self.sparsity_thresh)
        y_fft = Z
        y = torch.fft.irfft2(y_fft, s=x.shape[1:3], norm="ortho")
        return x + y  # residual
```

- **伪代码：主干模型**

```python
# Pseudocode: FourCastNet backbone (one-step forecast)

class FourCastNetBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = PatchEmbed(
            in_channels=cfg.in_channels,  # 20
            embed_dim=cfg.embed_dim,      # 768
            patch_size=cfg.patch_size     # 8
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, cfg.num_patches, cfg.embed_dim)
        )
        self.layers = nn.ModuleList([
            AFNOTokenBlock(cfg.embed_dim, cfg) for _ in range(cfg.depth)
        ])
        self.head = PatchDecoder(
            embed_dim=cfg.embed_dim,
            out_channels=cfg.in_channels,
            patch_size=cfg.patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        tokens = self.patch_embed(x)      # (B, N, d)
        tokens = tokens + self.pos_embed
        for blk in self.layers:
            tokens = blk(tokens)          # AFNO + MLP blocks
        y = self.head(tokens)             # (B, C, H, W)
        return y
```

---

### 组件 B：两阶段训练流程

- **子任务**
  - 使用 ERA5 训练 AFNO 主干网络，先学单步，再精调两步预测。

- **伪代码**

```python
# Pseudocode: Pre-training (one-step)

model = FourCastNetBackbone(cfg)
optimizer = AdamW(model.parameters(), lr=lr1)
scheduler = CosineLRScheduler(optimizer, T_max=80)

for epoch in range(80):
    for X_k, X_kp1 in train_loader:
        pred = model(X_k)
        loss = mse_loss(pred, X_kp1)
        loss.backward()
        optimizer.step(); optimizer.zero_grad()
    scheduler.step()

# Fine-tuning (two-step)
optimizer = AdamW(model.parameters(), lr=lr2)
scheduler = CosineLRScheduler(optimizer, T_max=50)

for epoch in range(50):
    for X_k, X_kp1, X_kp2 in train_loader_2step:
        X1 = model(X_k)          # step 1
        X2 = model(X1.detach())  # step 2 (can detach or not, paper sums both losses)
        loss1 = mse_loss(X1, X_kp1)
        loss2 = mse_loss(X2, X_kp2)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step(); optimizer.zero_grad()
    scheduler.step()
```

---

### 组件 C：降水诊断 AFNO 模型

- **子任务**
  - 给定 backbone 预测的 20 变量场，诊断 6 小时累计 TP；
  - 处理 TP 稀疏、长尾分布特性。

- **输入 / 输出**
  - 输入：$X\in\mathbb{R}^{B\times C\times H\times W}$，backbone 输出；
  - 输出：$TP\in\mathbb{R}^{B\times 1\times H\times W}$，6 小时累计降水深度。

- **伪代码**

```python
class PrecipModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = AFNOBackboneForTP(cfg)  # similar to main backbone but tailored
        self.conv_out = nn.Conv2d(
            in_channels=cfg.embed_dim,
            out_channels=1,
            kernel_size=3,
            padding="same"
        )

    def forward(self, x):
        # x: (B, C, H, W) prognostic variables
        h = self.backbone(x)
        tp_log = self.conv_out(h)         # (B, 1, H, W)
        tp_log = F.relu(tp_log)           # non-negative in log-space
        return tp_log

# training loop
for epoch in range(25):
    for X_k, TP_true in train_loader_tp:
        tp_true_log = torch.log1p(TP_true / 1e-5)
        tp_pred_log = precip_model(X_k)
        loss = mse_loss(tp_pred_log, tp_true_log)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
```

---

### 组件 D：自回归推理与集合预报

- **子任务**
  - 从单一 ERA5 初始场生成多步自回归预测；
  - 通过扰动初始场生成大规模集合预报，并计算 ensemble mean。

- **单轨迹自回归推理**

```python
# Pseudocode: Free-running autoregressive inference

def forecast_single(model, X_init, n_steps=28):  # 7 days
    preds = []
    X_curr = X_init
    for _ in range(n_steps):
        X_next = model(X_curr)
        preds.append(X_next)
        X_curr = X_next
    return torch.stack(preds, dim=1)  # (B, n_steps, C, H, W)
```

- **集合预报**
  - 采用高斯噪声扰动已标准化的初始场：
    $$
    X^{(e)}(k) = \hat{X}_{true}(k) + \sigma \xi,\;\xi\sim \mathcal{N}(0,1),\;\sigma=0.3
    $$

```python
# Pseudocode: Ensemble forecast

def generate_ensemble_inits(X_true, E=100, sigma=0.3):
    X_std = standardize(X_true)  # zero mean, unit var
    noise = sigma * torch.randn(E, *X_std.shape[1:], device=X_true.device)
    X_ens = X_std.unsqueeze(0) + noise
    return X_ens  # (E, C, H, W)

# batched inference: ensemble dimension as batch
X_ens0 = generate_ensemble_inits(X0, E=100)
Y_ens = forecast_single(model, X_ens0, n_steps=28)  # (E, steps, C, H, W)
Y_mean = Y_ens.mean(dim=0)  # ensemble mean over E
```

---

### 组件 E：评估与对比（ACC, RMSE, RQE）

- **子任务**
  - 对指定变量/层/区域，按 GraphCast/DLWP/WeatherBench 等协议计算 ACC/RMSE；
  - 区分 land/sea，计算 $\mathrm{ACC}_{land}, \mathrm{ACC}_{sea}$；
  - 对极端值使用 RQE 对比 FourCastNet 与 IFS。

- **伪代码（轮廓）**

```python
# Pseudocode: ACC & RMSE

def compute_lat_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()

def acc(pred, true, clim, lat_weights):
    # pred,true,clim: (T, H, W)
    a = true - clim
    b = pred - clim
    w = lat_weights[:, None]
    num = (w * a * b).sum(axis=(-2, -1))
    den = np.sqrt((w * a**2).sum(axis=(-2, -1)) *
                  (w * b**2).sum(axis=(-2, -1)))
    return (num / den).mean()  # average over time

def rmse(pred, true, lat_weights):
    diff2 = (pred - true) ** 2
    w = lat_weights[:, None]
    return np.sqrt((w * diff2).mean(axis=(-2, -1))).mean()

# Land/sea ACC

def masked_acc(pred, true, clim, lat_weights, mask):
    a = (true - clim) * mask
    b = (pred - clim) * mask
    w = lat_weights[:, None] * mask
    num = (w * a * b).sum(axis=(-2, -1))
    den = np.sqrt((w * a**2).sum(axis=(-2, -1)) *
                  (w * b**2).sum(axis=(-2, -1)))
    return (num / den).mean()
```

---

### 组件 F：与 DLWP / 其他 DL 模型对比

- **子任务**
  - 将 FourCastNet 与 IFS 输出降采样到 2°，使用与 DLWP 一致的 ACC/ RMSE 定义（基于日平均 climatology）评估；
  - 证明在相同分辨率下 FourCastNet 的优势，并强调其原生高分辨率能力。

- **伪代码**

```python
# Pseudocode: downsampling & comparison

def downsample_field(field, factor=8):
    # field: (T, H, W)
    H_new, W_new = H // factor, W // factor
    return resize_bilinear(field, (H_new, W_new))

Z500_fcn_ds = downsample_field(Z500_fcn)
Z500_ifs_ds = downsample_field(Z500_ifs)

acc_fcn = acc(Z500_fcn_ds, Z500_true_ds, clim_ds, lat_w_ds)
acc_dlwp = load_dlwp_acc()
# compare acc_fcn vs acc_dlwp
```

---

## 小结

- FourCastNet 通过 AFNO+ViT 主干，在 0.25° 全球高分辨率下，对 20 个关键大气变量实现了接近 IFS 的短–中期预报技巧，并在降水、近地面风等小尺度变量上具备明显优势；
- 独立的降水诊断模型成功在全球尺度上实现对 6 小时累计降水的高分辨率 DL 预报，首次在该尺度上与 IFS 可比；
- 极高的推理速度和低能耗使得 FourCastNet 能在秒级时间内生成 100–1000 成员的大集合预报，极大拓展了概率预报和不确定性分析的空间；
- 在极端事件（飓风、大气河、极端降水）和风能资源预测等应用中，FourCastNet 展示出强大的应用潜力，为后续基于 AFNO 的更高分辨率、物理约束和同化增强版本奠定了基础。