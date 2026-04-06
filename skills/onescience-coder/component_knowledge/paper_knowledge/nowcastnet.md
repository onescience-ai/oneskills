# NowcastNet 模型知识提取

信息来源：Zhang et al., “Skilful nowcasting of extreme precipitation with NowcastNet”（内容严格基于原文，不作主观推断；对文中未给出的细节标明“未在文中明确给出”）。

---

## 维度一：整体模型与任务知识

### 1. 任务与问题设定

- **任务类型：极端降水雷达回波的短时临近预报（nowcasting）**
  - 目标：在高空间分辨率和最长约 3 小时的时效内，对极端降水事件（尤其是对流和锋生系统）进行具有物理合理性的雷达回波预报。
  - 输入：过去一段时间的雷达合成反射率/降水率序列（美国 MRMS、以及中国 CMA 雷达网合成）。
  - 输出：未来一段时间（最长 3 小时）的雷达场序列，用于支持业务部门（预报员、风险管理）对极端降水和相关灾害的判断。

- **时空分辨率与覆盖范围**
  - 原始雷达：
    - 空间分辨率：约 0.01°（~1 km）
    - 时间分辨率：最高 5 min（论文中基于 MRMS 和中国复合雷达）。
  - 训练/评估中的统一设置：
    - 为权衡计算成本与技巧，采用 10 min 时间步长；空间上在原 0.01° 网格基础上适当下采样（保持 1–2 km 级分辨率）。
    - 训练 crop：空间尺寸 256 × 256，时间长度 270 min（9 过去 + 18 未来，10 min 一步）。
    - 测试 crop：空间尺寸 512 × 512，时间长度同上。
  - 论文中展示的大场景示例：2048 km × 2048 km 视野内的高分辨率 nowcasting（通过多 crop 拼接/滑动窗口实现）。

- **时间窗口与序列设定**
  - 输入长度：$T_0 = 9$（9 个过去时间步，覆盖 90 min）。
  - 输出长度：$T = 20$（用于训练），实际评估使用前 18 帧，对应 3 小时时效。
  - 记号：
    - 过去：$\mathbf{x}_{-T_0:0}$（含当前时刻）。
    - 未来：$\mathbf{x}_{1:T}$。

- **物理与统计目标**
  - 在保证 **对流尺度细节（1–2 km）** 的同时，保持 **中尺度结构（~20 km）** 的物理合理演变；
  - 在 3 小时内维持较高的 CSI（Critical Success Index）和真实降水谱（Power Spectral Density, PSD）；
  - 能对极端事件（龙卷、飑线、强对流、台风雨带等）给出有用预报，并获得专业预报员的主观高评价。

---

### 2. 传统方案与现有方法的痛点

- **数值天气预报（NWP）+ 雷达同化**
  - 优点：基于原始动力方程，物理一致性高；
  - 局限：
    - 预报时效和更新周期一般为小时级；
    - 分辨率主要为中尺度（> 10 km），难以解析对流细节；
    - 对临近预报（10 min–3 h）需求，更新频率不足。

- **传统雷达平流/外推方法（如 DARTS, pySTEPS）**
  - 以连续方程为启发：
    - 显式估计运动场 $\mathbf{v}$ 和强度残差 $\mathbf{s}$；
    - 使用 Lagrangian 持续性模型迭代地将过去雷达场平流到未来。
  - 优点：在 1 h 内对中尺度平移过程具有一定技巧；
  - 局限：
    1. 常用实现不可导，难以嵌入端到端神经网络训练框架；
    2. 多采用稳态/线性假设，不足以表示降水的非线性演变；
    3. 自回归误差积累不可控，导致位置误差、结构丢失，超过 ~1 h 技巧快速衰减。

- **纯数据驱动深度学习方法（PredRNN、DGMR 等）**
  - PredRNN：卷积 RNN，以 CSI 等格点指标在低/中等降水强度下表现良好；
  - DGMR：深度生成模型（GAN），可生成时空一致的雷达场，并提供 ensemble 不确定性表征；
  - 局限：
    - 只依赖雷达数据，未显式编码物理守恒约束；
    - 在强对流/极端降水情形下出现 **不自然的运动与强度演变**、大位置误差、云团快速消散等问题；
    - 难以保证多尺度结构（中尺度+对流尺度）的同时保真。

---

### 3. NowcastNet 的整体建模范式

- **核心思想：物理演化 + 条件生成 的统一框架**
  - 将 **物理启发的 2D 连续方程** 嵌入可微分的 **神经演化网络（evolution network, 参数 $\phi$）**，学习平流与强度残差；
  - 再在其上叠加 **条件生成网络（generative network, 参数 $\theta$）**，以 GAN 方式生成对流尺度细节；
  - 使用 **物理条件化（physics-conditional）机制** 进行多尺度解耦和融合：
    - 演化网络提供 20 km 尺度的物理一致 mesoscale 预测 $\mathbf{x}_{1:T}''$；
    - 生成网络在此基础上，利用雷达细节和随机潜变量 $\mathbf{z}$ 生成 1–2 km 对流尺度结构 $\hat{\mathbf{x}}_{1:T}$。

- **整体概率形式**
  $$
  P(\hat{\mathbf{x}}_{1:T} \mid \mathbf{x}_{-T_0:0}, \phi; \theta)
  = \int P(\hat{\mathbf{x}}_{1:T} \mid \mathbf{x}_{-T_0:0},
           \phi(\mathbf{x}_{-T_0:0}), \mathbf{z}; \theta) P(\mathbf{z})\,d\mathbf{z},
  $$
  - $\mathbf{z}$：标准高斯潜变量，使 NowcastNet 支持 ensemble 生成，刻画对流混沌不确定性。

- **多尺度分解与条件化机制**
  - 演化网络：在约 20 km 尺度上实现物理一致的平流 + 强度演变，得到 $\mathbf{x}_{1:T}''$；
  - 生成网络：
    - 通过 **Spatially Adaptive Normalization（类似 SPADE）** 将 $\mathbf{x}_{1:T}''$ 作为条件注入解码器各层；
    - 在特征归一化后，用从 $\mathbf{x}_{1:T}''$ 计算的空间相关均值/方差替换当前特征的统计量，实现物理引导的细节生成；
  - 这样可以 **显式解耦 20 km 以上 mesoscale 平移** 与 **1–2 km 对流细节**，减弱多尺度误差的上下传递。

- **训练范式**
  1. 先训练演化网络（物理部分），独立优化 $J_{\text{evolution}}$；
  2. 再固定演化网络，训练生成网络 + 时序判别器的 GAN：
     - 对抗损失 $J_{\text{adv}}$，推动生成细节；
     - 池化正则 $J_{\text{pool}}$，在空间池化尺度上约束 ensemble 统计与观测一致；
  3. 在中国数据上通过 transfer learning 微调，两部分网络学习率区分设置，防止遗忘物理知识。

---

### 4. 模型架构与关键公式

#### 4.1 物理基础：二维连续方程与演化算子

- **二维连续方程（经修正适配降水演变）**
  $$
  \frac{\partial \mathbf{x}}{\partial t} + (\mathbf{v}\cdot\nabla)\mathbf{x} = \mathbf{s}.
  $$
  - $\mathbf{x}$：降水率（由雷达反射率转换）、
  - $\mathbf{v}$：运动场（风矢量的投影，或抽象运动场），
  - $\mathbf{s}$：强度残差场（增长/衰减等非平流机理）。
  - 表示：一个时间步内的雷达场可视为 **平流** + **强度加法** 的合成。

- **神经演化算子**
  - 输入：当前（或上一时间步）预测雷达场 $\mathbf{x}_{t-1}''$ 以及所有未来步的运动场 $\mathbf{v}_{1:T}$ 和残差 $\mathbf{s}_{1:T}$；
  - 单步演化：
    1. 半拉格朗日反向平流：
       $$
       \mathbf{x}_t' = \mathcal{A}(\mathbf{x}_{t-1}''; \mathbf{v}_t),
       $$
       其中 $\mathcal{A}$ 为 backward semi-Lagrangian 算子。
    2. 强度残差叠加：
       $$
       \mathbf{x}_t'' = \mathbf{x}_t' + \mathbf{s}_t.
       $$
  - 实现细节：
    - 为减少多次双线性插值带来的模糊，**前向平流使用 nearest interpolation**；
    - 但 nearest 不可导，因此为优化运动场，额外计算一份双线性插值 $\mathbf{x}_t^{\prime}\_{\mathrm{bili}}$ 只用于损失项。

- **演化网络结构**
  - 主干：二路径 U-Net：
    - 共享 encoder：提取 $\mathbf{x}_{-T_0:0}$ 的上下文特征；
    - motion decoder：预测 $\mathbf{v}_{1:T}$；
    - intensity decoder：预测 $\mathbf{s}_{1:T}$；
  - 归一化：所有卷积层使用 spectral normalization；
  - skip 连接：在 U-Net 的跳连中，将所有输入/输出序列在时间维上拼接（提升对完整时序的建模能力）。

- **演化网络损失：累积损失 + 运动正则**
  - 累积损失（accumulation loss）：
    $$
    J_{\text{accum}} = \sum_{t=1}^{T}
    \Big( L_{\mathrm{wdis}}(\mathbf{x}_t, (\mathbf{x}_t')_{\mathrm{bili}})
        + L_{\mathrm{wdis}}(\mathbf{x}_t, \mathbf{x}_t'')\Big),
    $$
    其中加权 $L_1$ 距离：
    $$
    L_{\mathrm{wdis}}(\mathbf{x}_t, \mathbf{x}_t')
    = \big\|(\mathbf{x}_t - \mathbf{x}_t') \odot \mathbf{w}(\mathbf{x}_t)\big\|_1,\quad
    \mathbf{w}(x) = \min(24, 1 + x).
    $$
    - 权重随降水强度增加，缓解 log-normal 分布导致的极端降水样本稀疏问题；
  - 运动正则项：
    $$
    J_{\text{motion}} = \sum_{t=1}^T
    \Big(\|\nabla \mathbf{v}_t^1 \odot \sqrt{\mathbf{w}(\mathbf{x}_t)}\|_2^2
        + \|\nabla \mathbf{v}_t^2 \odot \sqrt{\mathbf{w}(\mathbf{x}_t)}\|_2^2\Big),
    $$
    - 通过 Sobel 滤波近似空间梯度，鼓励在高降水区域上的运动场更平滑（反映大尺度系统寿命更长的经验事实）。
  - 总目标：
    $$
    J_{\text{evolution}} = J_{\text{accum}} + \lambda J_{\text{motion}},\quad \lambda = 10^{-2}.
    $$
  - 为提高数值稳定性，每一步演化后 **截断时间步间梯度（stop-gradient）**，避免多次插值导致的端到端反向传播不稳定。

- **演化网络训练配置**
  - 采样：输入 9 帧，预测 20 帧（前 18 帧用于评估）。
  - 优化器：Adam，batch size = 16，初始 lr = 1e-3；
  - 学习率调度：在 3×10^5 次迭代内，从 1e-3 下降到 1e-4（2×10^5 步时衰减）；
  - 其它未在文中明确给出的细节（如卷积核大小、通道数等）不展开。

#### 4.2 生成网络与物理条件化机制

- **生成网络输入/输出**
  - 输入：
    - 过去雷达序列 $\mathbf{x}_{-T_0:0}$；
    - 演化网络输出的 20 km 尺度预测 $\mathbf{x}_{1:T}''$（上采样/下采样到各层所需分辨率）；
    - 潜在高斯向量 $\mathbf{z} \sim \mathcal{N}(0, I)$。
  - 输出：未来 1–2 km 尺度雷达序列 $\hat{\mathbf{x}}_{1:T}$。

- **结构：U-Net 型 encoder–decoder**
  - nowcast encoder：结构与演化 encoder 类似，将 $[\mathbf{x}_{-T_0:0}, \mathbf{x}_{1:T}'']$ 在通道维拼接后编码为多尺度特征；
  - nowcast decoder：独立的卷积解码器，通过 skip 连接融合 encoder 特征，并在各层施加物理条件化与噪声信息。

- **噪声投影器（noise projector）**
  - 将向量 $\mathbf{z}$ 投影为较小空间尺寸（如 H/8 × W/8）的特征图，与 encoder 特征同形；
  - 通过数层卷积/上采样构成，用于在 decoder 中注入随机性，实现 ensemble 预报。

- **空间自适应归一化（Spatially Adaptive Normalization, Spatial Norm）**
  - 对 decoder 的每一层特征 $F$：
    1. 用 instance normalization 去除通道内均值/方差：$\tilde{F}$；
    2. 将 $\mathbf{x}_{1:T}''$ 通过池化调整到与当前层同一空间尺寸，并拼接到 $\tilde{F}$ 上；
    3. 使用一个两层小卷积网络从条件特征中预测新的通道均值/方差 $(\mu, \sigma)$；
    4. 以 $(\mu, \sigma)$ 恢复尺度：$F' = \sigma \odot \tilde{F} + \mu$。
  - 这样在不破坏演化网络空间结构的前提下，将其输出注入生成网络，实现 **物理知识的条件化生成**。

- **时序判别器（temporal discriminator）**
  - 输入：真实序列 $\mathbf{x}_{1:T}$ 与生成序列 $\hat{\mathbf{x}}_{1:T}$；
  - 首层采用多种时间核长（4 到全时长）的 3D 卷积以捕捉多时间尺度一致性；
  - 后续为多层 3D/2D 卷积网络，所有层使用 spectral normalization；
  - 输出：对输入序列为真/假的置信分数。

- **GAN 损失**
  - 判别器目标：
    $$
    J_{\text{disc}} = L_{\mathrm{ce}}(D(\mathbf{x}_{1:T}), 1)
                     + L_{\mathrm{ce}}(D(\tilde{\mathbf{x}}_{1:T}), 0),
    $$
  - 生成器（nowcast decoder）对抗损失：
    $$
    J_{\text{adv}} = L_{\mathrm{ce}}(D(\hat{\mathbf{x}}_{1:T}), 1).
    $$
  - 池化正则（pool regularization）：
    - 对每个样本从 $k$ 个潜变量 $\{\mathbf{z}_i\}_{i=1}^k$ 生成 $k$ 条预报 $\hat{\mathbf{x}}_{1:T}^{\mathbf{z}_i}$；
    - 在空间维度上使用 kernel=5, stride=2 的 max-pooling $Q$，得到 coarse 版本：
      $$
      Q(\mathbf{x}_{1:T}),\quad Q(\hat{\mathbf{x}}_{1:T}^{\mathbf{z}_i});
      $$
    - 定义加权距离：
      $$
      J_{\text{pool}} = L_{\mathrm{wdis}}\Big(
        Q(\mathbf{x}_{1:T}),
        \frac{1}{k}\sum_{i=1}^k Q(\hat{\mathbf{x}}_{1:T}^{\mathbf{z}_i})
      \Big).
      $$
    - 该正则在空间池化尺度上对 ensemble 均值施压，使其统计特性接近观测，容忍小尺度混沌不确定性。
  - 生成网络总损失：
    $$
    J_{\text{generative}} = \beta J_{\text{adv}} + \gamma J_{\text{pool}},
    $$
    其中 $k=4, \; \beta = 6, \; \gamma = 20$。

- **生成网络训练配置**
  - 输入/输出长度同演化网络（T0=9, T=20）；
  - 优化器：Adam，batch size = 16；
  - 初始 lr = 3×10^-5（encoder/decoder/discriminator 相同尺度）；
  - 训练迭代：5×10^5 步；
  - 其它结构超参（通道宽度、block 数量）详见原文 Extended Data Fig. 1，未在此展开。

#### 4.3 迁移学习与训练流程

- **总体流程**
  1. 在美国 MRMS 数据集上预训练演化网络 + 生成网络（完整 NowcastNet）。
  2. 在中国雷达数据上进行 fine-tuning：
     - 使用相同损失 $J_{\text{evolution}}$ 与 $J_{\text{generative}}$；
     - 对 **演化网络使用比生成网络小 10 倍的学习率**，以避免遗忘通用物理知识；
     - 反向传播时，梯度在两部分之间解耦（decoupled backprop），使其各自收敛。

- **微调配置**
  - 优化器：Adam；
  - 迭代数：2×10^5；
  - 其它细节（具体学习率数值等）未在文中逐一给出。

---

### 5. 数据集与采样策略

- **USA 数据集（MRMS）**
  - 覆盖区域：20°N–55°N, 130°W–60°W；
  - 原始网格：3500 × 7000，0.01° 分辨率；
  - 时间范围：2016–2021；
    - 训练：2016–2020；
    - 测试：2021；
    - 验证：训练集中每月第一天的样本。
  - 预处理：
    - 时间分辨率设为 10 min；
    - 空间下采样为原始分辨率的一半；
    - 缺测赋为负值，并在评估时遮蔽。

- **China 数据集（CMA 雷达合成）**
  - 覆盖区域：17°N–53°N, 96°E–132°E；
  - 网格：3584 × 3584，0.01° 分辨率；
  - 时间范围：
    - 训练：2019-09-01 至 2021-03-31；
    - 测试：2021-04-01 至 2021-06-30（汛期，高度富含极端降水）；
  - 时间/空间下采样与雨率上限设定与 USA 数据集保持一致。

- **重要性采样（importance sampling）构造训练/测试集**
  - 为强调极端降水事件，对每个 spatiotemporal crop 定义接受概率：
    $$
    \Pr(\mathbf{x}_{-T_0:T}) = \sum_{t=-T_0}^{T} \|\mathbf{g}(\mathbf{x}_t)\|_1 + \epsilon,
    $$
    - 对训练集：$g(x) = 1 - e^{-x}$（仅在有效格点上，缺测为 0）；
    - 对测试集：$g(x) = x$（更直接地反映降水总量）；
  - 训练时采用层次采样（先采样整幅，再采样 crop）；
  - 测试集中再从这些 crop 中选取 **极端事件子集**（≥20 mm h^-1 阈值），用于专业气象员评价。

- **均匀采样协议**
  - 为全面评估轻–重降水，在补充实验中也采用 **不带权的均匀采样** 协议（样本数扩大约 3 倍），结果见补充材料，本文不细述。

---

### 6. 评估指标与结果

- **核心定量指标**
  1. **CSI with neighbourhood（邻域 CSI）**：
     - 在 16, 32, 64 mm h^-1 等降水阈值下评估命中率与假警率；
     - 使用邻域措施缓解微小位置偏差带来的惩罚；
  2. **PSD（Power Spectral Density）**：
     - 在多种空间波长下比较模型预报与雷达观测的降水谱；
     - 衡量多尺度结构（大尺度 mesoscale vs 小尺度 convective）的一致性。

- **案例分析（代表性事件）**
  - 美国 2021-12-11 龙卷爆发事件：
    - 大范围强对流带（convective fine line），伴随龙卷和直线风暴；
    - pySTEPS：细节清晰但位置误差大，线状回波扭曲；
    - PredRNN：轮廓模糊，多尺度结构丢失；
    - DGMR：局地对流细节尚可，但出现不自然的云团消散和位置偏移；
    - NowcastNet：在 1–3 h 内准确维持强对流线的位置和形状，高阈值 CSI 和全尺度 PSD 明显优于其它方法。
  - 中国 2021-05-14 江淮一带强降水：
    - 三个演化路径不同的对流单体/飑线，伴随红色暴雨预警；
    - 仅 NowcastNet 能在 3 h 时效内给出三单体演变的大致合理形态与路径，其它方法明显过度消散或错位；
    - 定量上，在高阈值 CSI 和多尺度 PSD 上都有显著优势。

- **专业气象员主观评价**
  - 评估协议：沿用 UK Met Office 框架，扩展为 posterior / prior 两类评估：
    - posterior：预报员在看到未来观测的前提下客观打分；
    - prior：只给过去，不给未来观测，模拟真实业务场景下的主观模型选择；
  - 参与者：来自中国中央和 23 个省级台站的 62 名高级预报员，每人随机评估 15 个极端事件个例（USA/China 各 1200 个事件子集）。
  - 结果（first-choice 比例及 95% 置信区间）：
    - posterior：
      - USA：NowcastNet 为首选的比例约 75.8%（[72.1, 79.3]）；
      - China：约 67.2%（[63.1, 71.1]）；
    - prior：
      - USA：约 71.9%（[66.6, 76.8]）；
      - China：约 64.4%（[58.9, 69.7]）。
  - 结论：在有/无未来观测参考的两种情形下，NowcastNet 均显著优于包括 pySTEPS、PredRNN、DGMR 在内的现有方法（p < 1e-4）。

- **整体定量结果（CSI 与 PSD）**
  - 在极端降水子集上，NowcastNet 在所有 lead time（1–3 h）和高阈值（≥16 mm h^-1 等）下的 CSI 均为最佳或并列最佳；
  - PSD 上，NowcastNet 在所有波长上均最接近雷达观测，意味着其预报在大尺度与小尺度上均保持正确的能谱分布，而其它方法要么过度平滑（谱偏弱），要么在某些尺度上失真。

---

### 7. 创新点与局限

- **创新点**
  1. **可微的物理演化网络**：
     - 将 2D 连续方程以神经演化算子形式实现，学习运动场与强度残差，并支持端到端梯度优化；
  2. **多尺度解耦的物理条件化生成框架**：
     - 通过 Spatial Norm 将物理演化输出作为条件，显式区分 20 km 尺度的 mesoscale 演化与 1–2 km 对流细节；
  3. **面向极端降水的权重与正则设计**：
     - 使用雨率加权的 L1 距离和基于 Sobel 梯度的运动正则，使模型更加关注重降水区域的演化与运动；
  4. **时序判别器 + 空间池化正则**：
     - 时序判别器约束整个轨迹的时空一致性；
     - 池化正则在 ensemble 统计层面对齐观测，缓解纯 GAN 与物理演化之间的冲突；
  5. **大规模迁移学习实践**：
     - 利用 USA 大数据集预训练，再迁移到更复杂但数据较少的中国天气系统，展示了物理知识的可迁移性；
  6. **系统性的专家评估**：
     - 通过 62 名高级预报员在多国数据上的主观评价，系统证明模型在业务价值上的优势。

- **局限**
  - 物理完备性：
    - 当前仅显式编码质量守恒相关的连续方程，动量、能量等其它守恒律尚未纳入；
  - 输入变量单一：
    - 只使用雷达降水/反射率，未融合卫星、多普勒风廓线、地面观测等多源信息；
  - 预报时效：
    - 技巧评估集中在 3 小时内，长期演化与不稳定性未被充分研究；
  - 业务落地：
    - 论文侧重研究性质的评估；将其嵌入实时业务系统仍需处理数据延时、系统稳定性等工程问题；
  - 未来气候与泛化：
    - 针对显著不同的未来气候场景（例如气候变暖后的统计分布偏移），模型泛化能力有待进一步验证。

---

## 维度二：基础组件 / 模块级知识

> 说明：以下伪代码基于论文描述提炼出的“高保真伪代码骨架”，用于刻画模块间的数据流与核心计算逻辑；不代表作者公开实现的完整细节，也不包含论文中未给出的主观推断实现。

### 组件 A：神经演化网络（Evolution Network）

- **子任务**
  - 从过去雷达序列 $\mathbf{x}_{-T_0:0}$ 中，联合学习：
    - 未来各步运动场 $\mathbf{v}_{1:T}$；
    - 未来各步强度残差 $\mathbf{s}_{1:T}$；
    - 并通过神经演化算子得到 20 km 尺度的雷达预报 $\mathbf{x}_{1:T}''$。

- **输入 / 输出**
  - 输入：$x\_hist \in \mathbb{R}^{B \times T_0 \times H \times W}$（过去 9 帧雷达）；
  - 输出：
    - $v\_{1:T} \in \mathbb{R}^{B \times T \times 2 \times H \times W}$（水平运动场两个分量）；
    - $s\_{1:T} \in \mathbb{R}^{B \times T \times 1 \times H \times W}$（强度残差）；
    - $x\_{1:T}^{\prime\prime} \in \mathbb{R}^{B \times T \times 1 \times H \times W}$（演化后雷达场）。

- **伪代码：演化算子**

```python
# Pseudocode: Neural evolution operator (per sample)

def advect_semi_lagrangian_nearest(x_prev, v_t):
    """Backward semi-Lagrangian advection with nearest interpolation.
    x_prev: (H, W), v_t: (2, H, W)
    """
    H, W = x_prev.shape
    # grid of destination points
    y, x = torch.meshgrid(
        torch.arange(H, device=x_prev.device),
        torch.arange(W, device=x_prev.device),
        indexing="ij",
    )
    # backward departure points
    dy = v_t[0]  # vertical component
    dx = v_t[1]  # horizontal component
    y_dep = torch.clamp((y - dy).round().long(), 0, H - 1)
    x_dep = torch.clamp((x - dx).round().long(), 0, W - 1)
    return x_prev[y_dep, x_dep]


def advect_semi_lagrangian_bilinear(x_prev, v_t):
    """Same as above but with bilinear interpolation for gradient flow.
    This is only used to build (x_t')_bili for the accumulation loss.
    """
    # Implementation uses grid_sample or custom bilinear; omitted here.
    raise NotImplementedError  # 论文未给出具体代码细节


def evolution_operator(x0, v_seq, s_seq):
    """Iteratively evolve radar field using neural evolution operator.
    x0: (H, W) initial field, v_seq/s_seq: lists length T.
    Returns: list of x_t'' for t=1..T, and x_t'_bili for loss.
    """
    x_prev = x0
    xs_dd = []       # x_t'' sequence
    xs_bili = []     # (x_t')_bili for loss

    for t, (v_t, s_t) in enumerate(zip(v_seq, s_seq), start=1):
        # differentiable path for v_t via bilinear advection
        x_t_prime_bili = advect_semi_lagrangian_bilinear(x_prev, v_t)
        xs_bili.append(x_t_prime_bili)

        # forward prediction uses nearest to avoid excessive blur
        x_t_prime = advect_semi_lagrangian_nearest(x_prev, v_t)
        x_t_dd = x_t_prime + s_t
        xs_dd.append(x_t_dd)

        # stop gradient between time steps to improve stability
        x_prev = x_t_dd.detach()

    return xs_dd, xs_bili
```

- **伪代码：演化网络前向与损失**

```python
class EvolutionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = UNetEncoder(cfg)
        self.motion_decoder = MotionDecoder(cfg)
        self.intensity_decoder = IntensityDecoder(cfg)

    def forward(self, x_hist):
        # x_hist: (B, T0, H, W)
        feat = self.encoder(x_hist)  # shared context
        v_seq = self.motion_decoder(feat)    # (B, T, 2, H, W)
        s_seq = self.intensity_decoder(feat) # (B, T, 1, H, W)
        return v_seq, s_seq


def weighted_l1(x, y, w):
    # x, y: (...), w: same shape or broadcastable
    return torch.sum(torch.abs(x - y) * w)


def compute_evolution_loss(x_hist, x_future, model, lambda_motion=1e-2):
    """One training step loss for evolution network.
    x_hist: (B, T0, H, W), x_future: (B, T, H, W)
    """
    B, T, H, W = x_future.shape
    v_seq, s_seq = model(x_hist)  # shapes: (B, T, ...)
    loss_accum = 0.0
    loss_motion = 0.0

    for b in range(B):
        x0 = x_hist[b, -1]  # last input frame
        v_list = [v_seq[b, t] for t in range(T)]
        s_list = [s_seq[b, t, 0] for t in range(T)]  # drop channel dim
        xs_dd, xs_bili = evolution_operator(x0, v_list, s_list)

        x_prev = x0
        for t in range(T):
            x_true_t = x_future[b, t]
            w = torch.clamp(1 + x_true_t, max=24.0)

            # accumulation loss with bilinear and final field
            loss_accum += weighted_l1(x_true_t, xs_bili[t], w)
            loss_accum += weighted_l1(x_true_t, xs_dd[t],   w)

            # motion regularization using Sobel gradients
            vy = v_list[t][0]; vx = v_list[t][1]
            grad_vy = sobel_filter(vy)  # approx ∇v_y
            grad_vx = sobel_filter(vx)  # approx ∇v_x
            loss_motion += torch.sum((grad_vy**2 + grad_vx**2) * torch.sqrt(w))

    loss = loss_accum + lambda_motion * loss_motion
    return loss / B
```

> 注：`sobel_filter`、U-Net 具体结构等在原文 Extended Data 中有详细给出，但不影响此处逻辑骨架。

---

### 组件 B：生成网络与物理条件化（Generative Network + Physics Conditioning）

- **子任务**
  - 在演化网络 20 km 预测 $\mathbf{x}_{1:T}''$ 的基础上，条件生成 1–2 km 细节；
  - 引入潜在变量 $\mathbf{z}$，实现 ensemble 预报与不确定性刻画；
  - 利用时序判别器和池化正则学习对流尺度多样性与谱特性。

- **伪代码：Spatial Norm（物理条件化归一化）**

```python
class SpatialNorm(nn.Module):
    def __init__(self, num_features, cond_channels):
        super().__init__()
        # small conv net to predict (gamma, beta) from evolution output
        self.mlp = nn.Sequential(
            nn.Conv2d(cond_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 2 * num_features, kernel_size=3, padding=1),
        )
        self.inst_norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, cond):
        # x: (B, C, H, W) from decoder
        # cond: (B, C_cond, H, W) evolution outputs resized
        x_norm = self.inst_norm(x)
        h = self.mlp(cond)  # (B, 2C, H, W)
        gamma, beta = torch.chunk(h, 2, dim=1)
        return gamma * x_norm + beta
```

- **伪代码：生成网络主干**

```python
class NoiseProjector(nn.Module):
    def __init__(self, z_dim, out_channels, out_h, out_w):
        super().__init__()
        self.fc = nn.Linear(z_dim, out_channels * out_h * out_w)
        self.out_channels = out_channels
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, z):
        # z: (B, z_dim)
        h = self.fc(z)
        return h.view(-1, self.out_channels, self.out_h, self.out_w)


class GenerativeNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = NowcastEncoder(cfg)  # similar to evolution encoder
        self.decoder = NowcastDecoder(cfg)  # uses SpatialNorm internally
        self.noise_proj = NoiseProjector(
            z_dim=cfg.z_dim,
            out_channels=cfg.noise_c,
            out_h=cfg.noise_h,
            out_w=cfg.noise_w,
        )

    def forward(self, x_hist, x_evol, z):
        # x_hist: (B, T0, H, W)
        # x_evol: (B, T, H, W) evolution outputs (coarse scale)
        # z: (B, z_dim)
        enc_feat = self.encoder(x_hist, x_evol)
        noise_feat = self.noise_proj(z)
        y_hat = self.decoder(enc_feat, x_evol, noise_feat)
        # y_hat: (B, T, H, W)
        return y_hat
```

- **伪代码：时序判别器与 GAN 训练步骤（逻辑骨架）**

```python
class TemporalDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # multi-kernel 3D convs at first layer (temporal kernels: 4..T)
        self.first_multiscale = MultiKernel3DConv(cfg)
        self.backbone = DiscBackbone(cfg)  # 3D/2D convs with spectral norm

    def forward(self, seq):
        # seq: (B, T, H, W)
        x = self.first_multiscale(seq.unsqueeze(1))  # add channel dim
        return self.backbone(x)  # logits


def adversarial_step(gen, disc, evol_net, batch, cfg):
    x_hist, x_future = batch  # shapes: (B, T0, H, W), (B, T, H, W)

    # 1) 更新判别器
    with torch.no_grad():
        v_seq, s_seq = evol_net(x_hist)
        x0 = x_hist[:, -1]
        x_evol_dd, _ = batched_evolution(x0, v_seq, s_seq)  # (B, T, H, W)
        z = torch.randn(x_hist.size(0), cfg.z_dim, device=x_hist.device)
        x_hat = gen(x_hist, x_evol_dd, z)

    logits_real = disc(x_future)
    logits_fake = disc(x_hat.detach())
    loss_disc = ce_loss(logits_real, torch.ones_like(logits_real)) \
              + ce_loss(logits_fake, torch.zeros_like(logits_fake))
    loss_disc.backward(); disc_opt.step(); disc_opt.zero_grad()

    # 2) 更新生成网络
    z = torch.randn(x_hist.size(0), cfg.z_dim, device=x_hist.device)
    x_hat = gen(x_hist, x_evol_dd, z)
    logits_fake = disc(x_hat)
    loss_adv = ce_loss(logits_fake, torch.ones_like(logits_fake))

    # pool regularization with k ensemble members
    x_hat_ens = []
    for _ in range(cfg.k_ens):
        z_i = torch.randn_like(z)
        x_hat_i = gen(x_hist, x_evol_dd, z_i)
        x_hat_ens.append(x_hat_i)
    x_hat_ens = torch.stack(x_hat_ens, dim=0)  # (k, B, T, H, W)

    def pool(x):
        # spatial max pooling with kernel=5, stride=2
        return F.max_pool2d(x, kernel_size=5, stride=2)

    x_pool_true = pool(x_future.view(-1, 1, H, W))
    x_pool_ens = pool(x_hat_ens.mean(dim=0).view(-1, 1, H, W))

    w_true = torch.clamp(1 + x_pool_true, max=24.0)
    loss_pool = weighted_l1(x_pool_true, x_pool_ens, w_true)

    loss_gen = cfg.beta * loss_adv + cfg.gamma * loss_pool
    loss_gen.backward(); gen_opt.step(); gen_opt.zero_grad()
```

> 注：以上训练伪代码仅刻画损失结构与信息流向，具体的维度处理（时间维放在 batch 还是通道）、判别器结构等细节参见原文 Extended Data 说明。

---

### 组件 C：集成（ensemble）生成与不确定性刻画

- **子任务**
  - 通过多次采样潜变量 $\mathbf{z}$ 对同一输入 $\mathbf{x}_{-T_0:0}$ 生成 ensemble；
  - 在推理阶段不需要 pool 正则，只需前向传播即可获得 ensemble 轨迹。

- **伪代码：推理阶段 ensemble 生成**

```python
@torch.no_grad()
def forecast_ensemble(gen, evol_net, x_hist, n_members=20, T=18):
    """Generate ensemble nowcasts for a single batch of histories.
    x_hist: (B, T0, H, W)
    Returns: (n_members, B, T, H, W)
    """
    v_seq, s_seq = evol_net(x_hist)
    x0 = x_hist[:, -1]
    x_evol_dd, _ = batched_evolution(x0, v_seq, s_seq)  # (B, T, H, W)

    members = []
    for _ in range(n_members):
        z = torch.randn(x_hist.size(0), gen.cfg.z_dim, device=x_hist.device)
        x_hat = gen(x_hist, x_evol_dd, z)[:, :T]  # first 18 frames
        members.append(x_hat)
    return torch.stack(members, dim=0)
```

- **应用示例**
  - 对单次强对流事件生成 20–50 个 ensemble 成员，计算：
    - ensemble mean（平均降水/雷达反射率）；
    - ensemble spread（成员 dispersions）；
  - 辅助预报员评估不确定性：例如对飑线位置的 spread、极端降水范围的不确定区域等。

---

### 组件 D：迁移学习流程（USA → China）

- **子任务**
  - 在 USA 数据上获得具有强物理泛化能力的初始模型；
  - 在 China 数据上以较小步长微调，适应更复杂的地形与天气系统。

- **伪代码：迁移微调骨架**

```python
# Assume evolution_net, gen_net already pre-trained on USA dataset

# Different learning rates to prevent forgetting
optim = Adam([
    {"params": evolution_net.parameters(), "lr": lr_evol},
    {"params": gen_net.parameters(),       "lr": lr_gen},
])

for step, batch in enumerate(china_loader):
    x_hist, x_future = batch

    # evolution loss
    loss_evol = compute_evolution_loss(x_hist, x_future, evolution_net)

    # generative loss (adversarial + pool)
    loss_gen = compute_generative_loss(
        gen_net, evolution_net, disc, x_hist, x_future
    )

    # decoupled gradients: can choose to stop-grad between components
    loss = loss_evol + loss_gen
    loss.backward()
    optim.step(); optim.zero_grad()
```

> 注：论文中只给出“evolution 部分 lr 为 generative 的 1/10、迭代 2×10^5 步”等高层信息，具体实现方式如上伪代码所示是一种合理的对应骨架。

---

### 组件 E：数据采样与评估管线

- **子任务**
  - 按照重要性采样策略从完整雷达序列中构建训练/测试 crop；
  - 按指标协议计算 CSI with neighbourhood 与 PSD，并支持极端事件子集与均匀采样子集评估。

- **伪代码：构造训练/测试集（重要性采样）**

```python
def compute_accept_prob(crop_seq, mode="train"):
    # crop_seq: (T_total, H, W) rain rate sequence for one crop
    if mode == "train":
        g = 1.0 - torch.exp(-crop_seq.clamp(min=0.0))
    elif mode == "test":
        g = crop_seq.clamp(min=0.0)
    else:
        raise ValueError
    eps = 1e-6
    return g.sum() + eps


def build_dataset(full_frames, mode="train"):
    crops = []
    for seq in sliding_window(full_frames, window=270_min):
        for crop in spatial_crops(seq, size=(H_crop, W_crop), stride=32):
            p = compute_accept_prob(crop, mode)
            if random.uniform(0, 1) < normalize_prob(p):
                crops.append(crop)
    return crops
```

- **伪代码：CSI with neighbourhood（轮廓）**

```python
def csi_neighbourhood(pred, true, thr, radius):
    """Compute CSI with neighbourhood of given radius.
    pred,true: (T, H, W); thr: mm/h; radius: neighbourhood size.
    """
    pred_bin = (pred >= thr).float()
    true_bin = (true >= thr).float()

    # dilate both fields within radius to account for small location errors
    pred_dil = dilate(pred_bin, radius)
    true_dil = dilate(true_bin, radius)

    hits = (pred_dil * true_bin).sum()
    false_alarms = (pred_dil * (1 - true_bin)).sum()
    misses = ((1 - pred_dil) * true_bin).sum()

    return hits / (hits + false_alarms + misses + 1e-6)
```

- **伪代码：PSD 计算（轮廓）**

```python
def compute_psd(field):
    """Compute azimuthally averaged power spectral density.
    field: (H, W)
    """
    F = torch.fft.fft2(field)
    power = (F.real**2 + F.imag**2)
    # radially average power into bins of wavelength; details omitted
    return radial_average(power)
```

---

## 小结

- NowcastNet 将 2D 连续方程的物理框架与条件 GAN 生成网络结合，在 USA MRMS 与中国雷达数据上实现了对极端降水事件的 **3 小时高分辨率 nowcasting**，在 CSI 和 PSD 指标上系统优于包括 pySTEPS、PredRNN、DGMR 在内的主流方法；
- 演化网络通过可微分的神经演化算子联合学习运动场与强度残差，并配合权重与正则设计强化对重降水区域的物理一致演化；生成网络在其基础上利用 Spatial Norm 与时序 GAN 生成对流尺度细节，并可通过潜在变量实现 ensemble 预报；
- 借助重要性采样和迁移学习，NowcastNet 能在数据量有限但天气系统复杂的地区（如中国）保持较高预报技巧，并获得大规模专业气象员主观评估的支持，为后续引入更多物理约束、多源数据和更长时效预报提供了坚实基础。