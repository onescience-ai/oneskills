# XiHe 模型知识提取

信息来源：Wang et al., "XiHe: A Data-Driven Model for Global Ocean Eddy-Resolving Forecasting"（内容严格基于原文，不作主观推断；未在文中明确给出的实现细节在下文伪代码中以占位或注释标出）。

---

## 维度一：整体模型与任务知识

### 1. 任务与问题设定

- **任务类型：全球高分辨率数据驱动海洋预报（GOFS 替代）**
  - 目标：构建首个在全球 $1/12^\circ$（约 8–9 km）分辨率下、能解析中尺度涡（eddy-resolving）的数据驱动全球海洋预报模型 XiHe，在多变量、多深度层上给出数十天尺度的预报，并在准确度上超过现有主流数值全球海洋预报系统（GOFSs）。
  - 时间分辨率：逐日（1 天步长）；
  - 空间分辨率：$1/12^\circ$，对应 $W=4320$（经度）、$H=2041$（纬度）全球规则经纬网；
  - 预报提前量：
    - 主实验：1–10 天逐日预报；
    - 扩展实验：20、30、60 天预报，尤其关注 15 m 海流的长期技巧。

- **输入 / 输出变量与维度**
  - 预报目标变量（输出，$C_{out}=94$）：
    - 5 个海洋变量 × 23 层（共 115 通道，但文中只统计 23 层×4 变量 + SSH）：
      - 温度 $T$
      - 盐度 $S$
      - 海流 zonal 分量 $U$
      - 海流 meridional 分量 $V$
      - 垂向 23 个标准深度层：$0.49, 2.65, 5.08, 7.93, 11.41, 15.81, 21.60, 29.44, 40.34, 55.76, 77.85, 92.32, 109.73, 130.67, 155.85, 186.13, 222.48, 266.04, 318.13, 380.21, 453.94, 541.09, 643.57$ m；
    - Sea Surface Height (SSH)（海表高度）；
    - 输出张量：$\hat{\mathbf{X}}^{t+\tau} \in \mathbb{R}^{W\times H\times C_{out}}$，尺寸 $4320\times 2041\times 94$。
  - 模型输入变量（$C_{in}=96$）：
    - 海洋 2D 面场：
      - 海表温度 SST（来自 OSTIA 高分辨率卫星 SST）；
      - 海表高度 SSH；
    - 海洋三维变量（23 层）：
      - 海温 $T$（23 层）
      - 盐度 $S$（23 层）
      - 海流 U/V 分量（23 层）
    - 海气耦合外强迫：
      - ERA5 10 m 海表风 U10/V10（插值到 $1/12^\circ$ 和日尺度）；
    - 以上变量堆叠形成：
      $$
      \mathbf{X}^t \in \mathbb{R}^{W\times H\times C_{in}},\; W=4320, H=2041, C_{in}=96
      $$

- **预测问题形式化**
  - 给定某一日的海洋–大气状态张量 $\mathbf{X}^t$，XiHe 直接输出 $K$ 步前的所有预报：
    $$
    \hat{\mathbf{X}}^{t+\tau} = \mathcal{F}(\mathbf{X}^{t}, \theta), \quad \tau = 1,2,\dots, K
    $$
  - XiHe 为**非自回归**（non-autoregressive）预测：每个 lead time 的预测通过同一个模型结构直接从当前输入生成，而非逐步 AR 滚动；实际训练时是否一次性预测多步或逐步训练，论文仅给出损失定义为单步 MSE（文中未详述多步损失的实现细节）。

- **应用与关注场景**
  - 全局海温、盐度与流场：
    - 支持海洋环境监测、海洋资源开发等任务所需的多变量预报；
  - 15 m 深度洋流：
    - 在 10 天内 RMSE 明显小于 PSY4 等 GOFS；
    - XiHe 的 60 天洋流预报 RMSE 仍优于 PSY4 的 10 天预报，是论文的核心亮点之一；
  - 大尺度海流（Agulhas、黑潮、湾流等）：
    - 实验表明 XiHe 对主要西边界流与大洋环流的速度和方向再现良好；
  - 中尺度涡（mesoscale eddies, O(100 km)）：
    - 使用 SLA（海面高度异常）识别中尺度涡，对比 CMEMS 观测和 GLORYS12 再分析，XiHe 10 天内对涡的数目与轨迹保持较强一致性；

---

### 2. 数据、评估框架与对比基线

- **训练数据**
  - GLORYS12 再分析（CMEMS）：
    - 全球 $1/12^\circ$ 海洋再分析，50 垂向层；
    - 同化：沿轨 SLA、SST、SIC + in situ T/S 剖面；
    - 使用时间范围：1993–2020 年日尺度数据；
    - 作为 XiHe 的主训练数据集（目标变量与部分输入）；
  - ERA5 再分析海表风：
    - 10 m U/V 分量（U10/V10）；
    - 原始分辨率 $0.25^\circ$，插值到 $1/12^\circ$ 与 GLORYS12 的网格对齐；
  - OSTIA 卫星 SST：
    - 分辨率约 5 km，对应 $1/20^\circ$ 网格；
    - 提供高分辨率近实时 SST，用于提升 XiHe 对 SST 的预报技巧；
    - 同样插值到 $1/12^\circ$ 网格。

- **评估数据与框架**
  - GODAE OceanView IV-TT Class 4 框架：
    - 核心思想：将各预报系统输出插值到观测点位，使用统一观测进行对比评估；
    - 观测数据：
      - Argo 浮标 T/S 剖面；
      - USGODAE 漂流浮标：SST 与近表面流速；
      - Jason-1/2 与 Envisat ALTIMETER 沿轨 SLA；
    - 统一提供 climatology 与多系统预报结果，供对比评估使用。
  - 额外评估数据：
    - Global Tropical Moored Buoy Array（TAO/RAMA/PIRATA）：
      - 热带太平洋、印度洋、大西洋固定系泊浮标阵列，提供 T/S 剖面与其他要素；
      - 用于对比 XiHe 预报与 GLORYS12 再分析在 0–200 m 深度平均 RMSE。
    - CMEMS L4 海表高程与流场产品（SEALEVEL GLO PHY L4 MY 008 047）：
      - 基于多星组合 Altimeter 的栅格化 SLA 与地转流（geostrophic currents）；
      - 用于检验 XiHe 海表流场对地转/准地转大尺度过程的再现能力。

- **对比基线：主流数值 GOFSs**
  - PSY4（Mercator Ocean Physical System）：
    - 水平分辨率：$1/12^\circ$；
    - 垂直层数：50；
    - OGCM：NEMO 3.1；
    - 数据同化：SAM2 + 3DVAR；
    - 预报时长：10 天；
  - FOAM（UK Met Office）：
    - 分辨率：$1/4^\circ$，75 层；
    - NEMO 3.2 + NEMOVAR (3DVAR)；
    - 预报时长：6 天；
  - OceanMAPS (BLK, 澳大利亚)：
    - 分辨率：$1/10^\circ$，51 层；
    - OFAM3 (MOM5) + BODAS(EnOI)；
    - 预报时长：7 天；
  - GIOPS（加拿大）：
    - 分辨率：$1/4^\circ$，50 层；
    - NEMO 3.1 + SAM2；
    - 预报时长：10 天；
  - RTOFS：
    - 因 2019–2020 年仅提供 4 个月预报数据，文中未列详细对比。

- **IV-TT 评价指标**
  - 在观测点位上计算：
    - Bias；
    - RMSE：
      $$
      \mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(F_i - O_i)^2}
      $$
    - ACC（Anomaly Correlation Coefficient）：
      $$
      \mathrm{ACC} = \frac{\sum_{i=1}^{N}(F_i-C_i)(O_i-C_i)}{\sqrt{\sum_i(F_i-C_i)^2}\sqrt{\sum_i(O_i-C_i)^2}}
      $$
      其中 $F_i$ 为预报，$O_i$ 为观测，$C_i$ 为 climatology；
    - 论文中主要使用 RMSE 与 ACC 评估温度、盐度、SLA 与 15 m 深度流速。

---

### 3. 整体模型架构与关键机制

- **总体框架：层次化 Ocean-Specific Transformer**
  - 三大组件：
    1. Patch Partition 模块：
       - 使用核大小为 $p$ 的 2D 卷积（stride=$p$）将输入栅格 $W\times H\times C_{in}$ 划分为 $W'\times H'$ 的 patch，并映射到 $C$ 维嵌入空间；
       - 零填充保证 $W,H$ 可被 $p$ 整除；陆地区和缺测点预先填 0；
       - patch embedding 后接 LayerNorm 提升训练稳定性；
    2. Ocean-Specific Transformer：
       - 由 5 个 ocean-specific blocks + 一个 down-sampling block + 一个 up-sampling block 组成；
       - 每个 ocean-specific block：
         - 若为第 1、5 层：1 个 local SIE 模块 + 1 个 global SIE 模块；
         - 若为第 2–4 层：2 个连续的 local SIE 模块 + 1 个 global SIE 模块；
       - down-sampling 与 up-sampling：
         - 下采样参考 Swin Transformer 的 patch merging，将 $2\times2$ patch 合并为 1，特征维从 $C\to4C$ 再经线性层缩到 $2C$；
         - 上采样做逆操作，恢复回 $W'\times H'\times C$；
       - skip-connection：
         - 第 1 与第 5 ocean-specific block 之间做特征拼接或残差，提高信息流动；
    3. Patch Restoration 模块：
       - 利用 2D 反卷积（kernel=$p$, stride=$p$）将 $W'\times H'\times C$ 的特征图恢复到原始 $W\times H\times C_{out}$ 维度，输出预报场。

- **Local SIE 模块（局地空间信息提取）**
  - 基于 Swin Transformer 的 window-based multi-head self-attention (W-MSA)：
    - 将输入特征图 $\mathbf{Z}^{l-1}$ 按窗口大小 $M\times M$ 划分为不重叠窗口；
    - 在每个窗口内独立做多头自注意力，计算复杂度对数据规模为线性；
  - 数学形式：
    $$
    \hat{\mathbf{Z}}^{l} = \mathrm{W\text{-}MSA}(\mathrm{LN}(\mathbf{Z}^{l-1})) + \mathbf{Z}^{l-1}
    $$
    $$
    \tilde{\mathbf{Z}}^{l} = \mathrm{MLP}(\mathrm{LN}(\hat{\mathbf{Z}}^{l})) + \hat{\mathbf{Z}}^{l}
    $$
    - 注意力公式：
      $$
      \mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{Softmax}\big(\mathbf{Q}\mathbf{K}^T/\sqrt{D} + \mathbf{B}\big)\mathbf{V}
      $$
      其中 $D$ 为特征维度，$\mathbf{B}$ 为相对位置偏置矩阵。

- **Global SIE 模块（全局空间信息提取）**
  - 动机：仅使用 window attention 会阻断窗口之间的长程信息交流，不利于捕捉海洋 teleconnection（远程相关）结构；
  - 基于 Group Propagation ViT (GPViT) 思想，引入一组 learnable group vectors 表征一组 patch 的聚类中心：
    1. **Feature grouping**：
       - 随机初始化一组 group 向量 $\mathbf{G}^l$（数量为超参数）；
       - 通过 Multi-head Cross-Attention (MCA) 让 group vectors 作为 query，patch features 作为 key/value：
         $$
         \mathbf{G}'^{l} = \mathrm{Concat}_{h} \big(\mathrm{Attention}(W_h^Q \mathbf{G}^l_h, W_h^K\tilde{\mathbf{Z}}^l_h, W_h^V\tilde{\mathbf{Z}}^l_h)\big)
         $$
       - 在这一阶段，当地海洋特征根据与 group vectors 的相关性被聚合到若干 group，类似“海洋模式”聚类表示；
    2. **Group propagation**：
       - 使用 MLPMixer 在 group 维度和通道维度上对 $\mathbf{G}'^{l}$ 做 MLP 混合，实现 group 之间的信息传播与融合：
         $$
         \hat{\mathbf{G}}^{l} = \mathbf{G}'^{l} + \mathrm{MLP}_1(\mathrm{LN}(\mathbf{G}'^{l})^T)^T
         $$
         $$
         \tilde{\mathbf{G}}^{l} = \hat{\mathbf{G}}^{l} + \mathrm{MLP}_2(\mathrm{LN}(\hat{\mathbf{G}}^{l}))
         $$
       - 这一步在 group 之间传播全局 teleconnection 信息，使每个 group 能代表更全局的海洋动力学模式；
    3. **Feature ungrouping**：
       - patch features 反过来作为 query，group vectors 作为 key/value 做一次 cross-attention：
         $$
         \mathbf{U}^l = \mathrm{Concat}_{h}\big(\mathrm{Attention}(\tilde{W}_h^Q\tilde{\mathbf{Z}}_h^l, \tilde{W}_h^K\tilde{\mathbf{G}}_h^l, \tilde{W}_h^V\tilde{\mathbf{G}}_h^l)\big)
         $$
       - 将 $\mathbf{U}^l$ 与 $\tilde{\mathbf{Z}}^l$ 拼接并投影回原维度，得到 $\mathbf{Z}^{\bar{l}}$ 作为 global SIE 的输出；
       - 从而实现从局地窗口特征 → group 聚合 → 全局传播 → 反向下发到每个 patch 的信息闭环。

- **Ocean-land Masking（海陆掩膜机制）**
  - 问题：海洋数据中陆地面积占比约 29.2%，虽然在预处理时填 0，但仍参与注意力计算，既浪费算力又会干扰模型关注海洋区域；
  - 方案：
    1. 构造海陆掩膜矩阵 $M \in \mathbb{R}^{H\times W}$：海洋网格点赋值 1，陆地网格点赋值 0；
    2. 对 $M$ 进行同样的 patch partition 得到 patch 级掩膜 $M_1\in\mathbb{R}^{W'\times H'}$：
       - 如果某个 patch 内所有网格都是陆地，则该 patch 掩膜为 0，否则为 1；
       - 对下采样后的特征图生成更粗分辨率的掩膜 $M_2\in\mathbb{R}^{W'/2\times H'/2}$；
    3. 在 Local SIE 中：
       - 使用 $M_1$（block 1,5）与 $M_2$（block 2–4）对 attention score 做 masking，将陆地区 patch 的注意力得分强制为 0；
    4. 在 Global SIE 的 feature grouping：
       - 直接在 cross-attention 的 key/value 中**剔除**陆地区 patch，降低计算量并避免 group 被陆地噪声污染。
  - 效果：大幅降低注意力计算量（仅对 70.8% 海洋 patch 进行注意力计算），并令模型重点关注海洋动力学。

---

### 4. 训练与优化设置

- **损失函数**
  - 使用标准逐像素 Mean Square Error（MSE）损失：
    $$
    \mathcal{L}_{MSE} = \frac{1}{W\times H\times C_{out}} \sum_{i=1}^{W}\sum_{j=1}^{H}\sum_{c=1}^{C_{out}} \big(\hat{X}^{t+\tau}_{i,j,c} - X^{t+\tau}_{i,j,c}\big)^2
    $$
  - 文中损失定义为针对给定 lead time 的单步 MSE，未明示是否联合多步训练，如涉及多步损失叠加，这属于实现细节，文中未给出。

- **优化器与训练细节**
  - 框架：PyTorch；
  - 优化器：AdamW，参数：
    - $\beta_1 = 0.9$
    - $\beta_2 = 0.95$
    - 初始学习率 $5\times10^{-5}$；
  - DropPath：采用随 epoch 线性增大的 DropPath 比例，最大 0.2，用于防止过拟合；
  - 批大小：1（每 GPU），模型在多 GPU 训练；
  - 训练轮数：50 epochs。

---

### 5. 主要实验结果与结论

- **IV-TT Class 4 整体评估（2019–2020）**
  - 变量：
    - 温度（垂向平均）
    - 盐度（垂向平均）
    - 15 m 深度海流（U/V 分量）
    - SLA（海表高度异常）
  - Lead time：1–10 天（每日一步），另有 20/30/60 天扩展实验；
  - 综合结论：
    - 在 1–10 天全范围内，XiHe 在上述全部变量上**持续优于**所有对比的数值 GOFS（PSY4, GIOPS, FOAM, BLK）；
    - 以 6 天 lead（144 h）为例：
      - 垂向平均温度 RMSE 相比 FOAM, PSY4, GIOPS, BLK 的平均改进分别约为 9.94%, 11.10%, 8.94%, 24.70%；
      - 垂向平均盐度 RMSE 改进分别约为 5.90%, 9.77%, 11.52%, 22.98%。
  - 15 m 深度海流：
    - 比较 PSY4 与 XiHe 在 1–10 天 lead 上的 RMSE 增长斜率：
      - Zonal 分量（uo）：PSY4 斜率约 0.00189 m/(s·day)，XiHe 约 0.00064 m/(s·day)，约为 PSY4 的 1/3；
      - Meridional 分量（vo）：PSY4 约 0.00180 m/(s·day)，XiHe 约 0.00040 m/(s·day)，约为 PSY4 的 1/4.5；
    - XiHe 的 10 天 RMSE < PSY4 的 1 天 RMSE；
    - 进一步训练 20/30/60 天 lead 的 XiHe：
      - uo 的 RMSE 分别为 0.2038, 0.2093, 0.2165，小于 PSY4 10 天 RMSE (0.2214)；
      - vo 的 RMSE 分别为 0.1955, 0.2008, 0.2062，小于 PSY4 10 天 RMSE (0.2149)。

- **ACC 评估**
  - XiHe 的温度与盐度 ACC 在 20 天 lead 时分别约为 0.8743 与 0.6420，仍高于他系统 10 天 lead 的 ACC；
  - 显示出 XiHe 在长 lead 预报上的稳健性。

- **空间分布与时间序列 RMSE**
  - XiHe vs in situ T/S（IV-TT）：
    - 在大部分公海区域，温度/盐度垂向平均 RMSE 分别低于 0.5 ℃ 与 0.2 PSU；
    - 高 RMSE 区集中在：阿古拉斯、黑潮延伸、湾流、巴西洋流、南极绕极流及热带区域，反映这些高涡动能区本身的高变率；
  - 日度 RMSE 时间序列（2019–2020）：
    - 对 T, S, uo, vo, SLA 的 1/5/10 天 lead 结果：XiHe 在整个时段对 PSY4 等 GOFS 均保持明显优势；
    - 同时存在季节性变化：如 uo/vo 在北半球夏季 RMSE 较小、冬季较大；
    - XiHe 的日 RMSE 峰值相对更少、更平滑，表现出更稳定的预报性能。

- **Global Tropical Moored Buoy Array 评估（0–200 m T/S）**
  - 对比 XiHe 与 GLORYS12 再分析在大西洋、太平洋、印度洋 0–200 m 深度平均 RMSE：
    - 温度：
      - GLORYS12：约 0.810, 0.795, 0.465 ℃；
      - XiHe 1 天预报：约 1.094, 0.889, 0.550 ℃；
      - XiHe 10 天预报：约 1.134, 1.068, 0.643 ℃；
      - XiHe 1 天预报已接近 GLORYS12 再分析的误差水平；
    - 盐度：
      - GLORYS12：约 0.155, 0.173, 0.347；
      - XiHe 1 天：约 0.158, 0.347, 0.189；
      - XiHe 10 天：约 0.191, 0.365, 0.213；
    - 说明 XiHe 预报误差与当前最先进再分析产品接近，并在长 lead 上仍保持合理水平。

- **大尺度海流与地转流评估**
  - 定性评估：
    - XiHe 5/10 天预报的 Agulhas、Kuroshio、Gulf Stream 流速与方向场与 GLORYS12 再分析高度一致；
  - 使用 CMEMS 地转流产品（仅海表，过滤掉非地转信号）：
    - 5 天 lead 时，大部分海域 geostrophic current RMSE < 0.2 m/s（覆盖 >95% 海域）；
    - Spearman 相关系数（SCC）>0.6 的区域约占 44%；高 RMSE 区集中在 Kuroshio, Gulf Stream, Agulhas, 南大洋等强流区；
    - 三大洋区域统计（5 天 lead）：
      - Indian/Atlantic/Pacific 平均 RMSE 约 0.09–0.11 m/s，RMSE<0.2m/s 区域占比均 >92%；
      - SCC>0.6 区域在印度洋最大，在大西洋最小；
    - 说明 XiHe 能够较好预报大尺度/准地转过程。

- **中尺度涡（mesoscale eddy）评估**
  - 区域：西北太平洋 $(15^\circ N–50^\circ N, 110^\circ E–160^\circ E)$；
  - 基于 SLA 进行中尺度涡识别：
    - 数据源：CMEMS SLA（作为真值）、GLORYS12 再分析、XiHe 1/4/7/10 天预报；
    - 先对 SLA 做 11 点局部平滑滤波以去除小尺度噪声；
  - 以 2019-08-01 为例：
    - CMEMS：检测到 71 个气旋涡 + 67 个反气旋涡；
    - GLORYS12：66+56 个，分别与 CMEMS 有 42, 41 个涡的轮廓重叠；
    - XiHe 1 天预报：57+65 个，其中与 CMEMS 重叠的为 39, 40 个；
    - 随着 lead time 增大，XiHe 与 CMEMS 重叠涡数量逐渐下降，但 10 天内仍保持相当的重叠比例；
  - 涡轨迹：
    - 选取 2019–2020 年内，寿命 > 60 天且在 CMEMS 与 GLORYS12 数据中共同出现的中尺度涡轨迹；
    - 在 CMEMS–GLORYS12 间可匹配的长寿命涡轨迹仅 6 条；
    - 在 CMEMS–XiHe 间可匹配的长寿命涡轨迹达 11 条；
    - 指示 XiHe 在涡轨迹预报上具备很强能力，有望在更高质量再分析数据出现时进一步提升。

- **结论性陈述**
  - XiHe 是首个在全球 $1/12^\circ$ 分辨率上，能在 T/S/流场/SSH 多变量上全面压制现有数值 GOFS 的数据驱动模型；
  - 在 60 天 15 m 海流预报上仍优于 PSY4 的 10 天预报，在性能上展现出极强的长期稳定性；
  - XiHe 能解析从大尺度环流到中尺度涡的一系列海洋过程，适合用于实时运营级全球海洋预报系统的候选或补充。

---

### 6. 创新点与局限

- **创新点**
  1. **首个数据驱动全球 $1/12^\circ$ Eddy-Resolving GOFS**：
     - 在分辨率和变量丰富度上对标乃至超越 PSY4 等数值 GOFS；
  2. **Ocean-Specific Transformer 架构**：
     - 通过 local SIE（window attention）+ global SIE（group propagation）结构，高效建模海洋局地–全局多尺度空间信息；
  3. **海陆掩膜机制**：
     - 对 self-attention 与 cross-attention 做海洋区域 masking，显著降低计算量并增强模型对海洋区域的关注；
  4. **多数据源融合输入**：
     - 融合 GLORYS12（再分析）、ERA5 海表风与 OSTIA 卫星 SST，在保持 $1/12^\circ$ 分辨率前提下，提高 SST 等敏感变量的预报技巧；
  5. **与主流 GOFS 在统一 IV-TT 框架下的系统对比**：
     - XiHe 在温度、盐度、SLA 与近表海流等多变量、多 lead time 上系统性优于 PSY4/GIOPS/FOAM/BLK，展示了数据驱动 GOFS 的现实可行性。 

- **局限与未来方向（按论文讨论概括）**
  - 数据驱动–数值模型融合：
    - 当前 XiHe 完全基于再分析数据训练，未直接与数值 GOFS 的物理方程耦合；
    - 未来可探索：
      - 使用数值 GOFS 生成更大规模、更高质量的再分析数据提升 XiHe；
      - 反向将 XiHe 作为数值 GOFS 的快速 surrogate 或 bias-correction 模块；
  - 初始场与同化：
    - 当前 XiHe 的输入仍依赖数值 GOFS 的数据同化系统生成的再分析场；
    - 如何将数据同化方法与大模型相结合，以生成更优初始场和预报仍是开放问题；
  - 物理一致性与长期稳定性：
    - 论文侧重 60 天以内预报，尚未系统评估多年积分下的物理守恒与漂移问题；
  - 气候变化与泛化：
    - 训练/测试阶段涵盖 1993–2020 年的气候变率，但未来更极端气候情景下的泛化能力仍有待检验。

---

## 维度二：基础组件 / 模块级伪代码骨架

> 说明：以下伪代码严格依据论文中给出的结构与公式构建，仅作为结构与接口级别参考；不包含作者未公开的实现细节。未在文中明确的实现部分用 `NotImplementedError` 或注释标出。

### 组件 A：XiHe 顶层模型与前向推理

- **子任务**
  - 从当前日的 96 通道输入场 $\mathbf{X}^t$ 直接生成未来某一 lead time（或多 lead time）的 94 通道输出场 $\hat{\mathbf{X}}^{t+\tau}$。

- **伪代码：顶层模型骨架**

```python
class XiHeModel(nn.Module):
    """XiHe: global 1/12° eddy-resolving data-driven ocean forecast model.

    Inputs strictly follow the paper: 96 channels including
    SST, SSH, 23-layer T/S/U/V, and 10m U/V wind.
    Outputs: 94 channels (23-layer T/S/U/V and SSH).
    """

    def __init__(self, cfg):
        super().__init__()
        # Patch partition: 2D conv with kernel=stride=p
        self.patch_partition = PatchPartition(
            in_channels=cfg.in_channels,   # 96
            embed_dim=cfg.embed_dim,       # e.g. 96/128, not specified exactly
            patch_size=cfg.patch_size      # p
        )

        # Ocean-specific Transformer
        self.encoder = OceanSpecificTransformer(
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,              # 5 blocks + down/up sampling
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,  # M
            num_groups=cfg.num_groups,    # number of group vectors
            ocean_mask_cfg=cfg.ocean_mask_cfg
        )

        # Patch restoration: 2D transposed conv with kernel=stride=p
        self.patch_restoration = PatchRestoration(
            embed_dim=cfg.embed_dim,
            out_channels=cfg.out_channels,  # 94
            patch_size=cfg.patch_size
        )

    def forward(self, x, lead_time_idx=0):
        """Forward pass.

        Args:
            x: (B, C_in=96, H=2041, W=4320) tensor at time t.
            lead_time_idx: index of the target lead time (if multiple heads used).

        Returns:
            y: (B, C_out=94, H, W) forecast at t + τ.
        """
        # 1) Patch partition and embedding
        # tokens: (B, H', W', C)
        tokens = self.patch_partition(x)

        # 2) Ocean-specific Transformer (local+global SIE with ocean-land masking)
        features = self.encoder(tokens, lead_time_idx=lead_time_idx)

        # 3) Patch restoration to full grid
        y = self.patch_restoration(features)
        return y
```

> 说明：论文只给出了单步 MSE 损失和单一 XiHe 结构，未明示是否 “一模型多 lead” 还是 “多模型对应不同 lead time”；上例以 `lead_time_idx` 作为扩展钩子，实际实现需依据原始代码或补充材料，此处不作主观假设。

---

### 组件 B：Patch Partition 与 Patch Restoration

```python
class PatchPartition(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C_in, H, W); land/missing already filled with 0
        h, w = x.shape[-2:]
        # Zero padding to make H,W divisible by p (paper mentions this explicitly)
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h))

        feat = self.proj(x)  # (B, C_embed, H', W')
        B, C, H_p, W_p = feat.shape
        # Rearrange to (B, H', W', C)
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = self.norm(feat)
        return feat  # (B, H', W', C)


class PatchRestoration(nn.Module):
    def __init__(self, embed_dim, out_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.deproj = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, H', W', C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C_embed, H', W')
        y = self.deproj(x)                      # (B, C_out, H, W)
        # cropping back to original H,W if padding was used is implementation detail
        return y
```

---

### 组件 C：Ocean-Specific Transformer 与 Ocean-Specific Block

```python
class OceanSpecificTransformer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, window_size,
                 num_groups, ocean_mask_cfg):
        super().__init__()
        self.window_size = window_size

        # Build 5 ocean-specific blocks
        # Block 1: local + global
        # Block 2-4: local + local + global
        # Block 5: local + global
        self.blocks = nn.ModuleList()

        # Down-sampling and up-sampling blocks
        self.down = DownSampleBlock(embed_dim)
        self.up = UpSampleBlock(embed_dim)

        # Ocean masks at two resolutions
        self.mask_high = OceanMask(ocean_mask_cfg, level="high")   # M1
        self.mask_low = OceanMask(ocean_mask_cfg, level="low")    # M2

        # Example layout; exact dims/heads per block not specified in paper
        for i in range(5):
            if i in (0, 4):
                self.blocks.append(
                    OceanSpecificBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        num_groups=num_groups,
                        num_local_modules=1
                    )
                )
            else:
                self.blocks.append(
                    OceanSpecificBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        num_groups=num_groups,
                        num_local_modules=2
                    )
                )

    def forward(self, x, lead_time_idx=0):
        # x: (B, H', W', C)
        # Block 1 at high-res
        mask_high = self.mask_high(x)  # (1, H', W') in {0,1}
        x1 = self.blocks[0](x, mask=mask_high)

        # Down-sampling
        x_down = self.down(x1)
        mask_low = self.mask_low(x_down)

        # Blocks 2-4 at low-res
        x_mid = x_down
        for i in range(1, 4):
            x_mid = self.blocks[i](x_mid, mask=mask_low)

        # Up-sampling
        x_up = self.up(x_mid)

        # Block 5 with skip from Block1 (concatenation or residual, paper states skip)
        x5_input = x1 + x_up  # or torch.cat([...], dim=-1) then project
        x5 = self.blocks[4](x5_input, mask=mask_high)
        return x5
```

```python
class OceanSpecificBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size,
                 num_groups, num_local_modules):
        super().__init__()
        self.local_modules = nn.ModuleList([
            LocalSIEModule(embed_dim, num_heads, window_size)
            for _ in range(num_local_modules)
        ])
        self.global_module = GlobalSIEModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_groups=num_groups
        )

    def forward(self, x, mask):
        # x: (B, H', W', C)
        # mask: (1, H', W'), 1 for ocean, 0 for land
        for local_sie in self.local_modules:
            x = local_sie(x, mask)
        x = self.global_module(x, mask)
        return x
```

---

### 组件 D：Local SIE 模块（带海陆掩膜的窗口注意力）

```python
class LocalSIEModule(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowMSA(embed_dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x, mask):
        # x: (B, H, W, C), mask: (1, H, W) in {0,1}
        shortcut = x
        x_norm = self.norm1(x)
        x_attn = self.attn(x_norm, mask=mask)  # window-based self-attn with masking
        x = shortcut + x_attn

        shortcut2 = x
        x_norm2 = self.norm2(x)
        x_mlp = self.mlp(x_norm2)
        x = shortcut2 + x_mlp
        return x


class WindowMSA(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # Relative position bias B is omitted; exact form not specified in paper

    def forward(self, x, mask):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        M = self.window_size
        # Partition into non-overlapping windows of size MxM
        # Implementation of window partitioning/unpartitioning is standard Swin-style
        windows, win_mask = partition_windows(x, mask, M)
        # windows: (num_win * B, M*M, C)
        # win_mask: (num_win * B, M*M) in {0,1}

        # Construct attention mask so that land tokens (mask=0) have 0 attention scores
        # Exact masking scheme (e.g., large negative bias) is implementation-specific
        attn_mask = build_attn_mask_from_window_mask(win_mask)

        q = k = v = windows
        out, _ = self.attn(q, k, v, attn_mask=attn_mask)
        out = merge_windows(out, B, H, W, M)
        return out
```

> 说明：`partition_windows`/`merge_windows`/`build_attn_mask_from_window_mask` 等函数的具体实现为标准 Swin Transformer 技巧，论文未给出代码细节，此处不展开。

---

### 组件 E：Global SIE 模块（Group Propagation + Cross-Attention + 掩膜）

```python
class GlobalSIEModule(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.num_heads = num_heads

        # Learnable group vectors initialization
        self.group_vectors = nn.Parameter(
            torch.randn(1, num_groups, embed_dim) * 0.02
        )

        # Cross-attention: groups <- patches (feature grouping)
        self.ca_group = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # Group propagation via MLP-Mixer
        self.mlp_mixer = GroupMLPMixer(embed_dim, num_groups)

        # Cross-attention: patches <- groups (feature ungrouping)
        self.ca_patch = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x, mask):
        # x: (B, H, W, C), mask: (1, H, W) with 1 for ocean, 0 for land
        B, H, W, C = x.shape
        tokens = x.view(B, H * W, C)  # (B, N, C)
        mask_flat = mask.view(1, H * W)  # (1, N)

        # Select only ocean tokens to reduce cost (feature grouping)
        ocean_idx = mask_flat.bool()[0].nonzero(as_tuple=False).squeeze(-1)
        tokens_ocean = tokens[:, ocean_idx, :]  # (B, N_ocean, C)

        # Stage 1: feature grouping (groups attend to patch features)
        G0 = self.group_vectors.expand(B, -1, -1)  # (B, G, C)
        G_new, _ = self.ca_group(
            query=G0,
            key=tokens_ocean,
            value=tokens_ocean
        )

        # Stage 2: group propagation via MLP-Mixer
        G_mixed = self.mlp_mixer(G_new)  # (B, G, C)

        # Stage 3: feature ungrouping (patch tokens attend to groups)
        # For simplicity, we let all tokens (incl. land) receive global info;
        # paper explicitly states land patches can be excluded to save cost.
        tokens_all = tokens
        U, _ = self.ca_patch(
            query=tokens_all,
            key=G_mixed,
            value=G_mixed
        )  # (B, N, C)

        # Concatenate and project back
        concat = torch.cat([tokens_all, U], dim=-1)  # (B, N, 2C)
        out = self.proj(concat).view(B, H, W, C)
        return out


class GroupMLPMixer(nn.Module):
    def __init__(self, embed_dim, num_groups):
        super().__init__()
        self.norm1 = nn.LayerNorm(num_groups)
        self.mlp_token = nn.Sequential(
            nn.Linear(num_groups, num_groups),
            nn.GELU(),
            nn.Linear(num_groups, num_groups)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp_channel = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, G):
        # G: (B, G, C)
        B, Gnum, C = G.shape
        # Token-mixing MLP across group dimension (transpose to (B, C, G))
        x = G.transpose(1, 2)  # (B, C, G)
        x_norm = self.norm1(x)
        x_mixed = self.mlp_token(x_norm)
        x = x + x_mixed
        x = x.transpose(1, 2)  # (B, G, C)

        # Channel-mixing MLP across channel dimension
        x_norm2 = self.norm2(x)
        x_mixed2 = self.mlp_channel(x_norm2)
        x = x + x_mixed2
        return x
```

> 说明：
> - 上述 `GroupMLPMixer` 结构仅遵循论文中 “MLPMixer” 的描述（在 group 和 channel 两个维度上做 MLP 混合），具体隐藏层维度、激活等属于实现细节；
> - 真实代码中 group 数量、MLP 维度可能与此不同，此处仅为结构骨架。

---

### 组件 F：海陆掩膜构建与使用

```python
class OceanMask(nn.Module):
    def __init__(self, cfg, level="high"):
        super().__init__()
        # cfg may contain high-res mask M (H, W) in {0,1}
        self.level = level
        self.register_buffer("mask_full", cfg.mask_full)  # (H, W)
        self.patch_size = cfg.patch_size

    def forward(self, x_like):
        # x_like: (B, H', W', C) ; we want patch-level mask for this resolution
        # High-level: use M1 with patch_partition; low-level: use M2 with downsampled patches
        if self.level == "high":
            # Partition full-res mask into patches, assign 1 if any ocean point exists
            M1 = build_patch_mask(self.mask_full, self.patch_size)
            return M1.unsqueeze(0)  # (1, H', W')
        elif self.level == "low":
            # Downsample M1 by factor 2x2 (patch merging) to M2
            M1 = build_patch_mask(self.mask_full, self.patch_size)
            M2 = downsample_mask_2x2(M1)
            return M2.unsqueeze(0)
        else:
            raise NotImplementedError
```

> 说明：`build_patch_mask`/`downsample_mask_2x2` 具体实现依赖于 patch 划分方式，但原则上遵循论文描述：
> - 某 patch 内全为陆地 → 掩膜 0；否则 → 掩膜 1；
> - 对 2×2 patch 合并后，只要任一子 patch 为 1，则合并 patch 也为 1。

---

### 组件 G：训练循环与评估轮廓

```python
def train_xihe(model, train_loader, val_loader, cfg):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        betas=(0.9, 0.95)
    )
    # DropPath schedule, LR schedule etc. are implementation details

    for epoch in range(cfg.num_epochs):  # 50
        model.train()
        for X_t, X_target in train_loader:
            # X_t: (B, 96, H, W)
            # X_target: (B, 94, H, W) at t+τ
            pred = model(X_t)
            loss = F.mse_loss(pred, X_target)
            loss.backward()
            optimizer.step(); optimizer.zero_grad()

        # Validation loop omitted; early stopping / model selection also omitted


def evaluate_ivtt(model, ivtt_dataset, cfg):
    """Outline: interpolate model forecasts to observation locations,
    then compute RMSE and ACC following IV-TT Class 4 framework.
    """
    # 1. For each forecast case and lead time, run model once (non-AR)
    # 2. Interpolate grid forecast to obs locations
    # 3. Compute RMSE and ACC as defined in the paper
    raise NotImplementedError  # details depend on IV-TT data format
```

---

## 小结

- 从结构上看，XiHe 将 Vision Transformer/GPViT/Swin Transformer 中的窗口自注意力与 group propagation 思路迁移到全球海洋格点预报任务中，通过 ocean-specific block 与海陆掩膜机制在 $1/12^\circ$ 分辨率下高效建模局地–全局多尺度海洋动力学；
- 在实现层面，上述伪代码给出了：
  - 顶层 XiHe 模型接口；
  - Patch Partition / Restoration；
  - Ocean-Specific Transformer 及 ocean-specific block；
  - 带海陆掩膜的 local SIE（窗口注意力）；
  - 基于 group vectors 的 global SIE 模块（特征聚类–传播–反聚类）；
  - 海陆掩膜构造与训练/评估轮廓；
- 这些组件组合在一起，即构成了论文中描述的 XiHe 框架的高保真结构骨架，可作为后续实现、扩展（如多步输出、与数值 GOFS 混合）和方法对比的基础。