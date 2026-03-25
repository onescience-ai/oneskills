

### 【第一部分：物理先验注入范式】

#### 1.1 海陆地理掩膜约束机制 (Ocean-land Masking)
##### 1. 【原理描述】
- **核心定义**：由于全球网格中陆地面积占比接近 29.2%，模型通过构造显式的海陆掩膜（Mask），在注意力计算阶段强行屏蔽陆地网格的信息流动，迫使模型将算力与表征能力全部分配给真实的海洋动力学演化。
- **理论依据**：在纯海洋预报任务中，陆地区域是无效的物理边界（通常在预处理时填 0）。如果允许陆地 patch 参与全局注意力计算，不仅浪费近 30% 的计算资源，还可能引入破坏海洋遥相关结构的计算噪声。

##### 2. 【数据规格层】
- **海陆掩膜矩阵**：原始掩膜 $M \in \mathbb{R}^{H\times W}$（海洋赋 1，陆地赋 0）。
- **Patch 级掩膜**：经过划块后得到 $M_1 \in \mathbb{R}^{W'\times H'}$。对于下采样后的特征图，生成更粗分辨率的掩膜 $M_2 \in \mathbb{R}^{W'/2\times H'/2}$。

##### 3. 【架构层】
计算流程图: |
  原始特征场 [B, C, H, W] 与 原始掩膜场 M [H, W]
    ↓
  并行下采样/划块 -> 得到 Patch 特征图与 Patch 级掩膜 M_1 [H', W']
    ↓
  Local SIE (窗口注意力):
    若窗口内包含陆地 Patch，利用 M_1 将其对应的 Attention Score 强制置为 0
    ↓
  Global SIE (交叉注意力):
    在构建 Key/Value 时，直接剔除 M_1 中值为 0 的陆地 Patch
    ↓
  仅包含有效海洋信息的特征表达更新

##### 4. 【设计层】
- **痛点解决**：大幅降低了无效注意力计算量（仅对 70.8% 的海洋 patch 进行计算），并从结构上防止了陆地零值对海洋特征聚类（Feature grouping）的污染。

---

### 【第二部分：变量组织范式】

#### 2.1 多源海气耦合特征全堆叠策略
##### 1. 【原理描述】
- **核心定义**：将驱动海洋演变的大气边界条件（如海表风）、高频观测的 2D 面场（如卫星 SST、SSH）以及描述海洋内部动力结构的 3D 剖面场（多层温盐流），在输入端直接沿通道维度进行扁平化堆叠。

##### 2. 【数据规格层】
- **输入张量形状**：$\mathbf{X}^t \in \mathbb{R}^{W\times H\times C_{in}}$，其中 $W=4320, H=2041, C_{in}=96$。
- **通道语义 ($C_{in}=96$)**：包含 2D 卫星海表温度 SST、海表高度 SSH，23 层的海温 $T$、盐度 $S$、海流 $U/V$，以及降维插值后的 ERA5 10m 海表风 $U10/V10$。
- **输出张量形状**：$\hat{\mathbf{X}}^{t+\tau} \in \mathbb{R}^{W\times H\times C_{out}}$，其中 $C_{out}=94$（23 层 $T/S/U/V$ + SSH）。

---

### 【第三部分：主干网络类型范式】

#### 3.1 基于 Ocean-Specific Transformer 的层次化架构
##### 1. 【原理描述】
- **核心定义**：采用类似于 Swin Transformer 的层次化视觉 Transformer 架构，但在基础 Block 中定制化集成了专门针对海洋流体特征的计算单元，通过下采样/上采样模块构建特征金字塔。

##### 2. 【数据规格层】
- **网络超参数**：包含 5 个 ocean-specific blocks，1 个 down-sampling block，1 个 up-sampling block。具体的嵌入维度 $C$ 与 patch 大小 $p$ **原文未详细说明**。

##### 3. 【架构层】
计算流程图: |
  输入张量 [B, 96, 2041, 4320]
    ↓
  Patch Partition -> [B, H', W', C]
    ↓
  Ocean-Specific Block 1 (高分辨率: Local + Global SIE)
    ↓
  Down-sampling Block (2x2 Patch 合并) -> 维度变为 2C，尺寸减半
    ↓
  Ocean-Specific Block 2~4 (低分辨率: 2*Local + Global SIE)
    ↓
  Up-sampling Block -> 恢复至高分辨率
    ↓
  Ocean-Specific Block 5 (结合 Block 1 的 Skip Connection)
    ↓
  Patch Restoration -> 还原为输出物理场 [B, 94, 2041, 4320]

---

### 【第四部分：多尺度建模范式】

#### 4.1 局地-全局双轨空间信息提取 (Local-Global SIE)
##### 1. 【原理描述】
- **核心定义**：为了在 $1/12^\circ$（约 8–9 km）的极高分辨率下同时捕捉 $O(100\,\mathrm{km})$ 的中尺度涡旋（Mesoscale eddies）和跨越大洋的全局遥相关（Teleconnection），将自注意力解耦为“基于局部窗口的局地提取（Local SIE）”和“基于聚类传播的全局提取（Global SIE）”。
- **数学推导 (Global SIE 的核心步骤)**：
  1. **特征聚类 (Feature grouping)**：利用可学习的 Group 向量 $\mathbf{G}^l$ 作为 Query，提取大尺度海洋模式：
     $$\mathbf{G}'^{l} = \mathrm{Concat}_{h} \big(\mathrm{Attention}(W_h^Q \mathbf{G}^l_h, W_h^K\tilde{\mathbf{Z}}^l_h, W_h^V\tilde{\mathbf{Z}}^l_h)\big)$$
  2. **群组传播 (Group propagation)**：使用 MLPMixer 在 Group 之间传播全局信息：
     $$\hat{\mathbf{G}}^{l} = \mathbf{G}'^{l} + \mathrm{MLP}_1(\mathrm{LN}(\mathbf{G}'^{l})^T)^T$$
  3. **特征反聚类 (Feature ungrouping)**：将融合了全局信息的 Group 向量下发回各个局地 Patch：
     $$\mathbf{U}^l = \mathrm{Concat}_{h}\big(\mathrm{Attention}(\tilde{W}_h^Q\tilde{\mathbf{Z}}_h^l, \tilde{W}_h^K\tilde{\mathbf{G}}_h^l, \tilde{W}_h^V\tilde{\mathbf{G}}_h^l)\big)$$

##### 4. 【设计层】
- **设计动机**：如果仅使用 Local SIE（如标准 Swin 的窗口注意力），会阻断跨越数千公里的洋流长程信息交流。
- **创新突破**：Global SIE 基于 Group Propagation ViT (GPViT) 思想，通过数量极少的聚类中心（Group Vectors）完成了全局感受野的构建，在计算复杂度与全局表征能力之间取得了完美的权衡。

---

### 【第五部分：训练与优化范式】

#### 5.1 非自回归直接多步预测范式 (Non-autoregressive Forecasting)
##### 1. 【原理描述】
- **核心定义**：摒弃了气象领域常用的“以 $X^t$ 预测 $X^{t+1}$ 并不断滚动”的自回归（Autoregressive）方案，XiHe 选择直接建立从初始态 $\mathbf{X}^t$ 到目标时效 $\mathbf{X}^{t+\tau}$ 的端到端映射映射函数。
- **数学推导**：
  给定日尺度目标时效 $\tau$，直接优化单步均方误差（MSE）：
  $$\mathcal{L}_{MSE} = \frac{1}{W\times H\times C_{out}} \sum_{i=1}^{W}\sum_{j=1}^{H}\sum_{c=1}^{C_{out}} \big(\hat{X}^{t+\tau}_{i,j,c} - X^{t+\tau}_{i,j,c}\big)^2$$
- **理论依据**：海洋系统的变化相对于大气更加缓慢（具有高热惯性和大粘性）。直接预测中长期状态（如 10 天、60 天）可以有效规避自回归迭代带来的高频误差累积，这对于维持 15m 深度海流的长期稳定性极其关键。

##### 2. 【数据规格层】
- **训练细节**：使用 AdamW 优化器，初始学习率 $5\times10^{-5}$，最大 DropPath 比例 0.2，训练 50 个 epochs。

---

### 【第六部分：评估与验证范式】

#### 6.1 统一观测点位插值评估 (GODAE IV-TT Class 4)
##### 1. 【原理描述】
- **核心定义**：为公平对比纯数据驱动模型（XiHe）与传统数值同化系统（如 PSY4, FOAM 等），不直接在网格级与某一种再分析数据对比，而是将模型输出插值到真实的海洋观测点位（如 Argo 浮标、高度计轨迹），在观测空间中计算 RMSE 与 ACC。
- **对比分析**：在 1-10 天的评估中，XiHe 在所有变量上持续优于数值系统。尤其在 15m 深度海流上，XiHe 的误差增长斜率仅为 PSY4 的 1/3 到 1/4.5，其 10 天的误差甚至小于 PSY4 第 1 天的误差。

#### 6.2 海洋物理现象针对性诊断范式 (Phenomena-specific Diagnosis)
##### 1. 【原理描述】
- **核心定义**：跳出全域统计平均的局限，专门针对高影响力的海洋动力学过程（如大洋西边界流、中尺度涡旋）进行物理一致性提取与评估。
- **实现路径**：
  - **地转流评估**：利用独立 CMEMS 产品检验模型大尺度场对地转/准地转平衡的遵守程度。
  - **中尺度涡追踪**：基于海表高度异常（SLA）进行涡旋闭合轮廓识别。结果表明，XiHe 不仅在涡旋数量上与观测保持高度重合，在长寿命（>60天）中尺度涡的轨迹追踪匹配度上，甚至超越了 GLORYS12 再分析产品。

---

