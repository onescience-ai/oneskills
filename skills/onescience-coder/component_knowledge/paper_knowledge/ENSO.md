# ENSO CNN 模型知识提取（Deep learning for multi-year ENSO forecasts）

信息来源：Ham et al., “Deep learning for multi-year ENSO forecasts”（本文中内容仅基于原文显式描述，不做主观推断；若原文未说明则标明“未在文中明确给出”）。

---

## 维度一：整体模型与任务知识

### 1. 任务与问题设定

- **目标气候现象：ENSO（El Niño/Southern Oscillation）**
  - ENSO 通过改变热带太平洋海气相互作用，影响全球范围内的区域极端气候和生态系统。
  - 多年（>1 年）提前量的 ENSO 预报对政策和风险管理非常重要，但长期以来仍是难题。

- **任务类型**：
  1. **多年前（最长约 24 个月）ENSO 强度预测**：
     - 预测指标：Niño3.4 指数（170°–120°W, 5°S–5°N 的海表温度距平，3 个月滑动平均）。
     - 预测提前量：最长 24 个月，文中重点强调“最长 1.5 年（约 17 个月）仍有较高技巧”。
  2. **El Niño 类型预测（EP 型 / CP 型 / 混合型）**：
     - 预测目标：成熟期（DJF）El Niño 的类型（东太平洋型 EP、中心太平洋型 CP、混合型）。
     - 提前量：12 个月。

- **输入 / 输出形式**：
  - 主 ENSO 强度 CNN：
    - 输入预测因子（predictors）：
      - 海表温度（SST）距平；
      - 上 300 m 海洋热含量（heat content，垂直平均温度）距平；
      - 空间范围：经度 0°–360°E、纬度 55°S–60°N（全球海洋区域）；
      - 时间窗口：从当前时间记为 τ，对应使用 τ, τ-1, τ-2 三个月的距平场（3 个月滑动窗口）。
    - 输出预测量（predictand）：
      - Niño3.4 指数（3 个月平均距平），预测时间从 τ+1 个月到 τ+23 个月（最多 24 个月提前）。
      - 每个“目标季节 × 提前量”对应单独的 CNN 模型。
  - El Niño 类型 CNN：
    - 输入：与主 ENSO CNN 类似的 SST / HC 距平场（细节未完全展开，按文中理解为全球或关键流域 SST/HC 模式）。
    - 输出：三类 El Niño 类型的发生概率（EP, CP, Mixed），选最大概率作为预测类型。

- **时间段与样本划分**：
  - 观测/再分析：
    - SODA（Simple Ocean Data Assimilation v2.2.4）：1871–1973 年；
    - GODAS（Global Ocean Data Assimilation System）：1984–2017 年；
    - ERA-Interim：1984–2017 年，用于 925 hPa 风和降水等影响分析；
  - CMIP5 历史模拟（21 个模式）：
    - 时段一般为 1850/1861–2005/2012（具体见 Extended Data Table 1）。
  - 训练 / 验证：
    - 训练集：
      - CMIP5 历史模拟：约 1861–2004；
      - SODA 再分析：1871–1973（103 年）；
    - 验证集：
      - GODAS 再分析 Niño3.4：1984–2017；
    - 特别设置：训练集最后年份（1973）与验证集最早年份（1984）之间留有约 10 年空档，以避免“海洋记忆”带来的信息泄漏。

---

### 2. 传统方案的痛点与动机

- **现有 ENSO 预测系统的局限**：
  - 大气–海洋耦合数值模式（CMIP / 业务动态模式）：
    - 通常在 12 个月以内的季节–年际尺度预报上优于传统统计模型；
    - 但在超过 1 年的提前量上，技巧显著下降，多年 ENSO 预测仍然“困难”。
  - 传统统计模型 / 简单神经网络：
    - 使用 EOF 主成分或少量指数作为输入，难以充分利用三维空间结构和遥相关关系；
    - 在多月甚至多年提前量下技巧有限。

- **物理层面的可预测性启示**：
  - ENSO 中存在慢变振荡成分，与海洋大尺度热含量变化及其与大气的耦合相关，理论上支持多年前预测；
  - 多次 La Niña 事件的赤道太平洋异常可持续数年，表明存在缓变背景；
  - 高频赤道风难以预测，但其缓变部分与 SST 耦合具有一定可预报性；
  - 赤道外太平洋乃至印度洋、大西洋 SST 异常可以以 >1 年的时滞触发 ENSO 事件，这些遥相关机制未被现有统计模型充分利用。

- **深度学习引入的动机**：
  - CNN 擅长在空间栅格数据中提取局地与多尺度特征，具备平移/形变的部分不变性，可以容忍先兆信号的空间位置与形状发生偏移；
  - 可以直接在全场 SST/HC 图像上学习 ENSO 前兆，而不必事先设定少数指数或 EOF 模式。
  - 挑战：观测样本有限（每个日历月只有 <150 年），需要利用气候模式样本扩充训练数据。

---

### 3. 整体解决方案与范式

- **核心策略：CMIP5 + 再分析的迁移学习 CNN**
  1. 使用 CMIP5 历史模拟输出对 CNN 进行**预训练**：
     - 学习模式模拟中 ENSO 及其遥相关的统计关系；
     - 提供数千个样本用于深度网络的初始特征学习。
  2. 再使用 SODA 再分析数据对 CNN 进行**微调（transfer learning）**：
     - 以 CMIP5 预训练后的权重为初始值；
     - 在真实观测一致的再分析数据集上做短期训练，校正模式系统误差；
     - 最终得到面向真实世界的 ENSO 预测 CNN 模型。

- **模型家族与集成**：
  - 四个架构组合：
    - C30H30：30 个卷积滤波器，30 个全连接神经元；
    - C30H50：30 滤波器，50 神经元；
    - C50H30：50 滤波器，30 神经元；
    - C50H50：50 滤波器，50 神经元；
  - 每个日历月×提前量构建 4 个 CNN，预测 Niño3.4 后再**平均四个模型结果**，利用“多模型组合”减小单模型误差（技巧略有提升）。

- **El Niño 类型 CNN 系统**：
  - 另一个独立 CNN，仅使用 CMIP5 输出进行训练（不使用再分析迁移学习），用于预测成熟期 El Niño 类型（EP/CP/Mixed）。

---

### 4. 整体模型架构与关键公式

#### 4.1 CNN 结构概览

- **结构**：
  - 输入层（SST+HC，3 个月，全球网格）；
  - 3 个卷积层，每个后接若干的激活和第 1/2 层后接 2×2 max-pooling；
  - 1 个全连接层（30 或 50 个神经元）；
  - 1 个输出层（标量 Niño3.4 指数或 3 类概率向量）。

- **卷积过程的数学形式**：

  - 第 i 个卷积层中，第 j 个特征图在栅格点 $(x,y)$ 的值：

$$
\mathbf{v}_{i,j}^{x,y} = \tanh\Bigg(\sum_{m=1}^{M_{i-1}}\sum_{p=1}^{P_i}\sum_{q=1}^{Q_i}w_{i,j,m}^{p,q}\,v_{(i-1),m}^{(x+p-P_i/2,\,y+q-Q_i/2)} + b_{i,j}\Bigg),
$$

  其中：
  - $P_i, Q_i$：第 i 层卷积核在经向/纬向的尺寸；
    - 第 1 卷积层：$P_1=8, Q_1=4$；
    - 第 2, 3 卷积层：$P_i=4, Q_i=2$；
  - $M_{i-1}$：第 i-1 层特征图数目；
  - $w_{i,j,m}^{p,q}$：连接上一层第 m 个特征图与当前层第 j 个特征图的卷积权重；
  - $b_{i,j}$：偏置；
  - 激活函数：$\tanh(\cdot)$；
  - 使用零填充（padding）保持空间维度不变，后续再通过 max-pooling 降采样。

- **输出层及热力图基础公式**：

  - 设第三个卷积层的特征图为 $v_{L,m}^{x,y}$，空间大小 $(X_L, Y_L)=(18,6)$，全连接层神经元数 N，输出标量 V：

$$
V = \sum_{n=1}^{N}\Bigg\{\tanh\Big[\sum_{m=1}^{M_L}\sum_{y=1}^{Y_L}\sum_{x=1}^{X_L}W_{F,m,n}^{x,y}v_{L,m}^{x,y} + b_{F,n}\Big]W_{O,n}\Bigg\} + b_O.
$$

  - 热力图 $h^{x,y}$（输出对第 L 层特征图在栅格 $(x,y)$ 的贡献）公式：

$$
h^{x,y} = \sum_{n=1}^{N}\Bigg\{\tanh\Big[\sum_{m=1}^{M_L}W_{F,m,n}^{x,y}v_{L,m}^{x,y} + \frac{b_{F,n}}{X_LY_L}\Big]W_{O,n}\Bigg\} + \frac{b_O}{X_LY_L}.
$$

  - 该热力图用于解释预测所依赖的空间先兆模式。

#### 4.2 ENSO 相关技巧评价公式

- **Niño3.4 指数相关技巧 C_l**：

  - 对每个 forecast lead l，定义跨年、跨月的异常相关系数：

$$
C_l = \sum_{m=1}^{12}
\frac{\sum_{y=s}^{e}\big(Y_{y,m}-\bar{Y}_m\big)\big(P_{y,m,l}-\bar{P}_{m,l}\big)}
{\sqrt{\sum_{y=s}^{e}\big(Y_{y,m}-\bar{Y}_m\big)^2\sum_{y=s}^{e}\big(P_{y,m,l}-\bar{P}_{m,l}\big)^2}}.
$$

  - $Y_{y,m}$、$P_{y,m,l}$：观测与预测的 Niño3.4 指数；
  - $\bar{Y}_m, \bar{P}_{m,l}$：对月份 m 的长期气候态（对年份求平均）；
  - y：年份，从 s=1984 到 e=2017。

- **El Niño 统一复指数 UCEI**：

  - 通过 Niño3 指数 $N_3$（150°–90°W, 5°S–5°N）与 Niño4 指数 $N_4$（160°E–150°W, 5°S–5°N）构造：

$$
\mathrm{UCEI} = (N_3+N_4) + (N_3-N_4)i = r e^{i\theta},
$$

  其中：

$$
r = \sqrt{(N_3+N_4)^2 + (N_3-N_4)^2},
$$

$$
\theta = 
\begin{cases}
\arctan\frac{N_3-N_4}{N_3+N_4}, & N_3+N_4>0,\\
\arctan\frac{N_3-N_4}{N_3+N_4}+\pi, & N_3+N_4<0,\ N_3-N_4>0,\\
\arctan\frac{N_3-N_4}{N_3+N_4}-\pi, & N_3+N_4<0,\ N_3-N_4<0.
\end{cases}
$$

  - 分类规则：
    - EP 型：$15^\circ < \theta < 90^\circ$；
    - CP 型：$-90^\circ < \theta < -15^\circ$；
    - 混合型：$-15^\circ < \theta < 15^\circ$；
    - El Niño 发生条件：幅度 r 在 DJF 季节大于其标准差。

---

### 5. 数据集与训练细节

- **训练样本数量**：
  - CMIP5：每个目标月的样本数约为 2,961（21 模式 × 年数 × 单集合成员，具体见 Extended Data Table 1）。
  - 再分析：SODA 1871–1973 共 103 年，每月 1 样本。

- **训练流程（ENSO 强度 CNN）**：
  1. **第一阶段（CMIP5 预训练）**：
     - 批大小：400；
     - 训练 epoch：700（调到 600–1000 范围对技巧无明显影响）；
     - 学习率：0.005（固定，无调度）；
     - 损失：预测 Niño3.4 与 CMIP5 ENSO 指数的 MSE。
  2. **第二阶段（再分析微调）**：
     - 以上一阶段训练的权重为初始化权重；
     - 使用 1871–1973 SODA/再分析数据继续训练 20 epochs；
     - 用再分析实际 ENSO 统计校正 CMIP5 系统性偏差。

- **El Niño 类型 CNN 训练**：
  - 仅使用 CMIP5 历史模拟样本（因为再分析训练期内 ENSO 类型几乎为单一类型）；
  - 不采用迁移学习；
  - 样本数：872 个 El Niño 事件（CMIP5 中基于 UCEI 分类）。

- **对比模型：前馈神经网络 ENSO 预报**：
  - 输入：Indo-Pacific、Atlantic、North Pacific 区域的 SST / HC EOF 主成分（不同区域分别做 EOF）；
  - 模型：2 层隐藏层，每层 20 神经元，tanh 激活；
  - 训练数据同 CNN（CMIP5 + 再分析）；
  - 在 DJF、18 个月提前量上最优技巧约 0.52，而 CNN 为 0.64；且 NN 技巧对 EOF 数量非常敏感，不易稳定获得最优配置。

---

### 6. 技巧与结果概述

- **Niño3.4 相关技巧（1984–2017 验证期）**：
  - 全季节相关技巧随提前量：
    - CNN 模型在提前量 >6 个月时系统性优于所有 NMME 动力模式和 SINTEX-F；
    - 在约 17 个月提前量仍能保持相关 >0.5；
    - SINTEX-F 在 17 个月提前量的技巧约 0.37。
  - 按目标季节分解：
    - CNN 在几乎所有目标季节上都优于 SINTEX-F；
    - 尤其对晚春–秋季（例如 MJJ）目标季节，CNN 大幅减弱“春季可预报性障碍”，比如 MJJ 目标季节在 SINTEX-F 中相关>0.5 仅达 4 个月提前，而 CNN 可达 11 个月提前。

- **El Niño 类型预测技巧**：
  - 12 个月提前量、DJF 成熟期类型预测：
    - CNN 命中率约 66.7%；
    - 随机预报命中率的 95% 置信区间约在 12.5%–62.5% 之间，CNN 明显高于上界（p=0.016）；
    - 多个 NMME 和 SINTEX-F 模式在统计上并不显著优于随机预报；
    - CNN 克服了动力模式在 El Niño 类型（EP vs CP）预测上的长期弱点。

- **物理可解释性（热力图）**：
  - 1997/98 EP 型 El Niño 的 18 个月提前预测个例：
    - 1996 年 MJJ 季节的热力图显示，热含量正异常在热带西太平洋、SST 负异常在西南印度洋与北副热带大西洋是主要贡献区；
    - 这与已有物理机制吻合：
      - 热带西太平洋的正热含量异常可视为“recharge”态势；
      - 西南印度洋的负 SST 先触发负 IOD，再导致整个印度洋负 SST，引发赤道西太平洋上空的西风异常，从而促成次年 El Niño；
      - 北副热带大西洋负 SST 可经中纬度太平洋变率影响下一年 ENSO。
  - 对 CP 型 El Niño，热力图表明：
    - 北热带大西洋及南太平洋、印度洋 SST 模式可作为独特前兆（部分未在既有研究中报道）。

---

### 7. 创新点与局限

- **创新点**：
  1. **首次展示基于 CNN 的 ENSO 预报在 1.5 年提前量上显著优于最先进动力系统**；
  2. **迁移学习范式**：利用 CMIP5 模拟扩充样本，再用再分析微调，解决深度学习在气候样本短缺条件下的训练问题；
  3. **全场空间先兆学习**：CNN 直接在全球 SST/HC 图像上学习 ENSO 前兆，而不是依赖少量经验指数；
  4. **El Niño 类型预测**：同一 CNN 范式扩展到 EP/CP/Mixed 类型预测，显著优于动力模式与随机预报；
  5. **热力图解释法**：利用类似 Class Activation Map 的方法，对每个栅格位置贡献进行量化，连接统计模型与物理机制研究。

- **局限与未来方向**（原文指出）：
  - 物理机制仍需进一步研究，尤其是 CP 型 El Niño 新前兆模式；
  - 训练样本虽然通过 CMIP5 增加，但真实观测仍有限，可能限制部分结论的稳健性；
  - 目前模型为“回归式 + 单期预测”，未显式构造时序一致的多步序列模型；
  - 仍需探索如何与过程型模式结合（例如将 CNN 预测嵌入耦合模式框架）。

---

## 维度二：基础组件 / 模块级知识

> 说明：本节按组件拆解 ENSO CNN 系统。伪代码均为**根据论文文字与公式复原的高保真伪代码**，仅用于结构与逻辑理解，不代表作者给出的真实代码。

### 组件 A：主 ENSO CNN（Niño3.4 多年前预测）

- **子任务**：
  - 从全球 SST/HC 距平（3 个月窗口）学习并预测未来 1–24 个月的三个月平均 Niño3.4 指数。

- **输入 / 输出**：
  - 输入：
    - $X \in \mathbb{R}^{C\times H\times W}$，其中 C 包含 SST 与 HC 在 3 个连续月份（通道数约为 2×3=6，论文未给具体数值，仅说明“从 τ-2 到 τ 的 SST & HC”）；
    - 空间范围：0°–360°E, 55°S–60°N（插值至 5°×5° 网格，用于训练）。
  - 输出：
    - 标量 $\hat{y}$：对应某一预报提前量 l 和目标 3 个月平均 Niño3.4 指数。

- **结构要点**：
  - 3×卷积层：
    - Conv1：卷积核 8×4，tanh 激活，后接 2×2 max-pooling；
    - Conv2：卷积核 4×2，tanh 激活，后接 2×2 max-pooling；
    - Conv3：卷积核 4×2，tanh 激活，输出特征图尺寸 18×6；
  - FC 层：
    - 神经元数 N ∈ {30, 50}；
  - 输出层：
    - 单一线性神经元，输出 Niño3.4；
  - 模型组合：C30H30, C30H50, C50H30, C50H50 四种结构，对同一任务预测值取平均。

- **实现逻辑（文字）**：
  - 针对每个“目标季节 × 提前量”：
    - 构建一组 4 个 CNN（不同滤波器数和 FC 神经元数）；
    - 对训练集中的所有时间样本（CMIP5 + 再分析）训练网络；
    - 预测时，将上一年 OND/MJJ 等季节的 SST/HC 距平场输入网络，输出目标季节 Niño3.4 预测值；
    - 将四个 CNN 的输出取平均，作为最终预测。

- **（基于文中描述生成的伪代码）**：

```python
# Main ENSO CNN forward (for one lead & target season)

class ENSOCNN(nn.Module):
    def __init__(self, num_filters=30, fc_neurons=30):
        super().__init__()
        # conv1: kernel 8x4, tanh
        self.conv1 = nn.Conv2d(in_channels=C_in, out_channels=num_filters,
                               kernel_size=(8, 4), padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2: kernel 4x2, tanh
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                               kernel_size=(4, 2), padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3: kernel 4x2, tanh
        self.conv3 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                               kernel_size=(4, 2), padding="same")

        # fully connected & output
        self.fc = nn.Linear(num_filters * 18 * 6, fc_neurons)
        self.out = nn.Linear(fc_neurons, 1)

    def forward(self, x):
        # x: (B, C_in, H, W)
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))  # (B, F, 18, 6)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc(x))
        y_hat = self.out(x)  # Niño3.4 prediction
        return y_hat
```

---

### 组件 B：迁移学习与训练调度

- **子任务**：
  - 在有限观测样本下利用 CMIP5 大量气候模拟样本，通过迁移学习构建稳健的 ENSO CNN。

- **流程**：
  1. **预训练（CMIP5）**：
     - 使用 CMIP5 的 SST/HC 与对应 Niño3.4 时间序列训练 ENSO CNN；
     - 目标：最小化预测–目标之间的 MSE；
     - 获得初始权重集 $\theta_0$。
  2. **微调（再分析）**：
     - 初始化为 $\theta_0$；
     - 使用 SODA 再分析（1871–1973）训练 20 epochs；
     - 校正由 CMIP5 模式误差带来的系统性偏差。

- **实现伪代码**：

```python
# Transfer learning training loop (pseudocode)

# 1) Pretrain on CMIP5
model = ENSOCNN(num_filters=30, fc_neurons=30)
optimizer = Adam(model.parameters(), lr=0.005)

for epoch in range(700):
    for batch_X, batch_y in cmip5_loader:  # SST/HC -> Nino3.4
        y_pred = model(batch_X)
        loss = mse_loss(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save pretrained weights
theta_0 = deepcopy(model.state_dict())

# 2) Fine-tune on reanalysis
model.load_state_dict(theta_0)

for epoch in range(20):
    for batch_X, batch_y in reanalysis_loader:  # SODA 1871-1973
        y_pred = model(batch_X)
        loss = mse_loss(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

### 组件 C：El Niño 类型预测 CNN

- **子任务**：
  - 基于 SST/HC 前兆模式，以 12 个月提前量预测 DJF El Niño 类型（EP/CP/Mixed）。

- **输入 / 输出**：
  - 输入：类似 ENSO 强度 CNN 的全球 SST/HC 距平场；
  - 输出：长度为 3 的概率向量 $p=(p_{EP},p_{CP},p_{Mix})$；
  - 训练标签：根据 UCEI 指数对 CMIP5 中 El Niño 事件分类得到的类型标签。

- **结构**：
  - 与 ENSO 强度 CNN 类似的 3×Conv + 2×MaxPool + FC + Softmax 输出层；
  - 不使用再分析迁移学习，仅在 CMIP5 模拟上训练。

- **训练/推理伪代码**：

```python
# El Niño type CNN training (pseudocode)

for epoch in range(E_type):
    for batch_X, batch_labels in cmip5_elnino_loader:
        logits = type_cnn(batch_X)            # (B, 3)
        loss = cross_entropy(logits, batch_labels)
        loss.backward()
        optimizer.step(); optimizer.zero_grad()

# Inference for one DJF season
def predict_elnino_type(X_input):
    logits = type_cnn(X_input)
    probs = softmax(logits)
    type_idx = argmax(probs)  # 0:EP, 1:CP, 2:Mixed
    return type_idx, probs
```

---

### 组件 D：热力图（Heat Map）分析模块

- **子任务**：
  - 定量评估每个空间栅格对 Niño3.4 预测的贡献，帮助理解 CNN 依赖的物理前兆模式。

- **实现思路**：
  - 从最后一层卷积特征图 $v_{L,m}^{x,y}$ 及其与 FC/输出层之间的权重 $W_{F,m,n}^{x,y}$、$W_{O,n}$ 出发，利用前述公式计算 $h^{x,y}$；
  - 对 $h^{x,y}$ 随时间样本求标准差；
  - 在某一事件（如 1997/98 El Niño）前兆季节上绘制热力图：
    - 仅对 $|h^{x,y}|$ 大于 95% 置信水平的栅格着色；
    - 叠加 SST/HC 距平等值线以形成物理诊断图。

- **伪代码**：

```python
# Heat map computation (pseudocode)

def compute_heatmap(model, feature_maps_L, W_F, b_F, W_O, b_O):
    # feature_maps_L: v_L (F, X_L, Y_L)
    # W_F: (F, N, X_L, Y_L), W_O: (N,)
    X_L, Y_L = feature_maps_L.shape[1:]
    heatmap = np.zeros((X_L, Y_L))

    for x in range(X_L):
        for y in range(Y_L):
            s = 0.0
            for n in range(N):
                z = 0.0
                for m in range(F):
                    z += W_F[m, n, x, y] * feature_maps_L[m, x, y]
                z += b_F[n] / (X_L * Y_L)
                s += np.tanh(z) * W_O[n]
            heatmap[x, y] = s + b_O / (X_L * Y_L)
    return heatmap
```

---

### 组件 E：评估与统计检验模块

- **子任务**：
  - 计算相关技巧、置信区间与 El Niño 类型预测命中率；
  - 使用 bootstrap 与随机预报检验统计显著性。

- **相关技巧与置信区间（bootstrap）**：
  - 步骤：
    1. 对每个系统（CNN/NMME/SINTEX-F）随机重采样其集合成员 N 次（允许重复）；
    2. 对重采样集合平均，计算 Niño3.4 相关技巧；
    3. 重复 10,000 次；
    4. 取技巧分布的 2.5% 与 97.5% 分位数作为 95% 置信区间。

- **El Niño 类型命中率与随机检验**：
  - 对每种类型（EP/CP/Mixed）设定气候发生频率（来自 CMIP5：约 30%、26%、44%）；
  - 构造随机预报（按该气候概率随机抽样类型），重复 10,000 次，统计命中率分布；
  - CNN 实际命中率若高于 95% 置信上界，则认为显著优于随机。

- **伪代码**：

```python
# Bootstrap confidence interval for correlation skill (pseudocode)

def bootstrap_ci(members_predictions, observations, n_iter=10000):
    N = len(members_predictions)
    skills = []
    for _ in range(n_iter):
        # resample member indices with replacement
        idx = np.random.randint(0, N, size=N)
        mean_pred = np.mean([members_predictions[i] for i in idx], axis=0)
        skill = correlation(mean_pred, observations)
        skills.append(skill)
    lower = np.percentile(skills, 2.5)
    upper = np.percentile(skills, 97.5)
    return lower, upper
```

---

### 组件 F：对比基线（前馈神经网络 FNN）

- **子任务**：
  - 提供一个传统非线性统计基线，与 CNN 在 ENSO 多年前预报上的技能做对比。

- **输入 / 输出**：
  - 输入：
    - Indo-Pacific、Atlantic、North Pacific 三大区域 SST/HC 的 EOF 主成分：
      - Indo-Pacific：EOF PCs 数量 5–10 不等；
      - Atlantic+North Pacific：EOF PCs 数量 5–9 不等；
      - 最优配置：Indo-Pacific 9 PCs + Atlantic/N. Pacific 7 PCs；
  - 输出：
    - 标量 Niño3.4 指数（与 CNN 相同）。

- **结构**：
  - 2 个隐藏层，每层 20 神经元，tanh 激活；
  - 输出层为线性回归至 Niño3.4。

- **技巧对比**：
  - 在 DJF、18 个月提前量上：
    - 最优 FNN 技巧：0.52；
    - ENSO CNN 技巧：0.64；
  - FNN 技巧随预测因子数量微调高度敏感，难以稳定复现最优结果；
  - CNN 在整体相关技巧和稳定性方面均明显优于 FNN 基线。

---

## 小结

- 该 ENSO CNN 系统通过“CMIP5 预训练 + 再分析迁移学习 + 多模型集成”的范式，在 1.5 年提前量上实现了对 Niño3.4 指数的高技巧预报，并显著优于当时最先进的动力模式与传统神经网络基线；
- 另一个 CNN 成功实现了 El Niño 类型（EP/CP/Mixed）的 12 个月提前分类预报，统计意义上优于随机与动力模式；
- 通过热力图方法，模型挖掘到了一系列物理合理的 ENSO 前兆模式，包括印度洋、西太平洋和大西洋对 EP/CP 型 El Niño 的不同贡献，为后续机理研究提供了数据驱动的线索。