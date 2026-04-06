# 维度一：整体模型知识（Model-Level Knowledge）

## 1. 核心任务与需求

### 1.1 核心建模任务
- 任务：构建一个 **利用 Mixture-of-Experts (MoE) 的数据驱动全球天气预报模型 EWMoE**，在显著减少训练数据与计算资源的条件下，达到或接近 SOTA（Pangu-Weather/GraphCast）水平。
- 预测形式：
  - 输入：单时刻的大气场 \(X_i \in \mathbb{R}^{C\times H\times W}\)；
  - 输出：未来 8 天、6 小时间隔的预报序列：\(\{\hat X_{i+1},\dots,\hat X_{i+32}\}\)。
  - 通过 **自回归多步展开** 实现长时效预测：
    \(\hat X_{i+1}=f_\theta(X_i),\,\hat X_{i+2}=f_\theta(X_{i+1}),\dots\)。

### 1.2 解决了什么问题
- 相比传统 NWP：
  - NWP 基于 PDE 数值积分，算力开销巨大，参数化误差明显，难以随数据和任务灵活扩展。
- 相比现有 DL 模型（FourCastNet、ClimaX、Pangu-Weather、GraphCast）：
  - 大多需要 **数十年 ERA5 数据 + 大规模 GPU 资源**；
  - 标准 ViT/Swin 的位置编码对气象变量的 **地理/垂直位置信息建模不足**；
  - 模型容量依赖于密集 FFN 层，参数/计算成本高。
- EWMoE 解决的关键点：
  - 通过 **3D 绝对位置嵌入**，显式建模经度/纬度/高度对不同气象变量的影响；
  - 通过 **稀疏 MoE 层** 提升模型容量而不线性增加计算量；
  - 通过 **负载均衡辅助损失 + 位置加权损失**，
    - 稳定 MoE 训练、避免“少数专家过载”；
    - 更好强调重要纬带与变量位置的预报精度；
  - 在仅用 **2 年 ERA5 数据 + 有限 GPU 资源** 情况下超过 FourCastNet、ClimaX，并与 Pangu/GraphCast 在短期预报上接近。


## 2. 解决方案设计

### 2.1 核心思路
- 模型整体采用 **ViT 编码器-解码器架构**：
  - 对气象“图像”做 patch 切分与线性嵌入；
  - 通道维为多物理变量（非 RGB），需额外跨变量的交互建模；
- 在此基础上引入三大改进：
  1. **3D 绝对位置嵌入（longitude, latitude, altitude）**：
     - 针对每个补丁的位置构建 3 组可学习向量（维度各为 \(D/3\)），拼接成 \(D\)-dim 位置向量；
     - 让每个 token 感知其在地球坐标系的三维绝对位置，适配如高度相关的温湿度、风场等物理关系。
  2. **稀疏 Mixture-of-Experts (MoE) 层**：
     - 在编码器中，用 MoE 替换标准 FFN；
     - 共有 \(N=20\) 个专家，每个为独立 FFN；
     - gating network 使用 top-\(k\)（文中为 \(k=2\)）路由，每个 token 仅激活最相关的少量专家；
     - 借此在 **计算成本近似不变** 的前提下显著增加总参数量/容量。
  3. **两类训练优化损失**：
     - MoE 负载均衡辅助损失：保证 token 在专家之间的均匀分布与路由概率平衡；
     - 位置加权损失：结合可学习的变量权重 \(f(v)\) 与纬度权重 \(L(i)\)，对不同纬带/变量位置差错赋予不同惩罚。

### 2.2 结构模块映射
- 输入与预处理：
  - 输入张量：\(X \in \mathbb{R}^{C\times H\times W}\)，\(C=20\) 个变量通道；
  - patch 划分：大小 \(p\times p\)（文中为 8×8），得到 \(C\times(H/p)\times(W/p)\) 个 patch；
  - 对每个 patch 线性映射到 \(D\)-dim；
  - 使用 ClimaX 思路的 **跨通道 cross-attention**，聚合不同变量之间的物理关系，输出形状约为 \((H/p)\times(W/p)\) 的 token 序列。

- 3D 绝对位置嵌入：
  - 对每个 token 的经纬高位置 (lon, lat, level) 对应：
    - 高度位置嵌入向量：\(e_{z}\in\mathbb{R}^{D/3}\)；
    - 经度嵌入向量：\(e_{lon}\in\mathbb{R}^{D/3}\)；
    - 纬度嵌入向量：\(e_{lat}\in\mathbb{R}^{D/3}\)；
    - 拼接 \([e_z, e_{lon}, e_{lat}] \in \mathbb{R}^D\)，加到 token embedding 上。

- 编码器：
  - 深度为 6、维度 768；
  - 每层 Transformer encoder 中将标准 FFN 替换为 MoE 层；
  - Multi-head attention 结构遵循标准 Transformer。

- MoE 层：
  - 结构：gating network + \(N=20\) 个 FFN 专家；
  - 对每个输入 token \(x\)：
    - 计算门控 logits：\(xW + \epsilon\)，其中 \(W\) 为可训练矩阵、\(\epsilon\sim\mathcal{N}(0,1/e^2)\) 为高斯噪声；
    - top-2 路由：只保留前 2 大分量，其余置为 \(-\infty\)；
    - softmax 得到稀疏权重向量 \(g(x)\in\mathbb{R}^N\)；
    - 对被激活的专家前向，输出加权和 \(y = \sum_i g_i(x)E_i(x)\)。

- 解码器：
  - 深度 6、维度 512 的 Transformer decoder；
  - 接收 encoder feature，输出与 patch/变量布局对应的特征，最终经线性映射/reshape 恢复到 \(C\times H\times W\)。

- 训练损失：
  - 主损失：纬度加权 MSE + 变量位置权重；
  - 辅助损失：MoE 负载均衡损失 \(l_{aux}\)。


## 3. 模型架构总览

- 模型类型：
  - 编码器-解码器式 Vision Transformer，内部嵌入 **稀疏 MoE FFN**，并配有 **3D 绝对位置嵌入** 与 **特定损失**。

- 主要通路：
  1. 输入 ERA5 张量（20 通道，多层压、地表变量） → patch 切分与线性嵌入；
  2. 通过 cross-attention 融合变量信息，得到 token 序列；
  3. 加上 3D 绝对位置嵌入 → 送入 6 层 ViT 编码器（每层含 MoE）；
  4. 编码表示经 6 层解码器映射回 patch/变量空间；
  5. 恢复到 \(C\times H\times W\) 输出，并以自回归方式迭代推理未来 8 天（32 步）。

- 参数与计算：
  - 未在片段中给出精确参数量；总体思想是：
    - 通过 MoE 提升“可用参数规模”，但每 token 仅激活少数专家；
    - 相比 FourCastNet、ClimaX 等，**以更少训练数据（2 年 vs 30+ 年）和更少 GPU 时** 达到更好或相近性能。


## 4. 创新与优势

### 4.1 主要创新点
1. **3D 绝对位置嵌入**：
   - 针对气象变量对绝对经纬度/高度的物理依赖（如 geopotential vs 纬度、高度 vs 温度/风速），使用三维可学习 embedding，而非图像任务中常见的 1D/2D 相对位置编码；
   - 更贴合大气物理结构，提高位置信息利用效率。

2. **MoE 在天气预报中的有效应用**：
   - 将 NLP 领域成熟的稀疏 MoE 架构引入全球天气预报；
   - 证明在有限数据（2 年 ERA5）与有限算力下，MoE 架构可显著提升精度与稳定性；
   - 提供消融实验，验证 MoE 层对性能和资源效率的贡献。

3. **专门设计的两类损失**：
   - 单一、可微的负载均衡辅助损失 \(l_{aux}\)，同时控制 token 负载 \(h_i\) 与概率质量 \(P_i\)；
   - 位置加权 MSE 损失，结合变量位置重要度函数 \(f(v)\) 与标准纬度权重 \(L(i)\)，在物理关键区域提升精度。

4. **数据与算力效率**：
   - 仅使用 2015–2016 两年训练、2017 验证、2018 测试；
   - 在这样的低资源设定下，仍能全面超越 FourCastNet、ClimaX，并在 1–3 天时效上与 Pangu、GraphCast 相当，长时效更稳定。

### 4.2 性能表现（定性概述）
- 指标：纬度加权 ACC 与 RMSE；
- 关键变量：Z500, T2m, T850, U10；
- 结果（图 2 & 图 3）：
  - 对所有 lead time、所有分析变量，EWMoE 的 ACC 更高、RMSE 更低 于 FourCastNet 与 ClimaX；
  - 与 Pangu-Weather、GraphCast 对比：
    - 1–3 天时效准确度接近；
    - 随着时效延长，EWMoE 表现更稳定，退化更慢。


## 5. 关键公式与训练目标

### 5.1 MoE 前向与门控
1. MoE 输出（公式 (1)）：
\[
 y = \sum_{i=1}^N g_i(x) E_i(x),
\]
其中 \(E_i\) 为第 \(i\) 个专家 FFN，\(g_i(x)\) 为门控网络输出的权重。

2. top-\(k\) 路由与 softmax（公式 (2)(3)）：
\[
 g(x) = \mathrm{Softmax}(\mathrm{Top\!-
}k(xW + \epsilon, k)),
\]
\[
 \mathrm{Top\!-
}k(m,k)_i =
 \begin{cases}
 m_i, & m_i \text{ 在前 }k\text{ 大中},\\
 -\infty, & \text{否则},
 \end{cases}
\]
其中 \(W\) 为可训练矩阵，\(\epsilon \sim \mathcal{N}(0,1/e^2)\)。

### 5.2 MoE 负载均衡辅助损失
1. 辅助损失（公式 (4)）：
\[
 l_{aux} = E \cdot \sum_{i=1}^E h_i P_i,
\]
其中 \(E\) 为专家数（与 \(N\) 同义）。

2. token 分配比例 \(h_i\)（公式 (5)）：
\[
 h_i = \frac{1}{L} \sum_{x\in B} \mathbf{1}\{\arg\max g(x) = i\},
\]
其中 \(B\) 为 batch、\(L\) 为 token 总数。

3. 路由概率质量 \(P_i\)（公式 (6)）：
\[
 P_i = \frac{1}{L} \sum_{x\in B} g_i(x).
\]
- 通过最小化 \(l_{aux}\)，驱动 \(h_i\) 与 \(P_i\) 接近均匀分布，从而实现平衡路由和更高吞吐。

### 5.3 位置加权 MSE 损失
1. 总损失形式（公式 (7)）：
\[
 \mathcal{L} = \frac{1}{CHW} \sum_{c=1}^C\sum_{i=1}^H\sum_{j=1}^W
 f(v)\,L(i)\bigl(\hat X_{i+\Delta t}^{cij} - X_{i+\Delta t}^{cij}\bigr)^2,
\]
其中：
- \(f(v)\)：与变量绝对位置/类型相关的可学习权重函数；
- \(L(i)\)：纬度权重函数，见下式；
- \(\hat X\) 为预报，\(X\) 为真值。

2. 纬度权重（公式 (8)）：
\[
 L(i) = \frac{\cos(\mathrm{lat}(i))}{\frac{1}{H}\sum_{i'=1}^H \cos(\mathrm{lat}(i'))}.
\]

### 5.4 评估指标
1. ACC（公式 (9)）：
\[
 \mathrm{ACC}(v,l) =
 \frac{\sum_m L(m)\,\hat X_{pred}\,\hat X_{true}}
 {\sqrt{\sum_m L(m)\hat X_{pred}^2\,\sum_m L(m)\hat X_{true}^2}},
\]
其中 \(\hat X_{pred/true}\) 是减去长期平均后的异常量。

2. RMSE（公式 (10)）：
\[
 \mathrm{RMSE}(v,l) =
 \sqrt{\frac{1}{NM}\sum_m^M\sum_n^N L(m)(X_{pred}-X_{true})^2}.
\]


## 6. 数据规格

### 6.1 数据集与时间范围
- 数据集：ERA5（0.25° 分辨率，37 个压层，1940–至今）。
- 本文配置：
  - 空间：0.25°，网格为 721 × 1440（lat × lon）；
  - 时间：选取 6 小时间隔样本（T0, T6, T12, T18）；
- 划分：
  - 训练：2015–2016（2 年）；
  - 验证：2017（1 年）；
  - 测试：2018（1 年）。

### 6.2 变量与垂直层
- 每个样本为 20 个气象变量、5 个垂直层 + 地表/积分量（表 1）：
  - Surface：U10, V10, T2m, sp, mslp；
  - 1000 hPa：U, V, Z；
  - 850 hPa：T, U, V, Z, RH；
  - 500 hPa：T, U, V, Z, RH；
  - 50 hPa：Z；
  - Integrated：TCWV。


# 维度二：基础组件知识（Component-Level Knowledge）

## 组件 1：ViT 编码器-解码器骨干

1. 子需求定位
- 需要一个能在高分辨率全球网格上处理多通道气象“图像”的主干网络，支持 patch 化与注意力机制。

2. 技术细节
- 输入 \(X\in\mathbb{R}^{C\times H\times W}\) 划分为 \(p\times p\) patch；
- 每个 patch 线性映射到维度 \(D\)；
- 使用 learnable query 做 cross-attention 汇聚变量间信息（参考 ClimaX）；
- 编码器：6 层，dim=768，每层为 MHA + MoE FFN；
- 解码器：6 层，dim=512，输出映射回空间网格。

3. 耦合程度
- 与 3D 位置嵌入、MoE 层、损失函数紧密耦合，但整体结构为 **弱绑定-通用**（可迁移到其他地球系统任务）。


## 组件 2：3D 绝对位置嵌入

1. 子需求定位
- 子问题：气象变量强烈依赖 **经度、纬度、垂直高度**，如何在 token 表示中显式注入这些信息？

2. 技术与创新
- 每个 patch 拥有三个坐标：lon, lat, altitude（pressure level）；
- 为每个维度分别学习一个长度为 \(D/3\) 的 embedding 向量；
- 将三者拼接得到 \(D\)-dim 的 3D 绝对位置向量，加到 token embedding 上；
- 相比 1D/2D（如行列）位置编码，更贴近地球坐标与气象高度结构。

3. 上下文与耦合度
- 强绑定-气象特异：嵌入的三维坐标完全针对全球大气模型；
- 与位置加权损失在“位置重要性”建模上形成互补。

4. 伪代码（概念）
```pseudo
for each patch at (lon, lat, level):
    e_lon   = emb_lon_table[lon_index]
    e_lat   = emb_lat_table[lat_index]
    e_level = emb_level_table[level_index]
    pos_3d  = concat(e_level, e_lon, e_lat)  # dim = D
    token   = patch_embed + pos_3d
```


## 组件 3：稀疏 Mixture-of-Experts (MoE) 层

1. 子需求定位
- 子问题：在资源有限情况下，如何提升模型容量，以更好拟合复杂的大气动力学？

2. 技术与创新
- 结构：
  - N=20 个 FFN 专家，每个参数独立；
  - 门控网络 `g(x) = Softmax(Top-k(xW + ε, k))`，k=2；
  - 计算仅对被激活专家进行 FFN 前向；
- 创新点：
  - 将 NLP 大规模 MoE 思路迁移到全球天气预报；
  - 借由稀疏路由+负载均衡辅助损失实现高吞吐、稳定训练。

3. 耦合度
- 强绑定-实现层面：内嵌在每个 encoder block 中，直接替代标准 FFN；
- 与辅助损失强耦合，用于控制专家使用分布。

4. 伪代码
```pseudo
function moe_layer(x):
    logits = x @ W + epsilon  # [B*T, N]
    logits_topk = topk_mask(logits, k=2)  # others -> -inf
    gates = softmax(logits_topk, dim=-1)

    outputs = 0
    for i in active_experts(gates):
        y_i = experts[i](x)           # FFN_i(x)
        outputs += gates[:, i:i+1] * y_i
    return outputs
```


## 组件 4：MoE 负载均衡辅助损失

1. 子需求定位
- 子问题：防止 token 只集中流向少数“热门”专家，导致大部分专家欠训练、模型容量浪费。

2. 技术与创新
- 使用单一、可微的辅助损失（而非拆分的 load-balance & importance losses）：
\[
 l_{aux} = E \sum_i h_i P_i,
\]
- \(h_i\)：token 分配频率；\(P_i\)：门控概率质量；
- 目标：使 \(h_i, P_i\) 接近均匀，从而提升专家利用率与模型泛化。

3. 耦合度
- 与 MoE 层、主损失联合优化；
- 强绑定-架构特性（但思路可迁移至其他 MoE 模型）。


## 组件 5：位置加权损失（Position-weighted Loss）

1. 子需求定位
- 子问题：某些纬带/位置上的误差对社会影响更大（例如中纬度、人口密集区），如何在损失中显式体现？

2. 技术与创新
- 损失形式：
\[
 \mathcal{L} = \frac{1}{CHW}\sum_{c,i,j} f(v) L(i)(\hat X - X)^2,
\]
- \(f(v)\) 为与变量位置相关的 learnable 权重；
- \(L(i)\) 为标准纬度权重（cos(lat) 归一）；
- 提高对关键区域预测的优先级，有利于 ACC/RMSE 及实际应用价值。

3. 耦合度
- 与 3D 位置嵌入在“空间重要性”建模上形成协同；
- 强绑定-气象任务，但形式上是通用的加权 MSE。


## 组件 6：自回归多步推理框架

1. 子需求定位
- 子问题：如何从单步 6 小时预测扩展到 8 天（32 步）预测，同时控制训练与推理复杂度？

2. 技术与创新
- 自回归展开：
  - 仅训练单步映射 \(f_\theta\colon X_t\to X_{t+6h}\)；
  - inference 时将上一步预测作为下一步输入，实现长时序预测；
- 优点：
  - 避免为每个 lead time 单独训练网络；
  - 在固定网络下通过多步迭代覆盖 1–8 天范围。

3. 伪代码（概念）
```pseudo
X = X_init
forecasts = []
for step in range(32):
    X = f_theta(X)      # one-step forecast (6h)
    forecasts.append(X)
return forecasts
```


## 组件 7：评估指标与可视化

1. 子需求定位
- 定量评价 EWMoE 与 FourCastNet、ClimaX、Pangu、GraphCast 在多 lead time、多变量上的性能；

2. 技术与创新
- 使用纬度加权 ACC 与 RMSE（标准气象评估方式）；
- 对 Z500, T2m, T850, U10 等关键变量，绘制 lead time vs. ACC/RMSE 曲线；
- 提供空间误差场与预报场可视化，对比模型之间的空间结构差异。

3. 耦合度
- 作为评估模块，不参与梯度回传；
- 强绑定-气象标准，使结果可与 NWP 和其他 DL 模型直接对标。
