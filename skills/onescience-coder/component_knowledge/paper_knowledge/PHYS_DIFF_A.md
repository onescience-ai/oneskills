# PHYS-Diff：用于热带气旋预报的物理启发潜在扩散模型

---

## 一、整体模型知识（Model-level）

### 1.1 问题背景与任务定义

- **目标任务**：联合预测热带气旋（Tropical Cyclone, TC）的未来属性序列，包括：
  - 轨迹：纬度/经度坐标 \(\mathbf{x}_i \in \mathbb{R}^2\)
  - 最大持续风速：\(v_i \in \mathbb{R}\)
  - 中心最低气压：\(p_i \in \mathbb{R}\)
- **输入输出时序**：
  - 历史观测序列：\(\mathcal{H} = \{h_1, \dots, h_M\}\)，每个 \(h_i = (\mathbf{x}_i, v_i, p_i)\)
  - 目标未来序列：\(\mathcal{F} = \{f_1, \dots, f_N\}\)，预测时长取决于预报步长（6h、12h、…、120h）
  - 文中设定：\(M = 4\)，即使用前 4 个时间步的历史 TC 属性
- **环境场输入**：
  - 历史再分析场：\(\mathcal{E}_{\text{hist}} = \{E_1, \dots, E_M\}\)，源自 ERA5
  - 未来预报场：\(\mathcal{E}_{\text{fut}} = \{E'_1, \dots, E'_N\}\)，源自 FengWu 模型
  - 每个场：\(E \in \mathbb{R}^{C \times H \times W}\)，其中 \(C = 69\)，\(H = W = 80\)
- **总体目标**：
  - 在利用多模态信息（TC 历史属性 + ERA5 历史场 + FengWu 未来场）的前提下，生成物理一致的概率 TC 预报（轨迹 + 强度），并显式约束属性间的物理关系，降低长时效误差累积。

### 1.2 数据与任务设定

- **TC 真值**：
  - 轨迹与强度来自 IBTrACS（1980–2022），包含全球与西北太平洋（WP）盆地样本
- **环境场来源**：
  - ERA5 再分析资料（历史）
  - FengWu 大模型预报场（未来）
  - 空间分辨率：\(0.25^\circ\)，时间分辨率：6 小时
  - 变量：69 个（4 个地面变量 + 5 个在 13 个等压面上的变量）
- **裁剪策略**：
  - 以 TC 中心为圆心，裁剪半径 \(10^\circ\) 的区域
  - 对 ERA5 与 FengWu 均进行裁剪，得到统一大小张量 \(69 \times 80 \times 80\)
  - 未来 FengWu 裁剪中心由 TC 追踪算法给出（跟随预测轨迹移动）
- **数据集划分**：
  - 训练集：1980–2017
  - 验证集：2018
  - 测试集：2019–2022
- **归一化处理**：
  - 轨迹坐标：
    \[
      \mathbf{x}_{\text{rel}, i} = \frac{\mathbf{x}_i - \mathbf{x}_{\text{ref}}}{\sigma_{\text{coord}}}
    \]
    其中 \(\mathbf{x}_{\text{ref}}\) 为序列起点坐标，\(\sigma_{\text{coord}}\) 为坐标标准差，使模型对绝对位置不敏感
  - 强度与环境场：
    \[
      a_{\text{norm}} = \frac{a - \mu_a}{\sigma_a}
    \]
    对风速、气压以及全部环境变量分别使用训练集均值 \(\mu_a\) 和标准差 \(\sigma_a\) 标准化

### 1.3 模型总体结构

- **整体范式**：物理启发的潜在扩散模型（Latent Diffusion Model, LDM）
  - 在潜在空间而非原始数据空间上执行扩散-去噪过程
  - 结合条件编码器（多模态融合）与物理启发解码器（Physics-Inspired Decoder）
- **三个主要子模块**：
  1. **潜在编码/解码器**：\(\mathcal{E}, \mathcal{D}\)
     - 编码未来 TC 序列：\(\mathbf{x}_0 \in \mathbb{R}^{N \times 4}\)
     - 映射到潜在空间：
       \[
         z_0 = \mathcal{E}(\mathbf{x}_0), \quad z_0 \in \mathbb{R}^{N \times D_{\text{embedding}}}
       \]
     - 通过扩散模型在潜在空间生成 \(\hat{z}_0\)，随后解码为预测：
       \[
         \hat{\mathbf{x}}_0 = \mathcal{D}(\hat{z}_0)
       \]
  2. **条件编码器（Conditional Encoder）**：构造条件上下文 \(c\)
     - 历史 TC 属性：经 GRU 编码为 token \(H_{TC}\)
     - 历史+未来环境场：经 Swin Transformer 提取环境 token \(T_{env}\)
     - 时间步嵌入 \(t_{emb}\) 参与条件编码
     - 三者拼接后送入 Transformer Encoder，得到最终上下文记忆：
       \[
       c = \text{TransformerEncoder}\big([\text{GRU}(H),\; \text{SwinTransformer}([E_{\text{hist}}, E_{\text{fut}}]),\; t_{emb}]\big)
       \]
  3. **物理启发解码器（Physics-Inspired Decoder）**：核心是 PIGA 模块
     - 输入为带噪潜在变量 \(z_t\)
     - 堆叠若干解码块：自注意力（self-attn） + 跨注意力（cross-attn, 与 \(c\) 交互） + PIGA + FFN
     - 输出为噪声预测 \(\epsilon_\theta(z_t, t, c)\)

### 1.4 扩散过程与生成机理

- **前向扩散（forward diffusion）**：
  - 将初始潜在表示 \(z_0\) 逐步加噪至 \(z_T\)
  - 采用标准 DDPM 形式：
    \[
      q(z_t \mid z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t}\, z_0, (1 - \bar{\alpha}_t)\, \mathbf{I})
    \]
  - \(\bar{\alpha}_t\) 是预设噪声调度，\(T\) 为扩散步数（具体数值未在文中明确给出）

- **反向去噪（reverse denoising）**：
  - 从 \(z_T \sim \mathcal{N}(0, \mathbf{I})\) 出发，逐步去噪生成 \(z_0\)
  - 单步更新：
    \[
      z_{t-1} = \frac{1}{\sqrt{\alpha_t}} \Big( z_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(z_t, t, c) \Big) + \sigma_t \mathbf{w}
    \]
  - 其中 \(\alpha_t = 1 - \beta_t\)，\(\sigma_t\) 为固定方差，\(\mathbf{w}\) 为高斯噪声（\(t=1\) 时为 0）

### 1.5 训练目标与优化策略

- **扩散损失（噪声预测）**：
  - 主体为 \(L_{\text{diffusion}}\)，最小化真实噪声 \(\epsilon\) 与预测噪声 \(\epsilon_\theta\) 的均方误差：
    \[
      L_{\text{diffusion}} = \mathbb{E}_{t, z_0, \epsilon} \big[\,\| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \big]
    \]

- **重建损失（Reconstruction）**：
  - 在数据空间上评估预测序列误差：\(L_{\text{recon}}\)
  - 分解为三部分：
    \[
      L_{\text{recon}} = L_{\text{traj}} + L_{\text{wind}} + L_{\text{pres}}
    \]
  - 针对轨迹、风速、气压三任务分别计算误差（具体采用 MAE/MSE 等指标的形式在文中未逐项写出，默认为回归误差）

- **任务特异梯度路由（Task-specific Gradient Routing）**：
  - 反向传播时，每一任务的重建损失只更新其对应的 PIGA 投影层参数
  - 例如：\(L_{\text{traj}}\) 仅更新轨迹投影 \(\text{Proj}_{\text{traj}}\) 相关参数
  - 旨在强化特征解耦，避免不同任务梯度相互干扰

- **不确定性加权多任务损失**：
  - 最终损失为：
    \[
      L_{\text{total}} = \frac{1}{2\sigma_{\text{diff}}^2} L_{\text{diffusion}} + \frac{1}{2\sigma_{\text{recon}}^2} L_{\text{recon}} + \log(\sigma_{\text{diff}}\sigma_{\text{recon}})
    \]
  - \(\sigma_{\text{diff}}^2\)、\(\sigma_{\text{recon}}^2\) 为可学习参数，表示各子任务（扩散/重建）的不确定性
  - 属于典型的基于不确定性的多任务加权策略

- **优化细节**：
  - 框架：PyTorch
  - 优化器：Adam
  - 初始学习率：\(1 \times 10^{-4}\)，采用 cosine annealing
  - 批大小：64
  - 训练轮数：30
  - 硬件：单张 NVIDIA RTX 4090 GPU

### 1.6 预测与集成推理

- **单样本生成流程（概念性伪代码）**：

  1. 给定历史 TC 序列 \(H\) 与环境场 \(E_{\text{hist}}, E_{\text{fut}}\)
  2. 通过 GRU、Swin Transformer、Transformer Encoder 得到上下文 \(c\)
  3. 从高斯噪声采样 \(z_T \sim \mathcal{N}(0, \mathbf{I})\)
  4. 对 \(t = T, \dots, 1\)：重复使用解码器预测噪声 \(\epsilon_\theta\)，按公式 (2) 更新 \(z_{t-1}\)
  5. 得到 \(\hat{z}_0\) 后，经解码器 \(\mathcal{D}\) 映射到数据空间得到 \(\hat{\mathbf{x}}_0\)

- **集成（Ensemble）预测**：
  - 由于模型是生成式的，可通过不同初始化噪声采样形成集合预报
  - 文中设定：\(N = 50\) 个成员，从不同高斯噪声初值生成 50 组预测
  - 使用集合均值作为最终预测，可显著提升准确率并量化不确定性（见消融表中的 “Phys-Diff (Ensemble)”）

### 1.7 评价指标与实验设定

- **评价指标**（主指标为 MAE）：
  - 轨迹误差（km）：预测坐标与真值坐标间的大圆距离（Haversine 公式）
  - 气压误差（hPa）：预测与真值的最低海平面气压绝对差
  - 风速误差（m/s）：预测与真值最大持续风速的绝对差

- **对比方法**：
  - 传统/早期 DL：GRU、GBRNN
  - 近期 DL：MSCAR、VQLTI、MMSTN、MGTCF、TC-Diffuser
  - 大模型：FengWu
  - 业务 NWP：ECMWF

- **实验区域**：
  - 全球（Global）
  - 西北太平洋（WP）

### 1.8 关键结果与性能总结

- **全球盆地（Global）24h 结果（表 1）**：
  - 相比 ECMWF，24h 轨迹误差降低 \(25.0\%\)
  - 相比 MSCAR，24h 气压误差降低 \(57.1\%\)
  - 相比 MSCAR，24h 风速误差降低 \(71.2\%\)
  - 同时优于 FengWu 等深度学习/大模型方法

- **西北太平洋（WP）24h 结果**：
  - 相比 TC-Diffuser，24h 轨迹误差降低 \(28.5\%\)
  - 24h 气压、风速误差也显著低于 TC-Diffuser

- **消融实验（表 2）**：
  - 去掉 PIGA（w/o PIGA）：24h 轨迹、强度误差明显升高，验证 PIGA 的关键作用
  - 去掉 FengWu 未来场（w/o FengWu）：
    - 仅用历史 ERA5 时仍表现较强
    - 加入 FengWu 可进一步提升长期预报
  - 同时去掉 PIGA 和 FengWu（w/o both）：性能退化最严重
  - 集合预报（Phys-Diff Ensemble, N=50）：所有时效上误差进一步降低

- **定性分析**：
  - 通过典型个例（如 Haleh 2019, ISAIAS 2020, DIANE 2020），展示 Phys-Diff 在直线移动、急转弯、螺旋变化、登陆交互等复杂轨迹下，相比 FengWu 具有更高的路径拟合能力
  - 通过 t-SNE 展示 PIGA 学到的任务特异特征：
    - 轨迹（pink）、气压（black）、风速（purple）形成三个清晰簇
    - 气压与风速簇较近并有重叠，符合它们物理耦合特性
    - 轨迹簇相对远离，体现其位置属性的独特性

---

## 二、基础组件知识（Component-level）

### 2.1 潜在扩散建模组件

#### 2.1.1 潜在编码器与解码器（\(\mathcal{E}, \mathcal{D}\)）

- **输入输出**：
  - 输入：未来 TC 序列 \(\mathbf{x}_0 \in \mathbb{R}^{N \times 4}\)（2 维轨迹 + 风速 + 气压）
  - 输出潜在表示：\(z_0 \in \mathbb{R}^{N \times D_{\text{embedding}}}\)
  - 最终解码输出：\(\hat{\mathbf{x}}_0 = \mathcal{D}(\hat{z}_0)\)
- **结构形态**：
  - 文中仅给出“卷积编码器”字样（convolutional encoder），具体层数/通道数等未详细说明 → 记为“未明确说明”
  - 解码器为对称结构或 MLP 形式，同样未展开细节
- **作用**：
  - 将复杂的时序多变量回归问题映射到相对低维、语义更集中的潜在空间
  - 降低扩散建模的计算成本，提高表示能力

#### 2.1.2 扩散与去噪网络 \(\epsilon_\theta\)

- **网络类型**：
  - 条件 Transformer 编码器-解码器结构
  - 编码器用于融合多模态条件，解码器用于在潜在空间执行噪声预测

- **单步去噪更新（伪代码级逻辑）**：

  - 输入：当前带噪 latent \(z_t\)、步长 \(t\)、条件 \(c\)
  - 输出：更新后的 \(z_{t-1}\)
  - 更新公式如前所述：
    \[
      z_{t-1} = \frac{1}{\sqrt{\alpha_t}} \Big( z_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(z_t, t, c) \Big) + \sigma_t \mathbf{w}
    \]

- **噪声预测目标**：
  - 通过最小化 \(\| \epsilon - \epsilon_\theta(z_t, t, c) \|^2\) 学习在任意扩散步长下的去噪映射

### 2.2 条件编码器模块

#### 2.2.1 GRU 基于历史 TC 属性编码

- **输入**：
  - 历史 TC 序列 \(H = \{h_1, \dots, h_M\}\)，每个包含 \((\mathbf{x}_i, v_i, p_i)\)
- **结构与输出**：
  - 经过 GRU 编码，得到表示历史轨迹与强度演变的 token：
    \[
      H_{TC} = \text{GRU}(H)
    \]
  - 具体 GRU 隐层维度、层数未在文中给出 → 记为“未明确说明”

- **作用**：
  - 捕捉历史 TC 属性的时间依赖
  - 为后续解码器提供关于 TC 发展阶段的先验信息

#### 2.2.2 Swin Transformer 环境场编码

- **输入**：
  - 历史环境场：\(E_{\text{hist}}\)
  - 未来环境场：\(E_{\text{fut}}\)
  - 二者拼接后送入 Swin Transformer
- **输出**：
  - 环境特征 token 集合：
    \[
      T_{env} = \text{SwinTransformer}([E_{\text{hist}}, E_{\text{fut}}])
    \]
- **结构细节**：
  - 使用层级型窗口注意力结构，但具体 stage 数、窗口大小等未展开说明 → 标记为“未明确说明”

- **作用**：
  - 融合历史与未来大尺度环境信息（再分析 + 预报场）
  - 提供高维空间场的紧凑表示，帮助模型理解导引气流、环境剪切、涡度等对 TC 演变的影响

#### 2.2.3 Transformer Encoder 条件融合

- **输入序列**：
  - 历史 TC token：\(H_{TC}\)
  - 环境 token：\(T_{env}\)
  - 时间步嵌入：\(t_{emb}\)
- **融合公式**：
  \[
    c = \text{TransformerEncoder}([H_{TC}, T_{env}, t_{emb}])
  \]
- **作用**：
  - 在统一 token 序列上执行多头自注意力，使不同模态信息充分交互
  - 产生统一的上下文记忆 \(c\)，供解码器 cross-attention 调用

### 2.3 物理启发解码器与 PIGA 模块

#### 2.3.1 解码器基本结构

- **解码块组成**：
  1. Self-Attention：在当前 latent token 序列上执行自注意力，建模时间步之间的依赖
  2. Cross-Attention：以条件记忆 \(c\) 为 Key/Value，将环境与历史信息融入到 latent 表示中，得到 \(X_{\text{cross}}\)
  3. PIGA 模块：在 \(X_{\text{cross}}\) 上施加物理启发的任务间交互与 gating，输出 \(X_{\text{PIGA}}\)
  4. FFN：前馈网络进一步非线性变换

- **堆叠**：多个解码块按层堆叠，逐层 refine 去噪表示

#### 2.3.2 PIGA：Physics-Inspired Gated Attention

- **目标**：
  - 在潜在空间中显式建模 TC 属性之间的物理关系（轨迹、风速、气压）
  - 实现特征“解耦 + 物理耦合”：每个任务拥有独立特征流，但通过 cross-task 注意力实现物理一致性

- **输入**：
  - Cross-attention 输出的上下文感知特征：\(X_{\text{cross}} \in \mathbb{R}^{N \times D_{\text{model}}}\)

- **步骤 1：任务特异投影（Decomposition）**：
  \[
    f_{\text{traj}} = \text{Proj}_{\text{traj}}(X_{\text{cross}}), \quad
    f_{\text{wind}} = \text{Proj}_{\text{wind}}(X_{\text{cross}}), \quad
    f_{\text{pres}} = \text{Proj}_{\text{pres}}(X_{\text{cross}})
  \]
  - 每个 \(f_* \in \mathbb{R}^{N \times D_{\text{sub}}}\)

- **步骤 2：跨任务注意力（Interaction）**（以轨迹流为例）：
  \[
    A_{\text{traj}} = \text{Attention}(Q = f_{\text{traj}}, K, V = [f_{\text{wind}}, f_{\text{pres}}])
  \]
  - 轨迹特征通过关注风速、气压特征，学习其物理依赖
  - 类似地，对风速、气压流也构造各自的 cross-task attention

- **步骤 3：门控融合（Gating）**：
  - 计算自适应门值：
    \[
      g_{\text{traj}} = \sigma\big(\text{MLP}([f_{\text{traj}}, A_{\text{traj}}])\big)
    \]
  - 利用门控在“原始任务特征”和“物理增强特征”之间插值：
    \[
      f'_{\text{traj}} = (1 - g_{\text{traj}}) \odot f_{\text{traj}} + g_{\text{traj}} \odot A_{\text{traj}}
    \]
  - 对风速、气压分支同理

- **步骤 4：融合输出（Fusion）**：
  \[
    X_{\text{PIGA}} = \text{Conv}_{1\times 1}(\text{Concat}(f'_{\text{traj}}, f'_{\text{wind}}, f'_{\text{pres}}))
  \]

- **效果与直观解释**：
  - t-SNE 可视化显示：轨迹/气压/风速特征形成分离簇，其中气压与风速簇距离更近且部分重叠，反映强度变量间耦合更紧密
  - PIGA 的移除显著恶化性能（表 2），说明物理启发设计对性能贡献巨大

- **任务特异梯度路由**：
  - 每个任务的重建损失仅更新其对应投影层 \(\text{Proj}_{\text{traj}}, \text{Proj}_{\text{wind}}, \text{Proj}_{\text{pres}}\)
  - 进一步保证解耦特征不会在梯度上相互污染

### 2.4 损失与多任务不确定性加权

#### 2.4.1 扩散损失 \(L_{\text{diffusion}}\)

- **定义**：
  \[
    L_{\text{diffusion}} = \mathbb{E}_{t, z_0, \epsilon} \big[\,\| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \big]
  \]
- **作用**：
  - 保证在潜在空间的每个扩散步长，网络都能准确预测添加的噪声

#### 2.4.2 重建损失 \(L_{\text{recon}}\)

- **分解**：
  \[
    L_{\text{recon}} = L_{\text{traj}} + L_{\text{wind}} + L_{\text{pres}}
  \]
- **任务路由**：
  - \(L_{\text{traj}}\) → 仅更新轨迹分支相关 PIGA 投影
  - \(L_{\text{wind}}\) → 仅更新风速分支
  - \(L_{\text{pres}}\) → 仅更新气压分支

#### 2.4.3 总损失与不确定性加权

- **公式**：
  \[
    L_{\text{total}} = \frac{1}{2\sigma_{\text{diff}}^2} L_{\text{diffusion}} + \frac{1}{2\sigma_{\text{recon}}^2} L_{\text{recon}} + \log(\sigma_{\text{diff}}\sigma_{\text{recon}})
  \]
- **解释**：
  - \(\sigma_{\text{diff}}^2\)、\(\sigma_{\text{recon}}^2\) 为可学习不确定性参数
  - 不确定性越大，对应任务损失权重越小，实现自适应多任务平衡

### 2.5 数据预处理与评价组件

#### 2.5.1 轨迹误差计算（Haversine 公式）

- **球面大圆距离**：
  - 将经纬度转换为弧度 \((\phi, \lambda)\)
  - 设预测点为 \((\phi_1, \lambda_1)\)，真值为 \((\phi_2, \lambda_2)\)，则：
    \[
      d = 2R \arcsin\Bigg(\sqrt{\sin^2\Big(\frac{\phi_2 - \phi_1}{2}\Big) + \cos\phi_1\cos\phi_2\sin^2\Big(\frac{\lambda_2 - \lambda_1}{2}\Big)}\Bigg)
    \]
  - \(R\) 为地球半径（约 6371 km）

#### 2.5.2 归一化组件

- **坐标归一化**：
  \[
    \mathbf{x}_{\text{rel}, i} = \frac{\mathbf{x}_i - \mathbf{x}_{\text{ref}}}{\sigma_{\text{coord}}}
  \]

- **变量标准化**：
  \[
    a_{\text{norm}} = \frac{a - \mu_a}{\sigma_a}
  \]
  - 对风速、气压、所有环境变量分别统计均值/标准差

### 2.6 集成预报与不确定性量化组件

- **单成员推理**：
  - 从噪声 \(z_T\) 出发，按 DDPM 反向过程生成一次 \(\hat{\mathbf{x}}_0\)

- **集合生成**：
  - 采样 \(N = 50\) 个不同噪声初始值 \(z_T^{(k)} \sim \mathcal{N}(0, \mathbf{I})\)
  - 得到 50 组预测 \(\hat{\mathbf{x}}_0^{(k)}\)

- **集合统计**：
  - 集合均值作为最终预测：
    \[
      \bar{\mathbf{x}}_0 = \frac{1}{N} \sum_{k=1}^N \hat{\mathbf{x}}_0^{(k)}
    \]
  - 方差/分位数可用于不确定性量化（论文中主要展示均值效果）

- **实验发现**：
  - Ensemble 相比单次采样在所有时效、所有指标上均有提升（表 2）

---

## 三、创新点与适用场景小结

### 3.1 核心创新

- 在 **潜在扩散框架** 中显式注入物理先验，将 TC 轨迹、风速、气压之间的物理关系嵌入生成过程，而非仅在输出后进行约束
- 通过 **PIGA 模块** 实现任务特异特征解耦与跨任务物理一致性交互，并结合梯度路由保证每个任务分支独立优化
- 利用 **多模态条件**（IBTrACS 轨迹/强度 + ERA5 历史场 + FengWu 未来场）实现对 TC 演变的更全面刻画
- 引入 **不确定性加权多任务损失**，自动平衡扩散与重建目标
- 充分利用扩散模型的生成特性构造 **集合预报**，在 TC 预报中同时获得精度与不确定性量化

### 3.2 适用与扩展方向（基于论文已给信息）

- 适用于：
  - 需要联合预测轨迹与强度，并关心三者物理一致性的 TC 预报场景
  - 具备再分析数据与数值/AI 未来预报场的环境
- 潜在扩展：
  - 将 PIGA 思路迁移至其他多任务气象问题（如降水 + 风 + 温度联合预报），但论文未具体展开，属于推断方向，需额外验证。