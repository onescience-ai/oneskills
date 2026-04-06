# AI-GOMS 模型知识提取

> 数据来源：AI-GOMS: Large AI-Driven Global Ocean Modeling System（AI-GOMS.md）

---

## 维度一：整体模型知识（Model-Level Knowledge）

### 1. 核心任务与需求

#### 1.1 核心建模任务
- 目标：构建一个大规模 AI 驱动的全球海洋建模系统 AI-GOMS，实现**全球海洋日尺度预测与模拟**。
- 主任务：
  - 对全球海洋**5 个基本物理变量**进行最长 30 天的逐日预测：
    - 海温 T（15 层）
    - 海盐 S（15 层）
    - 流速 zonal velocity U（15 层）
    - 流速 meridional velocity V（15 层）
    - 海表高度 SSH（1 层）
  - 水平分辨率：$1/4^\circ$，15 个深度层（0–500 m）。
- 下游任务：在已训练骨干模型基础上，通过轻量下游模块完成：
  - 区域下采样/区域放大（Regional Downscaling）：在如黑潮区域从 $1/4^\circ$ 下采样到 $1/12^\circ$，解析中尺度涡旋与路径细节。
  - 波浪解码（Wave Decoding）：从特征张量和风场等闭合条件解码显著波高（Significant Wave Height, SWH），实现 30 天预测。
  - 生物地球化学耦合（Biochemistry Coupling）：融合物理特征与生物地球化学条件，预测 8 个生物地球化学变量的 30 天演变。

#### 1.2 解决了什么问题
- 针对传统数值海洋模式的瓶颈：
  - **非线性不稳定性**与数值算法复杂度高。
  - **计算开销大**：高保真全球模拟直接求解原始方程需要比当前超级计算机**高 10 亿倍的存储与计算能力**。
  - **复用性与可移植性差**：
    - 传统模式高度依赖特定方程组与参数化方案，
    - 模型改动和新任务适配成本高。
  - **高耦合成本**：物理–生物地球化学等多模式之间通过外部耦合器耦合，成本高且灵活性不足。
- 针对 AI 在地球系统模拟中的现有不足：
  - 尚无**大规模深度学习模型**专门面向全球海洋建模。
  - 海洋系统具备**封闭复杂边界几何约束**，现有大气模型难以直接迁移。
  - 现有大模型**可迁移性不足**，未充分展示在多种下游海洋应用上的统一可复用能力（大型模型未被证明可在多下游任务上高效微调复用）。
- 针对海洋观测与分辨率需求：
  - 观测数据**稀疏且分布不均衡**，深海大范围缺测。
  - 中尺度涡解析能力是高质量全球模拟的前提，需要**区域更高分辨率嵌套网格**以解析 Rossby 变形半径尺度过程。

---

### 2. 解决方案

#### 2.1 核心解决思路
- 提出 **AI-GOMS**：一个由**大规模骨干模型 + 轻量下游模块**组成的 AI 驱动全球海洋建模系统：
  - 骨干模型（Backbone）：
    - 采用**基于 Fourier 的 Masked Autoencoder 结构**，即**非对称编码器–解码器（asymmetric encoder-decoder）+ patch embedding/recovery + 1D-AFNO Fourier 块**。
    - 支持多源输入（2D/3D/稀疏）并统一为规则网格数据，通过 patch 表征与序列 token 化，对全球海洋基础物理变量进行逐日自回归预测。
    - 利用**随机掩码策略（random mask strategy）**，在训练阶段对部分 patch 随机遮蔽，迫使模型学习更本质的物理特征，而非仅做局地插值，缓解过拟合并提升长期预测能力。
  - 下游模块（Downstream Modules）：
    - 在冻结骨干权重的基础上，通过**轻量微调（lightweight fine-tuning models）**，将骨干模型的特征张量与特定任务条件合并，完成：
      - 区域下采样/分辨率提升（downscaling）
      - 波浪变量解码（wave decoding）
      - 生物地球化学变量耦合预测（biochemistry coupling）
    - 微调成本极低，保证**可迁移性与复用性**。
- 总体范式：
  - 从传统数值模式的“动力内核 + 外部耦合器”转向**“AI 骨干 + 下游轻量模块”的 backbone–downstream 范式**，作为新一代地球系统建模范式。

#### 2.2 结构映射（思路与架构对应关系）
- 大规模多源数据驱动 + 基础物理变量预测 →
  - 对应：**AI-GOMS Backbone**：
    - Patch Embedding 模块：多变量、2D/3D/sparse 数据统一映射为 1D token 序列。
    - Asymmetric Encoder–Decoder with 1D-AFNO：在 token 空间建模全局–局地动力学关系，支持不同长度序列。
    - Random Mask Strategy：用于训练时的自监督式掩码学习。
    - Patch-Recovery 模块：将 token 序列还原到物理场网格。
- 解决区域分辨率需求（如黑潮等区域需要 $1/12^\circ$ 分辨率）→
  - 对应：**Regional Downscaling Module**：
    - 输入骨干特征张量 + 区域低分辨率表面变量，
    - 经过多层残差卷积块（Residual Conv blocks）+ ConvTranspose2d 上采样层实现 3 倍空间超分辨率。
- 解码波浪/生化等“派生变量”→
  - 对应：**Wave Decoding Module / Biochemistry Coupling Module**：
    - 从骨干特征张量和任务特定条件（风场、生化初始场等）解码目标变量，
    - 使用两个 1D-AFNO block + 线性投影层完成。

---

### 3. 模型架构概览

#### 3.1 总体流程
- 架构类型：
  - **骨干–下游（Backbone–Downstream）范式**，骨干为**Fourier-based Masked Autoencoder 风格的 Transformer-like 结构**（patch embedding + 1D-AFNO），下游为针对不同任务的轻量 CNN/AFNO 模块。

- 骨干模型（Backbone）数据流：
  1. 数据预处理：
     - 将 HYCOM、ERA5、ETOPO 等多源数据统一为规则网格，
     - 2D/3D 变量统一为形状 $C \times 720 \times 1440$ 的张量，稀疏数据通过掩码转为网格数据。
  2. 输入组装：
     - 输入张量 $X \in \mathbb{R}^{C_{in} \times H \times W}$，其中：
       - $C_{in}$ 由多变量叠加得到，总通道数为 67（文中示例值）。
       - $H = 720$，$W = 1440$。
     - 包含：
       - 15 层 T、S、U、V；
       - 1 层 SSH；
       - 5 个大气强迫变量（U10, V10, T2m, MSL, SP）；
       - 地形 Topology。
  3. Patch Embedding：
     - 将 $X$ 按 $p \times p$（$p=8$）非重叠划分为 patch，
     - 映射为 token 序列：
       $$
       X_{tokens} \in \mathbb{R}^{(h \times w) \times C_{\text{embed\_dim}}},
       $$
       其中 $h = H/p$，$w = W/p$。
  4. 随机掩码 + 位置编码：
     - 对 token 序列施加随机掩码，保留部分 token；
     - 对保留 token 加上余弦位置编码（cosine positional encoding）。
  5. 编码器（Encoder）：
     - 若干层 1D-AFNO blocks（基于自适应傅里叶神经算子，Fourier-based attention blocks），对 token 序列进行全局–局地混合建模。
  6. 解码器（Decoder）：
     - 利用 token 的 idrank 信息，将编码结果重排并输入解码器中较浅的 1D-AFNO blocks，完成重建。
  7. Patch Recovery：
     - 通过 2D 反卷积（ConvTranspose2d）将 token 序列还原到物理空间：
       $$
       X_{out} \in \mathbb{R}^{C_{out} \times H \times W},
       $$
       其中 $C_{out}$ 为输出变量个数（SSH + 4 个 3D 变量的 15 层）。
  8. 自回归预测：
     - 使用自回归策略生成 T 步预测：
       - 时间步 $t$ 的预测 $X_o^t$ 与边界条件一起作为 $t+1$ 步的输入 $X_i^{t+1}$，
       - 迭代生成 30 天逐日预测。

- 下游模块（Downstream Modules）数据流：
  1. 从骨干模型中截取**解码器前、patch-recovery 之前**的特征张量：
     $$
     X_f \in \mathbb{R}^{C_{decoder\_dim} \times h \times w}.
     $$
  2. 将 $X_f$ 与任务特定条件张量 $X_{d_{in}} \in \mathbb{R}^{C_{d_{in}} \times H_d \times W_d}$ 在通道维或拼接映射后融合。
  3. 经各任务相应的轻量网络映射至任务输出：
     - Downscaling：Residual Conv blocks + ConvTranspose2d。
     - Wave Decoding：2 × 1D-AFNO blocks + projection layer。
     - Biochemistry Coupling：2 × 1D-AFNO blocks + projection layer。

---

### 4. 创新与未来

#### 4.1 创新点（论文声称的主要贡献）

- **首个大规模深度学习全球海洋建模系统**：
  - 针对海洋这一地球系统关键组成部分，提出专门的大规模 AI 模型（此前大模型主要聚焦大气/天气中期预报）。

- **Fourier-based Masked Autoencoder 骨干结构**：
  - 使用非对称编码–解码结构 + 随机掩码策略：
    - 支持**不同长度时间序列**输入输出；
    - 类似 MAE 的训练范式，使模型在缺失信息的情形下学习隐藏物理特征，减弱对局地插值的依赖，提升长期预测性能。
  - 引入 1D-AFNO（自适应傅里叶神经算子）作为 token mixer：
    - 更高效地在频域建模全球尺度的海洋动力过程。

- **多源数据统一建模与天然“模型级同化”能力**：
  - 支持 2D/3D/稀疏数据，通过统一 patch embedding 转为规则网格；
  - 模型可被多源数据驱动，天然支持**在模型级别进行数据同化**。

- **Backbone–Downstream 范式与可迁移性验证**：
  - 提出“骨干 + 轻量下游模块”的新范式：
    - 骨干负责基础物理变量的全局动力模拟；
    - 下游模块针对不同场景（下采样、波浪、生化）进行轻量微调。
  - 在三个示例下游任务上展示了**骨干模型的可迁移能力**与**极低微调成本**。

- **区域下采样支持中尺度涡解析**：
  - 提出区域 downscaling 模块，实现从 $1/4^\circ$ 到 $1/12^\circ$ 的嵌套格点，解析如黑潮路径和中尺度涡结构，而且具备较高 ACC（7 天流速和 SSH 预测 ACC > 0.6）。

- **解码波浪与生化变量的统一框架**：
  - 使用骨干特征张量 + 少量边界条件/初始条件，通过轻量 1D-AFNO 模块解码：
    - 显著波高（SWH）；
    - 8 个生物地球化学变量，包括 TCa、Chl、Dia、Coc、Cya、Irn、Nit、MLD。
  - 展示了物理–生化协同模拟中 AI 框架的可行性。

- **物理一致性与长期稳定性**：
  - 在 30 天预测中，AI-GOMS 不仅在 RMSE/ACC 指标上优于 FourCastNet 等基线，还能合理模拟：
    - 赤道太平洋的温度廓线、混合层与温跃层；
    - 受 Walker 环流与 Bjerknes 反馈调制的赤道海温结构；
    - 东太平洋沿岸上升流及对应的叶绿素高值带。

#### 4.2 后续研究方向

论文中明确提到的未来工作包括：

1. **稀疏观测数据同化与端到端训练**：
   - 架构支持稀疏数据作为输入，未来将探索：
     - 利用稀疏观测数据进行数据同化；
     - 端到端地用稀疏观测驱动训练。

2. **引入更多迁移学习技术**：
   - 在现有骨干–下游范式基础上，引入更丰富的迁移学习手段，进一步降低下游微调成本。

---

### 5. 实现细节与代码逻辑

#### 5.1 理论公式

##### 5.1.1 统计度量

- **纬向权重（Latitude Weighting）**

$$
L(j) = \frac{\cos(\text{lat}(j))}{\dfrac{1}{N_{lat}} \sum_{j}^{N_{lat}} \cos(\text{lat}(j))},
$$

其中 $N_{lat}$ 为纬向格点数，$\text{lat}(j)$ 为第 $j$ 个纬度的纬度值。

- **加权均方根误差（RMSE）**

对于变量 $c$ 在预报时刻 $t$：

$$
\mathrm{RMSE}(c, t) = \sqrt{\frac{1}{|G|} \sum_{i,j} L(i) \left(\mathbf{X}^{t}_{pred}[c, i, j] - \mathbf{X}^{t}_{true}[c, i, j]\right)^2},
$$

其中 $G$ 为海洋格点集合，$\mathbf{X}^{t}_{pred}$、$\mathbf{X}^{t}_{true}$ 分别为预测值和真值。

- **异常相关系数（ACC）**

对于变量 $c$ 在预报时刻 $t$：

$$
\mathrm{ACC}(c, t) = \frac{\sum_{i,j} L(i) \tilde{\mathbf{X}}^{t}_{pred}[c, i, j] \, \tilde{\mathbf{X}}^{t}_{true}[c, i, j]}{\sqrt{\sum_{i,j} L(i) \left(\tilde{\mathbf{X}}^{t}_{pred}[c, i, j]\right)^2 } \, \sqrt{\sum_{i,j} L(i) \left(\tilde{\mathbf{X}}^{t}_{true}[c, i, j]\right)^2 }},
$$

其中 $\tilde{\mathbf{X}}$ 表示减去气候平均值后的异常场，气候平均由 2000–2010 年每 2 年采样 HYCOM 重分析数据计算得到。

##### 5.1.2 自回归预测与输入输出定义

- 输入张量：

$$
X_i^t \in \mathbb{R}^{C_{in} \times H \times W},
$$

包含 11 个变量（见数据规格部分），由两部分组成：

1. 初始条件或模型输出（海洋预测变量）：海温、海盐、流速、SSH 等；
2. 边界条件（大气强迫与地形）：U10, V10, T2m, MSL, SP, Topology。

- 输出张量：

$$
X_o^t \in \mathbb{R}^{C_{out} \times H \times W},
$$

包含：海表高度 SSH、温度 T、盐度 S、zonal 流速 U、meridional 流速 V。

- 自回归更新：

$$
X_i^{t+1} = \text{concat}\big( X_o^t, \text{BC}^{t+1} \big),
$$

其中 $\text{BC}^{t+1}$ 为下一时刻的边界条件，concat 表示在通道维的拼接（逻辑来源于文中“feeding its own output for the preceding time step combined with boundary conditions”描述，具体实现未在原文给出）。

#### 5.2 实现描述（关键步骤）

##### 5.2.1 Patch 划分与嵌入

- 原始输入张量：$X \in \mathbb{R}^{C_{in} \times 720 \times 1440}$。
- Patch 尺寸：$p \times p$，$p=8$。
  - Patch 网格尺寸：$h = 720/8 = 90$，$w = 1440/8 = 180$。
- 步骤：
  1. 将 $X$ 按非重叠的 $8 \times 8$ 小块划分；
  2. 对每个 patch 展平成向量，并通过线性映射得到维度为 $C_{embed\_dim}$ 的 token；
  3. 所有 patch 形成 token 序列 $X_{tokens} \in \mathbb{R}^{(h\times w) \times C_{embed\_dim}}$。

##### 5.2.2 随机掩码策略（Random Mask Strategy）

- 训练阶段，对 token 序列施加随机掩码：
  - 随机选取一定比例的 token 作为“可见 token”；
  - 其余 token 被标记为“掩码 token”，在编码器输入时被丢弃；
  - 解码器阶段根据 idrank 信息恢复 token 位置，并对掩码 token 进行重建。
- 掩码比例随训练进行**逐渐减小**，以在初期加强对隐含物理结构的学习、后期增强精细重建能力。

##### 5.2.3 1D-AFNO 块（encoder/decoder 内部）

- 每个 1D-AFNO block 使用基于 Fourier 的 token 混合器：
  - 对 token 序列在某一维度（如序列维）进行 FFT；
  - 在频域对通道进行加权或线性变换（实现“Fourier-based attention”）；
  - 进行逆 FFT 返回时域；
  - 搭配残差连接与前馈网络构成完整 block。
- 该部分具体数学公式未在文中详细给出，仅指出引用 AFNO 相关工作作为“efficient token mixers for transformers”。

##### 5.2.4 Patch-Recovery 模块

- 输入：解码器输出的 token 序列，形状为 $(h \times w) \times C_{decoder\_dim}$。
- 操作：
  1. 将 token 序列重排为 $C_{decoder\_dim} \times h \times w$ 的中间特征图；
  2. 使用二维转置卷积（ConvTranspose2d），步幅为 $p$，将 $(h, w)$ 上采样回 $(H, W)$；
  3. 最后通过线性/卷积映射到 $C_{out}$ 个输出通道，得到 $X_{out}$。

##### 5.2.5 下游模块的输入合并逻辑

- 对于任意下游任务：
  - 骨干特征：$X_f \in \mathbb{R}^{C_{decoder\_dim} \times h \times w}$；
  - 任务条件：$X_{d_{in}} \in \mathbb{R}^{C_{d_{in}} \times H_d \times W_d}$。
- 一般处理流程：
  1. 如有需要，对 $X_f$ 或 $X_{d_{in}}$ 进行插值/上采样，以对齐空间分辨率；
  2. 沿通道维拼接或经过线性映射后融合；
  3. 经相应的轻量网络（Residual Conv 或 AFNO blocks）输出下一个时间步的任务变量。

#### 5.3 伪代码/代码片段

以下伪代码均为**基于文中描述生成的伪代码**，用于结构化呈现算法逻辑，并非论文原始代码。

##### 5.3.1 骨干模型：自回归预测主循环

```python
# 基于文中描述生成的伪代码

# 输入: 初始物理场 X_init (T, S, U, V, SSH 等),
#       边界条件序列 BC[t], t = 0..T-1
# 输出: 逐日预测序列 pred[t]

X_prev = X_init  # t=0 时刻初始条件

for t in range(T):
    # 1) 组装输入张量 X_i^t
    X_in = concat_channels(X_prev, BC[t])  # 形状: [C_in, H, W]

    # 2) Patch Embedding
    tokens = patch_embed(X_in, patch_size=8)  # [N_tokens, C_embed]

    # 3) 随机掩码 (训练阶段)
    if training:
        visible_tokens, mask_info = random_mask(tokens, mask_ratio(t))
    else:
        visible_tokens, mask_info = tokens, None

    # 4) 位置编码
    visible_tokens = add_cosine_pos_encoding(visible_tokens)

    # 5) 编码器: 1D-AFNO blocks
    z = visible_tokens
    for blk in encoder_blocks:  # 1D-AFNO
        z = blk(z)

    # 6) 解码器: 恢复被掩码的位置, 1D-AFNO blocks
    full_tokens = restore_tokens(z, mask_info)
    for blk in decoder_blocks:  # 1D-AFNO (较浅)
        full_tokens = blk(full_tokens)

    # 7) Patch-Recovery 到物理空间
    X_out = patch_recover(full_tokens, out_channels=C_out, patch_size=8)

    # 8) 保存并作为下一步输入的“初始条件部分”
    pred[t] = X_out
    X_prev = select_ocean_state_channels(X_out)  # T, S, U, V, SSH
```

##### 5.3.2 区域 Downscaling 模块

```python
# 基于文中描述生成的伪代码

# 输入:
#   X_f: 骨干特征张量, 形状 [C_f, h, w]
#   low_res_vars: 区域低分辨率表面变量, [5, 120, 230] at 1/4°
# 输出:
#   high_res_vars: 区域高分辨率预测变量, [5, 360, 690] at 1/12°

# 1) 区域裁剪与插值到相同分辨率
X_f_region = crop_global_feature(X_f, region="Kuroshio")       # [C_f, h_r, w_r]
X_f_up = bilinear_upsample(X_f_region, size=(120, 230))         # 对齐低分辨率网格

# 2) 通道融合
cond = concat_channels(low_res_vars, X_f_up)  # [C_cond, 120, 230]

# 3) 轻量微调网络: 若干 Residual Conv blocks + ConvTranspose2d(upscale=3)
z = cond
for blk in residual_conv_blocks:  # 8/16/32 层可选
    z = blk(z)                    # 保持 [C_mid, 120, 230]

high_res_vars = conv_transpose2d(z, out_channels=5, scale_factor=3)
# high_res_vars 形状约为 [5, 360, 690]
```

##### 5.3.3 Wave Decoding 模块

```python
# 基于文中描述生成的伪代码

# 输入:
#   X_f: 骨干特征张量 [C_f, h, w]
#   swh_init: 初始显著波高 [1, 360, 720]
#   wind10: 10m 风速场 (U10/V10) [2, 360, 720]
# 输出:
#   swh_next: 下一时间步显著波高 [1, 360, 720]

# 1) 将骨干特征插值到 1/2° 网格
X_f_wave = bilinear_upsample(X_f, size=(360, 720))  # [C_f, 360, 720]

# 2) 合并输入条件
cond = concat_channels(swh_init, wind10, X_f_wave)  # [C_cond, 360, 720]

# 3) 展平成 token 序列并输入 1D-AFNO blocks
wave_tokens = spatial_to_tokens(cond)  # [N_tokens, C_cond]
for blk in wave_afno_blocks:  # 共两层 AFNO
    wave_tokens = blk(wave_tokens)

# 4) 投影到 SWH 通道并还原空间
swh_tokens = linear_project(wave_tokens, out_dim=1)
swh_next = tokens_to_spatial(swh_tokens, size=(360, 720))

# 5) 应用海冰掩码
swh_next = apply_sea_ice_mask(swh_next, sea_ice_mask)
```

##### 5.3.4 Biochemistry Coupling 模块

```python
# 基于文中描述生成的伪代码

# 输入:
#   X_f: 骨干特征张量 [C_f, h, w]
#   bio_vars_init: 8 个初始生化变量 [8, 180, 360]
# 输出:
#   bio_vars_next: 下一时间步 8 个生化变量 [8, 180, 360]

# 1) 对骨干特征重网格至 1°×1°
X_f_bio = bilinear_upsample(X_f, size=(180, 360))  # [C_f, 180, 360]

# 2) 合并物理特征与生化初值
cond = concat_channels(bio_vars_init, X_f_bio)  # [C_cond, 180, 360]

# 3) 1D-AFNO + 投影
bio_tokens = spatial_to_tokens(cond)
for blk in bio_afno_blocks:  # 两层 AFNO
    bio_tokens = blk(bio_tokens)

bio_tokens = linear_project(bio_tokens, out_dim=8)

# 4) 还原到网格
bio_vars_next = tokens_to_spatial(bio_tokens, size=(180, 360))
```

---

### 6. 数据规格

#### 6.1 数据集名称与来源

- **HYCOM**（Hybrid Coordinate Ocean Model）全球重分析数据：
  - 用于骨干模型和 downscaling 模块训练。
- **ERA5** 大气再分析数据：
  - 提供 5 个大气强迫变量；
  - 提供显著波高 SWH 和 10m 风速场用于 Wave Decoding 模块。
- **ETOPO 2022** 15 秒全球地形/海底地形数据：
  - 提供数字高程与海底地形 Topology 变量。
- **NASA Ocean Biogeochemical Model**（吸收卫星叶绿素数据的海洋生物地球化学模式）：
  - 提供 8 个生物地球化学变量用于 Biochemistry Coupling 模块。

#### 6.2 时间范围与划分

- HYCOM 重分析：1994–2015 年，3 小时分辨率，41 垂直层（0–5000 m）。
- 骨干模型训练数据：
  - 训练集：2000–2010 年（每日 12:00 UTC 采样形成日尺度数据）。
  - 验证集：2011 年。
  - 测试集：2012 年（out-of-sample）。
- 下游模块数据划分：
  - 与骨干模型类似的训练/验证/测试划分方式，时间步长均为 1 天。

#### 6.3 变量列表

##### 6.3.1 骨干模型变量（Table 1）

- 预测目标变量（输出）：
  - T：15 层海温（°C），深度层：0, 6, 10, 20, 30, 50, 70, 100, 125, 150, 200, 250, 300, 400, 500 m。
  - S：15 层海盐（PSU）。
  - U：15 层海流 zonal 速度（m/s）。
  - V：15 层海流 meridional 速度（m/s）。
  - SSH：1 层海表高度（m）。

- 边界条件与额外输入变量：
  - U10：10 m 风 meridional 分量（m/s）。
  - V10：10 m 风 zonal 分量（m/s）。
  - T2m：2 m 近地面气温（°C）。
  - MSL：平均海平面气压（m）。
  - SP：地表气压（hPa）。
  - Topology：数字高程 + 海底地形（m）。

##### 6.3.2 下游模块变量（Table 2）

- 区域 Downscaling 任务（HYCOM, $1/12^\circ$）：
  - SST：海表温度（°C）。
  - SSS：海表盐度（PSU）。
  - SSU：海表 zonal 流速（m/s）。
  - SSV：海表 meridional 流速（m/s）。
  - SSH：海表高度（m）。

- Wave Decoding 任务（ERA5, $1/2^\circ$）：
  - SWH：显著波高（m）。
  - U10：10m 风 meridional 速度（m/s）。
  - V10：10m 风 zonal 速度（m/s）。

- Biochemistry Coupling 任务（NASA, $1^\circ$）：
  - Tca：总叶绿素 a 浓度（mg/m³）。
  - Chl：绿藻浓度（mg/m³）。
  - Dia：硅藻浓度（mg/m³）。
  - Coc：球石藻浓度（mg/m³）。
  - Cya：蓝藻浓度（mg/m³）。
  - Irn：铁浓度（nmol/L）。
  - Nit：硝酸盐浓度（µmol/L）。
  - MLD：混合层深度（m）。

#### 6.4 数据处理流程

- 空间重网格与分辨率处理：
  - HYCOM 原始分辨率 $1/12^\circ$ 通过预处理得到 $1/4^\circ \times 1/4^\circ$ 的日尺度数据，用于骨干模型训练，分辨率足以解析大多数海洋区域。
  - 区域 Downscaling：
    - 训练标签使用 Kuroshio 区域 $1/12^\circ$ HYCOM 数据；
    - 条件输入为 $1/4^\circ$ 的区域低分辨率表面变量；
    - 模块实现 3 倍超分辨率（$1/4^\circ \to 1/12^\circ$）。
  - Biochemistry Coupling：
    - 使用双线性插值（bilinear interpolation）将 NASA 生化变量重网格至 $180 \times 360$（$1^\circ$ 分辨率）。

- 时间采样：
  - 从 3 小时分辨率数据中，每天在 12:00 UTC 采样，构建日尺度序列。

- 稀疏数据处理：
  - 稀疏观测数据需预处理为网格数据，并配套掩码张量，以在模型输入中统一为规则网格形态。

- 归一化与缺失处理：
  - 论文正文未给出具体归一化方式和缺失值处理细节，仅说明更多预处理细节在补充材料中提供，此处留空。


---

## 维度二：基础组件知识（Component-Level Knowledge）

> 下列组件为从论文中直接识别或可明确抽象出的核心模块；若论文未给出内部数学细节，则公式部分留空或仅描述已知统计量。

### 组件 1：Patch Embedding（3D/多变量 Patch 嵌入）

#### 1. 子需求定位
- 对应的子需求：
  - 将多源、多尺度、2D/3D/稀疏的物理变量场统一映射为**一维 token 序列**，以便在 Transformer/AFNO 结构中进行序列建模。
- 解决的问题：
  - 在统一框架中处理不同形状与层数的变量（如 15 层 3D 变量与 1 层 2D 变量）。
  - 支持**多模态物理场融合**（温度、盐度、速度、SSH、风场、地形等）。

#### 2. 技术与创新
- 关键技术点：
  - 非重叠 patch 划分（$p=8$）将 $C_{in} \times 720 \times 1440$ 的输入分解为 $(h \times w)$ 个 patch；
  - 每个 patch 通过线性映射获得维度为 $C_{embed\_dim}$ 的 token 表征；
  - 统一处理 2D/3D/稀疏数据，只要能表示为规则网格 + 掩码即可。
- 创新点：
  - 将**复杂海洋多变量场**映射为**统一 token 序列**，从而实现多源驱动、天然支持模型级同化，与传统数值模式中的“格点场 + 方程组”表征方式不同。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：多源数据预处理与变量堆叠（构造 $X \in \mathbb{R}^{C_{in} \times H \times W}$）。
  - 后接：随机掩码策略、位置编码、1D-AFNO 编码器。
- 绑定程度：
  - **弱绑定–通用**：
    - 该组件并未引入海洋特有物理偏置，主要是通用 patch embedding 机制，可迁移至其他地球系统或图像类任务。

#### 4. 实现细节与伪代码

- 理论公式（抽象）：

设输入张量 $X \in \mathbb{R}^{C_{in} \times H \times W}$，patch 大小为 $p \times p$，则：

- patch 个数：$N = (H/p) \cdot (W/p)$；
- 第 $k$ 个 patch 的向量化表示：$x_k \in \mathbb{R}^{C_{in} \cdot p^2}$；
- 线性嵌入：

$$
\text{token}_k = W_E x_k + b_E, \quad W_E \in \mathbb{R}^{C_{embed} \times (C_{in} p^2)}.
$$

- 伪代码：

```python
# 基于文中描述生成的伪代码

def patch_embed(X, patch_size=8, embed_dim=C_embed):
    # X: [C_in, H, W]
    C_in, H, W = X.shape
    h, w = H // patch_size, W // patch_size

    patches = []
    for i in range(h):
        for j in range(w):
            patch = X[:,
                      i*patch_size:(i+1)*patch_size,
                      j*patch_size:(j+1)*patch_size]
            patch_vec = patch.reshape(C_in * patch_size * patch_size)
            token = linear(patch_vec, out_dim=embed_dim)  # W_E * x + b
            patches.append(token)

    tokens = stack(patches, dim=0)  # [N_patches, embed_dim]
    return tokens
```

---

### 组件 2：Random Mask Strategy（随机掩码策略）

#### 1. 子需求定位
- 对应的子需求：
  - 在训练中通过随机掩码提升模型对**隐含物理特征**的学习能力，而不是简单的局部插值；
  - 提升长期预测稳定性，减轻过拟合。
- 解决的问题：
  - 海洋观测与重分析中存在稀疏与缺测；
  - 仅在完整输入上训练可能导致模型偏向记忆局地模式，泛化与长期积分能力不足。

#### 2. 技术与创新
- 关键技术点：
  - 在 patch token 级别随机丢弃一定比例 token，仅对可见 token 进行编码；
  - 解码器利用 token 索引信息恢复完整序列并重建被掩码 token；
  - 训练过程中掩码比例**逐渐降低**，实现从“强自监督重建”到“更精细预测”的过渡。
- 创新点：
  - 将 MAE 风格的随机掩码策略应用于**全球海洋动力学建模**，借此增强模型对稀疏信息和隐含动力过程的学习。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：Patch Embedding 生成的 token 序列；
  - 后接：位置编码与 1D-AFNO 编码器；
  - 解码阶段需要与 Patch-Recovery 协同，利用 idrank 信息恢复 token 位置。
- 绑定程度：
  - **弱绑定–通用**：
    - 随机掩码是通用训练技巧，可应用于图像、语音等多领域，不依赖特定海洋物理属性。

#### 4. 实现细节与伪代码

```python
# 基于文中描述生成的伪代码

def random_mask(tokens, mask_ratio):
    # tokens: [N, C]
    N, C = tokens.shape
    num_keep = int(N * (1 - mask_ratio))

    # 1) 随机打乱索引
    perm = randperm(N)
    keep_idx = perm[:num_keep]
    mask_idx = perm[num_keep:]

    # 2) 选择可见 token
    visible_tokens = tokens[keep_idx]

    # 3) 保存掩码信息以便解码阶段恢复
    mask_info = {
        "keep_idx": keep_idx,
        "mask_idx": mask_idx,
        "N_total": N
    }
    return visible_tokens, mask_info


def restore_tokens(encoded_visible, mask_info):
    # encoded_visible: [N_keep, C]
    N = mask_info["N_total"]
    full_tokens = zeros(N, encoded_visible.shape[1])
    full_tokens[mask_info["keep_idx"]] = encoded_visible
    # 被掩码部分可初始化为 0 或可学习向量
    return full_tokens
```

---

### 组件 3：1D-AFNO Block（Fourier-based Attention Block）

#### 1. 子需求定位
- 对应的子需求：
  - 高效地在序列维度上混合空间–通道信息，捕捉全球与局地尺度的海洋动力学相关性。
- 解决的问题：
  - 传统自注意力在长序列上的计算复杂度高；
  - 海洋过程具有显著谱结构，适合在频域进行建模。

#### 2. 技术与创新
- 关键技术点（按论文描述与引用 AFNO 工作）：
  - 利用自适应傅里叶神经算子作为 token mixer，在频域进行线性或非线性变换；
  - 在编码器与解码器中均使用 1D-AFNO blocks，构成 Fourier-based attention 结构。
- 创新点：
  - 将 AFNO 作为 Transformer 风格海洋模型中的主力 token mixer，替代或弱化标准自注意力结构，以提升效率与尺度适配性。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：Patch Embedding + Random Mask + 位置编码；
  - 后接：解码器/patch-recovery 或下游 AFNO/Conv 模块。
- 绑定程度：
  - **弱绑定–通用**：
    - AFNO 本身为通用算子，适用于图像/时空预测等多任务；在该工作中用于海洋建模，但未显式引入海洋特有偏置。

#### 4. 实现细节与伪代码

> 注：论文未给出 AFNO 的具体公式，此处仅根据“在频域进行变换”的一般逻辑给出抽象伪代码，不代表原文实现细节。

```python
# 基于文中描述生成的伪代码

class AFNOBlock:
    def __init__(self, dim):
        self.dim = dim
        self.ffn = FeedForward(dim)
        # 频域线性变换参数省略

    def forward(self, tokens):
        # tokens: [N, C]
        x = tokens

        # 1) 频域变换 (沿序列维度做 FFT)
        X_f = fft(x, dim=0)   # [N, C] in Fourier domain

        # 2) 频域线性/非线性变换 (省略细节)
        X_f = fourier_mixing(X_f)

        # 3) 逆 FFT 返回时域
        x_mixed = ifft(X_f, dim=0).real

        # 4) 残差 + 前馈网络
        x = x + x_mixed
        x = x + self.ffn(x)
        return x
```

---

### 组件 4：Patch-Recovery 模块（反卷积恢复物理场）

#### 1. 子需求定位
- 对应的子需求：
  - 将在 token 空间中建模后的表示恢复成**规则网格上的物理场**，输出各层 T/S/U/V/SSH 等变量。
- 解决的问题：
  - 将 $(h \times w)$ 个 token 重排并上采样回 $(H, W)$ 网格；
  - 支持多变量、多通道的同时重建。

#### 2. 技术与创新
- 关键技术点：
  - 通过 2D ConvTranspose2d 实现从 patch 网格到原始分辨率的上采样；
  - 在上采样后通过卷积/线性映射输出所需的变量通道数 $C_{out}$。
- 创新点：
  - 将 Vision MAE 风格的 patch recovery 与全球海洋 3D 变量重建结合，完成多变量统一重建。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：解码器输出 token 序列；
  - 后接：自回归预测循环和下游模块（通过选定通道作为下一步初始条件/特征）。
- 绑定程度：
  - **弱绑定–通用**：
    - 通用的上采样与多通道映射机制，可用于任何图像/场重建任务。

#### 4. 实现细节与伪代码

```python
# 基于文中描述生成的伪代码

def patch_recover(tokens, out_channels, patch_size=8, H=720, W=1440):
    # tokens: [N, C_dec], N = (H/p)*(W/p)
    h, w = H // patch_size, W // patch_size

    # 1) 重排为 [C_dec, h, w]
    feat = tokens.reshape(h, w, -1).permute(2, 0, 1)  # [C_dec, h, w]

    # 2) 使用 ConvTranspose2d 上采样
    up = conv_transpose2d(feat, out_channels=out_channels,
                          kernel_size=patch_size, stride=patch_size)
    # up: [out_channels, H, W]
    return up
```

---

### 组件 5：Regional Downscaling Module（区域下采样/超分辨率模块）

#### 1. 子需求定位
- 对应的子需求：
  - 在如黑潮等“中尺度涡活动强烈”区域，从全局 $1/4^\circ$ 分辨率下采样（嵌套）到区域 $1/12^\circ$，解析中尺度涡与流路径细节。
- 解决的问题：
  - 全局统一高分辨率模拟计算代价过高；
  - 传统嵌套网格需要数值模式多网格耦合，开发维护复杂。

#### 2. 技术与创新
- 关键技术点：
  - 条件输入为：
    - 区域内 5 个低分辨率表面变量（SST, SSS, SSU, SSV, SSH），大小 $5 \times 120 \times 230$；
    - 全局骨干特征张量在该区域的裁剪并重采样；
  - 微调网络由 8/16/32 层残差卷积块组成，最后用 ConvTranspose2d 进行 3 倍上采样，输出大小 $5 \times 360 \times 690$；
  - 实现从 $1/4^\circ$ 到 $1/12^\circ$ 的超分辨率。
- 创新点：
  - 在大规模 AI 海洋模式中，首次引入**AI 嵌套网格 downscaling 模块**，无需更改骨干，依靠轻量下游模块即可实现高分辨率区域模拟。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：骨干模型特征张量与低分辨率表面变量；
  - 后接：用于评估中尺度涡与流系路径，或进一步下游应用。
- 绑定程度：
  - **强绑定–气象/海洋特异**：
    - 虽然从网络结构看是通用 CNN + 上采样，但设计目标与使用的变量（黑潮区域流场、SSH 中尺度涡）高度专门面向海洋学需求。

#### 4. 实现细节与伪代码

伪代码参考“5.3.2 区域 Downscaling 模块”，此处不再重复。

---

### 组件 6：Wave Decoding Module（显著波高解码模块）

#### 1. 子需求定位
- 对应的子需求：
  - 从骨干模型的物理特征和风场等条件解码**显著波高 SWH**，实现 30 天波浪预报。
- 解决的问题：
  - 传统波浪模式需要独立求解波谱方程并与海洋/大气模式耦合；
  - 本模块试图直接从物理特征张量和闭合条件中解码派生变量，简化系统结构与耦合成本。

#### 2. 技术与创新
- 关键技术点：
  - 输入：
    - 骨干特征张量（插值到 $1/2^\circ$ 分辨率）；
    - 初始 SWH；
    - 两个 10m 风速分量（U10, V10）；
  - 网络结构：
    - 两层 1D-AFNO blocks；
    - 线性投影层将 token 映射到单通道 SWH；
    - 使用全年平均海冰分布作为掩码，限制有效海域。
- 创新点：
  - 利用骨干物理特征 + 风场条件在单一 AI 框架下完成波浪解码，而非独立波浪模式。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：骨干模型与 ERA5 风场；
  - 后接：波浪预报评估与业务应用。
- 绑定程度：
  - **强绑定–气象/海洋特异**：
    - 使用显著波高与风场变量，是典型海洋–大气耦合场景；
    - 虽然结构为通用 AFNO，但任务与输入变量为海洋学特有。

#### 4. 实现细节与伪代码

伪代码参考“5.3.3 Wave Decoding 模块”。

---

### 组件 7：Biochemistry Coupling Module（生物地球化学耦合模块）

#### 1. 子需求定位
- 对应的子需求：
  - 将物理海洋信息与生物地球化学场耦合，预测 8 个生化变量的时间演化。
- 解决的问题：
  - 传统数值模型中，物理–生化模式通过外部耦合器耦合，结构复杂且计算开销大；
  - 本模块尝试在单一 AI 框架内实现物理–生化协同预测。

#### 2. 技术与创新
- 关键技术点：
  - 输入：
    - 8 个生化变量（Tca, Chl, Dia, Coc, Cya, Irn, Nit, MLD）初始场，分辨率 $1^\circ$；
    - 骨干物理特征张量重网格至同一分辨率；
  - 网络结构：
    - 两层 1D-AFNO blocks；
    - 投影层输出 8 通道预测；
  - 数据来源：NASA 生物地球化学模式 + 卫星叶绿素数据同化结果。
- 创新点：
  - 将物理特征张量与生化场在下游模块中统一处理，构成 AI 版“物理–生化耦合模式”，不再依赖传统外部耦合器。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：骨干模型特征张量与 NASA 生化数据重网格；
  - 后接：生化预报与生态应用分析。
- 绑定程度：
  - **强绑定–气象/海洋特异**：
    - 直接针对海洋生物地球化学变量，属于高度领域特定组件。

#### 4. 实现细节与伪代码

伪代码参考“5.3.4 Biochemistry Coupling 模块”。

---

### 组件 8：统计评估模块（纬向加权 RMSE/ACC）

#### 1. 子需求定位
- 对应的子需求：
  - 为全球海洋变量的预测提供合理的统计评估指标，考虑纬度带面积差异（纬向加权），并排除陆地格点。
- 解决的问题：
  - 直接在格点上计算 RMSE/ACC 会忽略纬度差异导致的面积权重偏差；
  - 需要统一评价不同变量在全球海洋上的预测性能。

#### 2. 技术与创新
- 关键技术点：
  - 采用纬向加权的 RMSE 和 ACC；
  - 使用气候平均场定义异常场；
  - 仅在海洋格点集合 G 上计算指标。
- 创新点：
  - 将气象/海洋社区常用的 ACC/RMSE 评价体系系统化应用于 AI-GOMS 的多变量、多深度分层预测评估。

#### 3. 上下文与耦合度
- 上下文组件：
  - 前接：AI-GOMS 的预测输出与 HYCOM/ERA5/NASA 真实数据；
  - 后接：模型指标展示与与 FourCastNet 等基线对比。
- 绑定程度：
  - **弱绑定–通用**：
    - ACC/RMSE 为通用统计指标，但纬向权重与仅海洋格点相关，更常见于地球科学应用。

#### 4. 实现细节与伪代码

```python
# 基于文中描述生成的伪代码

def latitude_weight(lat):
    # lat: [N_lat] in radians
    w = np.cos(lat)
    w = w / w.mean()  # 归一化
    return w  # [N_lat]


def compute_rmse(pred, true, lat, ocean_mask):
    # pred, true: [C, H, W]
    # lat: [H], ocean_mask: [H, W] bool

    L = latitude_weight(lat)  # [H]
    diff2 = (pred - true)**2

    # 只在海洋格点上计算
    diff2 = diff2[:, ocean_mask]  # [C, N_ocean]

    # 为每个纬度格点乘以权重
    # 这里假设 ocean_mask_lat: 每个有效格点对应的纬度索引
    w = L[ocean_mask_lat]
    rmse = np.sqrt((diff2 * w).mean(axis=-1))  # 按格点平均
    return rmse


def compute_acc(pred, true, clim, lat, ocean_mask):
    # 异常场
    pred_anom = pred - clim
    true_anom = true - clim

    # 其余步骤同 RMSE 中纬度权重和海洋掩码处理
    # 计算加权协方差与方差，再求相关系数
    # 细节略
    pass
```
