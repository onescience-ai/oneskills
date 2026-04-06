# Forecasting the Future with Future Technologies — 模型与组件知识提取

> 说明：本文是一篇关于“气象大模型（large meteorological models）”的综述型文章，重点盘点了 FourCastNet、Pangu-Weather、FengWu、FuXi、ClimaX、GraphCast 等代表性 AI 模型及其训练成本、适用时间尺度与未来发展方向。这里把“气象大模型家族”视作一个整体范式，并对每个代表模型拆成组件卡。

---

## 维度一：整体模型知识（Model-Level）

### 1. 目标任务与总体范式

- 总体目标：
  - 利用大规模深度学习模型，对地球大气–地表系统进行从“实时/短期”到“中长期”多时间尺度的数值替代或增强预测，覆盖：
    - 近实时与短期：小时–几天（nowcasting / short-range）；
    - 中期：5–15 天（medium-range）；
    - 更长：气候预测与情景模拟（通过 ClimaX 等气候基础模型）。

- 与传统 NWP 的对比：
  - 传统 NWP：基于 Navier–Stokes 等物理方程 + 参数化，求解 PDE，分辨率与时间步长受限于计算资源；
  - 气象大模型：
    - 以 ERA5、CMIP6 等再分析 / 气候资料为主输入，直接学习“状态到状态”的映射（数据驱动算子）；
    - 使用 CNN、GNN、Transformer、FNO 等深度架构，替代或补充 NWP 模式；
    - 在高分辨率（典型 0.25°）上实现快速推理和较高准确度，部分指标上已超越 IFS/HRES 等传统模式。

- 典型输入/输出与数据：
  - 再分析：ERA5（1979–至今，0.25°、1h/6h），是大多数组件（FourCastNet、Pangu、FengWu、FuXi、GraphCast）的训练基础；
  - 气候：CMIP6（多模式、多情景），支撑 ClimaX 的预训练；
  - 变量与层次：
    - 上空：Z500（500 hPa 位势高度）、温度、风（u/v）、比湿等 5+ 个变量在 13 层或更多层（FuXi 5×13=65 上空变量）；
    - 地面：T2M、10m 风、降水等若干表面变量；
  - 时空分辨率：
    - 空间多数为 0.25° 全球网格，也存在多分辨率预训练（ClimaX 可以低分辨预训练，高分辨下游任务微调）；
    - 时间多为 1h 或 6h 时间步。

### 2. 模型家族与适用时间尺度

- FourCastNet：
  - 架构：FNO + ViT（频域算子 + 视觉 Transformer）；
  - 数据：ERA5，小时级；
  - 适用尺度：短–中期（至一周），尤其擅长小尺度变量（风、降水、水汽）；
  - 亮点：
    - 采用 Fourier Neural Operator（FNO）捕捉全局谱结构；
    - 利用 ViT 处理高分辨时空场；
    - 能在数秒内生成周尺度预报，明显加速相较 IFS。

- Pangu-Weather：
  - 架构：3D Earth Specific Transformer（3DEST），三维（时间–纬度–经度）地球专用 Transformer；
  - 数据：ERA5，1979–2017；
  - 适用尺度：1 小时至 1 周中期预报；
  - 亮点：中期多变量、多层预报上整体技能超过传统 NWP。

- FengWu：
  - 架构：Transformer，带“模态定制 encoder–decoder”（不同变量/模态的专门编码器解码器）；
  - 数据：ERA5，6 小时间隔，1979–2015；
  - 适用尺度：中期（10 天左右）全球预报；
  - 亮点：提出区域自适应不确定性损失（region-adaptive uncertainty loss），在若干关键指标（如 10 日 z500 RMSE）上优于 GraphCast。

- FuXi：
  - 架构：级联机器学习系统，由 cube embedding + U-Transformer + 全连接层构成；
  - 数据：ERA5，39 年，0.25°，6 h，70 个变量（5 个上空变量×13 层 + 5 个地面变量）；
  - 适用尺度：最长至 15 天全球预报；
  - 亮点：通过“分时段级联”策略，显著延长关键变量的有效预报时间（如 Z500 从 9.25→10.5 天，T2M 从 10→14.5 天），减缓长时间段误差积累。

- ClimaX：
  - 架构：Transformer 基础模型，在 CMIP6 上进行预训练；
  - 数据：CMIP6（多模式、多情景），可扩展到其他再分析/观测；
  - 适用尺度：从 nowcasting 到短–中期、直至长期气候预测，多任务多时间尺度；
  - 亮点：
    - 作为气候/天气 Foundation Model，可在低分辨和有限算力下预训练，再适配不同下游任务；
    - 统一处理多尺度、多区域、多变量任务。

- GraphCast：
  - 架构：GNN（图神经网络），在球面网格或图结构上运行；
  - 数据：ERA5，39 年；
  - 适用尺度：中期（10 天）全球预报；
  - 亮点：
    - 通过图结构精细建模大气场的空间依赖；
    - 0.25°、13 层分辨率下，能在一分钟内产生 10 天预报；
    - 在多项技能指标上优于传统 NWP。

### 3. 训练成本与算力需求（Training Cost）

- 该综述对各模型训练代价进行了定性–定量对比：
  - FourCastNet：
    - 训练配置：约 64× A100，约 16 小时；
    - 特点：FNO 结构较高效，在保持高分辨率的同时，整体计算效率较高。
  - Pangu-Weather：
    - 训练配置：约 192× V100，持续约 15 天；
    - 特点：3DEST 架构三维注意力+大时间跨度数据，算力消耗极大。
  - FengWu：
    - 训练配置：约 32× A100，约 17 天；
    - 特点：多模态、多任务处理+不确定性损失带来较重训练负载。
  - FuXi：
    - 训练配置：约 8× A100，约 30 小时；
    - 特点：架构更轻量、级联策略与有效训练技巧带来较低训练成本。
  - GraphCast：
    - 训练配置：32× TPU v4，约 4 周；
    - 特点：GNN 在全球 3D 网格上的消息传递和多步预测代价不低。

- 结论：
  - 大多数大模型训练均需要数十至上百高端 GPU/TPU 级别算力与数天乃至数周时间；
  - 不同架构（FNO、GNN、Transformer、U-Transformer）的训练效率差异显著，结构设计与优化策略对算力需求有巨大影响；
  - 轻量级 yet 高效的级联/多阶段训练（如 FuXi）是未来一个重要方向。

### 4. 专长、性能与未来发展潜力

- 专长与性能谱系：
  - FourCastNet：短–中期、小尺度与快速推理优；
  - Pangu：中期多变量高精度，适合业务级全球中期预报；
  - FengWu：中期 10 天附近技能突出；
  - GraphCast：中期、高精度+快速推理；
  - ClimaX：多尺度、多任务，具“基础模型”潜质；
  - FuXi：长达 15 天的级联预报，减少误差积累。

- 发展潜力：
  - 算法方向：
    - 更高效的算子学习（改进 FNO、GNN）、结构优化（multi-scale transformer）、不确定性建模（ensemble、分布预测）；
    - 强化对极端天气与罕见事件的专门建模；
  - 工程与硬件方向：
    - 新一代 GPU/TPU 与专用加速器；
    - 模型压缩、蒸馏、多分辨率推理；
  - 与传统气象的融合：
    - 与 NWP 同化系统、集合预报系统深度耦合；
    - AI 模型作为 NWP 的后处理、初始场修正、集合校正模块；
  - 文章观点：气象大模型将成为未来业务预报体系的重要支柱，与传统方法形成“AI+物理”协同格局。

---

## 维度二：基础组件知识（Component-Level）

> 说明：下面将文中出现的关键“气象大模型”与其结构要素抽象为组件。伪代码均为**基于论文描述生成的高层伪代码**，用于表达训练与推理范式，而非任何论文源码。

### 组件 1：FourCastNet — FNO + ViT 全球高分辨天气模型

1. 子需求与目标：
   - 在全球 0.25° 网格上，对多变量气象场进行短–中期快速预报；
   - 保持高分辨率的同时，实现比传统 NWP 快数个数量级的推理速度。

2. 关键设计：
   - Fourier Neural Operator（FNO）：
     - 在频域学习从输入场到输出场的算子，捕捉全局谱结构；
   - Vision Transformer（ViT）：
     - 将地球网格划分为 patch/token，利用多头自注意力建模时空依赖；
   - 训练数据：ERA5 小时级气象场，多变量、多层；
   - 输出：1–7 天的多步预报，可逐步滚动或多步并行。

3. 伪代码：训练循环（抽象）：

```pseudo
# 高层伪代码：FourCastNet 训练
for batch in ERA5_loader:
    x_t     = batch.current_state   # 当前气象场
    y_future = batch.future_state   # 目标未来状态（多步）

    # 1. 频域算子层
    z_spectral = FNO_block(x_t)

    # 2. ViT block 进行 token 混合和时空建模
    tokens = patch_embed(z_spectral)
    tokens = ViT(tokens)
    y_hat  = patch_restore(tokens)

    loss = mse(y_hat, y_future)
    loss.backward()
    optimizer.step()
```

---

### 组件 2：Pangu-Weather — 3DEST（三维地球专用 Transformer）

1. 子需求与目标：
   - 在全球范围内，以 3D 结构（经度–纬度–高度/层次）建模大气状态，进行 1 小时–1 周的中期预报；
   - 高度利用 ERA5 长时序数据，超过传统 NWP 技能。

2. 关键设计：
   - 3DEST：
     - 将时间、纬度、经度（及高度）作为 3D token 结构输入；
     - 使用适合球面坐标与垂直层的注意力模式；
   - 多变量输出：Z500、温度、风、比湿等多变量、多层联合预测；
   - 可支持极端事件预测、集合预报等扩展能力。

3. 伪代码：单步前向（简化）：

```pseudo
# 高层伪代码：Pangu-Weather 单步推理
function Pangu_step(X_t):
    # X_t: [batch, levels, lat, lon, vars]
    tokens = embed_3D_tokens(X_t)      # 3D embedding (lat, lon, level)
    tokens = EarthSpecificTransformer(tokens)  # 3DEST
    Y_hat  = project_to_fields(tokens) # 恢复到多变量 3D 场
    return Y_hat
```

---

### 组件 3：FengWu — 模态定制 Transformer + 区域自适应不确定性损失

1. 子需求与目标：
   - 提升 10 天左右 lead time 的中期预报技能，尤其是关键高度场（如 z500）；
   - 同时处理多变量、多层多模态输入，平衡不同区域与变量的不确定性。

2. 核心设计：
   - 模态定制 encoder–decoder：
     - 不同变量或变量组使用专门的编码器，捕捉其物理与统计特性；
     - 解码器共享或部分共享，用于重构联合未来状态；
   - 区域自适应不确定性损失：
     - 在空间上对损失加权，使模型在高不确定区域更关注结构/相对误差；
     - 在时间/变量维度上可做类似加权。

3. 伪代码：带区域权重的损失（抽象）：

```pseudo
# 高层伪代码：FengWu 区域自适应不确定性损失
for batch in ERA5_loader:
    x, y = batch.inputs, batch.targets
    features = []

    # 1. 各模态编码
    for modality in modalities:
        f_m = encoder_m[modality](x[modality])
        features.append(f_m)

    fused = fuse(features)
    y_hat = decoder(fused)

    # 2. 区域加权损失
    W = compute_region_uncertainty_weights(x, y)  # 根据历史误差/气候不确定性等
    loss = mean(W * (y_hat - y)^2)

    loss.backward()
    optimizer.step()
```

---

### 组件 4：FuXi — 级联 U-Transformer 15 天全球预报系统

1. 子需求与目标：
   - 在 0–15 天范围内进行全球预报，尤其延长关键变量（Z500、T2M 等）的“有效预报时间”；
   - 通过级联多阶段模型减缓长时间滚动误差累积。

2. 关键结构：
   - 数据立方体嵌入（cube embedding）：
     - 将上空多层变量 + 地表变量整理成高维 data cube，降维压缩；
   - U-Transformer：
     - 在空间上具 U-Net 式下采样/上采样结构，在 token 维度施加多层自注意力；
   - 全连接预测头：
     - 将 U-Transformer 输出映射回原变量维度；
   - 级联策略：
     - 不同时间窗口（0–5d、5–10d、10–15d）训练不同子模型或在同一模型上使用不同 fine-tune；
     - 将多个时间窗输出拼接成完整 15 天预报。

3. 伪代码：级联推理管线（抽象）：

```pseudo
# 高层伪代码：FuXi 级联推理
function FuXi_forecast(X_init):
    # X_init: 初始气象状态（或短期 NWP 分析场）

    # 1. 0-5 天模型
    cube0 = cube_embed(X_init)
    z0    = UTransformer_0_5d(cube0)
    Y_0_5 = fc_head_0_5d(z0)

    # 2. 5-10 天模型
    cube1 = cube_embed(Y_0_5[-1])      # 以上一阶段末尾状态为起点
    z1    = UTransformer_5_10d(cube1)
    Y_5_10 = fc_head_5_10d(z1)

    # 3. 10-15 天模型
    cube2 = cube_embed(Y_5_10[-1])
    z2    = UTransformer_10_15d(cube2)
    Y_10_15 = fc_head_10_15d(z2)

    Y_full = concat(Y_0_5, Y_5_10, Y_10_15)
    return Y_full
```

---

### 组件 5：ClimaX — 基于 CMIP6 的气候/天气 Transformer 基础模型

1. 子需求与目标：
   - 作为“天气–气候基础模型（foundation model）”，统一支撑多种气象与气候任务：
     - nowcasting、短–中期预报、长期气候预测与情景模拟；
   - 利用大规模 CMIP6 多模式多情景数据进行预训练，降低下游任务标注/监督需求。

2. 设计要点：
   - Transformer 主干：
     - 通用 tokenization + 自注意力，适配不同网格与分辨率；
   - 预训练任务：
     - 多时间步预测、重构不同变量/层、可能结合掩码重建与自监督策略（由 ClimaX 论文定义，本文综述层面只给出高层描述）；
   - 迁移能力：
     - 在较低分辨率和有限算力下预训练，即可在后续任务（更高分辨率预报）中获得较好技能。

3. 伪代码：FM 预训练 + 下游微调框架（抽象）：

```pseudo
# 高层伪代码：ClimaX 式基础模型
# 预训练
for batch in CMIP6_loader:
    x_ctx, x_target = build_pretrain_pair(batch)  # 如: 上下文 -> 未来或缺失变量
    tokens = tokenize(x_ctx)
    z = Transformer(tokens)
    x_hat = decode(z)
    L_pre = mse(x_hat, x_target)
    L_pre.backward()
    optimizer_pre.step()

# 下游任务微调
freeze(Transformer.parameters())  # 或部分冻结
for batch in downstream_loader:
    x, y = batch.inputs, batch.labels
    tokens = tokenize(x)
    z = Transformer(tokens)
    y_hat = task_head(z)
    L_task = loss_task(y_hat, y)
    L_task.backward()
    optimizer_head.step()
```

---

### 组件 6：GraphCast — 基于 GNN 的中期全球预报

1. 子需求与目标：
   - 利用图神经网络，在球面或不规则网格上进行高精度中期（10 天）预报；
   - 在 0.25°、13 层的分辨率下兼顾精度与推理速度。

2. 关键设计：
   - 图结构构建：
     - 将全球网格点视作节点，邻近网格之间建立边；
   - GNN 消息传递：
     - 每一步时间演化通过多层消息传递更新节点状态（包含多变量信息）；
   - 多步预测：
     - 可逐步递推（自回归）或采用多步结构。

3. 伪代码：单步 GNN 更新（抽象）：

```pseudo
# 高层伪代码：GraphCast 单步更新
function GraphCast_step(Graph_t):
    # Graph_t: 节点特征包含各层气象变量
    for layer in GNN_layers:
        for node in Graph_t.nodes:
            agg = aggregate([Graph_t.neighbor(h).feat for h in neighbors(node)])
            node.feat = GNN_update(node.feat, agg)
    return Graph_t
```

---

### 组件 7：训练成本与算力评估模块（横向比较）

1. 子需求：
   - 量化不同大模型在训练阶段对 GPU/TPU 数量、训练时间的需求；
   - 为模型选型与工程落地提供决策依据。

2. 综述中的关键信息：
   - GPU/TPU 数量：8–192 量级不等；
   - 训练时长：数十小时到数周不等；
   - 架构越复杂、数据量越大（如 3DEST/GNN、长时间 ERA5），训练代价越高。

3. 抽象流程：

```pseudo
# 高层伪代码：训练成本记录
for model in [FourCastNet, Pangu, FengWu, FuXi, ClimaX, GraphCast]:
    cfg = get_training_config(model)
    cost = estimate_cost(cfg.gpus, cfg.hours, cfg.dataset_size)
    log_training_cost(model.name, cost)
```

---

以上内容把这篇综述中的“气象大模型家族”抽象为一个整体范式，并为六个代表模型构建了统一格式的组件卡（目标、结构、伪代码）。如果你之后希望对某个具体模型（比如 FuXi 或 ClimaX）做更细粒度的拆解，我可以再对照它们的原始论文，把这里的高层组件细化成更具体的模块与公式。