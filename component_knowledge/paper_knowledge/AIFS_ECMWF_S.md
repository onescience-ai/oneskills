# 维度一：整体模型知识（Model-Level Knowledge）

## 1. 核心任务与需求

### 1.1 核心建模任务
- 构建 ECMWF 的数据驱动数值预报系统 AIFS（Artificial Intelligence Forecasting System），用于中期（最多 10 天）全球天气预报。
- 输入：ERA5 或 ECMWF 运行中的 IFS 分析场，在 \(t_{-6h}\)、\(t_0\)。
- 输出：\(t_{+6h}\) 的大气状态（多层气压面场和一组地面气象要素），通过自回归滚动（rollout）生成更长预报时效（最长到 72h 训练、10 天推理）。

### 1.2 解决了什么问题
- 精度方面：
  - 与 ECMWF 物理模式 IFS 相比，AIFS 在 500 hPa 位势高度等关键指标上表现出更高的 ACC 和更低的 RMSE，特别是在中长期（例如 6 天、10 天）有超过 12 小时以上的预报优势。
  - 上对流层和对流层整层（至 100 hPa）整体预报技能优于 IFS，地面 2 m 温度、10 m 风等指标在与分析场和观测（探空、SYNOP）对比时表现更好。
  - 热带气旋路径误差显著小于 IFS，主要源自减弱的移动速度慢偏差（slow bias）。
- 计算效率方面：
  - 数据驱动模型可在单块 GPU 上在几分钟内完成 10 天预报，相比基于物理方程的 NWP 模式需要约 \(O(1000)\) 个 CPU 核心的计算成本显著降低。
- 分辨率与格点设计问题：
  - 使用 N320 reduced Gaussian grid（约 31 km）代替常规经纬网格，减弱极区网格聚集带来的非均匀性，同时降低相同分辨率下的总格点数。
- 传统数据驱动模型的不足：
  - 现有基于加权 MSE 损失的模型普遍存在随预报时效增长而出现的场景“模糊化”（blurring）和活动度下降问题，以及类似集合平均的平滑行为。
  - 部分模型仅在较低分辨率或有限变量集上训练，地表要素、降水等预报技能有限。


## 2. 解决方案

### 2.1 怎么解决的（核心思路）
- 采用 Encoder–Processor–Decoder 结构：
  - 编码器和解码器基于图神经网络（GNN），在 ERA5 原生 N320 减少高斯格点与处理器网格之间进行信息聚合与映射。
  - 处理器部分为 pre-norm Transformer，采用沿纬向带的滑动窗口注意力（shifted window attention），在 O96 八面体 reduced Gaussian grid 上进行时序–空间特征建模，共 16 层。
- 使用高分辨率输入与灵活网格设计：
  - 原生 N320 reanalysis 网格（542,080 个格点），通过 encoder/decoder 与约 1° 分辨率的 O96 处理器网格（40,320 格点）耦合。
- 面向长时效的 rollout 训练：
  - 第一阶段预训练：仅预测 6 小时步长。
  - 第二阶段：在 ERA5 上进行最长到 72 h 的自回归 rollout 训练，梯度在整条时间链上传播。
  - 第三阶段：在 IFS 运行分析场上进一步 rollout 微调，提升对实际业务分析场的适配。
- 大规模并行与内存优化：
  - 通过数据并行和序列/张量并行（attention head 分片、节点/边分片）实现上千 GPU 的弱缩放训练。
  - 利用激活检查点（activation checkpointing）在前向中不保留全部中间激活，在反向时重计算，以降低显存占用。
- 面向概率预报的训练基础：
  - 实现了一种 device-to-device 通信模式，可在多 GPU 上分布式处理集合成员，实现基于公平概率评分（如 fair scores）的集合训练，为未来概率预报系统奠定基础。

### 2.2 结构映射
- 整体任务 → 子模块映射：
  - 高分辨率全球多变量 6 小时预报：
    - 输入/输出映射与降维/升维 → GNN 编码器与解码器。
    - 全局动力–物理演化建模 → 滑动窗口 pre-norm Transformer 处理器（16 层）。
  - 计算效率与可训练性：
    - 激活检查点 → 减少显存占用。
    - 数据并行 + 序列/张量并行（attention head 分片、节点/边分片）→ 支持高分辨率、长序列训练，扩展到 2048 块 GPU。
  - 概率预报与集合优化：
    - GPU 间的 ensemble 通信模式 → 允许在训练时直接优化概率评分。
  - 物理一致性与观测约束：
    - 训练数据来自 ERA5 重分析与运行 IFS 分析场；
    - 验证使用 NWP 分析场、探空和地面观测（SYNOP），并通过多种评分（ACC、RMSE、SEEPS、sdaf 等）评价。


## 3. 模型架构概览

- 整体结构：Encoder–Processor–Decoder 结构，全部以 Python 实现，依赖 PyTorch 与 PyTorch Geometric。
- 空间网格：
  - 输入/输出网格：N320 reduced Gaussian grid（约 31 km，542,080 格点）。
  - 处理器网格：O96 八面体 reduced Gaussian grid（约 1°，40,320 格点）。
- 编码器（Encoder）：
  - 将 ERA5/IFS 的输入场（含多层气压面和地面场）映射到处理器网格。
  - 使用基于图注意力/Transformer 的 GNN：
    - 节点：ERA5 网格点与处理器网格点。
    - 边：基于空间邻域构建；encoder 使用某个截断半径，将周围一定距离的 ERA5 格点连接到处理器格点；
    - 边特征：边长与方向。
  - 输入和边特征在进入 GNN 前通过线性层嵌入；同时在节点和边上添加 8 个可学习特征。
- 处理器（Processor）：
  - 在 O96 网格上运行的 pre-norm Transformer：
    - 注意力沿纬向带计算，使用局部 attention 窗口，并通过 shifted windows 机制扩展感受野；
    - 16 个处理器层；
    - 使用 GELU 激活函数；
    - 不显式使用边信息，而是通过注意力机制在纬向带上建模邻域信息。
- 解码器（Decoder）：
  - 将处理器网格上的潜在状态映射回 N320 网格：
    - 每个输入网格点连接到其最近的 3 个处理器网格点；
    - 使用 GNN attention/Transformer 型卷积更新节点和边；
    - 输出线性层将潜在表征投影回物理变量空间。
- 训练与推理流程：
  - 训练：
    - 初始阶段：6 h 单步预报；
    - Rollout 阶段：最长 72 h（12 步），梯度在整条链上传播；
    - 微调阶段：在 IFS 运行分析场上进行 rollout 训练。
  - 推理：
    - 给定 \(t_{-6h}\)、\(t_0\) 的状态，预测 \(t_{+6h}\)；
    - 以后续预测作为新的输入状态，自回归滚动到任意目标时效（例如 10 天）。


## 4. 创新与未来

### 4.1 创新点
- 业务级数据驱动预报系统：
  - AIFS 作为 ECMWF 的实验性数据驱动业务预报系统，与 IFS 并行运行，每天 4 次预报，并在 ECMWF 开放数据策略下向公众提供。
- 架构层面：
  - 在全球 N320 减少高斯网格上，采用 GNN encoder/decoder 与 O96 pre-norm Transformer 处理器组合，兼顾高分辨率输入与计算可行性。
  - 使用基于纬向带的 shifted window attention，窗口设计保证每个注意力窗口覆盖完整的局部网格邻域，并通过多层堆叠扩大信息传播范围。
  - 在 GNN encoder/decoder 和 Transformer 处理器中均加入 8 维可学习的节点和边特征，以补充输入物理变量中未显式给出的潜在有用特征。
- 并行与可扩展性：
  - 采用序列/张量并行：attention head 在多 GPU 之间分片，通过高效 all‑to‑all 通信实现同步；
  - GNN encoder/decoder 另一种模式为基于 1-hop 邻域的节点/边分片，经由图节点重排，将设备间通信压缩为前向的一次 gather 和反向的一次 reduce 操作；
  - 弱缩放测试显示可接近线性扩展至至少 2048 块 GPU。
- 概率预报训练基础设施：
  - 定制的 GPU 设备间通信模式允许在训练过程中对集合成员进行概率评分（如 fair scores）优化，为构建概率预报系统奠定基础。
- 预报性能：
  - 在对流层（至 100 hPa）、地表要素、热带气旋路径等方面总体优于 IFS；
  - 与 GraphCast 相比，在同一数据与类似训练策略下，二者性能接近，但在不同时间段可能交替优于对方，凸显不同架构和训练细节的影响。

### 4.2 后续研究方向与局限
- 模糊化与活动度降低：
  - 当前 AIFS 与其他基于加权 MSE 损失的模型一样，在较长预报时效会产生平滑/模糊化行为，与集合平均类似；
  - 模糊程度与训练时优化窗口长度相关；
  - 初步结果表明，基于概率目标的训练可以在全时效保持更“锐利”的预报场，这与文献中基于概率损失的结果一致。
- 微调策略：
  - 当前在 IFS 运行分析场上的微调过程较为 adhoc，未来可系统地探索不同微调策略以进一步提升性能。
- 目标量与归一化策略：
  - 当前损失作用在“完整的归一化状态场”上，而非对归一化预报增量（tendencies）建模；
  - 这可能导致最后线性层输出到物理空间的变量归一化不理想；
  - 未来版本计划改为类似“预测归一化 tendency”的方案，并与更系统的损失缩放策略（按变量）结合。
- 垂直方向损失加权：
  - 目前随高度线性减小损失权重，导致平流层（如 50 hPa）预报技能降低；
  - 预计通过优化垂直方向损失加权可改善平流层预报。


## 5. 实现细节与代码逻辑

### 5.1 理论公式

1. 面积加权均方误差损失（文字转写为公式）
- 文中描述：
  - 损失函数是目标大气状态与模型预测之间的“面积加权均方误差（MSE）”；
  - 每个输出变量有一个损失缩放因子，使不同变量对总损失贡献大致相当；
  - 垂直速度的权重被降低；
  - 随高度线性减小损失权重，使高层（如 50 hPa）对总损失贡献较小。
- 可用一般形式表示为：
\[
L = \sum_{v} \alpha_v \sum_{l} \beta_l \cdot \frac{\sum_{i} w_i \bigl(y_{v,l,i} - \hat y_{v,l,i}\bigr)^2}{\sum_{i} w_i},
\]
其中：
  - \(v\)：变量索引；\(\alpha_v\)：按变量的损失缩放因子；
  - \(l\)：垂直层索引；\(\beta_l\)：随高度线性衰减的垂直权重；
  - \(i\)：二维空间格点索引；\(w_i\)：与格点面积成正比的空间权重；
  - \(y_{v,l,i}\)、\(\hat y_{v,l,i}\)：目标值与预测值。

> 注：该公式为根据文中“面积加权 MSE + 按变量和高度缩放”的文字描述写出的标准化表达。

2. 其他公式
- 文中提到多头自注意力、GELU、AdamW 等，但未给出具体数学形式；此处不再补充推导公式。

### 5.2 关键实现逻辑（文字）

1. 网格与图构建：
- 输入 ERA5/IFS 网格：
  - 使用 N320 reduced Gaussian grid，约 542,080 个格点；
  - 相比 0.25° 规则经纬网（1,038,240 个格点）大幅减少格点数，同时在全球范围内提供更均匀的分辨率（极区不发生剧烈网格聚集）。
- 处理器网格：
  - O96 八面体 reduced Gaussian grid，约 1° 分辨率，40,320 个格点。
- 编码器图构建：
  - 节点：ERA5 网格点和处理器网格点；
  - 边：对每个处理器格点，在其一定截断半径内收集所有 ERA5 格点，建立从 ERA5 节点到处理器节点的边；
  - 边特征：边的几何长度和方向；
  - 输入变量与边特征先通过线性层嵌入，然后进入 GNN 模块。
- 解码器图构建：
  - 对每个 ERA5 网格点，找到距离最近的 3 个处理器格点，建立从处理器节点到 ERA5 节点的边；
  - 使用相同种类的边特征（长度和方向）与 GNN 更新策略。
- 可学习特征：
  - 在输入网格、处理器网格以及 encoder/decoder 边上均添加 8 维可学习参数，用作附加节点/边特征，捕获输入中未显式存在但对预报有用的统计模式。

2. 处理器中的 shifted window attention（描述层面）：
- 注意力计算沿纬向带（latitude bands）进行；
- 为每个格点定义一个 attention window，窗口大小与位置保证其覆盖一个完整的局部网格邻域；
- 使用 shifted windows 机制：
  - 在相邻层中平移窗口位置，使得多层堆叠后，信息可以在更大范围内传播（图中示例显示 6 层可覆盖相当大的区域，AIFS 实际使用 16 层）；
- 处理器不显式使用边信息：
  - 相比 encoder/decoder 的 GNN，处理器完全通过多头自注意力在 O96 网格上建立各节点间关系。

3. 训练范式：
- 预训练（ERA5，1979–2020）：
  - 任务：6 小时单步预报；
  - 优化器：AdamW，\(\beta=(0.9, 0.95)\)；
  - 学习率调度：
    - 260,000 步；
    - 前 1,000 步从 0 线性升至 \(10^{-4}\)；
    - 随后采用余弦退火降至最小学习率 \(3\times 10^{-7}\)。
- Rollout 训练（ERA5，1979–2018）：
  - 初始 rollout 长度较短，每 1,000 步增加一次 rollout 长度，直至最大 72 小时（12 个 6 小时时间步）；
  - 梯度沿 rollout 链全程传播；
  - 学习率为 \(6\times 10^{-7}\)。
- 微调（IFS 运行分析，2019–2020）：
  - 在 IFS O1280 分辨率场（约 0.1°）插值到 N320 后进行；
  - 使用与 ERA5 rollout 相同的自回归训练思路，对模型在业务分析场上的表现进行微调。
- 归一化与损失缩放：
  - 输入与输出状态在每个垂直层上标准化至零均值与单位方差；
  - 部分强迫变量（如地形）使用 min–max 归一化；
  - 不同变量的损失缩放因子通过经验设定，使各变量对总损失的贡献大致相等，垂直速度权重适当降低；
  - 垂直方向权重随高度线性减小，上层大气对损失贡献较小。

4. 并行与激活检查点：
- 激活检查点：
  - 在前向传播时，仅在关键位置保留激活，中间激活在反向传播时重算，以节省显存；
- 数据并行：
  - 批量大小 16，在多 GPU 间拆分；
  - 单个模型实例跨四个 40GB A100 GPU；
  - 整体训练使用 64 块 GPU，大约一周完成。
- 序列/张量并行：
  - attention head 分片：
    - 在 GNN encoder/decoder 和 Transformer processor 中，将多头注意力的不同 head 分配到不同 GPU；
    - 使用高效 all‑to‑all 通信进行同步和聚合。
  - 节点/边分片（GNN 专用备选方案）：
    - 基于 1‑hop 邻域将图节点/边切分到不同 GPU；
    - 通过节点重排使得前向只需一次 gather，反向只需一次 reduce 即可实现跨设备通信；
- 弱缩放测试表明，可近似线性扩展到至少 2048 块 GPU。

5. 概率预报训练机制：
- 为构建概率预报系统，设计了设备到设备的通信模式：
  - 集合成员在 GPU 间分布；
  - 允许在训练时直接针对概率评分（如 fair scores）进行优化；
  - 支持较大规模集合在训练阶段就进行概率目标的优化。

6. 推理时效与计算成本：
- 10 天预报：
  - 在单块 A100 上约 2 分 30 秒（包含输入/输出的数据处理时间）。

### 5.3 伪代码/代码片段

> 说明：下列伪代码均为“基于文中描述生成的伪代码”，用于结构化展示实现逻辑，非论文原文代码。

1. 6 小时步长的单步预测（Encoder–Processor–Decoder）

```pseudo
function AIFS_step(state_t_minus_6h, state_t):
    # 1. 构造输入
    #   - 包含两时刻的大气状态（多层气压面 + 地表场）
    #   - 以及静态或强迫变量（地形、经纬度、时间信息等）
    input_fields = concat(state_t_minus_6h, state_t, forcings)

    # 2. 在 N320 网格上做特征嵌入
    node_feats_era5 = linear_embed(input_fields.node_features)
    edge_feats_era5 = linear_embed(compute_edges_era5())
    node_feats_era5 = concat(node_feats_era5, learnable_node_feats_era5)
    edge_feats_era5 = concat(edge_feats_era5, learnable_edge_feats_era5)

    # 3. GNN Encoder：N320 -> O96 处理器网格
    graph_enc = build_encoder_graph(N320_nodes, O96_nodes, radius_cutoff)
    latent_processor_nodes = GNN_transformer_block(
        graph_enc, node_feats_era5, edge_feats_era5
    )

    # 4. Pre-norm Transformer Processor：在 O96 上迭代 16 层
    for layer in range(16):
        latent_processor_nodes = pre_norm_transformer_layer(
            latent_processor_nodes,
            attention_windows="latitude_band_shifted"
        )

    # 5. GNN Decoder：O96 -> N320
    graph_dec = build_decoder_graph(O96_nodes, N320_nodes, k_nearest=3)
    node_feats_dec = GNN_transformer_block(
        graph_dec,
        latent_processor_nodes,
        learnable_edge_feats_dec
    )

    # 6. 输出线性层：映射到物理量
    pred_state_t_plus_6h = linear_output(node_feats_dec)

    return pred_state_t_plus_6h
```

2. Rollout 训练流程（最长至 72 h）

```pseudo
function train_rollout(dataset, max_lead_hours=72, lead_step=6):
    # dataset 提供 (t_-6h, t_0, 后续真值序列)
    for batch in dataset:
        states_truth = batch.truth_sequence  # [t_+6h, t_+12h, ..., t_+max_lead]
        state_tm6, state_t0 = batch.state_tm6, batch.state_t0

        pred_states = []
        cur_tm6, cur_t0 = state_tm6, state_t0

        # 自回归 rollout
        for k in range(max_lead_hours // lead_step):
            pred_next = AIFS_step(cur_tm6, cur_t0)
            pred_states.append(pred_next)

            # 更新输入
            cur_tm6 = cur_t0
            cur_t0 = pred_next

        # 计算面积加权 MSE 损失（含变量和高度加权）
        loss = area_weighted_scaled_MSE(pred_states, states_truth)

        # 反向传播与参数更新（使用 AdamW）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

3. 多 GPU 序列/张量并行（attention head 分片示意）

```pseudo
function parallel_multihead_attention(x, num_heads, devices):
    # x: [batch, seq_len, hidden_dim]
    # 将 head 维度按设备划分
    heads_per_device = num_heads // len(devices)

    # 预先在每个设备上建立局部 head 投影
    for d in devices in parallel:
        local_heads = select_heads_for_device(d, heads_per_device)
        q_d, k_d, v_d = project_qkv(x, local_heads, device=d)
        attn_out_d = scaled_dot_product_attention(q_d, k_d, v_d)

    # all-to-all 通信聚合各设备 head 结果
    attn_out = all_to_all_concat([attn_out_d for d in devices])

    # 输出映射回 hidden_dim
    y = output_projection(attn_out)
    return y
```


## 6. 数据规格

### 6.1 数据集名称
- ERA5 重分析数据集（ECMWF / Copernicus）。
- ECMWF 运行中的 IFS 数值预报分析场（4D-Var）。
- 验证使用：
  - ECMWF 运行分析场；
  - 探空观测（radiosonde）中的位势高度、温度、风速；
  - SYNOP 地面观测中的 2 m 温度、10 m 风速、24 h 总降水量等。

### 6.2 时间范围
- ERA5 预训练阶段：
  - 1979–2020 年，用于 6 小时单步预报预训练（260,000 步）。
- ERA5 rollout 训练阶段：
  - 1979–2018 年，用于最长 72 h 的 rollout 训练。
- IFS 运行分析场微调阶段：
  - 2019–2020 年，用于在 IFS 分析场上进行 rollout 微调。
- 评估示例年份：
  - 主要结果展示 2022 年（ACC、RMSE 等）；
  - TC 结果覆盖 2022 年 1 月至 2023 年 12 月。

### 6.3 变量列表

按表格（Table 1）及正文描述整理：

1. 气压面变量（Pressure levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa）
- 既作为输入也作为输出：
  - Geopotential（位势高度）
  - Horizontal wind components（水平风分量，通常为 U、V）
  - Vertical wind component（垂直风分量）
  - Specific humidity（比湿）
  - Temperature（温度）

2. 地面/表面变量（Surface）
- 既作为输入也作为输出：
  - Surface pressure（地面气压）
  - Mean sea-level pressure（海平面气压，MSLP）
  - Skin temperature（地表皮肤温度）
  - 2 m temperature（2 m 温度）
  - 2 m dewpoint temperature（2 m 露点温度）
  - 10 m horizontal wind components（10 m 水平风分量）
  - Total column water（总柱水量）
- 仅作为输出：
  - Total precipitation（总降水）
  - Convective precipitation（对流性降水）
- 仅作为输入（强迫/静态变量）：
  - Land-sea mask（陆海掩膜）
  - Orography（地形高度）
  - Standard deviation of sub-grid orography（次网格地形标准差）
  - Slope of sub-scale orography（次网格地形坡度）
  - Insolation（辐射相关量/日照）
  - Latitude/longitude（经纬度）
  - Time of day / day of year（一天中的时间/一年中的日期）

### 6.4 数据处理
- 空间插值与网格：
  - IFS 运行分析场原始分辨率为 O1280（约 0.1°），在训练与推理中插值到 N320 网格（约 0.25°）；
  - AIFS encoder/decoder 在 N320 网格上工作，processor 在 O96 网格上工作；
  - 在可视化比较时，将 IFS 和 AIFS 预报场统一插值到 0.25° 规则经纬网格。
- 归一化：
  - 输入与输出状态在每个垂直层上进行标准化，使其具有零均值、单位方差；
  - 部分强迫变量（如 orography）采用 min–max 归一化。
- 损失缩放：
  - 每个输出变量有单独的损失缩放系数，通过经验方式设定，以使各预报量对损失贡献大致相等；
  - 垂直速度的损失权重降低；
  - 垂直方向损失权重随高度线性减小，高层（如 50 hPa）对总体损失贡献较小。
- 批量与精度：
  - 训练批量大小为 16；
  - 使用混合精度训练（mixed precision）。


# 维度二：基础组件知识（Component-Level Knowledge）

## 组件 1：GNN 编码器（ERA5 -> 处理器网格）

1. 子需求定位
- 对应的子需求：
  - 将高分辨率的 N320 ERA5/IFS 网格信息高效聚合到较低分辨率的 O96 处理器网格，以便在计算可承受的前提下保持足够的空间信息。
- 解决了什么问题：
  - 直接在高分辨率经纬网格/减少高斯网格上做全局注意力或卷积成本过高；
  - 需要在保持局地结构的同时实现跨网格信息投影与降维。

2. 技术与创新
- 关键技术点：
  - 使用图神经网络（基于多头图 Transformer 卷积）将 N320 网格映射到 O96 网格；
  - 编码器图通过截断半径构建邻接关系，使处理器节点只与空间邻近的 ERA5 节点相连；
  - 边特征包括边长与方向，显式编码空间几何结构。
- 创新点：
  - 在数据驱动天气模型中，结合 reduced Gaussian grid 与基于物理空间几何的 GNN 编码器，实现从高分辨率输入到较粗处理器网格的高效映射。

3. 上下文与耦合度
- 上下文组件位置：
  - 前接：输入变量嵌入与可学习节点/边特征；
  - 后接：O96 网格上的 pre-norm Transformer 处理器。
- 绑定程度：
  - 强绑定-气象特异：
    - 使用 ERA5/IFS 的特定网格（N320）、空间几何关系（边长、方向）以及大气物理变量；
    - 依赖于球面网格结构和大气场分布特征。

4. 实现细节与代码
- 理论公式：
  - 文中仅指出“多头图 Transformer 卷积”更新节点与边，未给出显式公式。
- 实现描述：
  - 构建二部图：源节点为 N320 格点，目标节点为 O96 处理器格点；
  - 每个 O96 节点连接其截断半径内所有 N320 节点；
  - 每条边计算长度与方向作为特征；
  - 节点和边特征经过线性映射后输入多层 GNN transformer block，输出为 O96 节点的潜在表示。
- 伪代码（基于文中描述生成的伪代码）：

```pseudo
function build_encoder_graph(N320_nodes, O96_nodes, radius):
    edges = []
    for j in O96_nodes:
        neighbors = find_neighbors_within_radius(N320_nodes, j, radius)
        for i in neighbors:
            length, direction = compute_edge_geometry(i, j)
            edges.append((i, j, length, direction))
    return Graph(nodes=(N320_nodes, O96_nodes), edges=edges)

function GNN_encoder(node_feats_N320, node_feats_O96, edges):
    # 初始化边特征
    edge_feats = linear_embed_edge([e.length, e.direction for e in edges])
    node_feats_N320 = concat(node_feats_N320, learnable_node_feats_input)
    node_feats_O96  = concat(node_feats_O96,  learnable_node_feats_proc)

    # 在二部图上应用多层 Graph Transformer block
    for l in range(num_gnn_layers):
        node_feats_N320, node_feats_O96, edge_feats = graph_transformer_layer(
            node_feats_N320, node_feats_O96, edge_feats, edges
        )

    return node_feats_O96
```


## 组件 2：GNN 解码器（处理器网格 -> ERA5 网格）

1. 子需求定位
- 对应的子需求：
  - 将 O96 处理器网格上的潜在状态精确映射回 N320 网格，以输出高分辨率预报场。
- 解决了什么问题：
  - 需要在较低分辨率上进行高效时空建模，同时最终输出需要与 ERA5/IFS 网格对齐以便评估和业务使用；
  - 通过局部邻域（最近 3 个处理器节点）可减少边数并保持局地一致性。

2. 技术与创新
- 关键技术点：
  - 解码器图中，每个 N320 节点连接到其最近的 3 个 O96 处理器节点；
  - 同样使用图 Transformer 卷积和边几何特征进行信息传播；
  - 输出线性层将潜在状态映射回物理变量。
- 创新点：
  - 使用少量最近邻（3 个）连接在保证局地一致性的同时控制计算量。

3. 上下文与耦合度
- 上下文组件位置：
  - 前接：O96 pre-norm Transformer 处理器输出；
  - 后接：线性输出层与解标准化操作（映射回物理空间）。
- 绑定程度：
  - 强绑定-气象特异：
    - 使用 N320/O96 的具体空间布局和几何关系；
    - 输出变量完全是气象物理量。

4. 实现细节与代码
- 实现描述：
  - 为每个 N320 节点寻找 3 个最近的 O96 节点，计算边长和方向；
  - 通过多层 GNN transformer block 传播信息到 N320 节点特征；
  - 最后通过线性层预测多变量物理量。
- 伪代码（基于文中描述生成的伪代码）：

```pseudo
function build_decoder_graph(O96_nodes, N320_nodes, k=3):
    edges = []
    for i in N320_nodes:
        neighbors = find_k_nearest(O96_nodes, i, k)
        for j in neighbors:
            length, direction = compute_edge_geometry(j, i)
            edges.append((j, i, length, direction))
    return Graph(nodes=(O96_nodes, N320_nodes), edges=edges)

function GNN_decoder(node_feats_O96, edges):
    edge_feats = linear_embed_edge([e.length, e.direction for e in edges])
    node_feats_O96 = concat(node_feats_O96, learnable_node_feats_proc)

    # 通过 GNN 将信息传播到 N320 节点
    for l in range(num_gnn_layers):
        node_feats_O96, node_feats_N320, edge_feats = graph_transformer_layer(
            node_feats_O96, None, edge_feats, edges
        )

    # 线性输出
    output_fields = linear_output(node_feats_N320)
    return output_fields
```


## 组件 3：Pre-norm Transformer 处理器（O96 网格上的滑动窗口注意力）

1. 子需求定位
- 对应的子需求：
  - 在全球尺度上对处理器网格上的大气状态进行时间–空间演化建模，捕获大尺度和中尺度的动力学特征，并为较长时效预报提供稳定的时序推理能力。
- 解决了什么问题：
  - 全局自注意力在高分辨率网格上的计算成本过高；
  - 需要一个在球面网格上既能利用局地结构又能通过层叠扩展感受野的注意力机制；
  - 同时避免显式依赖边信息，以简化处理器结构。

2. 技术与创新
- 关键技术点：
  - 使用 pre-norm Transformer 结构（LayerNorm 在子层前），搭配 GELU 激活；
  - 注意力沿纬向带计算，使用局部 attention windows；
  - 引入 shifted windows，使得不同层的窗口在纬向上偏移，从而在多层中扩展信息传递范围；
  - 16 层处理器堆叠，示意图表明 6 层已能覆盖较大邻域，实际更深以增强建模能力。
- 创新点：
  - 在 reduced Gaussian O96 网格上基于纬带的 shifted window attention，将球面结构与局部–全局信息融合；
  - 处理器不再显式使用图边信息，而是完全通过注意力机制建模空间依赖，这与 encoder/decoder 的显式图结构形成互补。

3. 上下文与耦合度
- 上下文组件位置：
  - 前接：GNN 编码器输出的 O96 潜在表示；
  - 后接：GNN 解码器，将处理器输出映射回 N320 网格。
- 绑定程度：
  - 强绑定-气象特异：
    - 注意力窗口沿纬向带定义，并根据 O96 网格的球面布局设计，以保证包含完整局地邻域；
    - 适配全球大气流场的纬向结构特征。

4. 实现细节与代码
- 实现描述：
  - 对 O96 网格进行排序或分组，以便形成纬向带序列；
  - 在每一层中，对每个节点从其所属窗口内的节点计算自注意力；
  - 层间通过窗口位移（shift）改变节点所属窗口，从而实现更大范围的跨窗口信息交换；
  - 使用 pre-norm 架构（LayerNorm -> Multi-head Attention -> 残差）和前馈网络（FFN）子层。
- 伪代码（基于文中描述生成的伪代码）：

```pseudo
function processor_transformer(latent_O96):
    x = latent_O96  # shape: [num_nodes, hidden_dim]

    for layer in range(16):
        windows = build_latitude_windows(O96_grid, shift=layer_is_odd(layer))

        # 多头自注意力（分窗口）
        x = pre_norm_with_windowed_attention(x, windows)
        x = pre_norm_with_ffn(x)

    return x

function pre_norm_with_windowed_attention(x, windows):
    x_norm = LayerNorm(x)
    out = zeros_like(x)
    for w in windows:
        nodes = w.nodes
        out[nodes] = multihead_attention(x_norm[nodes])
    return x + out
```


## 组件 4：可学习节点与边特征（8 维 learnable features）

1. 子需求定位
- 对应的子需求：
  - 在输入网格、处理器网格以及 encoder/decoder 边上增加容量，使模型可以自动学习对预报有用但未在物理变量中显式给出的特征。
- 解决了什么问题：
  - 仅依赖显式输入变量可能无法充分表达所有有用统计模式；
  - 预留的可学习维度为模型提供额外自由度。

2. 技术与创新
- 关键技术点：
  - 为每个节点和边添加 8 维可学习向量，作为额外特征；
  - 这些特征与物理变量/几何量一起输入线性层和 GNN/Transformer 模块。
- 创新点：
  - 在图结构的 encoder/decoder 以及处理器网格上统一引入 learnable features，为模型自动构造辅助特征提供空间。

3. 上下文与耦合度
- 上下文组件位置：
  - 前接：基础物理量和几何特征；
  - 后接：线性嵌入层与 GNN/Transformer 块。
- 绑定程度：
  - 弱绑定-通用：
    - 该机制本身与具体大气变量无关，可广泛用于其它图/网格模型；
    - 在此任务中与气象变量共同使用。

4. 实现细节与代码
- 实现描述：
  - 为每类节点/边初始化 8 维参数向量，并在训练过程中更新；
  - 与原始特征按维度拼接后输入后续网络。
- 伪代码（基于文中描述生成的伪代码）：

```pseudo
# initialization
learnable_node_feats_input  = Parameter(num_input_nodes,  8)
learnable_node_feats_proc   = Parameter(num_proc_nodes,   8)
learnable_edge_feats_encdec = Parameter(num_edges_encdec, 8)

# usage example in encoder
node_feats_era5 = concat(phys_node_feats_era5, learnable_node_feats_input)
edge_feats_enc  = concat(geom_edge_feats_enc,  learnable_edge_feats_encdec)
```


## 组件 5：激活检查点（Activation Checkpointing）

1. 子需求定位
- 对应的子需求：
  - 在保持模型规模和输入分辨率的前提下，降低单卡显存占用，以便在有限显存的 GPU 上进行训练。
- 解决了什么问题：
  - 高分辨率网格、长 rollout 和深层网络导致中间激活数量巨大，超出显存容量；
  - 必须在时间和显存之间进行权衡。

2. 技术与创新
- 关键技术点：
  - 在前向传播时不保存所有中间激活，而只保存检查点位置；
  - 在反向传播时重新计算未保存的激活，以换取显存节省。
- 创新点：
  - 在大规模数据驱动天气模型中系统地使用 activation checkpointing，以支持高分辨率 ERA5 输入和多步 rollout 训练。

3. 上下文与耦合度
- 上下文组件位置：
  - 跨越整个 AIFS 模型的前向和反向计算过程（GNN 和 Transformer 块）。
- 绑定程度：
  - 弱绑定-通用：
    - 该技术与任务无关，可应用于各种深度网络。

4. 实现细节与代码
- 实现描述：
  - 在若干大型模块（如成组的 Transformer 层或 GNN 块）之间设置检查点；
  - 使用深度学习框架提供的 checkpointing API 来自动重算。


## 组件 6：序列/张量并行（Attention head 分片）

1. 子需求定位
- 对应的子需求：
  - 扩展单模型实例到多块 GPU 上，以支持更高分辨率、更长 rollout 和更大 batch。
- 解决了什么问题：
  - 单卡显存和算力不足以支撑完整的 AIFS 模型与训练配置；
  - 需要高效通信模式减小并行带来的开销。

2. 技术与创新
- 关键技术点：
  - 在 GNN encoder/decoder 和 Transformer processor 中，将多头注意力的 head 在 GPU 之间进行分片；
  - 使用高效的 all‑to‑all 通信原语在设备间同步注意力结果；
  - 对 GNN 还提供节点/边分片模式，通过节点重排将通信简化为一次 gather 与一次 reduce。
- 创新点：
  - 将 head-level 张量并行与图结构/球面网格结合，支持 up to 2048 GPU 的 quasi-linear 弱缩放。

3. 上下文与耦合度
- 上下文组件位置：
  - 作用于 GNN encoder、GNN decoder、Transformer processor 的注意力计算与图更新中。
- 绑定程度：
  - 弱绑定-通用：
    - 张量并行方案可用于其它大型 Transformer/GNN 模型，但本工作中特别针对 AIFS 的图结构和网格布局进行了优化。

4. 实现细节与代码
- 伪代码：见“维度一”中 5.3 的 parallel_multihead_attention 示例。


## 组件 7：节点/边分片（基于 1-hop 邻域的 GNN 并行）

1. 子需求定位
- 对应的子需求：
  - 在 GNN encoder/decoder 中减少跨 GPU 通信，提升并行效率。
- 解决了什么问题：
  - 图结构天然跨越空间区域，直接切分会产生大量跨设备边；
  - 需要通过合适的节点重排减小通信次数。

2. 技术与创新
- 关键技术点：
  - 根据 1‑hop 邻域划分节点和边至不同 GPU；
  - 通过对图节点进行重新排序，使得在前向传播中只需一次 gather 操作，在反向传播中只需一次 reduce 操作。
- 创新点：
  - 针对 GNN 在全球网格上的特殊拓扑，设计低通信量的并行划分策略。

3. 上下文与耦合度
- 上下文组件位置：
  - 主要用于 GNN encoder 和 decoder 的可选并行模式，替代 head 分片方案或与之结合。
- 绑定程度：
  - 弱绑定-通用但针对图结构：
    - 可用于其它图神经网络，但分片策略依赖具体图的邻域结构。

4. 实现细节与代码
- 文中仅给出高层描述，未提供伪代码；此处不再补充。


## 组件 8：概率预报训练通信模式（Ensemble-based Training Communication Pattern）

1. 子需求定位
- 对应的子需求：
  - 在训练中直接基于集合预报成员优化概率评分（如 fair scores），构建面向概率预报的 AIFS 训练框架。
- 解决了什么问题：
  - 传统 deterministic 训练无法直接针对概率评分进行优化；
  - 大规模集合在训练阶段的计算和通信成本较高。

2. 技术与创新
- 关键技术点：
  - 将不同 ensemble 成员分布在多个 GPU 上；
  - 设计设备到设备的通信模式，以计算集合层面的概率评分；
  - 支持更大规模集合的训练。
- 创新点：
  - 在大规模数据驱动天气预报模型中，面向 probabilistic scores（包括 fair scores）实现训练时的 ensemble 优化，而不仅仅是 deterministic MSE。

3. 上下文与耦合度
- 上下文组件位置：
  - 主要作用于训练阶段的损失计算与梯度回传；
  - 可与数据并行、序列/张量并行结合使用。
- 绑定程度：
  - 强绑定-气象特异：
    - 目标评分（如 fair scores）来自概率预报评价文献（Gneiting & Raftery, Ferro 等），特定于气象概率预报领域。

4. 实现细节与代码
- 文中未给出具体实现或公式，仅说明存在此通信模式并可用于 ensemble-based training；此处不添加伪代码。


## 组件 9：Rollout 训练策略

1. 子需求定位
- 对应的子需求：
  - 提升长时效预报性能，减轻自回归误差累积和分布偏移问题。
- 解决了什么问题：
  - 若仅训练单步预测，模型在长时效 rollout 时会遭遇严重误差积累；
  - 需要在训练中显式暴露模型于自身预测驱动的序列。

2. 技术与创新
- 关键技术点：
  - 先进行单步（6 h）预训练，再逐步增加 rollout 长度直至 72 h；
  - 梯度在整个 forecast chain 中传播；
  - 最后在 IFS 分析场上做相同的 rollout 微调。
- 创新点：
  - 结合 ERA5 和 IFS 分析场的两阶段 rollout 训练，延续并扩展了 GraphCast 等模型的训练原则。

3. 上下文与耦合度
- 上下文组件位置：
  - 涉及训练数据管线和损失计算；
  - 与 AIFS_step（单步预测）函数相结合完成多步 rollout。
- 绑定程度：
  - 强绑定-气象特异：
    - 目标为 6 h 步长、最长 72 h 的中期天气预报；
    - 与 ERA5/IFS 的时间分辨率与业务需求相关。

4. 实现细节与代码
- 伪代码：见“维度一”中 5.3 的 train_rollout 示例。


## 组件 10：归一化与损失缩放策略

1. 子需求定位
- 对应的子需求：
  - 统一不同气象变量的数值尺度，稳定训练过程，避免某些变量主导损失；
  - 控制不同高度层在损失中的相对权重。
- 解决了什么问题：
  - 各变量物理量纲不同，直接使用原始值会导致梯度不平衡；
  - 平流层变量在观测和分析中误差特征不同，需要权重调整。

2. 技术与创新
- 关键技术点：
  - 对每个垂直层进行标准化（零均值、单位方差）；
  - 对部分强迫变量使用 min–max 归一化；
  - 为每个输出变量设置损失缩放因子，使各变量对总损失贡献近似相等；
  - 垂直方向权重随高度线性减小，弱化高层贡献；
  - 垂直速度损失权重进一步降低。
- 创新点：
  - 明确针对平流层技能和垂直速度预报进行权重设计，使模型更关注对流层和近地层预报。

3. 上下文与耦合度
- 上下文组件位置：
  - 贯穿数据预处理（归一化）和训练损失计算阶段。
- 绑定程度：
  - 强绑定-气象特异：
    - 高度权重与大气分层结构直接相关；
    - 垂直速度等变量的权重选择源自物理与预报需求。

4. 实现细节与代码
- 理论公式：参见“维度一”中 5.1 的损失函数表达式。
- 实现描述：
  - 在数据预处理阶段记录每个变量和高度层的均值和标准差；
  - 在训练时，用这些统计量进行标准化和反标准化；
  - 损失中按变量和高度乘以对应缩放因子和高度权重再求和。