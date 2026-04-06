# 维度一：整体模型知识（Model-Level Knowledge）

## 1. 核心任务与需求

### 1.1 核心建模任务
- 构建并分析一个基于 Swin Transformer V2（SwinV2）的全球中期（1–7 天）数据驱动天气预报模型：
  - 输入：ERA5 的大气状态 \(\mathbf{X}_t\)。
  - 输出：下一时刻（6 小时后）的状态 \(\mathbf{X}_{t+\Delta t}\)，其中 \(\Delta t = 6\,\mathrm{h}\)。
  - 通过自回归 roll-out 在推理阶段生成多步预报（最高 7–10 天，评估到 7 天、WeatherBench2 到 10 天）。
- 在“全 ERA5 分辨率” \(0.25^\circ\) 下，系统性地研究训练 recipe（损失函数、patch 大小、模型深度/宽度、多步微调等）对预报技能（主要是 RMSE）的影响。
- 与 ECMWF 物理模式 IFS 比较，关注关键变量（如 z500、t2m、u10m）RMSE 的优劣。

### 1.2 解决了什么问题
- 架构复杂性与训练开销：
  - 现有 DL–NWP 模型架构多样、训练设置差异巨大、超参昂贵难调，缺乏全面消融，难以判断哪些设计真正关键。
  - 本文展示：在中等算力预算+相对“开箱即用”的 SwinV2 架构+简单训练流程下，仍可在确定性均方误差上整体优于 IFS。
- 分辨率与细节：
  - 许多消融工作在较低分辨率（\(\sim 1.4^\circ\)）进行，而高分辨率（\(0.25^\circ\)）预报更有实际价值，同时深度学习模型普遍存在“模糊”“缺乏高频细节”问题。
  - 本文直接在全分辨率 ERA5 上训练与分析。
- 训练 recipe 不透明：
  - loss 形式（普通/纬度加权 MSE、通道加权）、multistep fine-tuning、patch 大小、模型规模等彼此纠缠，缺少系统对比。
  - 本文提供系统消融，降低未来 DL–NWP 模型超参搜索和调参门槛。


## 2. 解决方案

### 2.1 核心解决思路
- 使用“最小修改”的 SwinV2 架构：
  - 只做三点关键修改：
    1) 针对球面经度周期性调整 window shifting 的边界掩膜策略；
    2) 去掉相对位置编码，改用标准 ViT 式绝对位置嵌入；
    3) 取消层间分辨率层次结构（非分层、分辨率保持不变）。
- 使用标准单步训练 + 多步微调：
  - 先训练 1-step 预测（\(\mathbf{X}_t \to \mathbf{X}_{t+1}\)），然后进行 4 步、8 步 roll-out 微调，提升长时效 RMSE。
- 系统探索关键训练因子：
  - 模型规模：深度（12/24 层）、宽度（embed_dim 768/1536）。
  - patch 大小：4、8、16 以及对应的 attention window 尺度。
  - 通道加权：沿用 Lam et al. 方案，对高空、方差较大的变量等重新加权，并采用“残差预测”形式。
  - 纬度加权：是否使用 cos(latitude) 加权 MSE。
- 使用开放评分管线：
  - Earth2Mip 进行标准化评分（lat-weighted RMSE、能谱、lagged ensemble spread/skill、CRPS 等）。

### 2.2 结构映射（思路对应模块）
- “最小修改 SwinV2 也能达到高技能” → 模型架构：
  - SwinV2 主体 + 三处结构性修改（window shifting、pos embedding、hierarchy）。
- “系统比较训练 recipe” → 训练流程与损失：
  - 不同模型规模/patch 组合 → 不同 SwinV2 配置（见表 1）。
  - 通道加权/残差预测 → 损失函数与输出头设计。
  - multistep fine-tuning → 训练 schedule（额外 15 epoch、多步 roll-out）。
  - 纬度加权 → 损失权重机制。
- “探究高频细节、ensemble spread 代价” → 评估模块：
  - 功率谱分析、高波数能量；
  - lagged ensemble 构造与 spread–skill、CRPS 计算。


## 3. 模型架构概览

- 总体框架：
  - 单步 SwinV2 模型：\(\mathbf{X}_t \mapsto \hat{\mathbf{X}}_{t+\Delta t}\)。
  - 推理时自回归 roll-out 产生多步预报：\(\hat{\mathbf{X}}_{t+2\Delta t}, \hat{\mathbf{X}}_{t+3\Delta t}, \dots\)。
- 输入输出：
  - 输入通道数：73（13 层 z, u, v, t, q 共 65 通道 + 8 个地面/单层变量）+ 3 个静态附加输入（land–sea mask、orography、cos(太阳天顶角)）作为额外输入通道。
  - 空间分辨率：纬向 720 × 经向 1440（0.25°）。
- SwinV2 主干：
  - patch embedding：将 73 通道场切成 patch（基线 patch_size = 4，经纬维度各缩小 4 倍，到 180×360 patch 网格）。
  - 绝对位置编码：在 patch 嵌入后加上二维绝对位置嵌入。
  - 非分层：全 12（或 24）层保持同一空间分辨率和通道数（例如基线 embed_dim = 768）。
  - 局部 self-attention：
    - attention window 尺寸：基线为 9×18 patch；
    - 每层内先在窗口内做自注意力，再在下一层通过 shift windows 机制在空间上平移窗口，实现跨窗口的信息传播。
  - 输出头：将最后一层 patch 表征通过线性层映射回网格上每个点的 73 通道预测值（或预测残差）。
- 修改点：
  1) window shifting：对经度维不做“防越界”mask（经向周期），只对纬度边界（两极）做 mask。
  2) 相对位置编码 → 绝对位置编码：简化位置偏置实现，因为只考虑固定分辨率场景。
  3) 取消层级下采样：维持同分辨率结构，参考 Earth science 时空预测中非分层 Transformer 已证实有效的做法。


## 4. 创新与未来

### 4.1 论文声称的主要贡献/创新
- 模型层面：
  - 在几乎“开箱即用”的 SwinV2 架构上，仅通过三处小改动，在 ERA5 全分辨率上训练出在确定性均方误差指标上整体优于 IFS 的模型。
- 训练 recipe 层面：
  - 系统消融：
    - 模型深度 vs 宽度：发现加宽（embed_dim 1536）整体优于加深（24 层），尤其在 7 天 lead 的 z500、t2m、u10m 等上有 10–15% RMSE 差距。
    - patch 大小：patch=4 在 t2m 上始终最好，而在 z500、u10m 的后期时效上，patch=8/16 稍有优势。
    - 通道加权：对多数 lead time、变量的 RMSE 有提升，即便是权重中被下调的通道（如 z500）也受益。
    - multistep fine-tuning：4 步/8 步 fine-tune 明显改善中长期 RMSE，且 8 步在 5 天以上 lead 时优于 4 步。
    - 纬度加权：其效果依赖于是否结合 multistep、通道加权等；8-step fine-tuned 模型中，lat-weighted loss 通常更优，而仅 1-step 训练时则可能更差。
- 质量折衷分析：
  - 通过功率谱和 lagged ensemble 分析展示：
    - multistep fine-tuning 虽提升 RMSE，但明显降低高波数能量（wavenumber > 10 的 u10m），造成平滑/模糊化；
    - lagged ensemble 的 spread–skill 比例下降，但 CRPS 与 ensemble RMSE 依然受益（8-step 模型在所有 lead time 上好于 baseline）。
- 社区资源：
  - 提供开源代码、评分管线、模型权重，帮助未来工作复现与扩展，减少昂贵超参搜索的必要性。

### 4.2 后续方向 / 局限
- multistep fine-tuning 的折衷：
  - 高技能但更模糊、ensemble spread 减小，如何在 RMSE 与 sharpness/不确定性度量之间取得更好平衡？
- 通道加权 vs 直接/残差预测的拆解：
  - 当前 channel weighting 同时改变了权重和预测形式（残差预测 vs 直接预测），文中未完全解耦这两者，未来可分别控制以更清晰分析其影响。
- 纬度加权与其它设置的耦合：
  - 实验表明纬度加权效果依赖于是否 multistep、是否 channel weighting，未来需要在更系统的网格上联合搜索和理解。


## 5. 实现细节与代码逻辑

### 5.1 损失函数与权重（概念公式）

1. 基础（未加权）\(\mathcal{L}_2\) 损失
- 描述：标准逐变量、逐网格点的均方误差（RMSE 对应的训练目标）。
- 一般形式：
\[
L_{\text{MSE}} = \frac{1}{N} \sum_{c=1}^{C} \sum_{i=1}^{N} \bigl( x_{c,i}^{\text{true}} - x_{c,i}^{\text{pred}} \bigr)^2,
\]
其中 \(c\) 为通道（变量×层）索引，\(i\) 为空间网格点。

2. 纬度加权 \(\mathcal{L}_2\)
- 描述：沿用 WeatherBench 传统，用纬度的余弦作权重，补偿高纬格点面积更小的问题。
- 一般形式可写为：
\[
L_{\text{lat}} = \frac{1}{Z} \sum_{c=1}^{C} \sum_{i=1}^{N} w_{\phi(i)} \bigl( x_{c,i}^{\text{true}} - x_{c,i}^{\text{pred}} \bigr)^2,
\]
其中：
  - \(\phi(i)\) 为第 \(i\) 个网格点的纬度；
  - \(w_{\phi} = \cos(\phi)\)；
  - \(Z = \sum_i w_{\phi(i)}\) 为归一化因子。

3. 通道加权 + 残差预测（概念）
- 描述：
  - 权重由三部分组成：
    1) 随高度减小（高层权重更小）；
    2) 与时间差分标准差 \(\sigma_{\delta\mathbf{X}}\) 成反比（方差大的变量权重更小）；
    3) 对某些地面变量（如 t2m）手工上调权重；
  - 同时模型预测的是 \(\Delta \mathbf{X} = \mathbf{X}_{t+1} - \mathbf{X}_t\)（残差预测），损失在 \(\Delta \mathbf{X}\) 上计算。
- 抽象形式：
\[
L_{\text{chan}} = \frac{1}{Z} \sum_{c=1}^{C} \alpha_c \sum_{i=1}^{N} w_{\phi(i)} \bigl( \Delta x_{c,i}^{\text{true}} - \Delta x_{c,i}^{\text{pred}} \bigr)^2,
\]
其中：
  - \(\alpha_c\) 为通道权重（包含高度因子、方差因子、手工加权等）；
  - \(\Delta x_{c,i} = x_{c,i}(t+1) - x_{c,i}(t)\)。

> 上述公式为根据文中对“MSE、纬度加权、通道加权和残差预测”的文字描述整理出的通用表达；论文未给出具体解析式。

### 5.2 训练配置与流程

- 数据预处理：
  - 时间步长：从 ERA5 小时数据中以 \(\Delta t = 6\,\mathrm{h}\) 采样。
  - 通道选择：73 变量（13 层 z,u,v,t,q + 8 个表面/单层变量）+ 静态输入（land–sea mask, orography, cos(zenith angle)）。
  - 归一化：对每个变量按全局时空均值和标准差做标准化（zero mean, unit std）。
- 训练集/验证集/测试：
  - 训练：1979–2015 + 2019 年；
  - 验证：2016–2017；
  - 评估：2018 年（与 GraphCast、Pangu 等工作对齐）。
  - 6 小时间隔总计产生 55,480 个训练样本。
- 基线 SwinV2 模型（swin_depth12_embedding768_batch4）：
  - embed_dim = 768；
  - depth = 12；
  - patch_size = 4；
  - num_heads = 8；
  - window_size = 9×18 patch；
  - 参数量约 136,666,464。
- 训练细节：
  - 优化器：Adam；
  - 损失：纬度加权 \(\mathcal{L}_2\)；
  - DropPath rate = 0.1；
  - 初始学习率 0.001，cosine annealing 调度；
  - batch size = 64；
  - 训练 70 个 epoch，采用 PyTorch DDP 在 64 张 A100（Perlmutter 超算）上分布式训练；
  - 基线模型：
    - 每个 epoch 训练约 8 分 40 秒；
    - 验证约 37 s；
    - 总墙钟时间约 11 小时；
  - 多步 fine-tuning 最多 8 步时，总训练时间可达约 48 小时；
  - 激活检查点：在 fine-tuning 配置下为适配显存而启用。

### 5.3 多步 fine-tuning 伪代码（基于文中描述生成的伪代码）

```pseudo
# 单步模型：输入 X_t，输出预测 X_{t+1}
function SwinV2_step(X_t):
    patches = patch_embed(X_t)                      # [B, H/4, W/4, C_emb]
    patches = patches + absolute_pos_embed          # 加绝对位置编码
    for layer in range(num_layers):                 # 12 或 24
        patches = swin_block_with_shifted_windows(
            patches,
            window_size=(9, 18),                    # 或 (5,10), (3,6)
            periodic_longitude=True
        )
    out = patch_unembed_and_head(patches)           # 映射回 73 通道网格
    return out

# 1-step 预训练
for epoch in range(70):
    for X_t, X_tp1 in train_loader:
        X_pred = SwinV2_step(X_t)
        loss = lat_weighted_L2(X_pred, X_tp1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# K-step fine-tuning（K=4 或 8）
for epoch in range(15):
    for X_t, [X_tp1, ..., X_tK] in train_loader:
        X_cur = X_t
        for k in range(1, K+1):
            X_next = SwinV2_step(X_cur)
            X_cur = X_next
        loss = lat_weighted_L2(X_cur, X_tK)         # 只对最终步计算 loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```


## 6. 数据规格

### 6.1 数据集名称
- 训练/验证/测试：
  - ERA5 重分析数据集（ECMWF IFS 同化生成，NCAR RDA 提供）。
- 评估对比：
  - IFS_HRES（WeatherBench2 中 IFS 高分辨率产品）；
  - Pangu-Weather；
  - GraphCast。

### 6.2 时间范围
- ERA5 时间覆盖：1979 至今（数据特性）。
- 本文具体使用：
  - 训练：1979–2015 + 2019（6 小时步长，55,480 个样本）；
  - 验证：2016–2017；
  - 评估：2018（与其他 DL–NWP 工作对齐）。
- Earth2Mip 评分：
  - 11 个起报时（2018 年均匀选取），lead time 0–7 天，6 小时步长。
- WeatherBench 2 评估：
  - 使用 12 小时间隔的起报时，共 731 个起报时，lead time 至 10 天。

### 6.3 变量列表

1. 多层大气变量（13 个气压层：50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa）
- 变量：
  - Geopotential height z（位势高度）；
  - Winds (u, v)（水平风分量）；
  - Temperature t（温度）；
  - Specific humidity q（比湿）。

2. 单层/地面变量（8 个）
- 10 m 风（u10, v10）；
- 100 m 风（u100, v100）；
- 2 m 温度（t2m）；
- Surface pressure (SP)（地面气压）；
- Mean sea level pressure (MSL)（海平面气压）；
- Total column water vapor (TCWV)（整层水汽）。

3. 静态/额外输入
- Land–sea mask（陆海掩膜）；
- Orography（地形高度）；
- Cosine of zenith angle（太阳天顶角余弦，表示一天中的时间/年度周期）。

### 6.4 数据处理
- 时间下采样：
  - 从 ERA5 小时数据中每隔 6 小时取样（\(\Delta t = 6\,\mathrm{h}\)）。
- 空间处理：
  - 使用 ERA5 原生 0.25° 经纬度网格（720×1440），不再下采样。
  - patch_embedding：按 patch_size {4,8,16} 划分空间 patch，与 window_size（9×18, 5×10, 3×6）配套调整。
- 标准化：
  - 每个变量按全局均值和标准差标准化；
  - 文中未提及额外的纬度权重标准化或按层标准化。
- 损失加权：
  - 纬度加权：按 cos(latitude) 加权；
  - 通道加权：参考 Lam et al.（基于气压高度、时间差分标准差、手工权重）。


# 维度二：基础组件知识（Component-Level Knowledge）

## 组件 1：最小修改版 Swin Transformer V2 主干

1. 子需求定位
- 对应子需求：
  - 在高分辨率（0.25°）全球网格上高效建模多变量大气状态的空间相关性；
  - 使用现成、成熟、易获取的 Transformer 架构，减少自定义复杂度。
- 解决的问题：
  - 替代复杂的自定义架构，在较少修改的前提下达到高预报技能；
  - 通过局部 attention + shifted windows 捕获逐步扩大的空间关联。

2. 技术与创新
- 核心技术点：
  - patch embedding + window-based multi-head self-attention；
  - shifted window attention 扩展感受野；
  - 残差连接 + LayerNorm + MLP（标准 Swin 块结构）。
- 相对标准组件的改动：
  - 见后续 组件 2–4 的三点修改（窗口边界处理、位置编码、非分层结构）。

3. 上下文与耦合度
- 上下文组件：
  - 前接：ERA5/静态变量标准化后的输入场 → patch embedding → 绝对位置编码；
  - 后接：线性头（回到 73 通道网格），用于计算损失或残差输出。
- 绑定程度：
  - 弱绑定-通用：
    - SwinV2 本身是通用视觉 Transformer 架构；
    - 只在空间分辨率和窗口大小等超参与天气任务相关。

4. 实现细节与伪代码
- 架构层面：标准 SwinV2 block 堆叠，对实现不再展开。


## 组件 2：窗口移位与经度周期性处理（Window Shifting for Periodic Longitude）

1. 子需求定位
- 子需求：
  - 在球面经度周期性网格上使用窗口 attention 时，正确处理经度边界（180°/360° 接壤），避免错误的“图像边界”假设。
- 解决问题：
  - 原始 SwinV2 假设输入是非周期 2D 图像，窗口移位后在图像边缘使用 mask 防止跨图像边界的 attention；
  - 对全球经度而言，0° 与 360° 相邻，应允许 attention wrap-around。

2. 技术与创新
- 关键技术：
  - 在 SwinV2 的 shifted window 实现中，经度维使用 `torch.roll` 后不对“左右边界”做 mask，只对纬度边界（极区）做 mask；
  - 这样 attention 可跨越经度 0/360° 边界，相当于在经度维上周期拓扑。
- 创新点：
  - 简单地利用 SwinV2 已有 `torch.roll` 实现，一行逻辑变更即可实现球面经度周期性支持。

3. 上下文与耦合度
- 上下文组件：
  - 内嵌于 SwinBlock 中的 window attention 子模块；
  - 与 patch embedding 与绝对位置编码配合使用。
- 绑定程度：
  - 强绑定-气象特异：
    - 该修改明确利用了全球经度的周期性特性；
    - 非地球图像/非球面任务一般不需此修改。

4. 实现描述与伪代码（基于描述生成）

```pseudo
function shifted_window_attention(x, window_size, shift_size):
    # x: [B, H, W, C]

    # 1. 循环移位
    x_shifted = torch.roll(x, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    # 2. 构造 mask
    #   - 经度维 (W) 不做 mask，允许 wrap-around
    #   - 仅在纬度维 (H) 两端构造 mask，防止跨越极区边界
    attn_mask = build_latitude_only_mask(H, W, window_size, shift_size_h)

    # 3. 划分窗口并进行窗口内自注意力
    windows = partition_windows(x_shifted, window_size)
    attn_windows = window_multihead_self_attention(windows, attn_mask)

    # 4. 将窗口还原到全图，并逆移位
    x_out = merge_windows(attn_windows, H, W)
    x_out = torch.roll(x_out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

    return x_out
```


## 组件 3：绝对位置嵌入（替代 SwinV2 相对位置偏置）

1. 子需求定位
- 子需求：
  - 对固定分辨率 ERA5 网格，引入空间位置信息以帮助模型感知经纬位置，同时避免相对位置偏置的复杂实现和多分辨率泛化需求。
- 解决问题：
  - SwinV2 原版相对位置偏置是为多尺度/多分辨率迁移设计的，当前任务只在固定 0.25° 分辨率上训练和推理，简化为绝对位置编码即可。

2. 技术与创新
- 关键技术：
  - 在 patch embedding 输出后，直接加上和 patch 网格同尺寸的可学习二维绝对位置向量（与 ViT 做法相同）。
- 创新点：
  - 非传统意义的新算法，但在 DL–NWP 语境下表明复杂的相对位置偏置并非必要，简化实现即可获得高技能。

3. 上下文与耦合度
- 上下文组件：
  - 前接：patch embedding（线性映射）；
  - 后接：Swin 模块堆叠。
- 绑定程度：
  - 弱绑定-通用：
    - 绝对位置编码是通用 Transformer 技术；
    - 此处只是在 ERA5 空间网格上具体应用。

4. 实现伪代码

```pseudo
# H_p, W_p: patch 网格尺寸
pos_embed = Parameter(shape=[1, H_p, W_p, C_emb])

function forward_patches(x):
    patches = patch_embed(x)            # [B, H_p, W_p, C_emb]
    patches = patches + pos_embed       # 加绝对位置编码
    return patches
```


## 组件 4：非分层 Swin 结构（Nonhierarchical SwinV2）

1. 子需求定位
- 子需求：
  - 在所有层保持相同空间分辨率和通道数，便于高分辨率时空预测，避免下采样带来的过度平滑与 aliasing。
- 解决问题：
  - 传统 SwinV2 通过 layer 之间 patch merging 下采样形成多尺度层次结构，适合图像分类，但可能损伤精细空间结构，对天气预测不一定有利。

2. 技术与创新
- 关键技术：
  - 移除 patch merging 层和分辨率变化，所有层使用相同 H_p×W_p 和 C_emb；
  - 参考 Earth science 预测任务中非分层 Transformer 的成功实践（Gao et al. 2022）。
- 创新点：
  - 将 SwinV2 从“图像分类风格层次结构”转化为“固定分辨率、全深度局部注意力”结构，更适配时空预报任务。

3. 上下文与耦合度
- 上下文组件：
  - 在整个 Swin 主干中生效，影响所有层的维度；
  - 不改变 patch embedding 和输出头的接口。
- 绑定程度：
  - 弱绑定-通用：
    - 概念可用于其它时空序列任务，但在天气任务中尤为自然。

4. 实现描述
- 在 timm SwinV2 实现中，去除 stage 间 patch_merging 调用，仅重复相同分辨率的 SwinBlock。


## 组件 5：patch 大小与 attention window 设计

1. 子需求定位
- 子需求：
  - 在 0.25° 分辨率下控制计算成本与可解析尺度之间的平衡；
  - 通过 patch 尺度和窗口大小共同决定 attention 的物理覆盖范围。
- 解决问题：
  - 过小 patch 导致计算成本高；过大 patch 则损失局地结构并引入 aliasing。

2. 技术与创新
- 关键技术：
  - patch_size ∈ {4, 8, 16}：
    - patch=4（基线）：空间分辨率最高，窗口 9×18 对应较小但细致的物理范围；
    - patch=8：窗口改为 5×10（保证总物理覆盖区适中）；
    - patch=16：窗口为 3×6。
  - 实验发现：
    - patch=4 在 t2m RMSE 上始终最优；
    - 在 z500、u10m 长 lead 上，patch=8/16 有时稍优于 patch=4。
- 创新点：
  - 系统对比分辨率/patch/窗口组合与技能的关系，为未来 high-res DL–NWP patch 设计提供经验指导。

3. 上下文与耦合度
- 上下文组件：
  - 影响 patch embedding 输出尺寸以及后续窗口 attention 的实现。
- 绑定程度：
  - 强绑定-气象特异（在本文场景下）：
    - 所选 patch 和窗口大小直接对应 0.25° 网格下的物理尺度。

4. 实现伪代码

```pseudo
if patch_size == 4:
    window_size = (9, 18)
elif patch_size == 8:
    window_size = (5, 10)
elif patch_size == 16:
    window_size = (3, 6)

patches = PatchEmbed(patch_size=patch_size)(X)
patches = swin_transformer(patches, window_size=window_size)
```


## 组件 6：通道加权损失（Channel-Weighted Loss）

1. 子需求定位
- 子需求：
  - 通过差异化权重控制不同变量/高度层在损失中的重要性，提升关键变量（尤其地面变量）的预报技能。
- 解决问题：
  - 直接使用统一权重会让方差较大的变量主导损失；
  - 需要兼顾物理重要性（如 t2m、地面风）和数值尺度差异。

2. 技术与创新
- 关键技术：
  - 依据 Lam et al. 方案：
    - 高度越高权重越小；
    - 通道权重与时间差分标准差 \(\sigma_{\delta \mathbf{X}}\) 成反比；
    - 对地面变量（如 t2m）附加人工权重；
  - 损失在残差 \(\Delta \mathbf{X}\) 上计算，进一步控制尺度。
- 创新点：
  - 验证通道加权在 SwinV2/ERA5 全分辨率下的效果，并量化其与 multistep fine-tuning、纬度加权的耦合。

3. 上下文与耦合度
- 上下文组件：
  - 在训练损失计算阶段生效；
  - 与输出头（直接/残差）紧密耦合。
- 绑定程度：
  - 强绑定-气象特异：
    - 权重依赖气压层、高度、时间差分统计等大气特有属性。

4. 实现伪代码（基于描述生成）

```pseudo
# 预先计算每个通道的 sigma_deltaX 和高度信息
alpha = compute_channel_weights(levels, sigma_deltaX, manual_surface_boost)

function channel_weighted_loss(pred, target, prev):
    # 残差预测
    delta_pred   = pred   - prev
    delta_target = target - prev

    loss = 0
    Z = 0
    for c in channels:
        for i in grid_points:
            w_lat = cos(latitude(i))
            diff2 = (delta_pred[c,i] - delta_target[c,i]) ** 2
            loss += alpha[c] * w_lat * diff2
            Z    += alpha[c] * w_lat
    return loss / Z
```


## 组件 7：多步微调（Multistep Fine-Tuning）

1. 子需求定位
- 子需求：
  - 提高中长期（多于 1–2 天）的预报 RMSE 技能，缓解自回归 roll-out 的误差累积和分布偏移。
- 解决问题：
  - 仅使用 1-step 训练时，模型在长 lead 上性能不足；
  - 需要让模型在训练中暴露于自身预测驱动的状态。

2. 技术与创新
- 关键技术：
  - 在 70 epoch 单步训练后，以较小学习率（\(1\times 10^{-4}\)）进行 15 epoch 的多步 fine-tuning；
  - fine-tuning 期间：
    - K=4 或 8 步 roll-out，使用 \(\hat{\mathbf{X}}_{t+K}\) 与真实 \(\mathbf{X}_{t+K}\) 计算损失；
  - 结果：K=8 在 5 天及以上 lead 的 RMSE 大幅优于 K=4 和 baseline；
  - 代价：高波数能量、spread–skill 比例下降（模糊、spread 减少）。

3. 上下文与耦合度
- 上下文组件：
  - 在训练后期（fine-tuning 阶段）启用，对模型参数进行额外适配；
  - 与通道加权、纬度加权联合使用时效果更明显。
- 绑定程度：
  - 强绑定-气象特异：
    - 设计针对 6 小时 time step 和 1–7 天 lead 的 NWP 需求。

4. 实现伪代码
- 已在 维度一 5.3 中给出 train_rollout / fine-tuning 示例。


## 组件 8：纬度加权损失（Latitude-Weighted Loss）

1. 子需求定位
- 子需求：
  - 反映不同纬度格点的真实物理面积，使损失与地表真实面积一致，避免高纬小格点被“过度计数”。
- 解决问题：
  - 规则经纬度网格上，高纬格点面积比低纬小；不加权时，高纬误差对总体损失的影响被放大。

2. 技术与创新
- 关键技术：
  - 权重 \(w(\phi) = \cos\phi\)（WeatherBench 经典做法）；
  - 在所有模型中默认使用，除非明确关闭以做对比实验。
  - 发现其与 channel weighting 与 multistep 细节存在耦合：
    - 在 8-step fine-tune 模型中，lat-weighted loss 更优；
    - 在仅 1-step 训练模型中，lat-weighted loss 反而显著变差。

3. 上下文与耦合度
- 上下文组件：
  - 损失计算阶段；
- 绑定程度：
  - 强绑定-气象特异：
    - 权重来源于球面几何与地球半径假设。

4. 实现公式
- 见 维度一 5.1 中的 \(L_{\text{lat}}\) 表达式。


## 组件 9：DDP 分布式训练与激活检查点

1. 子需求定位
- 子需求：
  - 在 64×A100 上高效训练大模型，缩短墙钟时间；
  - 通过激活检查点在有限显存下支撑多步 fine-tuning。
- 解决问题：
  - 高分辨率（0.25°）、大 batch（64）、深网络导致单卡显存压力；
  - 单机训练时间过长。

2. 技术与创新
- 关键技术：
  - 使用 PyTorch Distributed Data Parallel (DDP) 做数据并行；
  - 激活检查点用于 fine-tuning 配置，减少前向过程中中间激活存储。

3. 上下文与耦合度
- 上下文组件：
  - 贯穿训练 pipeline（数据加载、梯度同步、优化）。
- 绑定程度：
  - 弱绑定-通用：
    - 通用于各种大模型训练。

4. 实现描述
- 文中仅提及使用 DDP 和 activation checkpointing，不提供更细代码细节。


## 组件 10：评估与 lagged ensemble 机制

1. 子需求定位
- 子需求：
  - 提供标准化、多维度的评估指标，量化模型在确定性和概率意义上的表现；
  - 利用 deterministic 预报构建无参数的 lagged ensemble 评估 spread/skill 与 CRPS。
- 解决问题：
  - 单纯依赖 RMSE/ACC 无法体现模糊度和不确定性；
  - 构造真正的大集合在计算上昂贵，lagged ensemble 提供折中方案。

2. 技术与创新
- 关键技术：
  - 使用 Earth2Mip：
    - lat-weighted RMSE；
    - 能量谱（检视高波数能量与 aliasing）；
    - lagged ensemble spread–skill 比、CRPS；
  - lagged ensemble 构造：
    - 9 个 ensemble 成员，起报时间间隔 6 h，分布在 48 h 窗口内，以目标 lead 时间为中心；
    - 遵从 Brenowitz et al. (2024) 方法。
- 创新点：
  - 将以上评估体系系统用于 SwinV2 DL–NWP 模型，并量化 multistep fine-tuning 对 sharpness 和 spread 的影响。

3. 上下文与耦合度
- 上下文组件：
  - 仅在评估阶段使用，对模型训练不产生直接反馈（除非未来扩展为训练目标）。
- 绑定程度：
  - 强绑定-气象特异：
    - spread–skill、CRPS、能量谱等评估均源自 NWP/概率预报传统。

4. 实现描述
- 文中给出构造规则与指标解释，具体实现由 Earth2Mip 提供，不在论文中展开。
