# DB-RNN: An RNN for Precipitation Nowcasting Deblurring — 模型与组件知识提取

> 说明：本文提出 DB-RNN，一种在递归 ConvRNN 预报框架上叠加多尺度去模糊网络和额外损失（GDL + 对抗损失）的端到端结构，用于缓解降水 Nowcasting 中因均值损失和自回归带来的模糊累积。以下按“整体模型维度 + 组件维度”整理。

---

## 维度一：整体模型知识（Model-Level）

### 1. 任务与问题设定

- 任务类型：
  - 雷达回波序列上的降水 Nowcasting（雷达视频预测），典型设置：
    - 输入历史序列 $X = \{X_0, \dots, X_{m-1}\}$；
    - 输出未来序列 $Y = \{X_m, \dots, X_{m+n-1}\}$；
    - 单帧 $X_t \in \mathbb{R}^{c \times h \times w}$。
  - 形式化目标：

    $$
    \hat{Y}^* = \arg\max_Y P(Y \mid X) \tag{1}
    $$

- 业务场景：
  - 短时高分辨率降水预报（0–1 小时及更长多步），应用于交通、农业、航空等。

- 现有痛点（以 MS-RNN / ConvRNN 为代表）：
  - 模糊来源一：均值损失（$L_1$ / $L_2$ / 其组合）在多模态未来下逼近“平均态”，预测图像视觉上模糊；
  - 模糊来源二：自回归结构（RNN）带来的误差逐步累积，长时多步预测误差和模糊呈指数放大；
  - 虽然 MS-RNN 等多尺度 ConvRNN 提升了时空建模能力，但“预测逐帧变糊”的问题仍然突出。

### 2. 核心思想与整体结构

- 总体思路：
  - 保留 MS-RNN 这一强表达的多尺度 ConvRNN 作为**基础模块**，在其上：
    - **串联两个 MS-RNN**：前一个作为 Forecasting Module（FM），后一个作为 Deblurring Module（DM）；
    - 在 FM 输出和 DM 之间引入多尺度跳连，融合同尺度特征；
    - 为两部分设计带权重的多项损失（$L_{for}, L_{deb}, L_{gdl}, L_{adv}$），并采用**动态损失权重**实现“先预测后去模糊”的渐进训练策略；
    - 从视频预测和视频去模糊两个领域同时借力，利用 autoregression 的“误差累积”特性，叠加“帧内去模糊”的累积，形成“去模糊累积”。

- one-step 递推结构：
  - 记 $\operatorname{FM}$ 与 $\operatorname{DM}$ 分别为预测与去模糊子网络，则：

    $$
    \hat{X}_{t+1} = \operatorname{FM}(X_t / \hat{\hat{X}}_t)
    $$

    $$
    \hat{\hat{X}}_{t+1} = \operatorname{DM}(\hat{X}_{t+1}, (\operatorname{FM}^{out})_H, (\operatorname{FM}^{out})_M) \tag{2}
    $$

  - 其中：
    - $X_t$：真实雷达帧（历史阶段）或上一时刻去模糊帧（预测阶段）；
    - $\hat{X}_{t+1}$：FM 输出的“模糊预测帧”；
    - $\hat{\hat{X}}_{t+1}$：DM 输出的“去模糊预测帧”；
    - $(\cdot)^{out}$：子网络所有层输出张量集合；
    - $(\cdot)_H, (\cdot)_M$：多层隐藏态与记忆态张量（MS-RNN 特有：$H^l_t, M^l_t$）。

- 内部多尺度结构（以 FM 为例）：
  - 预测编码器 FE：3 层 RNN block，间隔 2×2 max-pooling（$P_{2\times2}$），构造多尺度特征：

    $$
    FE_0^{out} = FE_0(X_t / \hat{\hat{X}}_t)
    $$

    $$
    FE_1^{out} = FE_1((FE_0^{out})^{\downarrow})
    $$

    $$
    FE_2^{out} = FE_2((FE_1^{out})^{\downarrow}) \tag{3}
    $$

  - 预测解码器 FD：3 层 RNN block，使用 2×2 双线性上采样（$B_{2\times2}$）+ 与 FE 对应层的 skip 连接：

    $$
    FD_2^{out} = FD_2(FE_2^{out})
    $$

    $$
    FD_1^{out} = FD_1(((FD_2^{out})_H)^{\uparrow} + (FE_1^{out})_H, ((FD_2^{out})_M)^{\uparrow})
    $$

    $$
    FD_0^{out} = FD_0(((FD_1^{out})_H)^{\uparrow} + (FE_0^{out})_H, ((FD_1^{out})_M)^{\uparrow})
    $$

    $$
    \hat{X}_{t+1} = (FD_0^{out})_H \tag{4}
    $$

- DM 结构：
  - 去模糊编码器 DE：3 层 RNN block，输入包含 $\hat{X}_{t+1}$ 以及 FM 解码端对应尺度的特征（$FD_i^{out}$），下采样同 FE；
  - 去模糊解码器 DD：3 层 RNN block，上采样同时接收 DE 输出与 FE 输出的 skip 特征，最后输出 $\hat{\hat{X}}_{t+1}$：

    $$
    \hat{\hat{X}}_{t+1} = (DD_0^{out})_{\tilde{H}} \tag{6}
    $$

  - 通过 FE/FD 与 DE/DD 之间的大量同尺度 skip 连接，让预测特征与去模糊特征彼此交互，从而实现逐帧去模糊与误差抑制。

### 3. 损失函数与训练策略

- 基础像素损失（预测与去模糊各自一份）：

  - 预测子网损失 $L_{for}$：

    $$
    L_{for} = \frac{1}{m+n-1}\sum_{t=1}^{m+n-1}\Big(\lVert X_t - \hat{X}_t \rVert_1 + \lVert X_t - \hat{X}_t \rVert_2^2\Big)
    $$

  - 去模糊子网损失 $L_{deb}$：

    $$
    L_{deb} = \frac{1}{m+n-1}\sum_{t=1}^{m+n-1}\Big(\lVert X_t - \hat{\hat{X}}_t \rVert_1 + \lVert X_t - \hat{\hat{X}}_t \rVert_2^2\Big)
    $$

- 梯度差损失 GDL（仅作用在去模糊输出上，用于强调边缘清晰度）：

  - 对每帧 $(i,j)$ 位置的水平/垂直梯度差异进行惩罚：

    （论文给出完整三重求和形式，这里概念上可理解为真实帧与预测帧在 $(i+1,j)$ 与 $(i,j+1)$ 方向上的梯度差的绝对差之和。）

- 对抗损失 $L_{adv}$：

  - 判别器 $D$：小型 CNN 编码 + 线性解码器（3 个 Conv2d+LeakyReLU+MaxPool2d 编码块 + 2 层全连接，表中列出若干 in/out 通道与 kernel/stride/padding）；
  - 判别器损失：

    $$
    L_D = BCE(D(X_t), Real) + BCE(D(\hat{\hat{X}}_t), Fake) \tag{8}
    $$

  - 生成器端对抗损失：

    $$
    L_{adv} = \frac{1}{m+n-1}\sum_{t=1}^{m+n-1} BCE(D(\hat{\hat{X}}_t), Real)
    $$

- 总损失：

  $$
  L_{all} = \lambda_1 L_{for} + \lambda_2 L_{deb} + \lambda_3 L_{gdl} + \lambda_4 L_{adv} \tag{7}
  $$

  - 推荐权重调度：
    - 前 20 个 epoch：
      - $\lambda_1$ 从 1 线性减到 0.01；
      - $\lambda_2$ 从 0.01 线性增到 1；
    - 之后保持：$\lambda_1 = 0.01, \lambda_2 = 1$；
    - $\lambda_3 = 0.001, \lambda_4 = 1$ 在整个训练中保持不变。
  - 含义：训练前期主要优化预测网络，稍微激活去模糊；中期两者共同训练；后期主要优化去模糊网络，同时保留预测网络少量更新避免退化。

### 4. 数据、实现与评估

- 数据集：
  - HKO-7：
    - 2009–2015 年单雷达数据，覆盖 512×512 km²；6 min 分辨率，1.07 km 网格；
    - 只用有雨天：812 天训练、50 天验证、131 天测试；
    - 输入输出：5 历史帧 → 5 未来帧，空间下采样到 160×160。
  - DWD-12：
    - 2006–2017 年 17 部雷达，覆盖约 900×900 km²，5 min 分辨率、1 km 网格；
    - 2006–2013 训练，2014–2015 验证，2016–2017 测试；

- 像素–物理量转换：
  - 像素值 $P \in [0,255]$ 与反射率 dBZ（0–60）：

    $$
    P = \left\lfloor 255 \times \frac{dBZ}{60} \right\rfloor \tag{9}
    $$

  - dBZ 与降雨强度 $R$（mm/h）：

    $$
    dBZ = 10\,\lg(58.53\,R^{1.56}) \quad\text{(HKO-7)}
    $$

    $$
    dBZ = 10\,\lg(256\,R^{1.42}) \quad\text{(DWD-12)} \tag{10}
    $$

- 训练与实现：
  - 基础 RNN 模块：ConvLSTM、TrajGRU、PredRNN/PredRNN++、MIM、MotionRNN、PrecipLSTM 等，统一通过 MS-RNN 多尺度化；
  - 超参数（统一设置）：
    - RNN kernel 3×3，隐藏通道=24；MS-RNN 堆叠 6 层；
    - 优化器 Adam，初始 lr=3e-4，batch size=4；
    - HKO-7 上：MS-RNN 25 epoch，DB-RNN 50 epoch；
    - 训练硬件：NVIDIA A100；
    - 判别器 LeakyReLU 负斜率 0.01。

- 评估指标：
  - 分类型：CSI、HSS、POD，多个雨强阈值（HKO-7: 0.5/2/5/10/30 mm/h；DWD-12: 0.5/2/5 mm/h）；
  - 回归型：B-MSE、B-MAE（平衡版本，以提高对大雨的敏感度）。

- 实验结论（高层概括）：
  - 所有基础 RNN 上，DB-RNN 在 HKO-7、DWD-12 的几乎所有指标（尤其是高阈值 CSI/HSS/B-MAE）均显著优于对应 MS-RNN；
  - 去模糊收益在高雨强阈值和长 lead time 上更明显，体现“逐帧去模糊累积”效应；
  - 相比 SimVP、Earthformer、LPT-QPN、MS-LSTM 等 SOTA 模型，DB-PrecipLSTM 以较少参数达到或超过它们的性能。

### 5. 复杂度、可扩展性与适用性

- 复杂度分析（HKO-7 上示例）：
  - DB-RNN 使用两个 MS-RNN 串联 + GAN 判别器，带来：
    - 参数量约 ×2；
    - FLOPs ×2；
    - 显存约 ×2；
    - 训练时间约 ×4（对抗训练带来的额外成本明显）。
  - 但在相似参数规模下，DB-RNN 的预测性能优于更“大”的 MS-RNN 变体（例如 DB-ConvLSTM > MS-PredRNN，DB-PredRNN > MS-MIM）。

- 可扩展性：
  - 只要基础 ConvRNN 能嵌入 MS-RNN，多数变体（SA-ConvLSTM、MoDeRNN、CMS-LSTM、PredRNN-V2、MK-LSTM 等）都可被“DB-化”。

- 适用拓展场景：
  - 其他气象要素（温度、风等）的短时预报模糊问题；
  - 其他时空视频预测任务（交通流、人/机器人行为等）；
  - 视频去模糊任务（MC-Blur、RWBI 等数据集），尤其是使用 ConvRNN 类结构的模型。

---

## 维度二：基础组件知识（Component-Level）

> 说明：以下组件均基于论文文字与公式抽象而来，伪代码为**高保真伪代码**（并非论文源码），旨在捕捉算法逻辑与训练范式。

### 组件 1：多尺度 ConvRNN 骨干（MS-RNN 基模块）

1. 子需求与目标：
   - 在有限参数和显存下，提高 ConvRNN 对多尺度时空结构的建模能力；
   - 兼容各类 ConvRNN 单元（ConvLSTM、TrajGRU、PredRNN、MIM、MotionRNN、PrecipLSTM 等），作为一层“多尺度封装”。

2. 主要设计要点：
   - U-Net 风格 encoder–decoder 结构嵌入到 RNN 帧内：
     - Encoder：多层 RNN 单元 + 空间下采样（max pooling）；
     - Decoder：多层 RNN 单元 + 空间上采样（双线性插值）+ encoder skip；
   - 时间上：每一层 RNN 单元维护隐藏态 $H_t^l$ 和记忆态 $M_t^l$，在时间上递推；
   - 空间上：多尺度 $l=0,1,2,...$ 间存在“曲线”连接用于多尺度特征流动（详见 MS-PrecipLSTM 图示）。

3. 上下游关系：
   - 在 DB-RNN 中作为 FM 与 DM 的基础子网；
   - 可直接替换内部 RNN Cell 以得到不同 DB-ConvLSTM / DB-TrajGRU / DB-PredRNN 等。

4. 帧级前向伪代码（简化）：

```pseudo
# 基于文中描述生成的高层伪代码
function MS_RNN_step(X_t, H_prev_list, M_prev_list):
    # H_prev_list, M_prev_list: {l -> (H_t^l, M_t^l)} from previous time step

    # Encoder
    H0, M0 = RNN_cell_0(X_t, H_prev_list[0], M_prev_list[0])
    X1 = max_pool(H0)

    H1, M1 = RNN_cell_1(X1, H_prev_list[1], M_prev_list[1])
    X2 = max_pool(H1)

    H2, M2 = RNN_cell_2(X2, H_prev_list[2], M_prev_list[2])

    # Decoder with skip connections from encoder
    Hd2, Md2 = RNN_cell_3(H2, H_prev_list[3], M_prev_list[3])

    Hd1_up = bilinear_upsample(Hd2)
    Hd1_in = Hd1_up + H1
    Hd1, Md1 = RNN_cell_4(Hd1_in, H_prev_list[4], M_prev_list[4])

    Hd0_up = bilinear_upsample(Hd1)
    Hd0_in = Hd0_up + H0
    Hd0, Md0 = RNN_cell_5(Hd0_in, H_prev_list[5], M_prev_list[5])

    Y_t = Hd0   # output frame or feature

    H_list = [H0, H1, H2, Hd2, Hd1, Hd0]
    M_list = [M0, M1, M2, Md2, Md1, Md0]

    return Y_t, H_list, M_list
```

---

### 组件 2：DB-RNN 双网络结构（预测模块 FM + 去模糊模块 DM）

1. 子需求：
   - 在不改变基础 ConvRNN 动态的前提下，通过额外的去模糊网络减弱自回归误差累积和均值模糊；
   - 允许逐帧 deblur，并利用 autoregression 累积去模糊效果。

2. 结构与接口：
   - FM：一个 MS-RNN，用于生成模糊但物理合理的预测帧；
   - DM：另一个 MS-RNN，用于在每一步输入 FM 的输出和其中间特征，输出去模糊结果；
   - 长时预测时，DM 输出 $\hat{\hat{X}}_t$ 同时作为下一步 FM 的输入，实现“预测→去模糊→预测→去模糊”的串联。

3. 时序前向伪代码：

```pseudo
# 基于文中描述生成的高层伪代码
# 输入：X_seq[0..m-1] 历史真实雷达帧
# 输出：pred_seq_hat[t], pred_seq_deblur[t]

function DB_RNN_forward(X_seq, m, n):
    # 初始化 FM, DM 的隐状态列表
    H_FM, M_FM = init_states_for_all_layers()
    H_DM, M_DM = init_states_for_all_layers()

    # t = 0..m-1: 仅用真实帧驱动 FM（warmup）
    for t in range(0, m):
        X_in = X_seq[t]    # 真实帧
        Y_hat, H_FM, M_FM = MS_RNN_step_FM(X_in, H_FM, M_FM)
        # 可选：warmup 阶段 DM 不参与，或仅用于初始化

    # t = m..m+n-1: 自回归预测 + 去模糊
    X_prev_deblur = X_seq[m-1]
    for t in range(m, m + n):
        # 1) 预测网络：使用上一时刻的去模糊帧或真实帧
        X_in = X_prev_deblur
        Y_hat, H_FM, M_FM, FM_features = MS_RNN_step_FM_with_features(X_in, H_FM, M_FM)

        # 2) 去模糊网络：输入预测帧 + FM 多尺度特征
        Y_deblur, H_DM, M_DM = MS_RNN_step_DM(Y_hat, FM_features, H_DM, M_DM)

        store(pred_seq_hat[t], Y_hat)
        store(pred_seq_deblur[t], Y_deblur)

        # 3) 将去模糊帧作为下一步的驱动
        X_prev_deblur = Y_deblur

    return pred_seq_hat, pred_seq_deblur
```

---

### 组件 3：多项损失与动态加权策略

1. 子需求：
   - 同时训练两张网络（FM 与 DM），既要保留预测能力，又要增强去模糊效果；
   - 避免一开始就用强对抗和 GDL 对尚未收敛的网络施加过大约束。

2. 损失组成：
   - $L_{for}$：针对 FM 输出的 $\hat{X}_t$ 的像素级 L1+L2；
   - $L_{deb}$：针对 DM 输出的 $\hat{\hat{X}}_t$ 的像素级 L1+L2；
   - $L_{gdl}$：仅对 $\hat{\hat{X}}_t$ 使用的梯度差分损失（强调边缘/结构）；
   - $L_{adv}$：GAN 框架下生成器端对抗损失。

3. 动态权重：
   - 训练早期：$\lambda_1 \gg \lambda_2$，主训预测子网，使输出具备物理合理性；
   - 中期：$\lambda_1 \approx \lambda_2$，让两子网协同；
   - 后期：$\lambda_2 \gg \lambda_1$，加强去模糊效果并防止 FM 完全退化；
   - ablation 显示：动态权重略优于始终使用 $\lambda_1=\lambda_2=1$ 的静态策略。

4. 训练主循环伪代码：

```pseudo
# 基于文中描述生成的高层伪代码
for epoch in range(num_epochs):
    λ1 = schedule_lambda1(epoch)  # 1 -> 0.01 in first 20 epochs, then 0.01
    λ2 = schedule_lambda2(epoch)  # 0.01 -> 1 in first 20 epochs, then 1
    λ3 = 0.001
    λ4 = 1.0

    for batch in train_loader:
        X_hist, X_future = batch.hist_seq, batch.future_seq

        # 1) 前向：DB-RNN
        pred_hat, pred_deblur = DB_RNN_forward(X_hist, m=len(X_hist), n=len(X_future))

        # 2) 构造逐帧 GT 序列（对齐 pred_hat/pred_deblur）
        X_all = concat(X_hist[1:], X_future)  # 对应 t=1..m+n-1

        # 3) 计算子损失
        L_for = pixel_L1L2(pred_hat,  X_all)
        L_deb = pixel_L1L2(pred_deblur, X_all)
        L_gdl = gradient_difference_loss(pred_deblur, X_all)

        # 4) 生成器对抗损失（鼓励 deblur 输出更真实）
        L_adv = 0
        for t in range(len(pred_deblur)):
            L_adv += BCE(discriminator(pred_deblur[t]), label_real)
        L_adv /= len(pred_deblur)

        # 5) 总生成器损失
        L_all = λ1 * L_for + λ2 * L_deb + λ3 * L_gdl + λ4 * L_adv

        optimizer_G.zero_grad()
        L_all.backward()
        optimizer_G.step()

        # 6) 更新判别器
        L_D = 0
        for t in range(len(pred_deblur)):
            L_D += BCE(discriminator(X_all[t]),        label_real)
            L_D += BCE(discriminator(pred_deblur[t]), label_fake)
        L_D /= len(pred_deblur) * 2

        optimizer_D.zero_grad()
        L_D.backward()
        optimizer_D.step()
```

---

### 组件 4：判别器结构（用于对抗损失）

1. 子需求：
   - 对每个 deblur 帧进行“真/假”判别，为生成器提供方向，改善细节与清晰度；
   - 保持参数量小，避免过多额外复杂度。

2. 架构概要（按论文表 I & II）：
   - Encoder：
     - [Conv2d → LeakyReLU → MaxPool2d] × 3
   - Decoder：
     - [Linear → LeakyReLU → Linear → Sigmoid]
   - 不同 DB-RNN 变体（DB-ConvLSTM / DB-TrajGRU / DB-PredRNN / DB-MIM / DB-MotionRNN / DB-PrecipLSTM 等）共享架构，通过 in_c/out_c 设置略调参数量（约 0.012M–0.027M）。

3. 伪代码（单帧判别）：

```pseudo
# 基于文中描述生成的高层伪代码
function Discriminator(X_frame):
    h = Conv2d_1(X_frame); h = LeakyReLU(h); h = MaxPool2d_1(h)
    h = Conv2d_2(h);       h = LeakyReLU(h); h = MaxPool2d_2(h)
    h = Conv2d_3(h);       h = LeakyReLU(h); h = MaxPool2d_3(h)

    h = flatten(h)
    h = Linear_1(h); h = LeakyReLU(h)
    h = Linear_2(h); out = Sigmoid(h)  # probability of real

    return out
```

---

### 组件 5：长时自回归预测与性能分析

1. 子需求：
   - 评估 DB-RNN 相对于 MS-RNN 在 1–7 小时 lead time 下的稳定性与极端降水技能；
   - 使用相同训练好的权重，通过增加自回归步数实现长预测。

2. 长时递推策略：
   - 复用训练好的 MS-PrecipLSTM 与 DB-PrecipLSTM；
   - 不改变网络，只增加递推步数（1–7 h），以 HSS-10 与 POD-10 为指标评估性能衰减曲线；
   - 结果：两者初期性能急剧下降后趋于平缓，但 DB-RNN 衰减更慢，尤其在后期 lead time 上保持更高技能，进一步证明“去模糊累积”效应。

3. 伪代码（示意）：

```pseudo
# 基于文中描述生成的高层伪代码
function long_horizon_forecast(DB_RNN_model, X_hist, max_lead_hours):
    forecasts = {}
    for lead in range(1, max_lead_hours + 1):
        # 每种 lead 长度都从相同历史序列重新 roll out
        pred_hat, pred_deblur = DB_RNN_forward(X_hist, m=len(X_hist), n=lead_frames(lead))
        forecasts[lead] = pred_deblur  # 使用去模糊输出评估
    return forecasts
```

---

以上整理了 DB-RNN 在“整体结构 + 双网络 + 多项损失 + 对抗判别 + 长时自回归”上的关键设计与伪代码，可直接作为你后续对其他 ConvRNN 预测模型或视频去模糊模型进行“DB-化”的参考蓝本。若需要，我可以再帮你基于不同基础 RNN（ConvLSTM/PredRNN/MIM/PrecipLSTM 等）画一张“DB-化改造对照表”（输入/输出接口、参数量变化、性能增益）。