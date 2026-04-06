# 全球气象AI建模通用范式终极大纲
## 五流派跨范式深度融合版
### Global Weather AI Universal Paradigm — Ultimate Cross-Paradigm Integration

> **文档定位**：本文档是气象AI领域五大海峡流派（Transformer / Operator / Graph / Generative / Fourier）的跨流派终极融合指南。通过系统梳理21篇代表性论文，提炼全行业通用的建模范式框架。
>
> **文献覆盖**：
> - **Transformer流派**：FengWu, Swin V2, EWMoE, Stormer, AERIS, Local Off-Grid, ClimaX, HEAL-Swin
> - **Operator/Fourier流派**：FourCastNet (AFNO), SFNO (Spherical Fourier Neural Operator)
> - **Graph流派**：GraphCast, GenCast, GNADET, AGODE, AIFS, ClimODE, NowcastNet, TelePiT
> - **Generative扩散流派**：FuXi-TC, PreDiff, GenEPS, POSTCAST, CorrDiff, PHYS-Diff
>
> **文档完成时间**：2026-03-27
> **版本**：v4.0（五流派终极大融合版）

---

## 序言：气象AI宏观技术版图

### 五流派全景扫描

气象AI自2018年至今，已发展出五条核心技术路径，代表着不同的架构哲学与物理建模思想：

#### 1. Transformer流派（空间-语义建模派）

**代表模型**：FengWu, Swin V2, EWMoE, Stormer, AERIS, Pangu-Weather, ClimaX

**核心架构**：基于自注意力机制（Swin窗口注意力 / 全局注意力 / MoE稀疏专家），在2D/3D图像空间直接建模。

**典型范式**：
- **Patch Embedding** → **窗口/全局注意力** → **Token解码**
- 像素级（1×1，AERIS）到16×16 patch不等
- 多步微调（4步/8步）缓解自回归误差累积
- 概率输出（FengWu）或扩散生成（AERIS）

**核心复杂度**：$O(M^2 \cdot N)$（窗口注意力）至 $O(N^2)$（全局注意力）

**代表性能**：0.25°分辨率，10-90天预报，Z500 ACC ~0.98（3天）

---

#### 2. Fourier/Operator流派（频谱-全局建模派）

**代表模型**：FourCastNet (AFNO), SFNO (Spherical Fourier Neural Operator)

**核心架构**：通过Fourier变换或球谐变换将气象场映射到频域，在频域执行全局算子学习，再逆变换回空间域。

**典型范式**：
- **空间域 → FFT / SHT → 频域MLP → 软阈值化 → 逆变换 → 空间域**
- 2D FFT（Lat-Lon网格）或球谐变换（均匀球面）
- 软阈值化引入频域稀疏性，天然抑制噪声
- Patch embedding：固定8×8，token数降低约60倍

**核心复杂度**：$O(N \log N)$（FFT），显存占用仅10GB（同深度ResNet需83GB）

**代表性能**：0.25°分辨率，7-10天预报，极高推理速度（7秒/100集合成员）

---

#### 3. Graph流派（图结构-物理关联派）

**代表模型**：GraphCast, GenCast, GNADET, AGODE, AIFS, ClimODE, NowcastNet

**核心架构**：将气象场表示为图结构，节点对应空间位置（网格顶点/icosahedral节点/站点），边编码物理邻近或动力学关联，通过消息传递神经网络（MPNN）在节点间传播信息。

**典型范式**：
- **Encoder（Grid→Mesh） → Multi-Mesh Processor（多尺度消息传递） → Decoder（Mesh→Grid）**
- 正二十面体（icosahedral）网格实现均匀球面覆盖
- 多尺度边（Level 0-6）实现从局地（~数百km）到全球（~数千km）的信息传播
- 图拉普拉斯编码物理演化方程（GNADET）

**核心复杂度**：$O(E)$（稀疏图），多跳消息传递实现 $O(\log N)$ 层全球通信

**代表性能**：0.25°分辨率，10天预报，GraphCast推理仅需1分钟

---

#### 4. Generative扩散流派（概率-不确定性派）

**代表模型**：AERIS, FuXi-TC, PreDiff, GenEPS, POSTCAST, CorrDiff, PHYS-Diff

**核心架构**：利用扩散概率模型（Denoising Diffusion / Score-based）生成气象场预测，天然产生概率分布，支持集合预报和极端事件建模。

**典型范式**：
- **前向加噪 → 条件去噪（UNet/Transformer/Swin） → ODE求解**
- 条件化：历史轨迹、驱动场、大尺度环境
- PF-ODE积分（10步，DPM-Solver++）生成样本
- 或利用扩散进行跨分辨率映射（CorrDiff）

**核心复杂度**：推理需10+次前向（ODE求解），但长期稳定性好（90天稳定预报）

**代表性能**：AERIS：90天S2S预报，spread-skill比接近IFS ENS；CorrDiff：km级区域降尺度

---

#### 5. ODE/连续建模派（物理-连续时间派）

**代表模型**：ClimODE, AGODE, NowcastNet

**核心架构**：将大气演化建模为连续时间偏微分方程（PDE），通过Neural ODE实现任意时刻的外推预报，物理一致性极强。

**典型范式**：
- **状态演化：$d\mathbf{h}(t)/dt = f_\theta(\mathbf{h}(t), t)$**
- **物质输运：$du_k/dt = -v_k \cdot \nabla u_k - u_k \nabla \cdot v_k$**
- RK4数值积分，支持任意时间查询
- 二阶ODE（学习流速的导数）实现守恒约束

**代表性能**：ClimODE连续时间气候建模；NowcastNet降水临近预报（10min步长）

---

### 五流派宏观对比总表

| 对比维度 | Transformer | Fourier/Operator | Graph | Generative | ODE/连续 |
|---------|------------|-----------------|-------|-----------|---------|
| **核心范式** | 窗口/全局注意力 | FFT/SHT频域算子 | 图消息传递 | 扩散去噪 | Neural ODE |
| **空间处理** | 2D/3D图像空间 | 频域全局 | 图结构 | 2D/3D图像空间 | 连续场 |
| **复杂度** | $O(M^2N)$~$O(N^2)$ | $O(N\log N)$ | $O(E)$ | 推理10+步 | ODE求解 |
| **全局感受野** | 多层堆叠 | 单层天然 | 多跳传播 | 多层堆叠 | 积分外推 |
| **极区处理** | 窗口mask修改 | 周期性填充 | 无极区奇异 | 移除极点 | 网格相关 |
| **位置编码** | 2D/3D绝对或RoPE | Fourier基 | 图结构隐式 | 正弦/绝对 | 连续坐标 |
| **不确定性** | NLL输出/扩散 | 初始场扰动 | 集合/贝叶斯 | 天然概率分布 | 贝叶斯ODE |
| **训练成本** | 高 | 中等 | 高 | 很高 | 中等 |
| **推理速度** | 中等 | 极高（~7秒/100成员） | 快 | 慢（10+步） | 快 |
| **适用时效** | 1-90天 | 1-10天 | 1-10天 | 1-90天 | 任意时刻 |
| **物理一致性** | 弱（数据驱动） | 中（频域正则化） | 强（图拉普拉斯） | 中（条件化） | 强（连续PDE） |
| **极代表模型** | AERIS (1.3B-80B参数) | FourCastNet (10GB显存) | GraphCast (36.7M) | AERIS (90天稳定) | ClimODE (连续时间) |
| **核心优势** | 全局建模、架构灵活 | 计算高效、全局感受野 | 球面均匀、极区友好 | 不确定性量化 | 物理一致、连续外推 |
| **核心劣势** | 计算成本高、极区冗余 | 平移等变假设过强 | 图构造复杂 | 推理成本高 | 训练复杂、数值稳定性 |

**五大流派的核心分歧与共识**：

- **分歧1**：空间表示的选择——像素空间（Transformer/Generative）、频域空间（Fourier/Operator）、图空间（Graph）、连续空间（ODE）
- **分歧2**：全局信息聚合方式——注意力矩阵（Transformer）、FFT全局（Operator）、消息传递（Graph）、ODE积分（ODE）
- **共识1**：均使用ERA5再分析数据作为训练目标
- **共识2**：均采用6小时作为标准时间步长
- **共识3**：均需处理经纬度网格的几何约束
- **共识4**：均使用纬度加权损失：$w(\phi) = \cos(\phi)$

---

## 【第一部分：物理先验注入范式】

### 1.1 地球几何

#### 【原理描述】

**核心定义**：地球几何先验是指在神经网络架构中显式编码地球的球面几何特性——包括经纬度坐标系统、球面周期性边界条件、三维大气垂直结构以及不同网格类型的拓扑差异——确保模型理解并尊重地球物理空间的拓扑约束与几何特性。

**数学推导**：

##### 1. 球面坐标系统

地球表面任一点可表示为经纬度 $(\lambda, \phi)$，其中 $\lambda \in [-180°, 180°]$（或 $[0°, 360°)$），$\phi \in [-90°, 90°]$。垂直维度采用气压坐标 $p \in \{p_1, p_2, ..., p_L\}$（如1000 hPa至50 hPa）。

**大圆距离**（Haversine公式）：
$$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2 - \phi_1}{2}\right) + \cos\phi_1\cos\phi_2\sin^2\left(\frac{\lambda_2 - \lambda_1}{2}\right)}\right)$$
其中 $R \approx 6371$ km 为地球半径。

**网格面积随纬度变化**：
$$A(\phi) = R^2 \Delta\lambda \Delta\phi \cos\phi$$
赤道附近网格面积约为极区的2倍，导致规则经纬网格在极区过采样、在赤道欠采样的问题。

##### 2. 经度周期性

$$\lambda = 0° \equiv \lambda = 360°$$

物理含义：经度维度在空间上形成闭环，模型需支持跨越本初子午线的连续性。

##### 3. 正二十面体网格（Graph流派）

GraphCast/GenCast将经纬网格投影到六次细化的正二十面体网格：

$$N = 12 + 30 \times \frac{4^r - 1}{3}$$
- Level 6: 40,962 节点
- 优势：球面近似均匀覆盖，无极区奇异性
- 劣势：需要专门的Grid↔Mesh插值层

##### 4. 球谐函数表示（Fourier/Operator流派-SFNO）

$$f(\theta, \phi) = \sum_{l=0}^{\infty} \sum_{m=-l}^{l} \hat{f}_l^m Y_l^m(\theta, \phi)$$
其中 $Y_l^m$ 为球谐基函数，$\theta$ 为余纬度。

优势：天然具有旋转对称性，逆变换自动满足球面约束。

##### 5. 3D位置嵌入（Transformer流派-EWMoE）

$$\mathbf{e}_{\text{pos}} = [\mathbf{e}_{\text{lon}}, \mathbf{e}_{\text{lat}}, \mathbf{e}_{\text{alt}}] \in \mathbb{R}^{D}$$
$$\mathbf{e}_{\text{lon}, k} = \sin\left(\frac{k \lambda}{T}\right), \quad \mathbf{e}_{\text{lat}, k} = \cos\left(\frac{k \phi}{T}\right)$$

其中 $\mathbf{e}_{\text{lon}} \in \mathbb{R}^{D/3}$ 为经度位置嵌入，$\mathbf{e}_{\text{lat}} \in \mathbb{R}^{D/3}$ 为纬度位置嵌入，$\mathbf{e}_{\text{alt}} \in \mathbb{R}^{D/3}$ 为高度（气压层）位置嵌入。

##### 6. 相对坐标归一化（Generative流派-PHYS-Diff）

$$\mathbf{x}_{\text{rel}, i} = \frac{\mathbf{x}_i - \mathbf{x}_{\text{ref}}}{\sigma_{\text{coord}}}$$

用于热带气旋轨迹预测，使模型对绝对位置不敏感，提升泛化能力。

**理论依据**：
- 大气动力学方程（如地转平衡、静力平衡）在球面坐标系下推导
- 科氏力效应随纬度变化：$f = 2\Omega \sin\phi$
- 卷积操作在经纬网格上无法保持平移不变性
- 忽略球面几何会导致：极区网格过度聚集导致的数值不稳定、纬向/经向动力学的不对称处理、质量/能量守恒的破坏

#### 【数据规格层】

**张量形状与物理含义（五流派对比）**：

| 流派 | 模型 | 分辨率 | 张量形状 | 网格类型 | 通道数 |
|------|------|--------|---------|---------|-------|
| **Transformer** | FengWu | 0.25° | `[B, 189, 721, 1440]` | 规则经纬网格 | 189 |
| **Transformer** | Swin V2 | 0.25° | `[B, 73, 720, 1440]` | 规则经纬网格 | 73 |
| **Transformer** | AERIS | 0.25° | `[B, 70, 720, 1440]` | 规则经纬网格 | 70 |
| **Fourier** | FourCastNet | 0.25° | `[B, 20, 721, 1440]` | 规则经纬网格 | 20 |
| **Operator** | SFNO | 0.25° | `[B, V, P, L, M]` | 球谐系数 | - |
| **Graph** | GraphCast | 0.25° | `[B, N_mesh, D]` (Mesh) | Icosahedral | 227 |
| **Graph** | GenCast | 0.25° | `[B, V, N_mesh]` | Icosahedral | ~80 |
| **Graph** | AIFS | N320 | `[B, C, 542080]` | Reduced Gaussian | ~65 |
| **Generative** | CorrDiff | 台湾区域 | `[B, 12, 36, 36]` | Lambert投影 | 12 |
| **Generative** | PHYS-Diff | TC区域 | `[B, 69, 80, 80]` | 规则网格 | 69 |
| **ODE** | ClimODE | 5.625° | `[B, 5, H, W]` | 规则网格 | 5 |

**维度物理语义**：
- $H$：纬度格点数（721对应0.25°），从南极到北极
- $W$：经度格点数（1440对应0.25°），从本初子午线环绕一周
- $C$：变量通道数（多气压层 + 地表变量）
- $N$（Graph流派）：icosahedral节点数（40,962），近似均匀分布

#### 【架构层】

**计算流程图**（五流派通用范式）：

```
原始气象场数据（Lat-Lon / Icosahedral / Reduced Gaussian / Lambert投影）
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤1：网格/坐标系转换                                        │
│   Transformer: 直接使用规则Lat-Lon网格                        │
│   Fourier: Gaussian→规则网格插值（FourCastNet）              │
│   Operator: Lat-Lon→球谐系数（SFNO）                        │
│   Graph: Lat-Lon→icosahedral顶点（GraphCast/GenCast）        │
│   Generative: Lambert投影区域网格（CorrDiff）                 │
│   ODE: 规则网格或粒子表示（ClimODE）                         │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤2：经度周期性处理                                        │
│   Transformer: 窗口mask修改（Swin允许wrap-around）           │
│   Fourier: 周期性填充（periodic padding）+ FFT天然支持       │
│   Operator: 球谐变换天然满足周期性                           │
│   Graph: 图结构中无经度概念，边连接自动处理                   │
│   Generative: 区域模型无需处理全球周期                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤3：位置编码注入                                          │
│   Transformer: 2D/3D绝对位置嵌入或RoPE（AERIS/Pangu）       │
│   Fourier: 可学习2D位置编码（叠加到patch tokens）            │
│   Operator: Fourier基函数天然编码位置（SFNO）                │
│   Graph: xyz坐标初始化 + 多尺度边隐式编码（GraphCast）       │
│   Generative: 正弦位置编码（4通道，CorrDiff）                │
│   ODE: 连续坐标归一化（PHYS-Diff）                           │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤4：Patch/Token化                                        │
│   Transformer: patch_size=1/4/8/16                           │
│   Fourier: patch_size=8                                      │
│   Operator: patch_size=8或球谐阶数截断                       │
│   Graph: 节点化为token（每个节点=1个token）                   │
│   Generative: UNet编码器下采样                               │
│   ODE: 网格点或粒子系统                                      │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤5：主干网络处理（各流派核心差异）                         │
│   Transformer: 窗口/全局注意力                               │
│   Fourier: FFT + 共享MLP + 软阈值化                          │
│   Operator: 球谐域算子学习                                   │
│   Graph: 多尺度消息传递                                      │
│   Generative: 扩散去噪UNet/Transformer/Swin                  │
│   ODE: Neural ODE数值积分                                    │
└─────────────────────────────────────────────────────────────┘
  ↓
输出预测气象场
```

**各流派详细流程图**：

1. **GraphCast Graph流程**：
```
ERA5 Lat-Lon网格 [B, 227, 721, 1440]
  ↓
Encoder: 双线性插值到正二十面体网格顶点
  ↓ [V, 3C] (拼接 z, x_t, x_{t-1})
Grid2Mesh边构建（~1.6M边）
  ↓
Graph Transformer Processor (16层, k-hop邻域注意力)
  ↓ [V, D_hidden]
Multi-Mesh消息传递 (Level 0-6, 327,660条边)
  ↓
Decoder: 插值回经纬网格
  ↓ [C, H, W]
残差连接: x_{t+1} = x_t + Δz
```

2. **SFNO 球谐变换流程**：
```
Lat-Lon输入 [B, V, P, H, W]
  ↓
球谐变换 (Spherical Harmonic Transform)
  ↓ [B, V, P, L, M] (频域表示)
频域卷积/算子学习 (逐阶MLP)
  ↓ [B, V, P, L, M]
逆球谐变换
  ↓ [B, V, P, H, W]
输出预测场
```

3. **AERIS 扩散流程**（Generative流派）：
```
训练: x_0 → 采样z → x_t = cos(t)x_0 + sin(t)z
  → 条件输入 [x_t, x_{i-1}, forcings]
  → Swin Transformer
  → 预测速度场v_t

推理: 初始噪声 x_{π/2} ~ N(0, σ²)
  → PF-ODE积分 (DPM-Solver++ 10步)
  → 生成样本 x_0
  → 还原状态 x_i = x_0 + x_{i-1}
```

4. **ClimODE 连续演化流程**（ODE流派）：
```
初始状态 u(t₀), v(t₀)
  ↓
ODE系统:
  ├─ du_k/dt = -v_k·∇u_k - u_k∇·v_k
  └─ dv_k/dt = f_θ(u, ∇u, v, ψ)
  ↓
RK4数值积分
  └─ u(t₁), v(t₁) = ODESolve([u₀, v₀], [t₀, t₁])
  ↓
高斯发射: u_obs ~ N(u + μ, σ²)
```

#### 【设计层】

**设计动机**：
- **物理一致性**：确保模型预测遵守地球物理约束（如经度连续性、极区特殊性）
- **位置敏感性**：不同纬度的天气系统特征差异巨大（如热带辐合带 vs 中纬度西风带）
- **垂直结构**：大气变量随高度变化显著（如温度递减率、风速切变）
- **球面均匀性**：避免极区网格过度聚集导致的数值不稳定

**痛点解决（五流派对照）**：

| 问题 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **球面拓扑** | 窗口mask修改 | 周期性填充+FFT | 图结构天然处理 | 移除极点或投影 | 网格相关 |
| **极区奇异性** | mask/padding | 纬度加权评估 | 无（均匀网格） | Lambert投影 | 避免 |
| **垂直结构** | 3D位置嵌入（EWMoE） | 通道堆叠 | 节点特征编码 | 通道堆叠 | 连续高度 |
| **周期性边界** | 显式wrap-around | 周期性填充 | 无经度概念 | 区域模型无需 | 无 |
| **长程依赖** | 多层注意力堆叠 | FFT单层全局 | 多跳消息传递 | 多层堆叠 | ODE积分 |

**创新突破（五流派）**：

- **Transformer流派**：
  - Swin V2：最小修改实现球面适配（仅改窗口mask）
  - EWMoE：首次在气象Transformer中使用3D位置嵌入
  - AERIS：像素级（1×1 patch）+ 扩散模型，保留最高空间分辨率
  - Pangu-Weather：3D Earth-Specific Transformer，窗口大小 $2 \times 6 \times 12$

- **Fourier/Operator流派**：
  - FourCastNet：规则网格 + 周期性填充，AFNO显存占用仅10GB
  - SFNO：球谐变换天然满足旋转对称性，长时自回归预报稳定
  - FFT天然处理周期性边界条件，软阈值化抑制噪声

- **Graph流派**：
  - GraphCast/GenCast：首次在全球天气预报中系统性使用正二十面体网格 + 图Transformer，在10天预报上超越IFS
  - AIFS：Reduced Gaussian grid与GNN结合，兼顾均匀性和NWP系统兼容性
  - GNADET：学习的图拉普拉斯实现平流-扩散方程

- **Generative流派**：
  - PHYS-Diff：相对坐标归一化使模型对绝对位置不敏感，提升TC轨迹预测泛化能力
  - CorrDiff：Lambert投影避免全球球面问题，实现km级降尺度

- **ODE流派**：
  - ClimODE：连续时间建模，支持任意时刻查询
  - AGODE：自适应边权重，根据物理参数动态调整

#### 【对比层】

**五流派共性归纳**：
- 所有模型均显式考虑地球几何
- 均使用某种形式的位置编码（显式嵌入 / 隐式结构 / Fourier基）
- 所有在Lat-Lon网格上训练的模型均使用纬度加权：$\mathcal{L} = \sum_{i,j} w(\phi_i) \cdot \text{loss}(y_{ij}, \hat{y}_{ij})$
- 位置编码均包含球面坐标的正余弦变换

**五流派差异分析（大型对比表）**：

| 维度 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **网格类型** | 规则Lat-Lon | Lat-Lon/球谐 | Icosahedral/Reduced Gaussian | 区域投影/规则网格 | 规则网格/粒子 |
| **经度周期性** | 窗口mask修改（显式） | 周期性填充（隐式） | 无经度概念 | 区域无需处理 | 无 |
| **位置编码方式** | 2D/3D绝对或RoPE | 可学习2D/Fourier基 | xyz坐标隐式 | 正弦4通道 | 连续归一化 |
| **Patch大小** | 1×1至16×16（多样） | 固定8×8 | N/A（节点=token） | UNet下采样 | N/A |
| **极区处理** | 移除极点或mask | 纬度加权评估 | 天然无奇异性 | 投影到区域 | 避免 |
| **网格预处理** | 直接使用ERA5规则网格 | Gaussian→规则网格插值 | Lat-Lon→icosahedral插值 | Lambert投影 | 直接使用 |
| **坐标系统** | 笛卡尔2D/3D | 频域/球谐域 | 图拓扑+xyz坐标 | 笛卡尔投影 | 连续坐标 |
| **参数量** | 137M-80B（差异巨大） | 适中（~100M） | 36.7M（GraphCast） | 80M（CorrDiff） | 适中 |
| **推理速度** | 中等 | 极高 | 快 | 慢（10+步） | 快 |
| **理论复杂度** | $O(M^2N)$~$O(N^2)$ | $O(N\log N)$ | $O(E)$ | 推理10+步 | ODE求解 |

**权衡评估**：

**正二十面体网格（Graph流派）**：
- ✅ 优点：球面近似均匀覆盖，无极地奇异性，精度最高
- ❌ 缺点：需要Grid↔Mesh插值层，数据预处理复杂，实现需专门图操作库

**周期性填充（Fourier/Operator流派）**：
- ✅ 优点：实现简单，FFT天然支持，计算效率极高
- ❌ 缺点：对极点奇异性处理不如球谐函数自然，频域操作假设全局平移等变

**像素级Swin（Transformer-AERIS）**：
- ✅ 优点：保留最高分辨率，无信息损失
- ❌ 缺点：计算成本高，token数量达 $720 \times 1440 = 1,036,800$，需SWiPe等高级并行策略

**球谐表示（Operator-SFNO）**：
- ✅ 优点：数学优雅，天然具有旋转对称性，长时预报稳定
- ❌ 缺点：高分辨率（高球谐阶数）下计算成本急剧上升

**区域投影（Generative-CorrDiff）**：
- ✅ 优点：完全避免全球球面问题，适合固定区域降尺度
- ❌ 缺点：无法用于全球预报，需预先定义区域

**适用场景**：
- **Transformer流派**：适合需要显式建模垂直结构和多尺度特征的场景
- **Fourier/Operator流派**：适合需要极高计算效率和全局建模的场景
- **Graph流派**：适合追求极致精度和球面均匀性的全球中期预报
- **Generative流派**：适合需要概率预报和极端事件建模的场景
- **ODE流派**：适合需要物理一致性和连续时间外推的场景

---

### 1.2 球面均匀性

#### 【原理描述】

**核心定义**：球面均匀性是指在球面上不同位置的物理量应具有一致的统计权重和物理意义，模型在处理全球数据时对不同纬度、经度位置应具有平移不变性或等变性。需通过纬度加权、权重共享、均匀网格等机制校正经纬网格的非均匀性。

**数学推导**：

##### 1. 纬度加权函数（五流派通用）

$$L(\phi) = \frac{\cos(\phi)}{\frac{1}{H}\sum_{i=1}^{H} \cos(\phi_i)}$$

物理含义：补偿高纬度网格点面积更小的问题，$\cos(\phi)$ 与纬度圈周长成正比。

**球面积分**：
$$\iint_S f(\lambda, \phi) \, dS = \int_0^{2\pi} \int_{-\pi/2}^{\pi/2} f(\lambda, \phi) \cos\phi \, d\phi \, d\lambda$$

##### 2. 加权损失函数（各流派对比）

**Transformer流派**（通用形式）：
$$\mathcal{L} = \frac{1}{CHW} \sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W} w(c) \cdot L(\phi_i) \cdot |\hat{X}_{cij} - X_{cij}|^2$$
其中 $w(c)$ 为变量/层权重，$L(\phi_i)$ 为纬度权重。

**Graph流派**（GraphCast方案）：
$$w(p) \propto p / p_{\text{surface}}$$
气压加权：近地面层（高气压）权重更大，高空层（低气压）权重更小。

**Fourier/Operator流派**（FourCastNet方案）：
- 评估时应用纬度加权
- 频域权重共享体现平移等变性：
$$z_{m,n} = [\text{DFT}(\mathbf{X})]_{m,n}, \quad \tilde{z}_{m,n} = S_{\lambda}(\text{MLP}(z_{m,n}))$$
MLP权重在所有 $(m,n)$ 位置共享，体现全局平移等变性。

**Generative流派**（FengWu方案）：
$$\mathcal{L}_{\text{NLL}} = \frac{1}{2}\log(\sigma^2) + \frac{(X_{t+\Delta t} - \mu)^2}{2\sigma^2}$$
不确定性损失自动学习变量权重，无需手工调参。

##### 3. 格点间距随纬度变化（规则经纬网格）

$$\Delta x(\phi) = R \cos(\phi) \Delta\lambda, \quad \Delta y = R \Delta\phi$$

在赤道附近 $\Delta x \approx R\Delta\phi$，而在极区 $\Delta x \to 0$，纬向分辨率极不均匀。

##### 4. 面积加权损失（Graph流派）

$$\mathcal{L}_{\text{MSE}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} w(i) (\hat{\mathbf{x}}_i - \mathbf{x}_i)^2$$
其中 $w(i) \propto \cos(\text{lat}_i)$ 为面积权重。

**理论依据**：
- 物理定律在球面上是均匀的，模型应在所有纬度带具有相似的学习能力
- 标准MSE损失会导致极区（网格点密集）主导损失函数
- Transformer架构本身不具备位置感知能力，需要位置编码
- 均匀网格（icosahedral）保证：卷积核/图消息传递的感受野在全球范围内一致

#### 【数据规格层】

**权重张量形状（五流派对比）**：

| 流派 | 纬度权重 | 变量权重 | 综合权重 | 权重学习方式 |
|------|---------|---------|---------|------------|
| **Transformer** | $\mathbf{L} \in \mathbb{R}^{H}$ | $\mathbf{w} \in \mathbb{R}^{C}$ | $\mathbf{W} \in \mathbb{R}^{C \times H}$ | 固定或可学习 |
| **Fourier/Operator** | $\mathbf{L} \in \mathbb{R}^{H}$（评估时） | 隐式（标准化） | $\mathbf{P} \in \mathbb{R}^{1 \times N \times d}$ | 标准化 |
| **Graph** | 无需（均匀网格） | $\mathbf{w} \in \mathbb{R}^{P}$ | 节点级权重 | 固定或可学习 |
| **Generative** | $\mathbf{L} \in \mathbb{R}^{H}$ | 隐式（NLL损失） | $\mathbf{W} \in \mathbb{R}^{C \times H \times W}$ | 端到端学习 |
| **ODE** | $\mathbf{L} \in \mathbb{R}^{H}$ | 无 | 无 | 无 |

**维度物理语义**：
- 权重反映每个纬度带在球面上的实际物理面积占比
- 位置编码帮助模型区分不同地理位置的气候特征
- Graph流派中无需纬度加权（图节点分布均匀）

#### 【架构层】

**计算流程图（五流派通用范式）**：

```
【训练阶段-损失计算】

预测输出 [B, C, H, W] 与真值 [B, C, H, W]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. 计算逐点误差                                              │
│    error = (pred - target)²                                 │
│    输出: [B, C, H, W]                                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌��────────────────────────────────────────────────────────────┐
│ 2. 应用纬度权重 L(φ_i)                                       │
│    Transformer: weighted_error = error * L[i]                │
│    Graph: 无需（icosahedral网格天然均匀）                    │
│    输出: [B, C, H, W]                                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 应用变量/气压权重 w(c)                                    │
│    Transformer: w[c] = f(v, p)（固定或可学习）              │
│    Fourier: 隐式（通过逐通道标准化）                         │
│    Generative: σ(c)自动学习                                 │
│    Graph: w[p] = p/p_surface                                │
│    输出: [B, C, H, W]                                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 全局平均                                                  │
│    loss = mean(final_error)                                  │
│    输出: 标量                                                │
└─────────────────────────────────────────────────────────────┘

【推理/编码阶段-权重共享】

输入Token/节点 [B, N, d]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Transformer: 窗口注意力                                      │
│   - 窗口内权重共享                                           │
│   - 跨窗口参数一致                                           │
│ Fourier/Operator: FFT + 共享MLP                             │
│   - 频域MLP在所有位置共享权重                                │
│   - 体现平移等变性                                           │
│ Graph: 消息传递                                             │
│   - 边函数在所有边上共享权重                                  │
│   - 节点更新函数在所有节点上共享                              │
│ Generative: UNet/Swin残差                                   │
│   - 卷积核权重共享                                           │
│   - 注意力权重动态计算                                        │
│ ODE: ODE函数权重共享                                         │
│   - f_θ(·)在所有时间/空间位置共享                            │
└─────────────────────────────────────────────────────────────┘
```

**GraphCast Multi-Mesh均匀性处理详细流程**：
```
[GraphCast Multi-Mesh 构建]
初始化: 正二十面体 (12 顶点, 20 面)
  ↓
递归细分 (r = 1 to 6):
  ├─ 每个三角面 → 4 个子三角面
  │   └─ 新顶点投影到单位球面
  ├─ Level r 节点数: ~10 × 4^r
  └─ 边集合: 合并所有层级的边

最终 Multi-Mesh:
  ├─ 节点: [40962, 3] (xyz 坐标)
  ├─ 边: [327660, 2] (多尺度连接)
  │   ├─ Level 0-2: 长程连接 (大尺度)
  │   └─ Level 3-6: 短程连接 (局地)
  └─ 节点特征初始化:
      └─ (cos(lat), sin(lon), cos(lon))
```

#### 【设计层】

**设计动机**：
- **公平性**：避免极区小面积区域主导损失函数
- **物理重要性**：强调近地面和人口密集区域的预报精度
- **评估一致性**：与气象业务评估标准（如WMO）保持一致
- **全局建模**：需要一种既能全局建模又计算高效的机制

**痛点解决（五流派对照）**：

| 问题 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **高纬过度权重** | 纬度加权损失 | 纬度加权评估 | 无需（均匀） | 纬度加权损失 | 纬度加权评估 |
| **变量权重调参** | 不确定性自动学习 | 逐通道标准化 | 固定气压加权 | NLL自动学习 | 无 |
| **全局感受野** | 窗口注意力+多层堆叠 | FFT单层全局 | 多跳消息传递 | 多层堆叠 | ODE积分 |
| **计算复杂度** | $O(M^2 \cdot N)$ | $O(N\log N)$ | $O(E)$ | 层数相关 | ODE求解 |
| **极区偏差** | 极区mask | 纬度加权 | 无偏差 | 投影消除 | 网格相关 |

**创新突破（五流派）**：

- **Transformer流派**：
  - FengWu：通过NLL损失自动学习权重，避免手工调参
  - EWMoE：结合可学习变量权重 $f(v)$ 与固定纬度权重
  - AERIS：物理加权损失同时考虑纬度、气压层和变量类型

- **Fourier/Operator流派**：
  - FourCastNet：首次将AFNO应用于全球天气预报，FFT复杂度 $O(N\log N)$
  - 在 $721 \times 1440$ 分辨率下，AFNO显存占用约10GB，而同深度ResNet需83GB
  - SFNO：频域权重共享天然满足平移等变

- **Graph流派**：
  - GraphCast：首次在全球天气预报中系统使用多尺度icosahedral mesh
  - GenCast：通过图结构避免显式位置编码，简化设计
  - AIFS：将Reduced Gaussian grid与GNN结合，兼顾均匀性和NWP兼容性

- **Generative流派**：
  - AERIS：90天S2S预报中使用扩散模型的稳定性
  - CorrDiff：在降尺度任务中系统性使用4通道正弦位置编码

- **ODE流派**：
  - ClimODE：连续时间模型，隐式满足物理均匀性

#### 【对比层】

**五流派共性归纳**：
- 所有在Lat-Lon网格上训练的模型均使用某种形式的纬度加权
- 多数模型结合变量/层权重
- 均通过某种机制实现全局信息聚合
- 所有模型均使用位置编码（显式嵌入 / 隐式结构 / Fourier基 / 图拓扑）

**五流派差异分析（大型对比表）**：

| 维度 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **纬度加权位置** | 训练损失中显式应用 | 评估时应用 | 无需（均匀网格） | 训练损失中显式 | 评估时应用 |
| **变量权重方式** | 可学习或固定 | 隐式（标准化） | 固定气压加权 | NLL自动学习 | 无 |
| **全局建模机制** | 窗口注意力+多层堆叠 | FFT单层全局 | 多跳消息传递 | 多层堆叠 | ODE积分外推 |
| **权重共享范围** | 窗口内共享 | 频域全局共享 | 图边/节点共享 | 卷积核共享 | ODE函数共享 |
| **计算复杂度** | $O(M^2N)$（窗口） | $O(N\log N)$ | $O(E)$ | $O(N^2)$~层 | ODE求解 |
| **参数量** | 较大（多层Transformer） | 较小（共享MLP） | 中等 | 取决于UNet深度 | 适中 |
| **均匀性度量** | 差（极区密度 10×） | 差（隐式处理） | 最优（相对标准差<5%） | N/A（区域模型） | 中等 |
| **极区处理** | 需额外mask/padding | 纬度加权 | 无需处理 | 区域投影 | 避免 |
| **位置编码方式** | 2D/3D绝对或RoPE | Fourier基函数 | xyz坐标+边拓扑 | 正弦4通道 | 连续归一化 |
| **权重学习** | 端到端或固定 | 标准化 | 固定 | 端到端学习 | 无 |

**五流派代表性模型均匀性详细对比**：

| 模型 | 纬度加权 | 变量权重 | 气压加权 | 权重学习 | 全局机制 | 极区处理 |
|------|---------|---------|---------|---------|---------|---------|
| FengWu (Transformer) | ✅ | ✅（NLL自动） | ✅ | 端到端学习 | 窗口注意力 | mask |
| Swin V2 (Transformer) | ✅ | ✅（手工） | ✅ | 固定权重 | 窗口注意力 | mask |
| AERIS (Transformer) | ✅ | ✅ | ✅ | 固定权重 | 扩散+注意力 | mask |
| FourCastNet (Fourier) | ✅（评估） | 隐式（标准化） | 隐式 | 标准化 | FFT全局 | 纬度加权 |
| SFNO (Operator) | ✅ | 隐式 | 隐式 | 标准化 | SHT全局 | 自然处理 |
| GraphCast (Graph) | 无需 | ✅（气压） | ✅ | 固定权重 | 多跳消息传递 | 天然无 |
| GenCast (Graph) | 无需 | ✅ | ✅ | 固定权重 | 图Transformer | 天然无 |
| AIFS (Graph) | ✅ | ✅ | ✅ | 固定权重 | GNN+Transformer | 点数减少 |
| CorrDiff (Gen) | N/A | ✅ | ✅ | 固定权重 | UNet | 区域模型 |
| ClimODE (ODE) | ✅ | 无 | 无 | 无 | ODE积分 | 网格相关 |

**权衡评估**：

**固定权重（Transformer-Swin V2 / Graph流派）**：
- ✅ 优点：简单稳定
- ❌ 缺点：无法自适应不同数据分布

**学习权重（Transformer-FengWu / Generative）**：
- ✅ 优点：自适应，避免手工调参
- ❌ 缺点：增加训练复杂度

**FFT全局建模（Fourier/Operator流派）**：
- ✅ 优点：单层即全局感受野，计算高效，天然包含全局信息
- ❌ 缺点：假设空间平移等变性，对强非均匀场景可能不如局部注意力灵活

**Icosahedral均匀网格（Graph流派）**：
- ✅ 优点：理论最优，无需纬度加权，极区无奇异性
- ❌ 缺点：需要Grid↔Mesh插值，预处理复杂

**扩散模型稳定性（Generative-AERIS）**：
- ✅ 优点：90天预报稳定，长期均匀性由扩散过程保证
- ❌ 缺点：推理成本高

**适用场景**：
- **Transformer流派**：适合需要精细控制不同变量/区域权重的场景
- **Fourier/Operator流派**：适合需要极高计算效率和全局建模的场景
- **Graph流派**：适合追求极致均匀性和球面自然性的全球预报
- **Generative流派**：适合需要概率预报和长期稳定性的场景
- **ODE流派**：适合需要物理一致性和连续时间外推的场景

---

## 【第二部分：变量组织范式】

### 2.1 2D通道堆叠策略

#### 【原理描述】

**核心定义**：将多层次、多变量的三维大气场展平为二维图像的多通道表示，每个通道对应一个"变量-气压层"组合或单一地表变量，类似于RGB图像的通道维度扩展，是气象AI中最广泛采用的变量组织方式。

**数学推导**：

设原始大气状态为四维张量：
$$\mathbf{X}_{\text{4D}} \in \mathbb{R}^{V \times L \times H \times W}$$
其中 $V$ 为变量数（如Z, T, U, V, Q），$L$ 为气压层数，$H \times W$ 为空间网格。

通道堆叠后：
$$\mathbf{X}_{\text{2D}} \in \mathbb{R}^{C \times H \times W}, \quad C = V \times L + V_{\text{surface}}$$

堆叠顺序（常见方案）：
$$C = [\underbrace{Z_{50}, Z_{100}, ..., Z_{1000}}_{\text{位势高度}}, \underbrace{T_{50}, ..., T_{1000}}_{\text{温度}}, ..., \underbrace{T2m, U10, V10, MSLP}_{\text{地表变量}}]$$

**GraphCast通道详细组织**（227通道）：
- 0-36: Z (1-1000 hPa, 37层)
- 37-73: T (1-1000 hPa, 37层)
- 74-110: u (1-1000 hPa, 37层)
- 111-147: v (1-1000 hPa, 37层)
- 148-184: w (1-1000 hPa, 37层)
- 185-221: q (1-1000 hPa, 37层)
- 222: 2m温度
- 223-224: 10m风 (u, v)
- 225: 海平面气压
- 226: 总降水

**FourCastNet通道详细组成**（20通道）：
- 地表变量(5)：U10, V10, T2m, sp, mslp
- 1000 hPa(3)：U1000, V1000, Z1000
- 850 hPa(5)：T850, U850, V850, Z850, RH850
- 500 hPa(5)：T500, U500, V500, Z500, RH500
- 50 hPa(1)：Z50
- 积分变量(1)：TCWV

**理论依据**：
- 利用成熟的2D卷积/Transformer架构
- 通道间关系通过网络隐式学习（如1×1卷积、cross-attention）
- 与计算机视觉中的RGB图像类似，便于应用成熟的深度学习架构
- **计算效率**：2D操作比3D操作更高效
- **层间耦合**：通过通道维的卷积核或注意力自然学习垂直相关性

#### 【数据规格层】

**输入张量形状对比（五流派全览）**：

| 模型 | 流派 | 输入形状 | C | 通道组成 |
|------|------|---------|---|---------|
| FengWu | Transformer | `[B, 189, 721, 1440]` | 189 | 5变量×37层 + 4地表 |
| Swin V2 | Transformer | `[B, 73, 720, 1440]` | 73 | 5变量×13层 + 8地表 |
| AERIS | Transformer | `[B, 70, 720, 1440]` | 70 | 5变量×13层 + 5地表 + 强迫 |
| FourCastNet | Fourier | `[B, 20, 721, 1440]` | 20 | 精简变量+关键层 |
| GraphCast | Graph | `[B, 227, 721, 1440]` | 227 | 6变量×37层 + 5地表 |
| GenCast | Graph | `[B, ~80, 721, 1440]` | ~80 | 完整垂直结构 |
| AIFS | Graph | `[B, C, 542080]` | ~65 | Reduced Gaussian |
| CorrDiff | Generative | `[B, 12, 36, 36]` | 12 | 关键层（850+500hPa） |
| FuXi-TC | Generative | `[B, 71, H, W]` | 71 | 完整垂直结构+RH推导 |
| ClimODE | ODE | `[B, 5, H, W]` | 5 | t2m, t, z, u10, v10 |
| GNADET | Graph | `[B, 11, H, W]` | 11 | T2M + 辅助（9动态+2静态） |

**通道数分布统计**：
- **>200通道**：GraphCast（227）—— 最完整，信息损失最小
- **100-200通道**：FengWu（189）—— 多模态分组建模
- **50-80通道**：Swin V2（73）、AERIS（70）、Graph/GenCast（~80）
- **<30通道**：FourCastNet（20）、CorrDiff（12）、ClimODE（5）—— 精简高效

**维度物理语义**：
- $C$ 维度：混合了变量类型和垂直层次信息
- 通道顺序：通常按"变量分组"或"层次分组"
- 空间维 $(H, W)$：编码水平空间结构

#### 【架构层】

**计算流程图（五流派通用）**：

```
原始4D数据 [V, L, H, W]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. 通道展平                                                  │
│    reshape: [V×L + V_surf, H, W]                           │
│    Transformer: 按变量分组展平                                │
│    Graph: 拼接 z, x_t, x_{t-1} → [3C, H, W]                │
│    Generative: 历史+未来+强迫拼接                            │
│    输出: [C, H, W]                                          │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 逐通道标准化                                              │
│    X̃_c = (X_c - μ_c) / σ_c                                 │
│    μ_c, σ_c: 预计算的训练集统计量                            │
│    Transformer: LayerNorm per channel                        │
│    Fourier: 独立计算μ_c, σ_c                                │
│    Graph: 节点级标准化                                       │
│    输出: [C, H, W]                                          │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Patch Embedding                                          │
│    - 划分patch (p×p)                                        │
│    - 展平 + 线性投影到D维                                    │
│    Transformer: p=1/4/8/16                                  │
│    Fourier: p=8                                             │
│    Graph: 每个节点=1 token, 无patch                          │
│    Generative: UNet编码器下采样                              │
│    输出: [N_patch, D] 或 [N_node, D]                        │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 位置编码                                                  │
│    + 2D/3D位置嵌入 / Fourier基 / xyz坐标                     │
│    输出: [N_patch, D] + position_embedding                   │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 主干网络处理                                              │
│    Transformer: 窗口/全局注意力                               │
│    Fourier: FFT + 共享MLP                                    │
│    Graph: 多尺度消息传递                                     │
│    Generative: 扩散去噪                                      │
│    ODE: Neural ODE积分                                       │
│    通道交互: 隐式学习或cross-attention                       │
│    输出: [N_patch, D]                                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. 解码回2D                                                  │
│    - 线性投影回C通道                                         │
│    - reshape: [C, H, W]                                      │
│    Graph: Mesh→Grid插值                                      │
│    Generative: UNet解码器上采样                              │
│    输出: [C, H, W]                                          │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. 逐通道反标准化                                            │
│    X_c = X̃_c × σ_c + μ_c                                   │
│    输出: [C, H, W]（物理量）                                │
└─────────────────────────────────────────────────────────────┘
```

**GraphCast Encoder图构建详细流程**：
```
Grid → Mesh (Encoder):
  ├─ 对每个mesh节点
  │   └─ 连接半径内所有grid点
  ├─ 边数: ~1.6M
  └─ 输出: [N_mesh, D] ← [N_grid, D]

Mesh → Grid (Decoder):
  ├─ 对每个grid点
  │   └─ 连接最近3个mesh节点（所在三角面）
  ├─ 边数: 3 × N_grid
  └─ 输出: [N_grid, D] ← [N_mesh, D]
```

**CorrDiff UNet张量变化详细流程**：
```
输入 [16, 36, 36] (12通道ERA5 + 4通道位置编码)
  ↓
Encoder Level 1: [128, 36, 36]
  ↓ 下采样
Encoder Level 2: [256, 18, 18]
  ↓ 下采样
Encoder Level 3: [256, 9, 9]
  ↓ 下采样
...
Bottleneck: [C_max, H_min, W_min]
  ↓ 上采样
Decoder (对称结构 + skip connections)
  ↓
输出 [4, 448, 448] (降尺度后高分辨率)
```

#### 【设计层】

**设计动机**：
- **架构复用**：直接使用ViT/Swin/AFNO/UNet等成熟视觉模型
- **实现简单**：无需设计复杂的3D卷积或图结构
- **计算效率**：2D操作比3D操作更高效
- **联合预测**：在单一模型中联合预测多个物理量，捕捉动力学耦合

**痛点解决（五流派对照）**：

| 问题 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **变量间关系** | 隐式学习或cross-attn（Stormer） | 隐式学习（FFT+MLP） | 隐式（消息传递） | UNet skip connections | 共享状态 |
| **通道数过多** | 多模态划分（FengWu: 6模态） | 精简变量选择（20通道） | 完整保留（GraphCast） | 精简区域（12通道） | 精选变量（5） |
| **垂直结构** | 3D位置嵌入或通道堆叠 | 通道堆叠+标准化 | 节点特征编码 | 通道堆叠 | 无垂直 |
| **降水处理** | 部分直接预测 | 独立诊断模型（FourCastNet） | 部分直接预测 | 聚焦建模 | 无 |

**创新突破（五流派）**：

- **Transformer流派**：
  - Swin V2：证明最小修改的2D Swin即可达到SOTA
  - AERIS：像素级（1×1 patch）保留最大空间细节
  - Stormer：变量聚合层（cross-attention）显式建模变量关系
  - ClimaX：每个变量独立patch embedding，变量级token拼接

- **Fourier/Operator流派**：
  - FourCastNet：首次在0.25°分辨率下联合预测20个变量，包括难度极高的近地面风和降水
  - 精简通道设计（20 vs 189），平衡预报能力与计算成本
  - AFNO在通道混合的同时进行频域变换

- **Graph流派**：
  - GraphCast：227通道的大规模堆叠，证明2D架构可处理复杂3D大气
  - GenCast：227通道的大规模堆叠，证明2D架构可处理复杂3D大气
  - AIFS：在Reduced Gaussian grid上应用通道堆叠，兼顾效率和精度

- **Generative流派**：
  - CorrDiff：首次在km级降尺度中使用12通道ERA5 + 4通道位置编码
  - FuXi-TC：通过71通道输入捕获完整的大气垂直结构
  - PHYS-Diff：69通道聚焦TC相关层

- **ODE流派**：
  - ClimODE：精选5个关键变量，通过ODE流场实现连续演化

#### 【对比层】

**五流派共性归纳**：
- 所有全球模型均采用2D通道堆叠策略（或其图/网格等价形式）
- 均使用标准的patch embedding或节点化
- 均通过某种机制学习通道间关系
- 所有模型均对每个"变量-层"组合独立标准化

**五流派差异分析（大型对比表）**：

| 维度 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **通道数范围** | 20-189（多样） | 20（精简） | 5-227（全覆盖） | 5-71（精简） | 5（精选） |
| **Patch大小** | 1×1至16×16 | 固定8×8 | N/A（节点=token） | UNet下采样 | N/A |
| **通道交互机制** | 隐式/cross-attn | 隐式（FFT+MLP） | 消息传递 | skip connections | 共享状态 |
| **垂直层数** | 5-37层 | 5层（代表性） | 37层（完整） | 2-13层 | 无（精选变量） |
| **降水处理** | 部分直接预测 | 独立诊断模型 | 部分直接预测 | 聚焦建模 | 无 |
| **多模态划分** | 6模态（FengWu） | 无（统一） | 无 | 3+模态（PHYS-Diff） | 无 |
| **预处理复杂度** | 低 | 中 | 高（Grid→Mesh） | 中 | 低 |
| **通道组织** | 变量分组 | 精简变量 | 完整保留 | 关键层 | 精选变量 |

**五流派代表性模型通道配置详细对比**：

| 模型 | 通道数 | Patch | 通道交互 | 垂直层 | 降水 | 流派 |
|------|--------|-------|---------|--------|------|------|
| FengWu | 189 | 4/8 | 6模态+cross-attn | 37 | 直接预测 | Transformer |
| Swin V2 | 73 | 4/8/16 | 隐式 | 13 | **未详细说明** | Transformer |
| AERIS | 70 | 1×1 | 隐式 | 13 | 直接预测 | Transformer |
| Stormer | **未说明** | 2/4 | cross-attn | **未说明** | **未说明** | Transformer |
| FourCastNet | 20 | 8 | 隐式（FFT+MLP） | 5 | 独立诊断 | Fourier |
| SFNO | **未说明** | 球谐阶数 | 频域算子 | **未说明** | 直接预测 | Operator |
| GraphCast | 227 | N/A | 消息传递 | 37 | 直接预测 | Graph |
| GenCast | ~80 | N/A | 图Transformer | 13 | 直接预测 | Graph |
| AIFS | ~65 | N/A | GNN+Transformer | 13 | 直接预测 | Graph |
| CorrDiff | 12 | UNet | skip connections | 2 | 聚焦建模 | Generative |
| FuXi-TC | 71 | **未说明** | cross-attn | 13 | 聚焦TC | Generative |
| ClimODE | 5 | N/A | 共享流场 | 无 | 无 | ODE |

**权衡评估**：

**2D通道堆叠优势**：
- ✅ 实现简单，易于扩展
- ✅ 可利用预训练视觉模型
- ✅ 2D操作计算高效
- ✅ 便于应用成熟的深度学习架构
- ✅ 通道顺序固定，便于模型学习位置不变性

**2D通道堆叠劣势**：
- ❌ 丢失显式的3D结构信息（垂直依赖通过隐式学习）
- ❌ 通道数过多时计算负担重（如FengWu的189通道）
- ❌ 变量间物理关系需网络自行学习
- ❌ 相比IFS（>150变量、>50层），信息损失较大

**高通道数（GraphCast: 227 / FengWu: 189）**：
- ✅ 优点：完整大气状态，长期预报信息损失最小
- ❌ 缺点：计算成本高，过拟合风险

**精简通道数（FourCastNet: 20 / CorrDiff: 12 / ClimODE: 5）**：
- ✅ 优点：计算高效，训练快速，可聚焦关键变量
- ❌ 缺点：信息损失，可能遗漏关键信号

**适用场景**：
- **高通道数（>100）**：适合需要完整大气状态的长期预报
- **中等通道数（50-100）**：适合标准全球预报
- **低通道数（<30）**：适合需要快速推理的业务场景或区域任务
- **精选通道（<10）**：适合特定任务（如气候建模、TC预报）

---

### 2.2 3D体建模策略

#### 【原理描述】

**核心定义**：保持大气场的原生三维结构 $(V, L, H, W)$ 或 $(C, P, H, W)$，通过3D卷积、3D Transformer或专门的3D patch embedding直接建模体数据，显式保留垂直方向的结构信息。

**数学推导**：

**3D Patch Embedding**：
$$\mathbf{X} \in \mathbb{R}^{V \times L \times H \times W} \rightarrow \text{Patch}_{3D}(p_l, p_h, p_w) \rightarrow \mathbf{T} \in \mathbb{R}^{N_{\text{3D}} \times D}$$
其中：
$$N_{\text{3D}} = \frac{L}{p_l} \times \frac{H}{p_h} \times \frac{W}{p_w}$$

**3D卷积操作**：
$$\mathbf{Y}_{c,p,i,j} = \sum_{c'}\sum_{p'}\sum_{i'}\sum_{j'} \mathbf{W}_{c,c',p',i',j'} \cdot \mathbf{X}_{c',p+p',i+i',j+j'} + b_c$$
其中 $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in} \times K_p \times K_h \times K_w}$ 为3D卷积核。

**3D窗口注意力（Pangu-Weather 3DEST）**：
窗口大小如 $2 \times 6 \times 12$（层×纬度×经度），同时处理垂直和水平维度。

**PreDiff时空3D处理**：
将时间维度视为类似垂直维度的第三维：
- 输入历史：$y \in \mathbb{R}^{L_{in} \times H \times W \times C}$
- 未来序列：$x \in \mathbb{R}^{L_{out} \times H \times W \times C}$

**理论依据**：
- 大气是真实的3D连续介质，垂直方向的物理过程（对流、辐射传输）与水平过程耦合
- 3D卷积能直接捕获垂直-水平耦合的物理过程
- 大气物理过程本质上是三维的（对流、辐射、湍流混合等）
- 垂直方向的信息传递对于准确预测至关重要

#### 【数据规格层】

**输入/输出张量形状对比**：

| 模型 | 流派 | 输入形状 | 垂直处理 | 特殊设计 |
|------|------|---------|---------|---------|
| Pangu-Weather | Transformer | `[B, V, P, H, W]` | 3DEST（窗口2×6×12） | 3D窗口注意力 |
| FuXi | Transformer | `[B, V, P, H, W]` | Cube embedding | 压缩后处理 |
| PreDiff | Generative | `[B, L_in, H, W, C]` | Earthformer-UNet | 3D Cuboid Attention |
| 高分辨率降水 | Transformer | `[B, 5, 13, H, W]` | 3D Swin | 分层窗口注意力 |
| GenCast | Graph | `[B, V, N_mesh]` | 节点特征编码 | 图Transformer |
| FourCastNet | Fourier | `[B, C, H, W]` | 通道堆叠 | 隐式2D |
| GraphCast | Graph | `[B, V, P, H, W]` | 通道堆叠 | Encoder-Decoder |

#### 【架构层】

**计算流程图（Pangu-Weather 3DEST）**：

```
多变量多层气象数据 [B, V, P, H, W]
  ↓
3D Patch Embedding
  ↓ [B, N_patches, D]
3D Earth-Specific Transformer
  ↓ 3D窗口注意力，窗口大小 2×6×12
  ↓ 同时处理垂直和水平维度
  ↓ [B, N_patches, D]
3D Patch恢复
  ↓ [B, V, P, H, W]
输出预测场
```

**PreDiff Earthformer-UNet流程（Generative流派）**：

```
输入序列 [L_in, H, W, C]
  ↓
VAE Encoder (逐帧): [L_in, H_z, W_z, C_z]
  ↓
拼接条件和噪声潜在: [L_in+L_out, H_z, W_z, C_z]
  ↓
Earthformer-UNet Encoder:
  - 3D Cuboid Self-Attention (时空块注意力)
  - 多尺度下采样
  ↓ [L', H_z', W_z', C_hidden]
Bottleneck
  ↓
Earthformer-UNet Decoder:
  - 上采样 + Skip connections
  - 3D Cuboid Self-Attention
  ↓ [L_out, H_z, W_z, C_z]
VAE Decoder (逐帧): [L_out, H, W, C]
```

**FuXi Cube Embedding流程（Transformer流派）**：

```
3D气象数据 [B, V, P, H, W]
  ↓
Cube Embedding（压缩3D为2D）
  ↓ [B, C', H', W']
U-Transformer处理
  ↓ [B, C', H', W']
Cube恢复
  ↓ [B, V, P, H, W]
```

#### 【设计层】

**设计动机**：
- **显式保留垂直结构**：捕获层间的动力学和热力学耦合
- **更符合大气物理本质**：3D结构更符合大气动力学方程的三维本质
- **时空耦合**：3D cuboid attention在时空块内建模局部依赖
- **极端事件**：对流等三维过程的准确建模提升极端降水、强对流预报

**痛点解决**：
- **垂直信息丢失**：2D通道堆叠难以捕获层间关系，3D建模直接处理垂直维度
- **物理一致性**：3D结构更符合大气动力学方程的三维本质
- **计算效率**：相比全局3D attention，cuboid attention降低复杂度
- **计算成本**：3D操作的参数量和FLOPs显著高于2D（约为 $P$ 倍）

**创新突破（五流派）**：

- **Transformer流派**：
  - Pangu-Weather：3DEST在中期预报上首次超越IFS，部分归功于3D建模
  - FuXi：Cube embedding平衡效率与精度，通过压缩后处理3D数据

- **Generative流派**：
  - PreDiff：首次在降水临近预报中使用Earthformer-UNet替代标准2D UNet
  - Earthformer：3D Cuboid Attention在时空块内实现高效的3D建模

#### 【对比层】

**五流派3D建模应用情况（大型对比表）**：

| 模型 | 流派 | 3D建模方式 | 垂直/时间维度处理 | 计算成本 | 适用场景 |
|------|------|-----------|-----------------|---------|---------|
| Pangu-Weather | Transformer | 3D窗口注意力 | 显式3D Transformer | 高 | 全球中期预报 |
| FuXi | Transformer | Cube embedding | 压缩后处理 | 中 | 全球预报 |
| PreDiff | Generative | 3D Cuboid Attention | 时空块注意力 | 高 | 降水临近预报 |
| 高分辨率降水 | Transformer | 3D Swin | 分层窗口注意力 | 高 | 降水预报 |
| GraphCast | Graph | 通道堆叠 | 节点特征编码 | 中 | 全球中期预报 |
| FourCastNet | Fourier | 通道堆叠 | 隐式通过通道 | 低 | 快速预报 |
| GenCast | Graph | 通道堆叠 | 隐式通过通道 | 中 | 全球集合预报 |
| ClimODE | ODE | 无 | 流场+连续时间 | 低 | 气候建模 |
| CorrDiff | Generative | 通道堆叠 | 隐式2D | 低 | 区域降尺度 |

**权衡评估**：

**3D卷积/Attention**：
- ✅ 优点：直接建模时空耦合或垂直耦合，物理一致性最强
- ❌ 缺点：计算成本高（参数量约为2D的 $P$ 倍），需要更多训练数据以避免过拟合

**2D + 隐式（主流方案）**：
- ✅ 优点：计算高效，2D架构成熟，易于迁移学习
- ❌ 缺点：难以显式建模垂直方向的物理过程

**Cube Embedding（FuXi）**：
- ✅ 优点：平衡效率与精度，通过压缩处理3D数据
- ❌ 缺点：压缩可能损失细节

**3D Cuboid Attention（PreDiff）**：
- ✅ 优点：在时空块内实现高效的3D建模，支持时空耦合
- ❌ 缺点：计算成本仍然较高，短时效

**适用场景**：
- **纯3D建模**：对于需要准确垂直结构的任务（如对流预报、温度廓线预测）
- **2D隐式**：对于主要关注地面变量或快速推理的业务场景
- **Cube Embedding**：全球预报中平衡精度与效率
- **时空3D**：降水临近预报等高度时空耦合的任务

---

### 2.3 多模态划分策略

#### 【原理描述】

**核心定义**：将多变量大气场按物理变量类型划分为多个"模态"（modality），每个模态使用独立的编码器处理，再通过跨模态融合机制整合信息。或将不同来源、不同物理意义的数据作为独立模态处理。

**数学推导**：

**变量类型模态划分**（Transformer流派-FengWu）：
$$\mathbf{X} \in \mathbb{R}^{189 \times H \times W} \rightarrow \{\mathbf{X}_s, \mathbf{X}_z, \mathbf{X}_q, \mathbf{X}_u, \mathbf{X}_v, \mathbf{X}_t\}$$

模态编码：
$$\mathbf{Z}_m = f_{\text{enc}, m}(\mathbf{X}_m \mid \theta_m), \quad m \in \{s, z, q, u, v, t\}$$

跨模态融合：
$$\mathbf{Z}_{\text{fused}} = \text{CrossModalTransformer}(\text{concat}(\mathbf{Z}_s, \mathbf{Z}_z, ..., \mathbf{Z}_t))$$

模态解码：
$$\hat{\mathbf{X}}_m = f_{\text{dec}, m}(\mathbf{Z}_{\text{fused}} \mid \phi_m)$$

**多源异构模态划分**（Generative流派-PHYS-Diff）：
- 模态1：历史TC属性 $\mathcal{H} = \{h_1, ..., h_M\}$，$h_i = (\mathbf{x}_i, v_i, p_i) \in \mathbb{R}^4$
- 模态2：历史环境场 $\mathcal{E}_{\text{hist}} = \{E_1, ..., E_M\} \in \mathbb{R}^{M \times 69 \times 80 \times 80}$
- 模态3：未来环境场 $\mathcal{E}_{\text{fut}} = \{E'_1, ..., E'_N\} \in \mathbb{R}^{N \times 69 \times 80 \times 80}$

**常见融合方式**：
- **拼接（Concatenation）**：$f_{\text{fused}} = [f_1; f_2; ...; f_M]$
- **加权求和**：$f_{\text{fused}} = \sum_{m=1}^M \alpha_m f_m$，其中 $\alpha_m$ 可学习
- **注意力融合**：$f_{\text{fused}} = \sum_{i} \alpha_i f_i$，其中 $\alpha_i = \text{Attention}(f_i)$
- **Transformer融合**：$f_{\text{fused}} = \text{TransformerEncoder}([f_1; f_2; ...; f_M])$

**理论依据**：
- 不同物理变量遵循不同的动力学方程
- 模态专用编码器可学习变量特异的特征表示
- 不同模态具有不同的统计特性和物理意义
- 专门的编码器可以更好地提取各模态的特征

#### 【数据规格层】

**各流派模态划分案例对比**：

| 模型 | 流派 | 模态数量 | 模态划分方式 | 编码器类型 | 融合方式 |
|------|------|---------|-------------|-----------|---------|
| FengWu | Transformer | 6 | 变量类型（s/z/q/u/v/t） | 6个独立Swin | CrossModalTransformer |
| Stormer | Transformer | 隐式 | 变量聚合 | cross-attention | cross-attention |
| FuXi-TC | Generative | 2 | FuXi场/WRF场 | CNN+CNN | 扩散条件 |
| GenEPS | Generative | 3+ | ERA5/多模型/扩散先验 | 各模型自带 | SDEdit采样 |
| PHYS-Diff | Generative | 3 | TC属性/历史场/未来场 | GRU+Swin+Embedding | Transformer Encoder |
| 高分辨率降水 | Transformer | 4 | 上空/地面/静态/时间 | 非线性embedding | 拼接+共享主干 |
| ClimODE | ODE | 1 | 精选变量 | 共享流场 | 无需 |
| FourCastNet | Fourier | 1 | 统一通道堆叠 | 单一AFNO | 无需 |

**FengWu模态详细配置**（Transformer流派）：
- 地表模态 $C_s = 4$（MSLP, T2m, U10m, V10m）
- 位势高度模态 $C_z = 37$（37个气压层）
- 湿度模态 $C_q = 37$
- 经向风模态 $C_u = 37$
- 纬向风模态 $C_v = 37$
- 温度模态 $C_t = 37$

**PHYS-Diff多模态详细配置**（Generative流派）：
- 模态1：GRU编码历史TC轨迹，输出 $H_{TC} \in \mathbb{R}^{D_{hidden}}$
- 模态2+3：Swin编码历史+未来环境场，输出 $T_{env} \in \mathbb{R}^{N_{tokens} \times D_{hidden}}$
- 融合：拼接后通过Transformer Encoder，输出条件上下文 $c$

#### 【架构层】

**计算流程图（FengWu多模态，Transformer流派）**：

```
输入 [B, 189, H, W]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. 模态切分                                                  │
│    X_s [B, 4, H, W]                                        │
│    X_z [B, 37, H, W]                                       │
│    X_q [B, 37, H, W]                                       │
│    X_u [B, 37, H, W]                                       │
│    X_v [B, 37, H, W]                                       │
│    X_t [B, 37, H, W]                                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 模态编码（并行）                                          │
│    Z_s = Enc_s(X_s)                                         │
│    Z_z = Enc_z(X_z)                                         │
│    Z_q = Enc_q(X_q)                                         │
│    Z_u = Enc_u(X_u)                                         │
│    Z_v = Enc_v(X_v)                                         │
│    Z_t = Enc_t(X_t)                                         │
│    输出: 6个 [B, N, D_m]                                    │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 跨模态融合                                                │
│    Z = concat([Z_s,...,Z_t])                                │
│    Z_fused = CrossModalTF(Z)                                │
│    输出: [B, N_total, D]                                    │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 模态解码（并行）                                          │
│    μ_s, σ_s = Dec_s(Z_fused)                                │
│    μ_z, σ_z = Dec_z(Z_fused)                                │
│    ...                                                       │
│    输出: 6对 (μ, σ)                                         │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 模态拼接                                                  │
│    μ = concat([μ_s,...,μ_t])                                │
│    σ = concat([σ_s,...,σ_t])                                │
│    输出: [B, 189, H, W] ×2                                  │
└─────────────────────────────────────────────────────────────┘
```

**PHYS-Diff多模态融合流程（Generative流派）**：

```
模态1: 历史TC属性 [M, 4]
  ↓
GRU Encoder
  ↓ H_TC [D_hidden]

模态2+3: 历史+未来环境场 [M+N, 69, 80, 80]
  ↓
Swin Transformer
  ↓ T_env [N_tokens, D_hidden]

时间步嵌入 t
  ↓ t_emb [D_hidden]

拼接: [H_TC; T_env; t_emb]
  ↓
Transformer Encoder (多头自注意力)
  ↓ c [N_total_tokens, D_hidden]

条件上下文 c 用于扩散解码器的 Cross-Attention
```

**高分辨率降水多模态流程（Transformer流派）**：

```
原始多模态数据
  ↓
模态1: 上空变量 → Encoder_upper → z_upper
模态2: 地面变量 → Encoder_surface → z_surface
模态3: 静态特征 → Encoder_static → z_static
模态4: 时间特征 → Encoder_time → z_time
  ↓
多模态融合层（Concat/Attention/Cross-Attention）
  ↓ z_fused
共享主干网络（Transformer/CNN）
  ↓
任务特定解码器
  ↓
输出预测
```

#### 【设计层】

**设计动机**：
- **物理解耦**：不同变量的动力学特性差异大（如风场 vs 温度场）
- **专家系统**：每个模态编码器成为该变量的"专家"
- **可解释性**：模态划分对应物理变量分类
- **异构数据**：不同模态使用专门编码器（GRU for 序列，Swin for 图像）

**痛点解决**：
- **问题**：单一编码器难以同时学习189个通道的复杂关系
- **解决**：模态专用编码器 + 跨模态融合，分而治之
- **特征不匹配**：直接拼接不同模态可能导致特征尺度不一致
- **模态缺失**：模块化设计便于处理部分模态缺失的情况

**创新突破（五流派）**：

- **Transformer流派**：
  - FengWu：首次在气象AI中引入多模态+多任务框架
  - Stormer：轻量版变量聚合（通过cross-attention）
  - 高分辨率降水：非线性patch embedding分别处理不同模态

- **Generative流派**：
  - PHYS-Diff：首次系统性融合TC属性、ERA5历史场和FengWu未来场
  - GenEPS：通过扩散先验统一多模型输出的概率表示
  - FuXi-TC：通过71通道输入捕获完整的大气垂直结构

- **Fourier/Operator流派**：
  - FourCastNet：将所有20个变量统一处理，未采用多模态划分

- **Graph流派**：
  - GraphCast：所有227个通道通过单一Encoder-Decoder处理，未显式模态划分

#### 【对比层】

**五流派共性归纳**：
- 多模态模型均采用"独立编码 + 融合"的范式
- Transformer是最常用的融合机制
- 多模态划分策略在复杂气象任务中越来越常见

**五流派差异分析（大型对比表）**：

| 维度 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **模态数量** | 6个显式（FengWu）或隐式（Stormer） | 无（统一处理） | 无（统一处理） | 3+模态 | 无 |
| **模态划分依据** | 变量类型 | N/A | N/A | 数据来源+物理类型 | N/A |
| **编码器** | 6个独立或聚合 | 单一AFNO | 单一Encoder-Decoder | 异构（GRU+Swin等） | 共享 |
| **融合机制** | CrossModalTF或cross-attn | 隐式（FFT+MLP） | 隐式（消息传递） | Transformer Encoder | 共享状态 |
| **输出类型** | 均值+方差（FengWu）或确定性 | 确定性 | 确定性 | 扩散条件 | 确定性 |
| **参数量** | 较大（6编码器+6解码器） | 较小 | 中等 | 取决于编码器 | 共享最小 |
| **训练复杂度** | 高 | 低 | 中 | 高 | 低 |
| **适用场景** | 超高维输入+可解释性 | 中低维输入 | 全球统一建模 | 多源异构数据 | 精选变量 |

**权衡评估**：

**多模态划分优势**：
- ✅ 物理意义清晰
- ✅ 可解释性强
- ✅ 适合超高维输入（如189通道）
- ✅ 异构数据融合自然

**多模态划分劣势**：
- ❌ 参数量大（6个编码器+6个解码器）
- ❌ 训练复杂度高
- ❌ 需要合理的模态划分先验知识

**统一处理优势**（Fourier/Operator、Graph流派）：
- ✅ 实现简单
- ✅ 参数量小
- ✅ 训练高效
- ✅ 隐式学习变量间关系

**统一处理劣势**：
- ❌ 对超高维输入（如189通道）学习压力大
- ❌ 难以处理异构数据

**适用场景**：
- **多模态划分**：适合超高维输入（>100通道）、需要可解释性、多源异构数据融合
- **统一处理**：适合中等维度输入（<100通道）和需要快速训练的场景

---

### 2.4 图节点表示策略

#### 【原理描述】

**核心定义**：将大气场或观测站点表示为图结构，节点对应空间位置或观测点，边表示物理关联或空间邻近关系，通过图神经网络（GNN）的消息传递机制建模节点间的依赖关系。适用于不规则网格、稀疏观测网络和需要显式空间关系建模的任务。

**数学推导**：

**异质图定义**（Local Off-Grid）：
$$\mathcal{G} = (\mathcal{V}, \mathcal{E})$$
其中 $\mathcal{V} = \mathcal{V}_{\text{station}} \cup \mathcal{V}_{\text{grid}}$ 为站点节点 + 格点节点，$\mathcal{E} = \mathcal{E}_{\text{s-s}} \cup \mathcal{E}_{\text{g-s}}$ 为站点间边 + 格点到站点边。

**标准GNN消息传递**：
$$h_v^{(l+1)} = \phi\left(h_v^{(l)}, \bigoplus_{u \in \mathcal{N}(v)} \psi(h_u^{(l)}, e_{uv})\right)$$

其中：
- $h_v^{(l)}$：节点 $v$ 在第 $l$ 层的特征
- $\mathcal{N}(v)$：节点 $v$ 的邻居集合
- $\psi$：消息函数（通常为MLP）
- $\bigoplus$：聚合函数（sum, mean, max）
- $\phi$：更新函数（MLP）
- $e_{uv}$：边特征（如距离、方向）

**GraphCast Interaction Network**：
```
边更新: e_ij ← MLP_edge([e_ij, h_i, h_j])
节点聚合: m_j ← Σ_{i→j} e_ij
节点更新: h_j ← h_j + MLP_node([h_j, m_j])
残差连接: H_out = H + H', E_out = E + E'
```

**GraphCast图拉普拉斯（GNADET）**：
```
学习邻接矩阵:
  A_diff = A_mask ⊙ ReLU(tanh(α W₁W₁ᵀ))
  A_adv = A_mask ⊙ ReLU(tanh(α(W₁W₂ᵀ - W₂W₁ᵀ)))
图拉普拉斯:
  L_diff = D_diff - A_diffᵀ
  L_adv = D_adv - A_advᵀ
物理演化:
  dH/dt = -(L_diff H + L_adv H) + F_uncertainty(H)
```

**理论依据**：
- 大气是连续场，但观测是离散的
- 图结构自然表达不规则空间分布（如气象站）
- 消息传递机制模拟物理量的传播和扩散
- 图结构灵活适应不规则网格和多尺度结构
- **归纳偏置**：图结构编码空间邻域的物理意义
- **置换不变性**：聚合操作对邻居顺序不敏感

#### 【数据规格层】

**各流派图结构配置对比**：

| 模型 | 流派 | 节点类型 | 节点数 | 边构建方式 | 边数量 | 特殊设计 |
|------|------|---------|--------|---------|--------|---------|
| GraphCast | Graph | Icosahedral | 40,962 | 半径邻域+递归细分 | 327,660 | Multi-mesh（7级） |
| GenCast | Graph | Icosahedral | ~40,000 | k-hop邻域 | **未详细说明** | 图Transformer |
| AIFS | Graph | Reduced Gaussian | 40,320（O96） | GNN聚合 | N/A | 2级多尺度 |
| GNADET | Graph | 规则网格 | H×W | Moore邻域 | ~8×H×W | 物理拉普拉斯 |
| AGODE | Graph | 规则网格/粒子 | N | k-NN | N/A | 自适应边权重 |
| Local Off-Grid | Transformer | 异质（站点+格点） | 358+格点数 | Delaunay+最近邻 | ~1.6M | MPNN |
| NowcastNet | ODE | 规则网格 | H×W | 规则邻域 | 4×H×W | U-Net演化 |

**GraphCast Multi-Mesh详细结构**：
- Grid节点：`[N_grid, D_node]`，N_grid = 721 × 1440 = 1,038,240
- Mesh节点：`[N_mesh, D_node]`，N_mesh = 40,962
- Grid2Mesh边：`[E_g2m, D_edge]`，E_g2m ≈ 1.6M
- Mesh边：`[E_mesh, D_edge]`，E_mesh = 327,660
- Mesh2Grid边：`[E_m2g, D_edge]`，E_m2g = 3 × N_grid
- 分级：Level 0-2（粗，长程）→ Level 3-4（中）→ Level 5-6（细，局地）

**Local Off-Grid异质图（Transformer流派）**：
- 站点节点：$|\mathcal{V}_{\text{station}}| = 358$（MADIS站点）
- 格点节点：$|\mathcal{V}_{\text{grid}}| = H \times W$（ERA5/HRRR网格）
- 节点特征：站点 $\mathbf{h}_i \in \mathbb{R}^{d_{\text{station}}}$；格点 $\mathbf{h}_r \in \mathbb{R}^{d_{\text{grid}}}$
- 边连接：站点间（Delaunay三角剖分）；格点→站点（最近8个格点）

#### 【架构层】

**计算流程图（GraphCast，Graph流派）**：

```
输入: ERA5 Lat-Lon格点场 [B, C, H, W]
  ↓
展平为节点: [B, N_grid, C]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Encoder: Grid → Mesh                                         │
│   ├─ 对每个Mesh节点                                          │
│   │   └─ 连接半径内所有Grid节点                              │
│   ├─ 边特征: (距离, 方向向量)                                │
│   └─ 输出: Grid节点[M, D], Mesh节点[N_mesh, D]              │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Multi-Mesh Processor (16层GN)                                │
│   ├─ 边更新 (所有尺度Level 0-6):                             │
│   │   e_ij ← MLP([e_ij, h_i, h_j])                          │
│   ├─ 节点聚合 (跨尺度):                                       │
│   │   m_j ← Σ_{all levels} Σ_{i→j} e_ij                     │
│   └─ 节点更新:                                               │
│       h_j ← h_j + MLP([h_j, m_j])                           │
│   └─ 输出: [N_mesh, D_hidden]                                │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Decoder: Mesh → Grid                                         │
│   └─ 每个Grid节点从最近3个Mesh节点聚合                        │
│   └─ 输出: [N_grid, D] → reshape → [C, H, W]               │
└─────────────────────────────────────────────────────────────┘
  ↓
残差连接: x_{t+1} = x_t + Δz
```

**Local Off-Grid MPNN流程（Transformer流派）**：

```
输入：站点观测 + 格点预报
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. 节点编码                                                  │
│    f_i = Enc_station(w_i, p_i)                              │
│    h_r = Enc_grid(g_r, p_r)                                  │
│    输出: 节点特征                                            │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 消息传递（站点←站点）                                      │
│    μ_ij = β(f_i, f_j, Δp_ij)                                │
│    f_i += γ(f_i, AGG(μ))                                    │
│    输出: 更新后站点特征                                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 消息传递（站点←格点）                                      │
│    ν_ir = χ(h_r, f_i, Δp_ir)                                │
│    f_i += ω(f_i, h_r, AGG(ν))                               │
│    输出: 融合后站点特征                                      │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 节点解码                                                  │
│    ŵ_i = Dec(f_i)                                           │
│    输出: 站点预报 [N_station, 4]                             │
└─────────────────────────────────────────────────────────────┘
```

**NowcastNet神经演化流程（ODE流派）**：

```
历史雷达 [B, 9, H, W]
  ↓
共享Encoder (U-Net):
  ├─ Conv + Downsample × 4
  └─ 特征: [B, D, H/16, W/16]
  ↓
双路径Decoder:
  ├─ Motion Decoder: v[1:T] [B, T, 2, H, W]（运动场）
  └─ Intensity Decoder: s[1:T] [B, T, 1, H, W]（强度残差）
  ↓
神经演化算子:
  ├─ 对 t = 1..T:
  │   ├─ x'_t = Advect(x_{t-1}, v_t)
  │   └─ x_t = x'_t + s_t
  └─ 输出: [B, T, 1, H, W]
```

#### 【设计层】

**设计动机**：
- **不规则空间**：气象站点分布不均匀，无法用规则网格表示
- **多源融合**：需整合站点观测（局地、高精度）和格点预报（大尺度、平滑）
- **显式关系**：通过边显式建模空间邻近和物理关联
- **球面均匀**：正二十面体网格在球面上近似均匀分布

**痛点解决**：
- **问题**：Transformer对离网站点的全连接注意力计算成本高
- **解决**：GNN通过稀疏图结构降低复杂度
- **问题**：规则网格无法自然处理球面几何
- **解决**：Icosahedral网格 + 图结构

**创新突破（Graph流派为主）**：

- **GraphCast**：Encode-Process-Decode架构，Grid↔Mesh双向映射，10天预报超越IFS
- **GenCast**：将图神经网络与Transformer结合，在球面网格上实现高效的全球预报
- **GNADET**：学习的图拉普拉斯实现平流-扩散方程，可解释性强
- **AGODE**：自适应边权重，根据物理参数动态调整
- **AIFS**：Reduced Gaussian grid与GNN结合，兼顾均匀性和NWP兼容性

#### 【对比层】

**五流派图/节点表示应用情况（大型对比表）**：

| 模型 | 流派 | 图结构类型 | 图构建方式 | GNN类型 | 边更新 | 消息传递 | 特殊设计 | 适用场景 |
|------|------|-----------|-----------|---------|--------|---------|---------|---------|
| GraphCast | Graph | Icosahedral多尺度 | 正二十面体网格 | Interaction Network | 是 | 标准GNN | Multi-mesh | 全球中期预报 |
| GenCast | Graph | Icosahedral | k-hop邻域 | Graph Transformer | 是 | k-hop注意力 | 均匀采样 | 全球集合预报 |
| AIFS | Graph | Reduced Gaussian | GNN聚合 | Graph Attention | 是 | 注意力 | 2级多尺度 | 业务预报 |
| GNADET | Graph | 规则网格+Moore | Moore邻域 | Graph Transformer | 否 | 注意力 | 物理拉普拉斯 | 区域T2M |
| AGODE | Graph | k-NN | k-NN或距离阈值 | 自适应GNN | 动态权重 | 自适应 | 条件化边权 | PDE/流体 |
| Local Off-Grid | Transformer | 异质图 | Delaunay+最近邻 | MPNN | 否 | 标准MP | 站点+格点 | 局地订正 |
| NowcastNet | ODE | 规则网格 | 规则邻域 | U-Net | 否 | 卷积 | 神经演化算子 | 临近预报 |
| ClimODE | ODE | 规则网格 | 连续场 | Neural ODE | 否 | ODE积分 | 守恒约束 | 气候建模 |
| FengWu | Transformer | 无 | N/A | N/A | N/A | 注意力 | 多模态 | 全球预报 |
| FourCastNet | Fourier | 无 | N/A | N/A | N/A | FFT | 频域全局 | 快速预报 |

**Graph流派内GNN类型对比**：

| GNN类型 | 消息传递方式 | 计算成本 | 表达力 | 代表模型 |
|---------|------------|---------|--------|---------|
| Interaction Network | 边更新+节点聚合 | $O(E)$ | 高 | GraphCast |
| Graph Transformer | 多头注意力 | $O(N^2 \cdot h)$ | 最高 | GenCast, GNADET |
| Graph Attention | 加权聚合 | $O(N^2)$ | 高 | AIFS |
| 自适应GNN | 动态边权重 | 可变 | 最高 | AGODE |
| 标准MPNN | 固定聚合 | $O(E)$ | 中 | Local Off-Grid |

**Graph流派 vs Transformer性能对比（Local Off-Grid实验）**：

| 模型 | 风误差 (m/s) | T2m RMSE | 计算复杂度 |
|------|------------|----------|-----------|
| Transformer（全连接） | 0.48 | 最优 | $O(N^2)$ |
| GNN (MPNN) | 0.62 | 次优 | $O(E)$ |
| MLP（无空间） | 0.78 | 较差 | $O(N)$ |

**权衡评估**：

**GNN优势**：
- ✅ 适合不规则空间分布和稀疏观测
- ✅ 稀疏图降低计算成本
- ✅ 显式建模物理关联（边特征）
- ✅ 球面几何自然处理（Icosahedral）
- ✅ 多尺度建模（通过不同层级边）

**GNN劣势**：
- ❌ 性能略逊于全连接Transformer（Local Off-Grid任务）
- ❌ 图构造需要领域知识
- ❌ 难以大规模并行（相比Transformer）
- ❌ 需要Grid↔Mesh插值层（GraphCast）

**消息传递 vs 注意力**：
- **消息传递**：计算高效；难以捕获长程依赖
- **注意力**：更灵活但计算成本高（$O(N^2)$）

**固定图 vs 动态图**：
- **固定图**：训练稳定；适应能力有限
- **动态图**（AGODE）：适应性强但训练复杂

**适用场景**：
- **GNN**：适合不规则空间、多源融合、球面均匀性要求的全球预报
- **规则网格+Transformer**：适合规则网格、大规模、需要全局感受野的任务
- **规则网格+CNN/UNet**：适合区域任务、局部特征、U-Net天然适合多尺度

---

*（本文件涵盖【序言】【第一部分】【第二部分】【第三部分】【第四部分】，共计5大流派、21篇代表性论文的系统性跨流派融合分析。）*

---

## 【第三部分：主干网络类型范式】

### 3.1 基于Transformer系列

#### 【原理描述】

**核心定义**：Transformer通过自注意力机制建模序列中任意位置间的依赖关系，在气象AI中用于捕捉全球大气的长程空间相关性。不同变体通过窗口化、Fourier域处理等方式降低计算复杂度，从Swin窗口注意力到全局注意力再到MoE稀疏专家，构成气象AI领域最广泛的主干架构家族。

**数学推导**：

##### 1. 标准自注意力（Full Attention）

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q = XW_Q, \quad K = XW_K, \quad V = XW_V$
- $X \in \mathbb{R}^{N \times D}$ 为输入token序列
- $N$ 为token数量，$D$ 为隐藏维度，$d_k$ 为key维度

**复杂度**：$O(N^2 \cdot D)$，对全球气象网格（$N > 10^6$）不可行。

##### 2. 多头注意力（Multi-Head Attention）

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

其中 $h$ 为注意力头数，每个头关注不同的特征子空间。

##### 3. Swin Transformer窗口注意力

$$\text{W-MSA}(X) = \text{Attention}_{\text{window}}(X), \quad \text{SW-MSA}(X) = \text{Attention}_{\text{shifted}}(X)$$

W-MSA在 $M \times M$ 窗口内计算注意力，SW-MSA窗口平移 $\lfloor M/2 \rfloor$ 后计算，实现跨窗口信息流动：

- **窗口注意力**：$\text{Attention}_{\text{window}}: \mathbb{R}^{M^2 \times D} \rightarrow \mathbb{R}^{M^2 \times D}$
- **复杂度**：$O(M^2 \cdot N \cdot D)$，当 $M \ll \sqrt{N}$ 时显著降低
- 典型窗口大小：$M = 12$（Stormer）、$M = 8$（标准Swin）

##### 4. RoPE旋转位置编码

$$\text{RoPE}(x_m, m) = x_m \cdot e^{im\theta}$$

在气象场景中捕获相对位置关系：
- 2D RoPE：分别对经度维和纬度维应用旋转
- 优势：无需额外位置嵌入，直接注入相对位置信息

##### 5. AdaLN时间条件化

$$\mathbf{h}_i = \text{AdaLN}(\mathbf{x}_i, t) = \LayerNorm}\left(\mathbf{W}_{\text{scale}}(t) \cdot \mathbf{x}_i + \mathbf{W}_{\text{shift}}(t) + \mathbf{b}\right)$$

其中 $t$ 为时间步或扩散时间步，AERIS/Stormer用AdaLN实现时间条件化。

##### 6. MoE稀疏专家（EWMoE）

$$\mathbf{y} = \sum_{i=1}^{E} G(x)_i \cdot \text{Expert}_i(x)$$

其中：
- $E$ 为专家数量（如8-64个）
- $G(x) = \text{TopK}(\text{MLP}(x))$，每次激活 $K$ 个专家（稀疏）
- 路由函数 $G(x)$ 决定每个token分配给哪些专家
- **优势**：参数量大但计算成本低

**EWMoE配置**：
- 专家数：8-64个Swin专家
- 激活专家数：$K = 2$
- 容量因子：1.0-1.25

##### 7. AFNO的Fourier域处理（Fourier/Operator流派）

AFNO（Adaptive Fourier Neural Operator）是气象AI中最具代表性的Operator主干：

**2D离散Fourier变换**：
$$z_{m,n} = [\text{DFT}(\mathbf{X})]_{m,n} = \sum_{h=0}^{H-1}\sum_{w=0}^{W-1} \mathbf{X}_{h,w} e^{-2\pi i\left(\frac{mh}{H}+\frac{nw}{W}\right)}$$

**频域MLP + 软阈值化**：
$$\tilde{z}_{m,n} = S_\lambda(\text{MLP}(z_{m,n}))$$

软阈值函数（引入频域稀疏性）：
$$S_\lambda(x) = \text{sign}(x) \cdot \max(|x| - \lambda, 0)$$

**逆Fourier变换 + 残差连接**：
$$\mathbf{Y}_{m,n} = [\text{IDFT}(\tilde{\mathbf{Z}})]_{m,n} + \mathbf{X}_{m,n}$$

**AFNO vs 标准FNO的区别**：
- Block-diagonal MLP：在每个频率通道独立应用MLP，减少参数量
- 软阈值化：动态稀疏化频域表示，抑制噪声频率

**计算复杂度对比**：

| 注意力类型 | 复杂度 | 全局感受野 | 典型N（0.25°全球） | 实际复杂度 |
|-----------|--------|-----------|-------------------|-----------|
| 全局注意力 | $O(N^2 \cdot D)$ | 单层全局 | ~1M tokens | ~$10^{12}$（不可行） |
| 窗口注意力 | $O(M^2 \cdot N \cdot D)$ | 多层累积 | $M=12$ | ~$10^8$（可行） |
| AFNO（FFT） | $O(N\log N \cdot D)$ | 单层全局 | ~162k tokens | ~$10^6$（极快） |
| 稀疏注意力（MoE） | $O(E \cdot K \cdot N \cdot D)$ | 多层累积 | $E=64, K=2$ | ~$10^8$（可行） |

**理论依据**：
- 大气波动具有多尺度特性（从行星波到中尺度对流）
- 自注意力的全局感受野适合捕捉远程相关（如遥相关型）
- Fourier变换将空间域信号分解为频率分量，每个频率天然包含全局信息
- 窗口注意力通过多层堆叠逐步扩大感受野

#### 【数据规格层】

**输入输出形状对比（五流派）**：

| 模型 | 流派 | 输入形状 | Token数量N | D（隐藏维度） | 层数 | 注意力类型 | 头数h |
|------|------|---------|-----------|--------------|------|-----------|-------|
| Swin V2 | Transformer | `[B, 73, 720, 1440]` | (720/p)×(1440/p) | 768/1536 | 12/24 | 窗口注意力 | 8/16 |
| EWMoE | Transformer | `[B, 20, 721, 1440]` | (721/8)×(1440/8) | 768 | 6 | 全局+MoE | 8 |
| Stormer | Transformer | `[B, C, 128, 256]` | (128/p)×(256/p) | 1024 | 24 | 窗口注意力 | 16 |
| AERIS | Transformer | `[B, 70, 720, 1440]` | 720×1440 | **未详细说明** | **未详细说明** | 窗口注意力 | **未详细说明** |
| FengWu | Transformer | `[B, 189, 721, 1440]` | ~6480（p=8） | ~768 | 多 | 窗口注意力+cross-attn | **未说明** |
| FourCastNet | Fourier | `[B, 20, 721, 1440]` | (721/8)×(1440/8)≈16290 | 768 | 12 | FFT（频域MLP） | N/A |
| Pangu-Weather | Transformer | `[B, V, P, H, W]` | N_patches | ~384 | 16 | 3D窗口注意力 | **未说明** |
| AIFS | Graph | `[B, 40320, D]` | 40,320（O96网格） | ~512 | 16 | Shifted Window | 16 |
| TelePiT | Transformer | `[B, H, D_emb]` | H（纬度带） | ~512 | **未说明** | Teleconnection-aware | **未说明** |

**维度物理语义**：
- $N$：空间token数量，patch化后约为 $H/p \times W/p$
- $D$：隐藏维度，决定模型容量
- $h$：注意力头数，每个头关注不同的特征子空间（如不同变量、尺度的关系）

#### 【架构层】

**通用Transformer Block计算流程图（五流派统一框架）**：

```
输入 [B, N, D]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ LayerNorm / Pre-LN                                           │
│    x_norm = LayerNorm(x) 或 x（Pre-LN）                    │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 分支A: 注意力计算（三种变体）                                  │
│                                                           │
│ 【Transformer: 标准/窗口注意力】                             │
│   Q = x_norm W_Q, K = x_norm W_K, V = x_norm W_V          │
│   窗口: Attention_window = softmax(QK^T/√d_k)V              │
│   输出: attn_out [B, N, D]                                 │
│                                                           │
│ 【Transformer: 全局+MoE（EWMoE）】                          │
│   路由: G = TopK(MLP_router(x_norm), K)                    │
│   对激活的E个专家: y_e = Expert_e(x_norm)                  │
│   y = Σ G_e · y_e                                          │
│   输出: moe_out [B, N, D]                                  │
│                                                           │
│ 【Fourier: AFNO】                                          │
│   X [N, D] → reshape [h, w, D]                           │
│   Z = FFT2D(X) → [h, w, D]                                │
│   Z_reshaped = reshape [h·w, D]                          │
│   Z_mlp = MLP(Z_reshaped) → [h·w, D]                     │
│   Z_thresh = S_λ(Z_mlp)                                   │
│   Z_out = iFFT2D(reshape(Z_thresh))                       │
│   输出: afno_out [B, N, D]                                │
│                                                           │
│ 【Graph: AIFS纬向窗口注意力】                               │
│   Q, K, V投影: [B, N, D] → [B, N, h, d_k]                 │
│   沿纬向带划分窗口                                          │
│   窗口内: Attention = softmax(QK^T/√d_k)V                 │
│   输出: attn_out [B, N, D]                                │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 残差连接: x = x + α · attn_out                             │
│    （Transformer: α=1; Fourier: α<1的权重衰减）            │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ LayerNorm / Pre-LN                                           │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ FFN / MoE-FFN / Spectral FFN                                │
│   Transformer: FFN = Linear(D→4D) → GELU → Linear(4D→D)   │
│   Fourier: 无独立FFN（MLP在频域内完成）                     │
│   Graph: FFN = Linear(D→4D) → GELU → Linear(4D→D)         │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 残差连接: x = x + FFN(x)                                   │
└─────────────────────────────────────────────────────────────┘
  ↓
输出 [B, N, D]
```

**Stormer随机化动力学目标详细流程**：

```
【训练阶段】
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 随机采样时间步长: δt ~ U{6h, 12h, 24h}                    │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ AdaLN时间条件化                                              │
│   β, γ = time_embedding(δt)                               │
│   AdaLN(x, t) = LayerNorm(γ · x + β)                      │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 单步预测: X_{t+δt} = f_θ(X_t, δt)                          │
│ 损失: L = ||X_{t+δt} - X_true||²                          │
└─────────────────────────────────────────────────────────────┘

【推理阶段】
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 枚举时间步长组合                                             │
│   路径1: [6h, 6h, 6h, 6h] → 24h                            │
│   路径2: [12h, 12h] → 24h                                  │
│   路径3: [24h] → 24h                                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 多路径自回归滚动预报                                         │
│   for each path:                                            │
│     X_curr = X_0                                            │
│     for δt in path:                                         │
│       X_curr = f_θ(X_curr, δt)                             │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 集成平均: X_final = mean(all_paths)                        │
└─────────────────────────────────────────────────────────────┘
```

**AERIS扩散Transformer详细流程**：

```
【训练】
  真实状态: x_0 ~ p_data
    ↓
  采样: z ~ N(0, I), t ~ U[0, 1]
    ↓
  加噪: x_t = cos(t·π/2) · x_0 + sin(t·π/2) · z
    ↓
  条件输入: [x_t, x_{i-1}, forcings]
    ↓
  Swin Transformer + AdaLN(t)
    ↓
  预测: v_t = F_θ(x_t/σ, t)
    ↓
  损失: ||σ · F_θ(x_t/σ, t) - v_t||²

【推理】
  初始噪声: x_{π/2} ~ N(0, σ²)
    ↓
  PF-ODE积分 (DPM-Solver++ 10步):
    for step in 1..10:
      x_{t-Δt} = x_t - Δt · dF/dt
    ↓
  生成样本: x_0
    ↓
  状态还原: x_i = x_0 + x_{i-1}
```

**AIFS Pre-norm Transformer详细流程（Graph流派）**：

```
输入: O96网格token [B, N=40320, D]
  ↓
Layer l (共16层):
  ├─ LayerNorm
  ├─ Shifted Window Attention (纬向带):
  │   ├─ Q, K, V投影: [B, N, D] → [B, N, h, d_k]
  │   ├─ 沿纬向带划分窗口
  │   ├─ 窗口内: Attention = softmax(QK^T/√d_k)V
  │   └─ 输出: [B, N, D]
  ├─ 残差连接
  ├─ LayerNorm
  ├─ FFN (GELU):
  │   └─ Linear(D → 4D) → GELU → Linear(4D → D)
  └─ 残差连接
    ↓
输出: [B, N=40320, D]
```

**TelePiT Teleconnection-Aware Attention详细流程**：

```
输入: 纬向token [B, H, D_emb]
  ↓
全局状态: x̄ = mean(X, dim=1) ∈ R^{B×D_emb}
  ↓
Teleconnection向量计算:
  ├─ ω = softmax(W_p · x̄) ∈ R^{B×M}
  ├─ c = Σ ω_j · P_j ∈ R^{B×D_emb}
  │    (P_j为EOF模式基函数)
  └─ q_tel = c · W^Q ∈ R^{B×D_emb}
    ↓
标准注意力 + Teleconnection偏置:
  ├─ logits = QK^T/√d_k
  ├─ bias = q_tel · K/√d_k
  └─ attn = softmax(logits + λ·bias) ∈ R^{B×H×H}
    ↓
输出: attn · V ∈ R^{B×H×D_emb}
```

#### 【设计层】

**设计动机**：
- **全局感受野**：捕捉远程大气遥相关（如ENSO对东亚季风的影响）
- **计算效率**：窗口注意力或FFT降低复杂度
- **架构成熟**：利用NLP/CV领域的成功经验
- **动态权重**：根据当前状态调整不同区域的重要性

**痛点解决（五流派对照）**：

| 问题 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **全局注意力成本高** | 窗口注意力（Swin） | FFT（$O(N\log N)$） | Shifted Window（AIFS） | 多层堆叠 | N/A |
| **模型容量不足** | MoE稀疏专家（EWMoE） | 深度堆叠（12层） | GNN+Transformer | UNet深层 | 状态维度 |
| **时间条件化弱** | AdaLN（Stormer, AERIS） | 无显式条件 | 可学习时间嵌入 | 扩散时间步 | 连续时间t |
| **变量交互隐式** | cross-attn（Stormer） | 隐式（FFT+MLP） | 边特征更新 | skip connections | 共享状态 |
| **内存占用高** | 窗口化 | FFT（10GB vs 83GB） | GNN稀疏 | 分辨率限制 | ODE求解器 |
| **长期稳定性** | 多步微调 | 两步微调 | 自回归 | 扩散天然稳定 | ODE积分 |

**创新突破（五流派）**：

- **Transformer流派**：
  - Swin V2：证明最小修改的2D Swin即可达到SOTA，非层级结构在气象任务上优于层级结构
  - EWMoE：首次在气象中应用MoE，2年数据达SOTA，稀疏激活降低计算成本
  - Stormer：随机化动力学目标 + AdaLN + 多路径集成
  - AERIS：像素级Swin + 扩散模型，1.3B-80B参数，10ExaFLOPS训练

- **Fourier/Operator流派**：
  - FourCastNet：首次将FNO从PDE求解引入全球天气预报
  - AFNO：block-diagonal MLP + 软阈值化，引入自适应频域稀疏化
  - SFNO：球谐FNO，天然满足旋转对称性

- **Graph流派**：
  - AIFS：Shifted window attention沿纬向带，适配球面几何
  - TelePiT：Teleconnection-aware attention，显式建模气候模态（EOF模式）

- **Generative流派**：
  - PreDiff：Earthformer-UNet，3D Cuboid Attention替代标准注意力
  - CorrDiff：UNet扩散，去噪过程在隐空间进行

#### 【对比层】

**五流派共性归纳**：
- 均基于自注意力机制或其数学变体（FFT、消息传递、ODE）
- 均使用残差连接和归一化（LayerNorm / Pre-LN / RMSNorm）
- 均采用编码器-解码器或编码器-only结构
- 多头机制广泛使用（GELU激活函数）
- 层间累积感受野逐步扩大

**五流派差异分析（大型对比表）**：

| 维度 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **核心操作** | softmax(QK^T/√d)V | FFT + 频域MLP | 消息传递 | 去噪UNet | ODE积分 |
| **注意力范围** | 窗口/全局 | 全局（频域） | 局部k-hop | 局部（UNet） | 无 |
| **位置编码** | 2D/3D绝对或RoPE | Fourier基函数 | xyz坐标+边拓扑 | 正弦/绝对 | 无 |
| **FFN类型** | 标准FFN或MoE | 无独立FFN | 标准FFN | ResBlock | Neural网络 |
| **归一化** | LayerNorm/RMSNorm | LayerNorm | LayerNorm | GroupNorm | 无 |
| **时间条件化** | AdaLN（部分模型） | 无 | 可学习嵌入 | 扩散时间步t | 连续时间t |
| **复杂度** | $O(M^2N)$~$O(N^2)$ | $O(N\log N)$ | $O(E)$ | 层数相关 | ODE求解 |
| **显存占用** | 中等至高 | 极低（10GB） | 中等 | 取决于UNet | 低 |
| **参数量** | 137M-80B | 适中（~100M） | 36.7M | 80M | 适中 |
| **感受野扩展** | 多层堆叠 | 单层全局 | 多跳传播 | 多层UNet | 积分时长 |
| **长程依赖** | 灵活但昂贵 | 天然全局 | 多跳可达 | 有限 | 积分控制 |
| **推理速度** | 中等 | 极高 | 快 | 慢 | 快 |
| **典型配置** | 12-24层, D=768-1536 | 12层, D=768 | 16层, D=512 | 6层UNet | RK4积分 |

**五流派代表性模型性能对比**：

| 模型 | 流派 | Z500 ACC (3天) | T2m RMSE (3天) | 训练数据 | 参数量 | 显存占用 |
|------|------|----------------|----------------|---------|--------|---------|
| Swin V2 | Transformer | 优于IFS | 优于IFS | 1979-2015 | 137M | **未说明** |
| EWMoE | Transformer | 与Pangu相当 | 与Pangu相当 | 2015-2016（2年） | **未说明** | **未说明** |
| Stormer | Transformer | 优于Pangu | 优于Pangu | 1979-2018 | **未说明** | **未说明** |
| AERIS | Transformer | 与IFS ENS相当 | 与IFS ENS相当 | 1979-2018 | 1.3B-80B | **未说明** |
| FengWu | Transformer | ~0.98 | 优于GraphCast | **未说明** | **未说明** | **未说明** |
| FourCastNet | Fourier | 与IFS相当 | 与IFS相当 | 1979-2015 | **未说明** | ~10GB |
| GraphCast | Graph | 超越IFS（10天） | 优于IFS | **未说明** | 36.7M | **未说明** |
| IFS HRES | 物理模式 | ~0.98 | 基准 | 物理模式 | **未说明** | **未说明** |

**权衡评估**：

**Transformer流派优势**：
- ✅ 全局感受野（多层堆叠），适合大气遥相关
- ✅ 架构灵活，易于扩展（MoE、扩散、时间条件化）
- ✅ ���行化友好
- ✅ 窗口注意力在高分辨率下可行

**Transformer流派劣势**：
- ❌ 计算成本高（需窗口化或稀疏化）
- ❌ 缺乏显式物理约束
- ❌ 需要大量训练数据

**Fourier/Operator流派优势**：
- ✅ 计算和内存效率极高（$O(N\log N)$，10GB显存）
- ✅ 单层即具有全局感受野
- ✅ 频域表示具有物理可解释性（不同频率对应不同尺度的大气波动）
- ✅ 软阈值化机制天然抑制噪声频率

**Fourier/Operator流派劣势**：
- ❌ 假设空间平移等变性，对强非均匀场景可能不够灵活
- ❌ FFT在极点附近的周期性假设与球面几何不完全匹配
- ❌ 频域MLP的表达能力可能不如全连接注意力

**Graph流派优势**：
- ✅ 稀疏图计算高效，$O(E)$
- ✅ 边特征可编码几何信息（距离、方向）
- ✅ 多跳消息传递实现 $O(\log N)$ 层全球通信
- ✅ 球面几何自然处理（Icosahedral）

**Generative流派优势**：
- ✅ 自然生成概率分布，量化不确定性
- ✅ 扩散过程天然稳定（90天预报）
- ✅ UNet保留多尺度细节

**ODE流派优势**：
- ✅ 连续时间建模，物理一致性最强
- ✅ 支持任意时刻查询
- ✅ 守恒约束自然编码

**适用场景**：
- **Transformer**：适合需要灵活建模、多尺度特征、长期预报（>7天）的场景
- **Fourier/Operator**：适合需要极高计算效率、快速推理、中短期预报（1-7天）的场景
- **Graph**：适合需要球面均匀性、极区友好、全球中期预报的场景
- **Generative**：适合需要集合预报、长期预报（>30天）、不确定性量化的场景
- **ODE**：适合需要物理一致性、连续时间外推的场景

---

### 3.2 基于图神经网络（GNN）

#### 【原理描述】

**核心定义**：将大气场或观测网络表示为图结构，节点对应空间位置或观测点，边表示物理关联或空间邻近关系，通过图神经网络（GNN）的消息传递机制在节点间传播信息，适合不规则空间分布和多尺度建模，是Graph流派的主干架构。

**数学推导**：

##### 1. 标准消息传递框架

$$\mathbf{m}_{j \to i}^{(l)} = \phi_e^{(l)}(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ij})$$

消息聚合：
$$\mathbf{m}_j^{(l)} = \bigoplus_{i \in \mathcal{N}(j)} \mathbf{m}_{j \to i}^{(l)}$$

节点更新：
$$\mathbf{h}_j^{(l+1)} = \phi_v^{(l)}(\mathbf{h}_j^{(l)}, \mathbf{m}_j^{(l)})$$

其中：
- $\mathbf{h}_i^{(l)}$：节点 $i$ 在第 $l$ 层的隐藏状态
- $\mathbf{e}_{ij}$：边 $(i,j)$ 的特征（距离、方向等）
- $\phi_e$：边更新函数（通常为MLP）
- $\phi_v$：节点更新函数（通常为MLP）
- $\bigoplus$：聚合函数（sum / mean / max / attention）

##### 2. 图卷积网络（GCN）

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \frac{1}{\sqrt{d_v d_u}} W^{(l)} h_u^{(l)}\right)$$

其中 $d_v, d_u$ 为节点度数，$W^{(l)}$ 为可学习权重矩阵。

##### 3. 图注意力网络（GAT）

$$\alpha_{uv} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[W h_u \| W h_v]))}{\sum_{k \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(\mathbf{a}^T[W h_k \| W h_v]))}$$

$$h_v' = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{uv} W h_u\right)$$

##### 4. GraphCast Interaction Network（核心架构）

**边更新**：
$$\mathbf{e}_{ij} \leftarrow \text{MLP}_{\text{edge}}([\mathbf{e}_{ij}, \mathbf{h}_i, \mathbf{h}_j])$$

**节点聚合**：
$$\mathbf{m}_j \leftarrow \sum_{i \to j} \mathbf{e}_{ij}$$

**节点更新**：
$$\mathbf{h}_j \leftarrow \mathbf{h}_j + \text{MLP}_{\text{node}}([\mathbf{h}_j, \mathbf{m}_j])$$

**残差连接**：
$$\mathbf{H}_{\text{out}} = \mathbf{H} + \mathbf{H}', \quad \mathbf{E}_{\text{out}} = \mathbf{E} + \mathbf{E}'$$

##### 5. 物理引导图拉普拉斯（GNADET）

**学习的扩散邻接矩阵**：
$$A_{\text{diff}} = A_{\text{mask}} \odot \text{ReLU}(\tanh(\alpha \mathbf{W}_1 \mathbf{W}_1^T))$$

**学习的平流邻接矩阵**：
$$A_{\text{adv}} = A_{\text{mask}} \odot \text{ReLU}(\tanh(\alpha(\mathbf{W}_1 \mathbf{W}_2^T - \mathbf{W}_2 \mathbf{W}_1^T)))$$

**图拉普拉斯**：
$$L_{\text{diff}} = D_{\text{diff}} - A_{\text{diff}}^T, \quad L_{\text{adv}} = D_{\text{adv}} - A_{\text{adv}}^T$$

**物理演化方程**：
$$\frac{d\mathbf{H}}{dt} = -(L_{\text{diff}} \mathbf{H} + L_{\text{adv}} \mathbf{H}) + F_{\text{uncertainty}}(\mathbf{H})$$

##### 6. 自适应GNN（AGODE）

边权重根据状态动态调整：
$$\alpha_{ij} = \frac{\exp(\mathbf{a}^T \cdot [\mathbf{h}_i, \mathbf{h}_j, \mathbf{e}_{ij}])}{\sum_{k \in \mathcal{N}(i)} \exp(\mathbf{a}^T \cdot [\mathbf{h}_i, \mathbf{h}_k, \mathbf{e}_{ik}])}$$

$$\mathbf{e}_{ij} \leftarrow \alpha_{ij} \cdot \mathbf{e}_{ij}$$

##### 7. 复杂度分析

| GNN类型 | 消息传递复杂度 | 边更新复杂度 | 全局感受野 |
|---------|-------------|------------|-----------|
| 标准MPNN | $O(E \cdot D)$ | 无 | $O(L)$ 跳 |
| Interaction Network | $O(E \cdot D)$ | $O(E \cdot D)$ | $O(L)$ 跳 |
| GAT | $O(E \cdot D)$ | 无 | $O(L)$ 跳 |
| k-hop Attention | $O(k \cdot E \cdot D)$ | 无 | $O(k)$ 跳 |
| 自适应GNN | $O(E \cdot D)$ | $O(E \cdot D)$ | $O(L)$ 跳 |

**理论依据**：
- 大气是连续场，但观测是离散的——图结构自然桥接这一矛盾
- 消息传递机制模拟物理量的传播和扩散
- 图的拓扑结构可编码物理邻近性和动力学关联
- **归纳偏置**：图结构编码空间邻域的物理意义
- **置换不变性**：聚合操作对邻居顺序不敏感
- **局部性**：每层只传播到直接邻居，多层实现长程传播

#### 【数据规格层】

**输入输出张量形状对比**：

| 模型 | 流派 | 节点特征形状 | 边索引形状 | 边特征形状 | 节点数N | 边数E | 隐藏维度 |
|------|------|------------|---------|---------|--------|-------|---------|
| GraphCast | Graph | `[B, N_mesh, D]` | `[2, E_mesh]` | `[E_mesh, D_edge]` | 40,962 | 327,660 | ~256 |
| GenCast | Graph | `[B, N_mesh, D]` | `[2, E]` | `[E, D_edge]` | ~40,000 | **未详细** | ~256 |
| AIFS | Graph | `[B, N, D]` | - | - | 40,320（O96） | N/A | ~512 |
| GNADET | Graph | `[B, N, D]` | `[2, E]` | - | H×W | ~8×H×W | **未说明** |
| AGODE | Graph | `[B, N, D]` | `[2, E]` | `[E, D]`（可学习） | N | N/A | **未说明** |
| Local Off-Grid | Transformer | `[B, N, D]` | `[2, E]` | `[E, D_edge]` | 358+格点 | ~1.6M | **未说明** |
| NowcastNet | ODE | `[B, T, D, H, W]` | - | - | H×W | 4×H×W | **未说明** |

**GraphCast Multi-Mesh详细张量规格**：
- Grid节点特征：`[B, N_grid=1038240, D_node]`
- Mesh节点特征：`[B, N_mesh=40962, D_node]`
- Grid2Mesh边索引：`[2, E_g2m≈1.6M]`
- Grid2Mesh边特征：`[B, E_g2m, D_edge]`
- Mesh边索引：`[2, E_mesh=327660]`
- Mesh边特征：`[B, E_mesh, D_edge]`
- Mesh2Grid��索引：`[2, E_m2g=3114520]`
- 多尺度层级：Level 0-2（粗，长程，跨半球通信）→ Level 3-4（中）→ Level 5-6（细，局地）

**维度物理语义**：
- `N`：空间采样点总数
- `D_node`：节点特征维度（多变量×多层+位置编码）
- `E`：边数，取决于邻域定义（k-NN、距离阈值、网格拓扑）
- `D_edge`：边特征维度（距离、方向等几何信息）
- 1-hop：局部邻近（~数百km）
- k-hop：远程依赖（~数千km）

#### 【架构层】

**通用GNN Block计算流程图**：

```
输入: 节点特征 H [B, N, D_v], 边特征 E [B, E, D_e], 边索引 [2, E]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: 边更新（可选，如Interaction Network）              │
│   for each edge (i→j):                                     │
│     e_ij ← MLP_edge([e_ij, h_i, h_j])                    │
│   输出: E' [B, E, D_e]                                    │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: 消息聚合                                           │
│   for each node j:                                         │
│     ├─ 收集所有入边消息: {m_ij | i→j}                     │
│     ├─ m_j = AGGREGATE({m_ij})                           │
│     │   ├─ sum: m_j = Σ m_ij                             │
│     │   ├─ mean: m_j = Σ m_ij / |N(j)|                   │
│     │   ├─ max: m_j = max_i m_ij                        │
│     │   └─ attention: m_j = Σ α_ij · m_ij               │
│     └─ 输出: 聚合消息 m_j [B, D_e]                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: 节点更新                                           │
│   for each node j:                                         │
│     h_j ← MLP_node([h_j, m_j])                            │
│     h_j ← h_j + residual(h_j_old)                        │
│   输出: H' [B, N, D_v]                                    │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: LayerNorm                                          │
│   H_out = LayerNorm(H')                                    │
└─────────────────────────────────────────────────────────────┘
  ↓
重复上述过程 L 次（感受野逐步扩大）
  ↓
输出: [B, N, D_v]
```

**GraphCast Interaction Network详细流程图**：

```
输入: 节点 H [B, N_mesh, D_v], 边 E [B, E_mesh, D_e], 边索引 [2, E_mesh]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Edge Update (所有尺度 Level 0-6 并行):                       │
│   for each edge (i→j):                                     │
│     e_ij ← MLP([e_ij, h_i, h_j])                         │
│   ├─ Level 0-2 边: 长程连接                               │
│   ├─ Level 3-4 边: 中程连接                               │
│   └─ Level 5-6 边: 短程连接                               │
│   输出: E' [B, E_mesh, D_e]                               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Node Aggregation (跨所有尺度):                               │
│   for each node j:                                         │
│     m_j ← Σ_{i→j, all levels} e_ij                       │
│   输出: m_j [B, D_e]                                      │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Node Update:                                                │
│   for each node j:                                         │
│     h_j ← h_j + MLP([h_j, m_j])                          │
│   输出: H' [B, N_mesh, D_v]                               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 残差连接:                                                   │
│   H_out = H + H'                                           │
│   E_out = E + E'                                           │
└─────────────────────────────────────────────────────────────┘
  ↓
重复16次（GraphCast典型配置）
  ↓
输出: [B, N_mesh, D_v]
```

**GNADET物理约束图拉普拉斯详细流程图**：

```
输入: 节点特征 H [B, N, D]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. 学习邻接矩阵                                              │
│   ├─ A_diff = A_mask ⊙ ReLU(tanh(α W₁W₁ᵀ))               │
│   │    (扩散邻接：编码空间平滑性)                          │
│   └─ A_adv = A_mask ⊙ ReLU(tanh(α(W₁W₂ᵀ - W₂W₁ᵀ)))       │
│        (平流邻接：编码方向性传输)                           │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 图拉普拉斯                                               │
│   ├─ L_diff = D_diff - A_diffᵀ                            │
│   │    D_diff = diag(Σ A_diff)                           │
│   └─ L_adv = D_adv - A_advᵀ                              │
│        D_adv = diag(Σ A_adv)                             │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 物理演化                                                 │
│   ├─ 平流项: -L_adv H (数据驱动)                          │
│   ├─ 扩散项: -L_diff H (数据驱动)                          │
│   └─ 不确定性项: F_uncertainty(H)                         │
│   dH/dt = -(L_diff H + L_adv H) + F_uncertainty(H)        │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 离散化时间积分                                            │
│   H(t+Δt) = H(t) + Δt · dH/dt                            │
└─────────────────────────────────────────────────────────────┘
```

**AGODE自适应边权重流程图**：

```
输入: 节点 H [B, N, D], 边 E [B, E, D], 边索引 [2, E]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 动态��权重计算                                              │
│   for each edge (i→j):                                    │
│     α_ij = softmax_j(MLP_att([h_i, h_j, e_ij]))          │
│     e_ij ← α_ij · e_ij                                   │
│   输出: E' [B, E, D]（权重归一化）                        │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 标准消息传递                                                │
│   m_ij = f_msg(e_ij, h_i)                                │
│   m_j = Σ_{i→j} m_ij                                     │
│   h_j ← f_update(h_j, m_j)                                │
└─────────────────────────────────────────────────────────────┘
```

**GraphCast Encoder-Decoder（Grid↔Mesh）详细流程**：

```
【Encoder: Grid → Mesh】
  Lat-Lon网格节点 [B, N_grid=1038240, C]
    ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Grid2Mesh边构建                                         │
  │   ├─ 对每个Mesh节点                                     │
  │   │   └─ 连接半径r内所有Grid节点                       │
  │   │       └─ 边特征: (Δx, Δy, Δz, 距离)              │
  │   ├─ 边数: ~1.6M                                        │
  │   └─ 边索引: [2, E_g2m]                               │
  │   └─ 边特征: [E_g2m, D_edge]                           │
  │   └─ Grid节点: [N_grid, D]                            │
  └─────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────┐
  │ GNN消息传递 (Grid → Mesh):                              │
  │   for l in 1..L:                                        │
  │     e_ij ← MLP([e_ij, h_i, h_j])                      │
  │     m_j ← Σ_{i→j} e_ij                                │
  │     h_j ← h_j + MLP([h_j, m_j])                       │
  └─────────────────────────────────────────────────────────┘
    ↓
  输出: Mesh节点特征 [B, N_mesh=40962, D]

【Decoder: Mesh → Grid】
  Mesh节点特征 [B, N_mesh=40962, D]
    ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Mesh2Grid边构建                                         │
  │   ├─ 对每个Grid节点                                     │
  │   │   └─ 连接最近3个Mesh节点（所在三角面顶点）          │
  │   ├─ 边数: 3 × N_grid                                 │
  │   └─ 输出: Grid节点 [N_grid, D] ← Mesh节点 [N_mesh, D]│
  └─────────────────────────────────────────────────────────┘
    ↓
  输出: [B, N_grid, D] → reshape → [B, C, H, W]

【残差连接】
  x_{t+1} = x_t + Δx（GraphCast/GenCast采用）
```

#### 【设计层】

**设计动机**：
- **几何灵活性**：GNN天然支持不规则网格和多尺度结构
- **物理邻域**：边连接反映物理上的空间相关性（平流、波动传播）
- **可解释性**：消息传递对应物理量的扩散/平流过程
- **球面均匀性**：Icosahedral网格无极点，节点分布近似均匀
- **长程依赖**：通过多跳消息传递或显式长程边建模遥相关

**痛点解决**：
- **不规则网格**：GNN无需规则采样，适合自适应网格
- **球面几何**：通过图边定义球面距离关系，无需极区处理
- **多尺度**：通过不同层级的边实现从局地到全球的传播
- **物理约束**：可以在消息函数中注入物理定律（GNADET的图拉普拉斯）
- **可扩展性**：可以轻松调整邻域大小（k-hop）

**创新突破**：

- **GraphCast**：Encode-Process-Decode架构，Grid↔Mesh双向映射，在10天预报上超越IFS，推理仅需1分钟；首次系统性使用正二十面体网格
- **GenCast**：将图神经网络与Transformer结合，在球面网格上实现高效的全球预报；k-hop邻域注意力
- **GNADET**：学习的图拉普拉斯实现平流-扩散方程，物理可解释性强；扩散邻接+平流邻接分别编码
- **AGODE**：自适应边权重，根据物理参数动态调整边连接强度
- **AIFS**：Shifted window attention沿纬向带 + GNN编码/解码，适配球面几何
- **Local Off-Grid**（Transformer流派）：异质图（站点+格点）MPNN，首次用GNN做局地天气订正

#### 【对比层】

**五流派共性归纳**：
- 均使用消息传递 + 节点更新的框架
- 边特征包含几何信息（距离、方向）
- 多层堆叠扩大感受野
- 通常与其他架构（Transformer、CNN）结合使用

**五流派差异分析（大型对比表）**：

| 维度 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **核心操作** | softmax注意力 | FFT+频域MLP | 消息传递 | 去噪ResBlock | ODE积分 |
| **邻域定义** | 窗口/k-NN | 全局（频域） | 拓扑/k-hop | 局部（卷积） | 网格邻域 |
| **边特征** | 无 | 无 | 距离/方向 | 无 | 网格索引 |
| **边更新** | 无 | 无 | 是/否 | 无 | 无 |
| **感受野** | 多层累积 | 单层全局 | 多跳传播 | 多层UNet | 积分步数 |
| **计算复杂度** | $O(M^2N)$~$O(N^2)$ | $O(N\log N)$ | $O(E)$ | $O(CHW)$ | $O(N)$ |
| **参数量** | 大（注意力参数） | 中（MLP） | 中（GNN层） | 取决于UNet | 中（ODE函数） |
| **显存占用** | 高 | 低 | 中 | 中 | 低 |
| **全局建模** | 多层堆叠 | FFT天然 | 多跳传播 | 多层堆叠 | 积分控制 |
| **极区处理** | 需额外处理 | 周期性填充 | 无需 | 区域模型 | 避免 |
| **空间均匀性** | 不均匀（经纬网格） | 不均匀 | 均匀（icosahedral） | 均匀 | 不均匀 |

**Graph流派内GNN类型深度对比**：

| 模型 | GNN类型 | 边更新 | 聚合方式 | 多尺度设计 | 特殊机制 | 节点数 | 边数 |
|------|---------|--------|---------|-----------|---------|-------|------|
| GraphCast | Interaction Network | 是（MLP） | Sum | Multi-mesh（7级） | Grid↔Mesh映射 | 40,962 | 327,660 |
| GenCast | Graph Transformer | 是 | Attention | Icosahedral | k-hop邻域注意力 | ~40,000 | **未详细** |
| AIFS | Graph Attention | 是 | Attention | 2级（网格） | Shifted Window | 40,320 | N/A |
| GNADET | Graph Transformer | 否 | Attention | Moore邻域（8邻域） | 物理拉普拉斯 | H×W | ~8×H×W |
| AGODE | 自适应GNN | 动态权重 | MLP | k-NN | 条件化边权 | N | N/A |
| Local Off-Grid | MPNN | 否 | Sum | Delaunay+最近邻 | 异质图（站点+格点） | 358+格点 | ~1.6M |

**GNN vs Transformer vs CNN性能对比**：

| 模型 | 架构 | 精度 | 计算复杂度 | 可解释性 | 适用场景 |
|------|------|------|-----------|---------|---------|
| GraphCast | GNN | 超越IFS | $O(E)$ | 中（边特征可查） | 全球中期 |
| Transformer | 全连接注意力 | 最优 | $O(N^2)$ | 低（黑盒） | 规则网格 |
| GNN（Local Off-Grid） | MPNN | 次优 | $O(E)$ | 高（显式消息） | 局地订正 |
| CNN（U-Net） | 卷积 | 良好 | $O(CHW)$ | 低 | 区域/临近预报 |

**Graph流派vs其他流派的关键权衡**：

| 维度 | Graph流派 | Transformer流派 | Fourier/Operator流派 | Generative流派 | ODE流派 |
|------|---------|---------------|-------------------|--------------|--------|
| **图构造开销** | 高（需预定义拓扑） | 无 | 无 | 无 | 无 |
| **极区友好度** | ★★★★★ | ★★☆☆☆ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ |
| **实现复杂度** | 高（需图操作库） | 中 | 中 | 中 | 中 |
| **推理效率** | 快（稀疏计算） | 中 | 极快 | 慢 | 快 |
| **物理约束注入** | ★★★★★ | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| **不规则数据支持** | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ |
| **大规模并行** | ★★★☆☆ | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★☆ |

**权衡评估**：

**GNN优势**：
- ✅ 适合不规则空间分布（气象站、卫星轨道等）
- ✅ 稀疏图计算高效（$O(E)$，远低于 $O(N^2)$）
- ✅ 显式建模物理关联（边特征可编码距离、方向、拓扑）
- ✅ 球面几何自然处理（Icosahedral无极点奇异性）
- ✅ 多尺度建模（通过不同层级边）

**GNN劣势**：
- ❌ 图构造开销：需要预定义图拓扑，网格转换复杂
- ❌ 性能在规则网格任务中可能不如专门设计的Transformer
- ❌ 大规模并行化困难（消息传递的依赖关系限制并行度）
- ❌ 需要专门的图操作库（PyG、DGL等）

**边更新 vs 无边更新**：
- **边更新（GraphCast）**：提升表达力，但计算量增加 $O(E)$ 的MLP开销
- **无边更新（GNADET）**：计算高效，但表达力受限

**消息传递 vs 注意力聚合**：
- **消息传递**：计算高效，适合规则邻域；难以捕获长程依赖
- **注意力聚合**：更灵活，但注意力权重计算 $O(E \cdot D)$ 额外开销

**固定图 vs 动态图**：
- **固定图**：训练稳定，收敛快；适应能力有限
- **动态图（AGODE）**：适应性强但训练复杂，可能不稳定

**适用场景**：
- **GNN**：全球球面预报（GraphCast/GenCast）、局地订正（Local Off-Grid）、不规则观测网络、需物理约束的建模
- **Transformer**：规则网格、大规模并行、需灵活全局建模
- **Fourier/Operator**：快速推理、计算资源受限、中短期预报
- **Generative**：概率预报、极端事件、长期稳定性
- **ODE**：物理一致性、连续时间、长期演化建模

---

### 3.3 其他主干架构（Operator, Generative, ODE等）

#### 【原理描述】

本节覆盖三大补充主干架构：Fourier/Operator神经网络（频域算子学习）、Generative扩散模型（概率生成）、以及Neural ODE/连续建模（物理微分方程）。这些架构共同构成气象AI的完整技术版图。

##### 3.3.1 Fourier/Operator神经网络

**核心定义**：通过Fourier变换将气象场映射到频域，在频域执行全局算子学习（通过MLP或卷积），再逆变换回空间域。算子学习的目标是逼近从输入函数空间到输出函数空间的非线性映射。

**SFNO（Spherical Fourier Neural Operator）数学推导**：

**球谐变换（SHT）**：
$$\hat{f}_l^m = \int_S f(\theta, \phi) \cdot \overline{Y_l^m(\theta, \phi)} \, dS$$

**球面Fourier核**：
$$\mathcal{K}_\theta: L^2(S^2) \rightarrow L^2(S^2)$$
通过在球谐域逐系数应用可学习的线性变换：
$$\widehat{(\mathcal{K}_\theta f)}_l^m = \kappa_l^m \cdot \hat{f}_l^m$$

其中 $\kappa_l^m$ 是可学习的谱系数（按球谐阶数 $l$ 和次数 $m$ 索引）。

**完整SFNO层**：
$$f \mapsto \sigma\left(\mathcal{K}_\theta(f)\right)$$

其中 $\sigma$ 为非线性激活函数（如GELU）。

**复杂度**：
- SHT正变换：$O(N \cdot L^2)$ 或使用FFT加速
- 谱域操作：$O(L^2)$（仅前 $L$ 个球谐阶数）
- SHT逆变换：$O(N \cdot L^2)$
- 总复杂度：$O(N \cdot L^2)$，其中 $N$ 为空间点数，$L$ 为截断阶数

**SFNO vs AFNO**：

| 维度 | SFNO（球谐FNO） | AFNO（自适应FNO） |
|------|--------------|-----------------|
| 变换类型 | 球谐变换（SHT） | 2D离散Fourier变换（DFT） |
| 基函数 | 球谐基 $Y_l^m(\theta,\phi)$ | Fourier基 $e^{2\pi i(kx+ly)}$ |
| 域 | 球面 $S^2$ | 平面/经纬网格 |
| 旋转对称性 | 天然满足 | 需要特殊处理 |
| 极区处理 | 自然（无奇异性） | 需周期性填充/加权 |
| 截断 | 按球谐阶数 $l$ 截断 | 按波数 $k$ 截断 |
| 适用网格 | 任意（需SHT实现） | 规则Lat-Lon |
| 复杂度 | $O(N \cdot L^2)$ | $O(N\log N)$ |
| 长期稳定性 | 高（谱域正则化） | 中（软阈值化正则化） |

##### 3.3.2 Generative扩散模型

**核心定义**：扩散概率模型通过逐步加噪和去噪学习数据分布，生成高质量的气象场预报样本。扩散模型天然生成概率分布，支持集合预报和不确定性量化。

**DDPM（Denoising Diffusion Probabilistic Models）数学推导**：

**前向加噪过程**（$T$ 步）：
$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

其中 $\beta_t \in (0, 1)$ 为噪声调度。联合分布：
$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

其中 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$，$\alpha_t = 1 - \beta_t$。

**逆向去噪过程**：
$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})$$

**损失函数**（简化形式）：
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}, t)\|^2\right]$$

其中 $\epsilon_\theta$ 为去噪网络。

**AERIS速度场预测**：

AERIS预测的是速度场而非噪声场：
$$\mathcal{L} = \|\sigma \cdot F_\theta(\mathbf{x}_t / \sigma, t) - \mathbf{v}_t\|^2$$

其中 $\mathbf{v}_t = \cos(t) \cdot \mathbf{x}_0 + \sin(t) \cdot \boldsymbol{\epsilon}$ 为速度场。

**PF-ODE（Probability Flow ODE）推理**：

将随机微分方程转化为确定性常微分方程进行快速采样：
$$\frac{d\mathbf{x}_t}{dt} = -\frac{\beta_t}{2(1-\bar{\alpha}_t)} \cdot \epsilon_\theta(\mathbf{x}_t, t)$$

使用DPM-Solver++等常微分方程求解器，可在10步内完成采样。

**条件化机制**：

- **AERIS**：条件 = $[x_t, x_{i-1}, \text{forcings}]$，通过concatenation注入
- **PHYS-Diff**：条件 = $[H_{TC}, T_{env}, t_{emb}]$，通过cross-attention注入
- **FuXi-TC**：条件 = FuXi粗分辨率预报场，通过channel拼接注入
- **GenEPS**：条件 = SDEdit，多个预报模型输出作为条件

##### 3.3.3 Neural ODE与连续建模

**核心定义**：将大气演化建模为连续时间微分方程，通过Neural ODE实现任意时刻的外推预报，物理一致性极强。

**ClimODE数学推导**：

**状态定义**：
- 流体状态：$u_k \in \mathbb{R}^{H \times W}$（标量场，如温度）
- 流速场：$v_k \in \mathbb{R}^{H \times W \times 2}$（每个标量场的速度场）

**守恒输运方程**：
$$\frac{\partial u_k}{\partial t} = -v_k \cdot \nabla u_k - u_k \nabla \cdot v_k$$

离散形式（欧拉前向）：
$$u_k(t+dt) = u_k(t) - dt \cdot (v_k \cdot \nabla u_k + u_k \nabla \cdot v_k)$$

**流速演化方程**（数据驱动）：
$$\frac{dv_k}{dt} = f_\theta(u, \nabla u, v, \psi)$$

其中 $\psi$ 为全局上下文（如静态地形）。

**RK4数值积分**：
$$k_1 = f_\theta(u_n, v_n)$$
$$k_2 = f_\theta(u_n + \frac{dt}{2}k_1, v_n + \frac{dt}{2}k_1)$$
$$k_3 = f_\theta(u_n + \frac{dt}{2}k_2, v_n + \frac{dt}{2}k_2)$$
$$k_4 = f_\theta(u_n + dt \cdot k_3, v_n + dt \cdot k_3)$$
$$u_{n+1} = u_n + \frac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

**NowcastNet神经演化算子**：

运动场预测 + 平流 + 强度残差：
$$\mathbf{x}_t' = \text{Advect}(\mathbf{x}_{t-1}, \mathbf{v}_t)$$
$$\mathbf{x}_t = \mathbf{x}_t' + \mathbf{s}_t$$

其中 $\mathbf{v}_t$ 为预测的运动场，$\mathbf{s}_t$ 为预测的强度残差。

#### 【数据规格层】

**三大补充主干架构数据规格对比**：

| 模型 | 流派 | 输入形状 | 主干架构 | 输出形状 | 特殊配置 |
|------|------|---------|---------|---------|---------|
| SFNO | Operator | `[B, V, P, H, W]` | 球谐FNO | `[B, V, P, H, W]` | 球谐阶数$L$截断 |
| FourCastNet | Operator | `[B, 20, 721, 1440]` | AFNO | `[B, 20, 721, 1440]` | 软阈值化λ |
| AERIS | Generative | `[B, 70, 720, 1440]` | Swin扩散 | `[B, 70, 720, 1440]` | 10步PF-ODE |
| CorrDiff | Generative | `[B, 16, 36, 36]` | UNet扩散 | `[B, 4, 448, 448]` | 12.5×降尺度 |
| PHYS-Diff | Generative | `[B, M+N, 69, 80, 80]` | Swin扩散 | 轨迹集合 | TC条件化 |
| FuXi-TC | Generative | 模态1+2 | CNN扩散 | 高分辨率TC | 跨分辨率 |
| ClimODE | ODE | `[B, K, H, W]` | Neural ODE | 连续时间 | 流速场 |
| NowcastNet | ODE | `[B, T_in, H, W]` | U-Net演化 | `[B, T_out, H, W]` | 运动+强度 |

#### 【架构层】

**三大补充主干详细计算流程图**：

**SFNO完整流程（Operator流派）**：

```
Lat-Lon输入 [B, V, P, H, W]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. 球谐变换 (SHT)                                          │
│    for each (v, p):                                       │
│      f[θ, φ] → FFT沿经度 → Legendre变换沿纬度             │
│    输出: \hat{f}_l^m [B, V, P, L, M]                     │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 谱域操作（逐系数MLP）                                    │
│    for each (l, m):                                       │
│      \hat{g}_l^m = GELU(W_l^m · \hat{f}_l^m)            │
│    截断: 仅保留 l ≤ L_max                                  │
│    输出: \hat{g}_l^m [B, V, P, L_max, M]                 │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 逆球谐变换 (ISHT)                                       │
│    for each (v, p):                                       │
│      \hat{g}_l^m → Legendre逆变换 → FFT逆变换            │
│    输出: g[θ, φ] [B, V, P, H, W]                         │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 残差连接 + 层归一化                                      │
│    x_out = LayerNorm(x + g)                               │
└─────────────────────────────────────────────────────────────┘
  ↓
重复N_SFNO层
  ↓
输出预测场 [B, V, P, H, W]
```

**CorrDiff UNet扩散完整流程（Generative流派）**：

```
【训练】
  ERA5条件 [12, 36, 36] + 位置编码 [4, 36, 36] → [16, 36, 36]
    ↓
  噪声: ε ~ N(0, I)
    ↓
  加噪: x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε
    ↓
  ┌─────────────────────────────────────────────────────────┐
  │ UNet去噪器 (6层, 含跳跃连接)                            │
  │   Encoder:                                              │
  │     L0: [128, 36, 36] → ↓ → [128, 18, 18]             │
  │     L1: [256, 18, 18] → ↓ → [256, 9, 9]               │
  │     L2: [256, 9, 9] → ↓ → [256, 4, 4]                 │
  │   Bottleneck: ResBlock + Attention (分辨率4处)          │
  │   Decoder: (对称上采样 + skip connection)               │
  │     ← skip1: [256+128, 18, 18] ← [256, 9, 9]           │
  │     ← skip0: [128+128, 36, 36] ← [128, 18, 18]         │
  │   输出: ε_θ [4, 36, 36]                                │
  └─────────────────────────────────────────────────────────┘
    ↓
  损失: ||ε - ε_θ||²

【推理】
  高分辨率条件 [4, 448, 448]（下采样）
    ↓
  初始噪声 x_T [4, 36, 36] ~ N(0, I)
    ↓
  PF-ODE积分 (T步):
    for t in T..1:
      x_{t-1} = x_t + Δt · dε_θ/dt
    ↓
  生成低分辨率: [4, 36, 36]
    ↓
  超分辨率插值: [4, 448, 448]
```

**ClimODE连续演化流程（ODE流派）**：

```
初始状态 u(t₀), v(t₀)
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 物质导数计算                                                │
│   for each variable k:                                     │
│     du_k/dt|_adv = -v_k · ∇u_k (平流项)                  │
│     du_k/dt|_div = -u_k · ∇·v_k (散度项)                 │
│     du_k/dt = du_k/dt|_adv + du_k/dt|_div                 │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 流速演化 (MLP网络)                                          │
│   dv_k/dt = f_θ(u, ∇u, v, ψ)                             │
│   其中 ψ 为全局上下文（地形、海温等静态特征）               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ RK4数值积分                                                 │
│   [u(t₁), v(t₁)] = ODESolve([u₀, v₀], [t₀, t₁], f_θ)    │
│   支持任意时刻查询: t ∈ R⁺                                │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 高斯发射（观测模型）                                        │
│   u_obs ~ N(u + μ, σ²)                                   │
│   支持概率预测                                              │
└─────────────────────────────────────────────────────────────┘
```

**NowcastNet双路径演化流程（ODE流派）**：

```
输入: 历史雷达序列 [B, T_in=9, H, W]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 共享U-Net编码器                                              │
│   Conv + BN + GELU (4层下采样)                             │
│   输出: [B, D, H/16, W/16]                                │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 双路径解码器                                                │
│   ├─ Motion Decoder:                                       │
│   │   输出运动场: v[1:T] ∈ [B, T, 2, H, W]                │
│   └─ Intensity Decoder:                                    │
│       输出强度残差: s[1:T] ∈ [B, T, 1, H, W]               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 神经演化算子 (每步)                                         │
│   for t in 1..T:                                          │
│     x'_t = Advect(x_{t-1}, v_t)  ← 空间平流              │
│     x_t = x'_t + s_t              ← 加残差               │
│   输出: [B, T, 1, H, W]                                   │
└─────────────────────────────────────────────────────────────┘
```

#### 【设计层】

**三大补充主干设计动机与创新对比**：

| 架构类型 | 设计动机 | 核心创新 | 代表模型 | 适用场景 |
|---------|---------|---------|---------|---------|
| **SFNO** | 球面旋转对称性天然满足 | 球谐域逐系数MLP | SFNO | 长时稳定预报 |
| **AFNO** | 高效全局建模 | 软阈值化频域稀疏 | FourCastNet | 快速推理 |
| **DDPM扩散** | 自然概率分布生成 | PF-ODE快速采样 | AERIS | 集合预报 |
| **UNet扩散** | 图像超分辨率天然适配 | 降尺度条件化 | CorrDiff | 区域降尺度 |
| **条件扩散** | 多源异构条件注入 | Cross-attention | PHYS-Diff | TC预报 |
| **Neural ODE** | 物理连续演化 | 守恒约束 | ClimODE | 气候建模 |
| **神经演化** | 时空耦合预报 | 运动+强度分离 | NowcastNet | 临近预报 |

#### 【对比层】

**三大补充主干 vs 主流Transformer/GNN（大型对比表）**：

| 维度 | Transformer | Graph | SFNO | AFNO | DDPM扩散 | UNet扩散 | Neural ODE | 神经演化 |
|------|-----------|-------|------|------|---------|---------|----------|---------|
| **核心范式** | 注意力 | 消息传递 | 球谐域MLP | FFT+MLP | 加噪去噪 | UNet去噪 | ODE积分 | 平流+残差 |
| **全局感受野** | 多层堆叠 | 多跳传播 | 单层全局 | 单层全局 | 多层UNet | 跳跃连接 | 积分外推 | 运动场 |
| **计算复杂度** | $O(M^2N)$ | $O(E)$ | $O(NL²)$ | $O(N\log N)$ | 10+步推理 | 10+步推理 | ODE求解 | 逐帧推理 |
| **不确定性量化** | ❌ | ❌ | ❌ | ❌ | ✅天然 | ✅天然 | ✅贝叶斯 | ❌ |
| **物理一致性** | 弱 | 中 | 强 | 中 | 中 | 中 | 极强 | 强 |
| **推理速度** | 中 | 快 | 中 | 极快 | 慢 | 慢 | 快 | 快 |
| **极区友好** | 中 | 优 | 优 | 中 | 中 | N/A | 中 | 中 |
| **通道/模态** | 多 | 多 | 多 | 多 | 多 | 少 | 精选 | 单 |
| **训练难度** | 高 | 高 | 中 | 中 | 很高 | 高 | 中 | 中 |
| **长期稳定性** | 中 | 中 | 高 | 中 | 高 | 高 | 高 | 中 |
| **代表性模型** | AERIS, Stormer | GraphCast | SFNO | FourCastNet | AERIS | CorrDiff | ClimODE | NowcastNet |
| **典型时效** | 1-90天 | 1-10天 | 长期 | 1-10天 | 1-90天 | 区域 | 任意 | 0-2小时 |
| **参数规模** | 1B-80B | 36.7M | ~100M | ~100M | 1B-80B | 80M | 适中 | 适中 |

**多生成式方法对比（Generative流派内）**：

| 维度 | AERIS | CorrDiff | PHYS-Diff | FuXi-TC | GenEPS | PreDiff |
|------|-------|---------|----------|---------|--------|--------|
| **生成方式** | 扩散 | 扩散 | 扩散 | 扩散 | 扩散 | 扩散 |
| **主干网络** | Swin | UNet | Swin | CNN | 各模型自带 | Earthformer |
| **条件化** | 隐式 | 通道拼接 | Cross-attn | 通道拼接 | SDEdit | Cross-attn |
| **推理步数** | 10步（PF-ODE） | T步 | T步 | T步 | 可变 | T步 |
| **分辨率** | 全球0.25° | 区域km | TC区域 | 区域 | 全球 | 区域 |
| **不确定性** | 集合 | 条件分布 | 集合轨迹 | 条件分布 | 集成 | 条件分布 |
| **主要任务** | S2S预报 | 降尺度 | TC预报 | TC强度 | 多模型集成 | 降水预报 |
| **典型时效** | 1-90天 | 未来时刻 | 未来时刻 | 未来时刻 | 任意 | 未来时刻 |

**权衡评估**：

**Operator（Fourier/SFNO）vs Transformer**：
- ✅ Operator：单层全局感受野，计算效率高（$O(N\log N)$）
- ❌ Operator：平移等变假设，灵活度不如注意力
- ✅ Transformer：灵活建模，内容依赖权重
- ❌ Transformer：计算成本高

**扩散模型 vs 确定性模型**：
- ✅ 扩散：自然概率分布，长期稳定（90天），不确定性量化
- ❌ 扩散：推理成本高（10+步），训练复杂
- ✅ 确定性：推理快速（单次前向）
- ❌ 确定性：无不确定性量化，长期误差累积

**Neural ODE vs 标准离散模型**：
- ✅ ODE：物理一致，连续时间，任意时刻查询，守恒约束自然编码
- ❌ ODE：数值稳定性需注意，训练可能收敛慢
- ✅ 离散：实现简单，训练稳定
- ❌ 离散：时间步长固定，插值不自然

**适用场景**：
- **SFNO**：需要球面旋转对称性保证的长时积分任务
- **AFNO**：需要极高推理速度的中短期全球预报
- **AERIS扩散**：需要集合预报和长期稳定性的S2S任务
- **CorrDiff UNet扩散**：区域高分辨率降尺度
- **Neural ODE**：需要物理一致性和连续时间外推的任务
- **神经演化**：降水临近预报（0-2小时）

---

## 【第四部分：多尺度建模范式】

### 4.1 U-Net式降采样/上采样

#### 【原理描述】

**核心定义**：U-Net是一种编码器-解码器架构，通过逐层降采样构建多尺度特征金字塔，在编码器路径中捕获大尺度特征（低分辨率、高通道数），在解码器路径中恢复空间分辨率（高分辨率、低通道数），跳跃连接融合不同尺度的细节信息。

**数学推导**：

##### 1. 编码器（降采样）路径

$$\mathbf{h}^{(l+1)} = \text{Downsample}(\text{Conv}(\mathbf{h}^{(l)}))$$

$$\mathbf{h}^{(l+1)} \in \mathbb{R}^{C_{l+1} \times H_l/2 \times W_l/2}$$

典型模式：每下采样一次，通道数翻倍，空间尺寸减半。

##### 2. 解码器（上采样）路径

$$\mathbf{h}^{(l-1)} = \text{Conv}(\text{Concat}(\text{Upsample}(\mathbf{h}^{(l)}), \mathbf{S}^{(l-1)}))$$

其中 $\mathbf{S}^{(l-1)}$ 为编码器对应层的跳跃连接特征（skip connection）。

##### 3. 跳跃连接（Skip Connection）

$$\mathbf{h}_{\text{dec}}^{(l)} = g^{(l)}([\mathbf{h}_{\text{dec}}^{(l+1) \uparrow}, \mathbf{h}_{\text{enc}}^{(l)}])$$

其中 $[\cdot, \cdot]$ 表示通道维拼接，$\uparrow$ 表示上采样。

##### 4. 下采样操作

**MaxPooling**：
$$\mathbf{Y}_{c,i,j} = \max(\mathbf{X}_{c,2i:2i+K, 2j:2j+K})$$

**Stride Convolution**：
$$\mathbf{Y}_{c,i,j} = \sum_{c',k,l} \mathbf{W}_{c,c',k,l} \cdot \mathbf{X}_{c', s\cdot i+k, s\cdot j+l}$$

其中 $s$ 为步长（stride），通常 $s=2$。

##### 5. 上采样操作

**双线性插值**：
$$\mathbf{Y}_{c,i,j} = \alpha \mathbf{X}_{c, \lfloor i/s \rfloor, \lfloor j/s \rfloor} + (1-\alpha) \mathbf{X}_{c, \lceil i/s \rceil, \lceil j/s \rceil}$$

**转置卷积（反卷积）**：
$$\mathbf{Y}_{c,i,j} = \sum_{c',k,l} \mathbf{W}_{c,c',k,l} \cdot \mathbf{X}_{c', i-s\cdot k, j-s\cdot l}$$

转置卷积可学习，但可能产生棋盘效应（checkerboard artifact）。

##### 6. ResBlock（U-Net中的基础残差块）

$$\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x}; \theta)$$

$$\mathcal{F}(\mathbf{x}) = \text{BN}(\text{Conv}(\text{GELU}(\text{BN}(\text{Conv}(\mathbf{x}))))$$

#### 【数据规格层】

**CorrDiff UNet张量详细变化**：

| 层级 | 输入尺寸 | 输出尺寸 | 通道数 | 操作 |
|------|---------|---------|-------|------|
| Input | [B, 16, 36, 36] | - | 16 | 拼接ERA5(12) + 位置编码(4) |
| Enc L0 | - | [B, 128, 36, 36] | 128 | Conv2d(16→128) + ResBlock |
| Enc ↓ | [B, 128, 36, 36] | [B, 128, 18, 18] | 128 | MaxPool/StrideConv |
| Enc L1 | - | [B, 256, 18, 18] | 256 | Conv2d(128→256) + ResBlock |
| Enc ↓ | [B, 256, 18, 18] | [B, 256, 9, 9] | 256 | MaxPool/StrideConv |
| Enc L2 | - | [B, 256, 9, 9] | 256 | Conv2d(256→256) + ResBlock |
| Enc ↓ | [B, 256, 9, 9] | [B, C_max, 4, 4] | C_max | (推测) 继续下采样 |
| Bottleneck | - | [B, C_max, 4, 4] | C_max | ResBlock + Attention |
| Dec ↑ | [B, C_max, 4, 4] | [B, 256, 9, 9] | 256 | Upsample + Concat(skip2) |
| Dec L2 | - | [B, 256, 9, 9] | 256 | Conv2d(512→256) + ResBlock |
| Dec ↑ | [B, 256, 9, 9] | [B, 128, 18, 18] | 128 | Upsample + Concat(skip1) |
| Dec L1 | - | [B, 128, 18, 18] | 128 | Conv2d(256→128) + ResBlock |
| Dec ↑ | [B, 128, 18, 18] | [B, 128, 36, 36] | 128 | Upsample + Concat(skip0) |
| Dec L0 | - | [B, 128, 36, 36] | 128 | Conv2d(256→128) + ResBlock |
| Output | - | [B, 4, 448, 448] | 4 | Conv2d(128→4) + 超分辨率插值(12.5×) |

**NowcastNet U-Net张量变化**：

| 层级 | 输入尺寸 | 输出尺寸 | 通道数 |
|------|---------|---------|-------|
| Input | [B, T_in=9, H, W] | - | 9 |
| Enc L0 | - | [B, D₀, H, W] | D₀ |
| Enc ↓ | [B, D₀, H, W] | [B, D₁, H/2, W/2] | D₁ |
| Enc ↓ | [B, D₁, H/2, W/2] | [B, D₂, H/4, W/4] | D₂ |
| Enc ↓ | [B, D₂, H/4, W/4] | [B, D₃, H/8, W/8] | D₃ |
| Enc ↓ | [B, D₃, H/8, W/8] | [B, D₄, H/16, W/16] | D₄ |
| Bottleneck | [B, D₄, H/16, W/16] | [B, D₄, H/16, W/16] | D₄ |
| Dec ↑+Motion | [B, D₄, H/16, W/16] | [B, T·2, H/8, W/8] | 2·T |
| Dec ↑+Intensity | [B, D₄, H/16, W/16] | [B, T·1, H/8, W/8] | 1·T |
| 输出 | [B, T, 1, H, W] | - | 1 |

#### 【架构层】

**U-Net完整计算流程图（含张量形状）**：

```
输入: [B, C_in, H, W]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Encoder路径 (4-6层)                                         │
│   L0: Conv(3×3) + BN + GELU → [B, 64, H, W]               │
│        (保存skip0: [B, 64, H, W])                         │
│       ↓ ↓ (Stride-2)                                      │
│   L1: Conv(3×3) + BN + GELU → [B, 128, H/2, W/2]          │
│        (保存skip1: [B, 128, H/2, W/2])                    │
│       ↓ ↓                                                  │
│   L2: Conv(3×3) + BN + GELU → [B, 256, H/4, W/4]          │
│        (保存skip2: [B, 256, H/4, W/4])                    │
│       ↓ ↓                                                  │
│   L3: Conv(3×3) + BN + GELU → [B, 512, H/8, W/8]          │
│        (保存skip3: [B, 512, H/8, W/8])                     │
│       ↓ ↓                                                  │
│   Bottleneck: [B, 1024, H/16, W/16]                      │
│       (可选: 添加注意力层，如Attention on 28分辨率)        │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Decoder路径 (4-6层)                                         │
│       ↑ Upsample/TransposeConv                             │
│   L3': Concat(skip3) + Conv → [B, 768, H/8, W/8]          │
│       ↑ Upsample/TransposeConv                            │
│   L2': Concat(skip2) + Conv → [B, 384, H/4, W/4]          │
│       ↑ Upsample/TransposeConv                             │
│   L1': Concat(skip1) + Conv → [B, 192, H/2, W/2]          │
│       ↑ Upsample/TransposeConv                             │
│   L0': Concat(skip0) + Conv → [B, 64, H, W]               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 输出层                                                       │
│   Conv(1×1) → [B, C_out, H, W]                            │
└─────────────────────────────────────────────────────────────┘
```

**CorrDiff完整扩散+UNet流程**：

```
【条件注入（Encoder输入层）】
  ERA5条件 [12, 36, 36]
    ↓
  4通道正弦位置编码 [4, 36, 36]
    ↓
  Concat → [16, 36, 36]

【Encoder】
  Level 1: [128, 36, 36] → ↓ → [128, 18, 18]
    ↓
  Level 2: [256, 18, 18] → ↓ → [256, 9, 9]
    ↓
  Level 3: [256, 9, 9] → ↓ → [256, 4, 4]（推测）
    ↓
  ...
    ↓
  Bottleneck: [C_max, H_min, W_min]
    ↓

【Bottleneck】（特殊设计）
  ResBlock + Attention (Attention gate on 28分辨率)
    ↓

【Decoder】（对称上采样）
  ...
    ↓
  Level 2': ↑ + Concat(skip2) → [256, 9, 9]
    ↓
  Level 1': ↑ + Concat(skip1) → [128, 18, 18]
    ↓
  Level 0': ↑ + Concat(skip0) → [128, 36, 36]
    ↓

【输出】
  Conv(1×1) → [4, 36, 36]  （低分辨率）
    ↓
  超分辨率插值 → [4, 448, 448]  （12.5×上采样）
```

**PreDiff Earthformer-UNet+Cuboid Attention流程**：

```
输入序列 [L_in+L_out, H, W, C]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ VAE Encoder (逐帧)                                          │
│   for each frame:                                           │
│     Conv + GELU → BN → [H_z, W_z, C_z]                    │
│   输出: [L_in, H_z, W_z, C_z]                              │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 条件+噪声拼接                                                │
│   concat([条件z_c, 噪声z_N]) → [L_in+L_out, H_z, W_z, C_z]│
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Earthformer-UNet Encoder                                    │
│   ├─ 3D Cuboid Self-Attention (时空块注意力)               │
│   │    在 [T_in+T_out, H_z, W_z] 三维时空块内计算注意力    │
│   ├─ 多尺度下采样                                          │
│   └─ 逐层通道递增: C_z → 2C_z → 4C_z → ...                │
│   输出: [L', H_z', W_z', C_hidden]                        │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Bottleneck                                                  │
│   ├─ 3D Cuboid Attention                                   │
│   └─ 可选: 物理约束损失                                     │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Earthformer-UNet Decoder                                    │
│   ├─ 上采样 + Skip connections                              │
│   ├─ 3D Cuboid Self-Attention                              │
│   └─ 逐层通道递减                                           │
│   输出: [L_out, H_z, W_z, C_z]                            │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ VAE Decoder (逐帧)                                           │
│   for each future frame:                                   │
│     Conv + GELU → BN → [H, W, C]                          │
│   输出: [L_out, H, W, C]                                  │
└─────────────────────────────────────────────────────────────┘
```

#### 【设计层】

**设计动机**：
- **多尺度特征**：气象现象具有多尺度特性（从局地对流到行星波）
- **信息保留**：跳跃连接避免降采样导致的高频信息丢失
- **计算效率**：在低分辨率上处理大尺度特征，减少计算量
- **细节恢复**：上采样+跳跃连接融合大尺度语义和细节纹理
- **大规模降尺度**：CorrDiff从36×36到448×448的12.5×上采样需要多尺度架构

**痛点解决**：
- **信息瓶颈**：跳跃连接提供编码器和解码器之间的直接信息通道
- **多尺度融合**：不同层捕获不同尺度的气象特征
- **梯度流动**：跳跃连接提供额外的梯度路径，缓解梯度消失
- **尺度耦合**：不同层级的特征融合实现跨尺度交互

**创新突破**：

- **CorrDiff**：6层深度UNet处理大倍率降尺度，80M参数大型UNet实现km级降尺度；在latent空间进行扩散，平衡质量与效率
- **PreDiff**：将Earthformer的3D Cuboid Attention嵌入UNet结构，首次在降水临近预报中使用时空联合注意力替代标准2D UNet
- **POSTCAST**：双U-Net结构（预测+去模糊），模糊核估计增强降水预报
- **NowcastNet**：双路径U-Net（运动场+强度残差分离），共享编码器
- **AIFS**：GNN编码器/解码器实现Grid↔Mesh的多尺度映射

#### 【对比层】

**五流派U-Net式架构对比（大型对比表）**：

| 模型 | 流派 | 编码器层数 | 基础通道 | 最大通道 | 下采样倍率 | 总降采样 | 跳跃连接 | 特殊设计 | 适用场景 |
|------|------|-----------|---------|---------|-----------|---------|---------|---------|---------|
| CorrDiff | Generative | 6+ | 128 | **未说明** | 2×每层 | ~9倍 | 通道拼接 | 注意力门(28分辨率) | 区域降尺度 |
| PreDiff | Generative | **未说明** | **未说明** | **未说明** | **未说明** | **未说明** | 通道拼接 | Earthformer 3D Cuboid | 降水临近预报 |
| POSTCAST | Generative | **未说明** | **未说明** | **未说明** | **未说明** | **未说明** | **未说明** | 双U-Net（预测+去模糊） | 去模糊降水 |
| NowcastNet | ODE | 4-5 | **未说明** | **未说明** | 2×每层 | 16× | 通道拼接 | 双路径输出（运动+强度） | 临近预报 |
| AIFS | Graph | 2(Grid↔Mesh) | **未说明** | **未说明** | 13×总 | 13×总 | GNN聚合 | 非规则网格 | 业务预报 |
| 高分辨率降水 | Transformer | 多层 | **未说明** | **未说明** | **未说明** | **未说明** | **未说明** | 3D Swin块 | 降水预报 |

**U-Net vs Patch-based多尺度对比**：

| 维度 | U-Net式降采样 | Patch Embedding（Transformer） | 直接全局（Graph/GN） |
|------|-------------|-------------------------------|-------------------|
| **尺度变化** | 逐层减半 | 一步降维 | 无尺度变化 |
| **细节保留** | 跳跃连接融合 | 中等（patch化损失部分细节） | 完整（节点=token） |
| **计算成本** | 低分辨率层计算量小 | 中等 | 高（全分辨率token） |
| **感受野** | 深层→大尺度 | 全局（FFT天然） | 多跳累积 |
| **适用场景** | 多尺度特征融合、降尺度 | 平衡精度与效率 | 高分辨率、全局建模 |
| **信息流** | 多尺度层级递进 | 一步到位 | 直接全局 |

**权衡评估**：

**跳跃连接方式**：
- **通道拼接**：保留更多信息，但通道数翻倍增加计算量
- **加法**：参数量少，但可能信息损失

**上采样方法**：
- **双线性插值**：简单快速，但无学习能力
- **转置卷积**：可学习，但可能产生棋盘效应
- **最近邻+卷积**：避免棋盘效应，计算开销适中

**注意力门（Attention Gate）**：
- 在跳跃连接处添加注意力，抑制无关特征
- 提升解码器对有意义特征的关注

**层级数与降采样倍率**：
- **层级数**：更多层级捕捉更大尺度，但增加参数和计算量
- **降采样率**：2×标准但可能不足；4×或更大需要更强的跳跃连接
- **极端情况**：CorrDiff从36→448（12.5×）需要6+层UNet

**适用场景**：
- **多尺度U-Net**：区域降尺度（CorrDiff）、临近预报（NowcastNet）、降水预报（PreDiff）
- **Patch Embedding**：全球预报（FourCastNet）、快速原型
- **直接全局处理**：全局中期预报（GraphCast）、需要完整分辨率的任务

---

### 4.2 多尺度图结构

#### 【原理描述】

**核心定义**：在图神经网络中构建包含多个尺度的边连接，通过不同层级的节点和边实现从局地到全球的多尺度信息传播。粗尺度边实现长程通信，细尺度边捕捉局地相互作用，是Graph流派实现多尺度建模的核心机制。

**数学推导**：

##### 1. k-hop邻域定义

$$\mathcal{N}_k(v) = \{u : d(u, v) \leq k\}$$

其中 $d(u, v)$ 为图上的最短路径距离（跳数）。

##### 2. 多尺度图定义

$$\mathcal{G} = (\mathcal{V}, \mathcal{E}_0 \cup \mathcal{E}_1 \cup \cdots \cup \mathcal{E}_L)$$

其中 $\mathcal{E}_l$：第 $l$ 级边集，连接距离约为 $2^l \cdot d_0$ 的节点。

##### 3. 多尺度图卷积

$$h_v^{(l)} = \text{AGGREGATE}_{k \in \{1, 2, \ldots, K\}} \left(\text{GCN}_k(h_v^{(l-1)}, \mathcal{N}_k(v))\right)$$

##### 4. GraphCast Multi-Mesh多尺度消息传递

$$\mathbf{h}_i^{(t+1)} = \phi\left(\mathbf{h}_i^{(t)}, \bigoplus_{l=0}^L \bigoplus_{j \in \mathcal{N}_l(i)} \mathbf{m}_{j \to i}^{(l)}\right)$$

其中 $\mathcal{N}_l(i)$ 是节点 $i$ 在第 $l$ 级的邻居。

##### 5. TelePiT多尺度频带分解

$$\mathbf{X}^{(0)}, \mathbf{X}^{(1)}, \ldots, \mathbf{X}^{(L)} = \text{MultiScaleSplit}(\mathbf{X})$$

其中每个频带 $\mathbf{X}^{(l)}$ 对应不同的空间尺度。

##### 6. 跨尺度感受野分析

| 层级 | 边类型 | 覆盖范围（球面距离） | 物理现象 | 对应大气尺度 |
|------|--------|-------------------|---------|------------|
| Level 0 | 超长程 | ~10000 km | 行星波、罗斯贝波 | 行星尺度 |
| Level 1 | 长程 | ~5000 km | 大尺度气旋/反气旋 | 天气尺度 |
| Level 2 | 中长程 | ~2500 km | 锋面系统 | 天气尺度 |
| Level 3 | 中程 | ~1250 km | 锢囚锋、中尺度对流 | 中尺度 |
| Level 4 | 中短程 | ~625 km | 对流单体、积云 | 中尺度 |
| Level 5 | 短程 | ~312 km | 积云对流、边界层 | 小尺度 |
| Level 6 | 超短程 | ~156 km | 边界层湍流、细对流 | 小尺度 |

##### 7. 复杂度分析

| 尺度设计 | 计算复杂度 | 通信深度 | 内存占用 | 适用场景 |
|---------|----------|---------|---------|---------|
| 单尺度（k=1） | $O(E)$ | $O(L)$层全局 | 低 | 简单任务 |
| 固定多尺度（k=1,2,4） | $O(3E)$ | $O(1)$层 | 中 | 标准GNN |
| GraphCast 7级 | $O(E_{total})$ | $O(1)$层 | 高 | 全球预报 |
| TelePiT频带 | $O((L+1) \cdot E)$ | $O(L+1)$ | 中 | S2S预报 |

**理论依据**：
- **物理多尺度性**：大气动力学本质上是多尺度耦合系统
- **高效传播**：粗尺度边减少信息传播的层数（从 $O(N)$ 到 $O(\log N)$ 层）
- **尺度解耦**：不同尺度的边可学习不同的物理过程
- 大气运动具有级联能量传递特性，从大尺度的罗斯贝波到小尺度的积云对流

#### 【数据规格层】

**各流派多尺度配置对比**：

| 模型 | 流派 | 网格层级 | 节点分布 | 总边数 | 尺度分级 | 融合策略 |
|------|------|---------|---------|--------|---------|---------|
| GraphCast | Graph | Icosahedral | Level 6: 40962 | 327,660 | 7级（0-6） | 同时消息传递 |
| TelePiT | Transformer | 纬向压缩 | Level 0: H个 | N/A | L+1频带 | 自上而下融合 |
| AIFS | Graph | Reduced Gaussian | Level 0: N320→O96 | N/A | 2级 | GNN编码/解码 |
| GNADET | Graph | 规则网格 | H×W | ~8×H×W | 1级（Moore 8邻域） | 注意力融合 |

**GraphCast Multi-Mesh层级详细规格**：

| 层级 | 节点数（估算） | 边连接范围 | 物理尺度 | 大气现象 |
|------|------------|----------|---------|---------|
| Level 0 | ~12 | 超长程 | 行星波 | 跨半球遥相关 |
| Level 1 | ~42 | 长程 | Rossby波 | 大尺度环流 |
| Level 2 | ~162 | 中长程 | 锋面 | 中纬度天气系统 |
| Level 3 | ~642 | 中程 | 锢囚锋 | 温带气旋 |
| Level 4 | ~2562 | 中短程 | 对流单体 | 中尺度对流 |
| Level 5 | ~10242 | 短程 | 积云 | 小尺度对流 |
| Level 6 | ~40962 | 超短程 | 湍流 | 边界层过程 |

#### 【架构层】

**GraphCast Multi-Mesh多尺度消息传递详细流程图**：

```
输入: Mesh节点特征 [B, N_mesh=40962, D]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer: GN层 (重复16次)                                     │
│                                                           │
│   【所有尺度Level 0-6 并行边更新】                          │
│   for each level l in {0,1,2,3,4,5,6}:                   │
│     for each edge (i→j) in Level l:                      │
│       e_ij^(l) ← MLP([e_ij^(l), h_i, h_j])             │
│   输出: E'^(l) [B, E_l, D_edge] ∀l                       │
│                                                           │
│   【跨尺度节点聚合】                                        │
│   for each node j:                                        │
│     m_j ← Σ_{l=0}^{6} Σ_{i→j in Level l} e_ij^(l)      │
│   输出: m_j [B, D_edge]                                  │
│                                                           │
│   【节点更新】                                             │
│   for each node j:                                        │
│     h_j ← h_j + MLP([h_j, m_j])                          │
│   输出: H' [B, N_mesh, D]                                │
│                                                           │
│   【残差】                                                 │
│   H ← H + H'                                              │
│   E^(l) ← E^(l) + E'^(l) ∀l                             │
└─────────────────────────────────────────────────────────────┘
  ↓
重复16层
  ↓
输出: [B, N_mesh=40962, D]
```

**TelePiT多尺度频带处理详细流程图**：

```
输入: 纬向token [B, H, D_emb]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. 多尺度分解 (可学习，类似小波变换)                        │
│   for l in 1..L:                                          │
│     [A_l, D_l] = split(MLP_l(A_{l-1}))                  │
│   其中 A_l 为低频分量，D_l 为高频分量                      │
│   输出: {X^(0), X^(1), ..., X^(L)}                       │
│       X^(0): 最低频（全球尺度）                           │
│       X^(l): 第l高频（递增细节）                          │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 对每个频带独立处理 (并行或串行)                          │
│   for l in 0..L:                                          │
│     ├─ Physics-Informed ODE:                               │
│     │    Ẋ^(l) = f_θ(X^(l))                             │
│     │    X̃^(l) = Euler/RK4(X^(l), Δt)                    │
│     ├─ Teleconnection-Aware Transformer:                   │
│     │    ω = softmax(W_p · mean(X̃^(l)))                  │
│     │    c = Σ ω_j · P_j (EOF基)                         │
│     │    q = c · W^Q                                      │
│     │    X̂^(l) = Attention(Q=q, K=X̃^(l), V=X̃^(l))     │
│     └─ 输出: {X̂^(0), ..., X̂^(L)}                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 跨尺度融合                                              │
│   Z_final = Σ_{l=0}^{L} W_l · X̂^(l) / (L+1)             │
│   其中 W_l 为可学习的融合权重                             │
└─────────────────────────────────────────────────────────────┘
```

**多尺度图层次化处理流程图**：

```
不同分辨率的三维气象场/节点特征
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 细网格到粗网格的特征聚合 (Pooling/Down-projection)          │
│   H_coarse = Pool(H_fine)                                  │
│   输出: [B, N_coarse, C']                                 │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 粗网格节点间消息传递 (GNN Message Passing)                  │
│   在粗网格上执行多层消息传递                                │
│   捕获大尺度长程依赖                                        │
│   输出: [B, N_coarse, C'']                                │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 粗网格到细网格的特征扩散 (Unpooling/Up-projection)          │
│   H_fine' = Up(H_coarse) ⊕ H_fine                        │
│   ⊕: 拼接或加法（保留细网格原始特征）                      │
│   输出: [B, N_fine, C''']                                  │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 细网格节点间消息传递                                        │
│   在细网格上执行多层消息传递                                │
│   捕获局地细节                                              │
│   输出: [B, N_fine, C'''']                                │
└─────────────────────────────────────────────────────────────┘
  ↓
输出: 多尺度融合特征 [B, N, D]
```

#### 【设计层】

**设计动机**：
- **物理多尺度性**：大气动力学本质上是多尺度耦合系统
- **高效传播**：粗尺度边减少长程信息传播的层数
- **尺度解耦**：不同尺度的边可学习不同的物理过程
- **球面几何**：图结构自然适应球面距离，无需极区处理

**痛点解决**：
- **长程依赖**：粗尺度边实现 $O(1)$ 层全局通信（GraphCast通过7级边同时传递）
- **尺度混叠**：显式分离不同尺度避免相互干扰
- **参数共享**：所有尺度共享GNN参数，减少模型复杂度
- **计算效率**：相比全局注意力，k-hop限制了计算复杂度

**创新突破**：

- **GraphCast**：Icosahedral multi-mesh实现均匀的多尺度覆盖，Level 0-2边用于长程通信，Level 5-6边用于局地细节；所有层级并行消息传递
- **TelePiT**：可学习的多尺度分解，类似小波但端到端训练；Teleconnection-aware attention显式建模气候模态
- **GenCast**：通过图结构在球面上实现多尺度建模，k-hop邻域注意力

#### 【对比层】

**五流派多尺度建模对比（大型对比表）**：

| 维度 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **多尺度机制** | 窗口+多层堆叠 | FFT天然多尺度 | 多级图边 | UNet层级 | ODE时间尺度 |
| **尺度数量** | 隐式多层 | 无限（频域分解） | 7级（GraphCast） | 4-6层UNet | 可配置 |
| **尺度融合** | 多层累积 | 逆变换融合 | 并行消息传递 | 跳跃连接 | 积分求和 |
| **大尺度建模** | 多层堆叠（慢） | 单层全局（快） | 粗边（快） | 深层（慢） | 慢变分量 |
| **小尺度建模** | 窗口注意力（快） | 高频分量 | 细边（精确） | 浅层+跳跃（快） | 快变分量 |
| **计算成本** | $O(L \cdot M^2 N)$ | $O(N\log N)$ | $O(E_{total})$ | $O(CHW)$ | ODE求解 |
| **感受野增长** | 线性（每层+M） | 无（单层全局） | 对数（多跳） | 指数（每层2×） | 积分时长 |
| **跨尺度交互** | 隐式 | 无 | 显式 | 显式（跳跃） | 隐式 |

**Graph流派内多尺度设计深度对比**：

| 模型 | 尺度数 | 多尺度实现 | 融合策略 | 并行度 | 边构建 | 优势 | 劣势 |
|------|-------|----------|---------|--------|-------|------|------|
| GraphCast | 7级 | Multi-mesh边 | 同时消息传递 | 完全并行 | 递归细分icosahedron | 均匀覆盖 | 实现复杂 |
| TelePiT | L+1 | 频带分解 | 加权求和 | 串行/并行 | N/A | 计算高效 | 丢失经向细节 |
| AIFS | 2级 | Reduced Gaussian | GNN编码/解码 | 串行 | 网格聚合 | 业务标准 | 尺度有限 |
| GNADET | 1级 | Moore邻域 | 注意力 | 隐式 | 固定8邻域 | 简单 | 无显式多尺度 |

**多尺度设计的核心权衡**：

| 设计选择 | 优势 | 劣势 | 适用场景 |
|---------|------|------|---------|
| **更多尺度边** | 表达力更强 | 边数增加，计算量↑ | 追求精度 |
| **更少尺度边** | 计算高效 | 表达力受限 | 实时推理 |
| **并行消息传递** | 感受野一步到位 | 内存占用高 | 全局预报 |
| **串行层次传播** | 内存高效 | 需多层才能全局 | 资源受限 |
| **频带分解** | 物理可解释 | 尺度数量受限 | S2S预报 |

**权衡评估**：

**GraphCast 7级多尺度**：
- ✅ 最完整的尺度覆盖（从行星波到边界层湍流）
- ✅ 并行消息传递，感受野一步到位
- ❌ 边数多（327,660条），内存占用高
- ❌ 实现复杂，需要专门的icosahedral网格处理

**TelePiT频带分解**：
- ✅ 计算高效，尺度数量可配置
- ✅ 物理可解释（不同频带对应不同尺度的大气波动）
- ❌ 仅保留纬向信息，丢失经向细节
- ❌ 串行处理尺度

**适用场景**：
- **GraphCast级别多尺度**：需要从局地到全球精确建模的全球中期预报
- **TelePiT频带分解**：计算资源受限但需要多尺度的大尺度S2S预报
- **单尺度+多层堆叠**：简单任务或规则网格

---

### 4.3 层次化时间模型

#### 【原理描述】

**核心定义**：层次化时间模型通过在不同时间尺度上建模气象演化过程，捕获从短期（小时）到长期（天/周/季节）的多时间尺度动力学特征。时间建模是气象AI的灵魂——大气混沌系统的时间演化直接决定了预报的不确定性随预报时效的增长规律。

**数学推导**：

##### 1. 单步自回归预测（最基础范式）

$$\hat{X}_{t+\Delta t} = f_\theta(X_t)$$

损失：$\mathcal{L} = |f_\theta(X_t) - X_{t+\Delta t}|^2$

##### 2. 多步自回归展开

$$\hat{X}_{t+k\Delta t} = f_\theta^{(k)}(X_t) = \underbrace{f_\theta \circ f_\theta \circ \cdots \circ f_\theta}_{k \text{ times}}(X_t)$$

##### 3. 随机化时间步长（Stormer创新）

$$\delta t \sim \mathcal{U}\{6\text{h}, 12\text{h}, 24\text{h}\}$$

$$\hat{X}_{t+\delta t} = f_\theta(X_t, \delta t)$$

优势：
- 数据增强：同一时间段构造多种训练样本
- 多时间尺度统一建模：单模型覆盖短期和长期
- 推理灵活性：多路径集成

**推理多路径**：
- 路径1：$[6h, 6h, 6h, 6h] \to 24h$
- 路径2：$[12h, 12h] \to 24h$
- 路径3：$[24h] \to 24h$
- 集成：$\hat{X}_T = \frac{1}{M}\sum_{m=1}^M \text{Rollout}_m(X_0)$

##### 4. 时间尺度分解

$$\mathbf{X}(t) = \mathbf{X}_{\text{fast}}(t) + \mathbf{X}_{\text{slow}}(t)$$

**快变分量（高频）**：
$$\frac{d\mathbf{X}_{\text{fast}}}{dt} = f_{\text{fast}}(\mathbf{X}_{\text{fast}}, \mathbf{X}_{\text{slow}})$$

**慢变分量（低频）**：
$$\frac{d\mathbf{X}_{\text{slow}}}{dt} = f_{\text{slow}}(\mathbf{X}_{\text{slow}})$$

##### 5. 连续时间演化（ClimODE）

$$\frac{d\mathbf{u}(t)}{dt} = f_\theta(\mathbf{u}(t), t)$$

$$\mathbf{u}(t_1) = \mathbf{u}(t_0) + \int_{t_0}^{t_1} f_\theta(\mathbf{u}(\tau), \tau) d\tau$$

支持任意时刻查询，无需离散时间步长。

##### 6. Replay Buffer机制（FengWu）

训练时混合真实输入和缓存预测：
$$X_{\text{input}} = \begin{cases} X_t & \text{概率 } 1-p \\ \hat{X}_{\text{buffer}} & \text{概率 } p \end{cases}$$

Buffer更新：$\text{Buffer} \leftarrow \text{Buffer} \cup \{\hat{X}_t\}$$

目的：缓解训练-推理不匹配（训练时输入真实场，推理时输入预测场）

##### 7. 多步微调策略

**阶段1（单步）**：
$$\mathcal{L}_1 = \mathbb{E}[|f_\theta(X_t) - X_{t+\Delta t}|^2]$$

**阶段2（K步）**：
$$\mathcal{L}_K = \mathbb{E}\left[\frac{1}{K}\sum_{k=1}^K |f_\theta^{(k)}(X_t) - X_{t+k\Delta t}|^2\right]$$

其中 $f_\theta^{(k)}$ 表示 $k$ 次自回归应用。

##### 8. 扩散时间步（Generative流派）

$$t \in [0, 1] \to \text{PE}(t) \to \text{AdaLN}$$

时间步编码注入到去噪网络的条件化中。

**时间步长物理对照表**：

| 时间步长 | 覆盖物理现象 | 适用任务 | 累积误差 |
|---------|------------|---------|---------|
| 10 min | 对流、降水 | 临近预报 | 严重 |
| 1 h | 边界层、日变化 | 超短期预报 | 较严重 |
| 6 h | 天气系统 | 中期预报 | 中等 |
| 12 h | 天气系统、日循环 | 中期预报 | 较少 |
| 24 h | 日变化、天气尺度 | 延伸期预报 | 较少 |
| 周平均 | 气候模态 | S2S预报 | 无（直接预测） |

**理论依据**：
- 大气具有多时间尺度：昼夜循环（24h）、天气系统（3-7天）、季节变化（季-年）
- 不同时间尺度的物理过程耦合（如慢变的海温强迫影响快变的对流）
- 数据驱动模型学习隐式的时间演化算子

#### 【数据规格层】

**各流派时间建模配置对比**：

| 模型 | 流派 | 时间步长 | 训练策略 | 推理方式 | 最大时效 | 时间条件化 |
|------|------|---------|---------|---------|---------|---------|
| FengWu | Transformer | 固定6h | 单步+Replay Buffer | 自回归56步 | 14天 | 无 |
| Swin V2 | Transformer | 固定6h | 单步70epoch + 4/8步微调 | 自回归 | **未说明** | 无 |
| Stormer | Transformer | 随机{6h,12h,24h} | 随机单步 + 多步微调 | 多路径集成 | **未说明** | AdaLN |
| AERIS | Transformer | 6h/24h（双模型） | 扩散单步 | 自回归90天 | 90天 | PF-ODE |
| GraphCast | Graph | 固定6h | 自回归12步 | 自回归 | 10天 | 无 |
| FourCastNet | Fourier | 固定6h | 单步80epoch + 两步微调 | 自回归 | 7-10天 | 无 |
| FuXi | Transformer | 固定6h | 自回归 | 自回归 | 15天 | 无 |
| CorrDiff | Generative | T步 | 扩散去噪 | ODE求解 | 未来时刻 | 扩散时间t |
| PHYS-Diff | Generative | 未来时刻 | 扩散去噪 | ODE求解 | TC预报 | t_emb |
| ClimODE | ODE | 连续 | Neural ODE | RK4积分 | 任意 | 连续时间t |
| TelePiT | Transformer | 周平均 | 直接预测 | 直接多周 | 6周 | 周索引 |

#### 【架构层】

**层次化时间建模完整流程图**：

```
【训练阶段：多策略对比】

┌─────────────────────────────────────────────────────────────┐
│ 策略A: 单步训练（最基础）                                    │
│   X_t → f_θ(X_t) → X̂_{t+Δt}                              │
│   Loss = ||X̂_{t+Δt} - X_{t+Δt}||²                        │
│   输出: 单步预测能力                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 策略B: 随机时间步长（Stormer）                             │
│   δt ~ U{6h, 12h, 24h}                                     │
│   X_t → f_θ(X_t, δt) → X̂_{t+δt}                         │
│   Loss = ||X̂_{t+δt} - X_{t+δt}||²                        │
│   输出: 多时间尺度统一建模                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 策略C: 多步微调                                             │
│   阶段1: 单步预训练70-80 epoch                             │
│   阶段2: 加载权重 → 降低学习率(1e-4)                      │
│           X_t → [f_θ]² → X̂_{t+2Δt}                       │
│           Loss = ||X̂_{t+2Δt} - X_{t+2Δt}||²              │
│           (FourCastNet: 两步; Swin V2: 4/8步; Stormer: K步)│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 策略D: Replay Buffer（FengWu）                            │
│   Buffer初始化为空                                          │
│   for iteration:                                            │
│     X_input = real(X_t) 或 buffer样本                      │
│     μ, σ = f_θ(X_input)                                    │
│     Loss = NLL(μ, σ, X_{t+Δt})                            │
│     更新Buffer: Buffer ← Buffer ∪ {μ}                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 策略E: 扩散时间步（AERIS/Generative）                     │
│   t ~ U[0, 1]                                              │
│   x_t = √(ᾱ_t)·x₀ + √(1-ᾱ_t)·ε                           │
│   v_t = F_θ(x_t, t)                                        │
│   Loss = ||v_t - v_true||²                                │
└─────────────────────────────────────────────────────────────┘

【推理阶段：多路径展开】

X_0（初始状态）
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 路径1: 短步长路径                                          │
│   [6h, 6h, 6h, ..., 6h] → T小时                          │
│   X_0 → f_θ(X_0) → X̂_1 → f_θ(X̂_1) → ...                 │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 路径2: 中步长路径（仅Stormer）                             │
│   [12h, 12h, ..., 12h] → T小时                            │
│   X_0 → f_θ(X_0, 12h) → X̂_1 → f_θ(X̂_1, 12h) → ...      │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 路径3: 长步长路径（仅Stormer）                             │
│   [24h] → T小时                                            │
│   X_0 → f_θ(X_0, 24h) → X̂_1                               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 集成平均                                                    │
│   X_final = mean(X̂_1^(路径1), X̂_1^(路径2), X̂_1^(路径3))│
└─────────────────────────────────────────────────────────────┘
```

**GraphCast自回归Rollout详细流程**：

```
初始: X_{t-6h}, X_t
  ↓
for step in 1..T (T=40, 即10天):
  ┌─────────────────────────────────────────────────────────┐
  │ 单步预测:                                                  │
  │   X_{t+6h} = Model(X_t, X_{t-6h})                        │
  │   其中 Model = Encoder-Process-Decoder(GNN)              │
  └─────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────┐
  │ 状态更新:                                                  │
  │   X_{t-6h} ← X_t                                        │
  │   X_t ← X_{t+6h}                                        │
  └─────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────┐
  │ 轨迹累积:                                                  │
  │   trajectory.append(X_{t+6h})                           │
  └─────────────────────────────────────────────────────────┘
    ↓
  输出: [X_{t+6h}, X_{t+12h}, ..., X_{t+T×6h}]
```

**ClimODE连续时间演化流程**：

```
初始状态 u(t₀), v(t₀)
  ↓
┌─────────────────────────────────────────────────────────────┐
│ ODE系统定义                                                 │
│   du_k/dt = -v_k·∇u_k - u_k∇·v_k                         │
│   dv_k/dt = f_θ(u, ∇u, v, ψ)                             │
│   其中 ψ 为全局上下文（静态特征）                          │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ RK4数值积分 (任意时刻t₁)                                   │
│   k₁ = f_θ(u_n, v_n, t_n)                                 │
│   k₂ = f_θ(u_n + dt/2·k₁, v_n + dt/2·k₁, t_n + dt/2)     │
│   k₃ = f_θ(u_n + dt/2·k₂, v_n + dt/2·k₂, t_n + dt/2)     │
│   k₄ = f_θ(u_n + dt·k₃, v_n + dt·k₃, t_n + dt)           │
│   u_{n+1} = u_n + dt/6·(k₁ + 2k₂ + 2k₃ + k₄)              │
│   v_{n+1} = v_n + dt/6·(k₁ + 2k₂ + 2k₃ + k₄)              │
│   任意时刻查询: t ∈ ℝ⁺                                    │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 高斯发射（观测模型）                                        │
│   u_obs ~ N(u + μ, σ²)                                     │
│   支持概率预测和不确定性量化                                │
└─────────────────────────────────────────────────────────────┘
```

**TelePiT直接多周预测流程**：

```
输入: X_t (初始日ERA5)
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Encoder + Multi-scale + Transformer                         │
│   多尺度分解: X → {X^(0), ..., X^(L)}                      │
│   每个频带独立Transformer处理                               │
│   Teleconnection-Aware Attention                          │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Forecast Head (多分支输出)                                   │
│   ├─ 分支1: 预测周 3-4 平均                                │
│   │    Y_{week3-4} = Head1(Z_final)                       │
│   └─ 分支2: 预测周 5-6 平均                                │
│        Y_{week5-6} = Head2(Z_final)                       │
└─────────────────────────────────────────────────────────────┘
  ↓
  输出: (Y_{week3-4}, Y_{week5-6})
```

#### 【设计层】

**设计动机**：
- **多时间尺度适配**：6h捕捉昼夜循环，24h关注天气尺度
- **数据增强**：同一时间段可构造多种 $\delta t$ 组合
- **误差平均**：多路径集成降低单一路径的累积误差
- **训练-推理一致**：Replay Buffer缓解训练-推理不匹配

**痛点解决**：
- **固定步长难以平衡**：短期精度 vs 长期稳定性
- **自回归误差累积**：多步微调/Replay Buffer缓解
- **时间离散化**：Neural ODE使用连续时间，消除时间步长依赖

**创新突破**：

- **Stormer**：首次在气象AI中使用随机化动力学目标 + AdaLN + 多路径集成；随机时间步长 $\delta t \in \{6h, 12h, 24h\}$ 提供数据增强
- **FengWu Replay Buffer**：首次在气象AI中应用，CPU缓存历史预测，显存友好
- **GraphCast**：12步自回归训练（3天），平衡训练成本和长期技巧
- **ClimODE**：二阶ODE（学习流速的时间导数），支持任意时刻查询
- **TelePiT**：直接预测多周平均，适配S2S任务特点，避免逐日累积误差

#### 【对比层】

**五流派时间建模对比（大型对比表）**：

| 维度 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **时间步长** | 固定6h/随机 | 固定6h | 固定6h | 扩散时间t | 连续 |
| **时间条件化** | AdaLN（部分） | 无 | 无 | 扩散时间步 | 连续t |
| **推理方式** | 自回归/多路径 | 自回归 | 自回归 | ODE求解 | ODE积分 |
| **训练-推理匹配** | 多步微调 | 两步微调 | 自回归 | 扩散训练 | ODE训练 |
| **累积误差处理** | 多步微调/Replay | 两步微调 | 自回归 | 扩散天然稳定 | 积分控制 |
| **时间灵活性** | 多路径（Stormer） | 多路径 | 固定步长 | 任意t | 任意t |
| **计算成本（推理）** | O(T) | O(T) | O(T) | O(T_diff) | O(T_ODE) |
| **长期稳定性** | 中（需微调） | 中 | 中 | 高 | 高 |

**各模型时间策略详细对比**：

| 模型 | 时间步长 | 训练方式 | 推理方式 | 累积误差缓解 | 最长时效 | 流派 |
|------|---------|---------|---------|------------|---------|------|
| FengWu | 固定6h | 单步+Replay | 自回归56步 | Replay Buffer | 14天 | Transformer |
| Swin V2 | 固定6h | 单步+4/8步微调 | 自回归 | 多步微调 | **未说明** | Transformer |
| Stormer | 随机{6h,12h,24h} | 随机单步+多步 | 多路径集成 | 随机化+多路径 | **未说明** | Transformer |
| AERIS | 6h/24h（双） | 扩散单步 | 自回归90天 | 扩散天然稳定 | 90天 | Transformer |
| FourCastNet | 固定6h | 单步+两步微调 | 自回归 | 两步微调 | 7-10天 | Fourier |
| GraphCast | 固定6h | 自回归12步 | 自回归 | 自回归 | 10天 | Graph |
| CorrDiff | T步 | 扩散 | ODE求解 | 扩散 | 未来时刻 | Generative |
| ClimODE | 连续 | Neural ODE | RK4积分 | ODE控制 | 任意 | ODE |
| TelePiT | 周平均 | 直接预测 | 直接 | 无累积 | 6周 | Transformer |
| NowcastNet | 固定10min | 单步 | 自回归 | 无 | 2小时 | ODE |

**多步微调收益量化**：

| 模型 | 基线（单步） | +多步微调 | 改善幅度 | 多步配置 |
|------|------------|----------|---------|---------|
| Swin V2 | Z500 RMSE 700m | 600m | **-14%** | 8步微调 |
| Stormer | **未说明** | SOTA | 最优 | 4/8步 |
| FourCastNet | **未说明** | 显著提升3天+ | 显著 | 两步微调 |
| FengWu | **未说明** | 优于GraphCast | 显著 | Replay Buffer |

**权衡评估**：

**随机时间步长（Stormer）**：
- ✅ 数据增强，同一样本构造多种训练目标
- ✅ 单模型覆盖多时间尺度
- ✅ 多路径推理提升稳定性
- ❌ 训练复杂度增加
- ❌ 推理需多路径计算（成本增加3倍）

**多步微调**：
- ✅ 直接优化长期目标，收益显著（RMSE降低10-15%）
- ❌ 显存需求高（8步需8倍中间激活）
- ❌ 训练时间长

**Replay Buffer（FengWu）**：
- ✅ 显存友好（Buffer存CPU，GPU只存batch）
- ✅ 近似多步训练效果
- ❌ 采样策略需调参
- ❌ Buffer质量依赖模型早期性能

**扩散模型（Generative/AERIS）**：
- ✅ 天然稳定，90天预报不发散
- ✅ 自然概率分布，支持集合预报
- ❌ 推理成本高（10+次ODE求解）
- ❌ 训练复杂（需调噪调度）

**连续时间（ClimODE）**：
- ✅ 任意时刻查询，无需离散化
- ✅ 物理一致性最强
- ❌ 训练可能收敛慢
- ❌ 数值稳定性需注意

**直接��周预测（TelePiT）**：
- ✅ 避免逐日累积误差
- ✅ 过滤高频天气噪声
- ❌ 丢失逐日变率信息
- ❌ 仅适用于S2S周平均预报

**适用场景**：
- **随机时间步长 + 多路径**：需要最高长期技巧且计算资源充足
- **多步微调（4-8步）**：需要显著提升中期预报（>3天）技巧
- **两步微调**：计算资源有限但希望提升中期技巧
- **Replay Buffer**：超高维输入（>100通道）和显存受限
- **扩散模型**：需要90天以上S2S预报和概率量化
- **连续时间ODE**：需要物理一致性和任意时刻查询
- **直接周预测**：S2S周平均预报

---

## 【第五部分：训练与优化范式】

### 5.1 时间演化策略范式

#### 【原理描述】

**核心定义**：时间演化策略定义了模型如何学习大气状态的时序推进机制——从单步预测到多步自回归，从确定性映射到概率分布，是气象AI实现任意时效预报的核心技术路径。

**数学推导**：

##### 1. 单步确定性预测（各流派通用基础）

$$\hat{X}_{t+\Delta t} = f_\theta(X_t)$$

损失函数：
$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{t \sim \mathcal{T}}\left[\left\|f_\theta(X_t) - X_{t+\Delta t}\right\|^2\right]$$

其中 $\Delta t$ 通常取6小时（天气预报的标准时间步长）。

##### 2. 残差预测（Transformer/Generative流派）

Transformer流派-Stormer、Generative流派-AERIS采用残差预测，模型不直接预测下一时刻状态，而是预测状态增量：

$$\Delta X = X_{t+\Delta t} - X_t$$

$$\hat{\Delta X} = f_\theta(X_t, \Delta t)$$

$$\mathcal{L}_{\text{residual}} = \left\|f_\theta(X_t, \Delta t) - (X_{t+\Delta t} - X_t)\right\|^2$$

物理意义：大气状态的绝对值变化范围大，但增量相对平滑，残差预测降低预测值的动态范围，利于网络学习。

##### 3. 自回归多步展开

$$\hat{X}_{t+k\Delta t} = \underbrace{f_\theta \circ f_\theta \circ \cdots \circ f_\theta}_{k \text{ 次}}(X_t)$$

训练时通常使用Teacher Forcing（输入真实场），推理时使用自回归（输入前一预测场），导致训练-推理不匹配（train-test mismatch）。

##### 4. 概率预测（Transformer流派-FengWu）

FengWu输出均值和方差，隐式建模预测不确定性：

$$p\left(X_{t+\Delta t} \mid X_t\right) = \mathcal{N}\left(\mu_\theta(X_t), \sigma_\theta^2(X_t)\right)$$

负对数似然损失：
$$\mathcal{L}_{\text{NLL}} = \frac{1}{2}\sum_{c,i,j}\left[\log\left(\sigma_{cij}^2\right) + \frac{\left(X_{t+\Delta t,cij} - \mu_{cij}\right)^2}{\sigma_{cij}^2}\right]$$

其中 $\mu = f_{\theta,\mu}(X_t)$，$\sigma = \text{softplus}(f_{\theta,\sigma}(X_t))$。

##### 5. 扩散生成（Generative/Transformer流派-AERIS）

AERIS采用扩散概率模型，在隐空间中直接生成状态增量：

$$\mathbf{x}_t = \cos(t) \cdot \mathbf{x}_0 + \sin(t) \cdot \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

推理时通过PF-ODE求解（DPM-Solver++，10步）从噪声生成预测场。

##### 6. 随机化时间步长（Transformer流派-Stormer）

$$\delta t \sim \mathcal{U}\{6\text{h}, 12\text{h}, 24\text{h}\}$$

$$\hat{X}_{t+\delta t} = f_\theta(X_t, \delta t)$$

物理依据：大气具有多时间尺度——6h捕捉日变化，12h/24h捕捉天气系统演变。随机化使模型接触多样化的动力学目标。

##### 7. 连续时间演化（ODE流派-ClimODE）

$$\frac{d\mathbf{h}(t)}{dt} = f_\theta(\mathbf{h}(t), t)$$

$$\mathbf{h}(t_1) = \mathbf{h}(t_0) + \int_{t_0}^{t_1} f_\theta(\mathbf{h}(\tau), \tau) d\tau$$

通过RK4数值积分实现任意时刻外推，无需离散时间步长的约束。

**理论依据**：
- 大气演化遵循偏微分方程（原始方程组），数据驱动模型学习隐式的时间演化算子
- 大气混沌敏感性要求概率建模（Butterfly Effect）
- 训练-推理不匹配是自回归模型的核心挑战
- 多时间尺度建模可增强模型的泛化能力

#### 【数据规格层】

**时间步长与预报时效配置对比（五流派全览）**：

| 模型 | 流派 | 基础步长 | 训练方式 | 推理方式 | 最长时效 | 可变步长 |
|------|------|---------|---------|---------|---------|---------|
| FengWu | Transformer | 6h | 单步 + Replay Buffer | 自回归56步 | 14天 | 否 |
| Swin V2 | Transformer | 6h | 单步 + 多步微调(4/8步) | 自回归 | **未说明** | 否 |
| Stormer | Transformer | {6,12,24}h | 随机单步 + 多步微调 | 多路径集成 | **未说明** | 是 |
| AERIS | Transformer/Generative | 6h/24h（双模型） | 扩散单步 | PF-ODE积分90天 | 90天(S2S) | 是 |
| FourCastNet | Fourier | 6h | 单步 + 两步微调 | 自回归 | 7-10天 | 否 |
| GraphCast | Graph | 6h | 自回归12步训练 | 自回归 | 10天 | 否 |
| GenCast | Graph | 6h | 自回归 | 自回归+集合 | 10-15天 | 否 |
| AIFS | Graph | 6h | 自回归 | 自回归+集合 | 10天 | 否 |
| CorrDiff | Generative | T步（不固定） | 扩散 | ODE求解 | 未来序列 | 否 |
| ClimODE | ODE | 连续 | Neural ODE | RK4积分 | 任意 | 是 |
| TelePiT | Transformer | 周平均 | 直接预测 | 直接 | 6周(S2S) | 否 |
| NowcastNet | ODE | 10min | 单步 | 自回归 | 2小时 | 否 |
| FuXi-TC | Generative | 6h | 单步 | 自回归 | TC强度 | 否 |
| PHYS-Diff | Generative | 6h | 扩散条件 | PF-ODE | TC轨迹 | 否 |

**训练数据配置**：

| 模型 | 训练数据年份 | 训练分辨率 | 验证/测试年份 | 数据集 |
|------|------------|-----------|--------------|--------|
| FengWu | 1979-2017 | 0.25° ERA5 | 2018 | WeatherBench |
| Swin V2 | 1979-2015 | 0.25° ERA5 | 2018 | WeatherBench |
| Stormer | 1979-2018 | 0.25° ERA5 | **未说明** | WeatherBench |
| AERIS | 1979-2018 | 0.25° ERA5 | 2018-2020 | WeatherBench |
| FourCastNet | 1979-2015 | 0.25° ERA5 | 2018 | WeatherBench |
| GraphCast | 1979-2017 | 0.25° ERA5 | 2018 | WeatherBench |
| GenCast | 1979-2018 | 0.25° ERA5 | **未说明** | WeatherBench |
| CorrDiff | **未说明** | 2km台湾区域 | **未说明** | ERA5+雷达 |
| ClimODE | **未说明** | 5.625°ERA5 | **未说明** | ERA5 |

#### 【架构层】

**计算流程图（五流派时间演化通用范式）**：

```
【训练阶段】

真实场序列 [X_{t-Δt}, X_t, X_{t+Δt}, ...]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. 输入构建                                                  │
│    Transformer: X_t (当前状态)                              │
│    Graph: [X_{t-Δt}, X_t] (双时刻输入)                     │
│    Fourier: X_t                                             │
│    Generative: x_t = cos(t)x_0 + sin(t)z                  │
│    ODE: u(t₀), v(t₀) (连续状态)                            │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 时间演化计算                                               │
│    Transformer-SwinV2: X̂ = f_θ(X_t)                        │
│    Transformer-Stormer: ΔX̂ = f_θ(X_t, δt), δt∈{6,12,24}h │
│    Transformer-FengWu: (μ,σ) = f_θ(X_t)                     │
│    Fourier-FourCastNet: X̂ = f_θ(X_t)                       │
│    Graph-GraphCast: X̂ = f_θ(X_t, X_{t-Δt})                │
│    Generative-AERIS: v̂_t = f_θ(x_t, x_{t-1}, forcings)    │
│    ODE-ClimODE: du_k/dt = -v_k·∇u_k - u_k∇·v_k            │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 损失计算                                                   │
│    确定性: L = ||X̂ - X_{t+Δt}||²                          │
│    概率(FengWu): L = 0.5[log(σ²) + (X-μ)²/σ²]              │
│    残差(Stormer): L = ||ΔX̂ - (X_{t+Δt}-X_t)||²             │
│    扩散(AERIS): L = ||σ·F_θ(x_t/σ,t) - v_t||²             │
│    ODE(ClimODE): L = ||u(t₁) - u_true(t₁)||²                │
└─────────────────────────────────────────────────────────────┘

【推理阶段】

初始状态 X_0
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 自回归滚动（确定性/概率模型）                                  │
│    for step in 1..N:                                        │
│      X̂_{step} = f_θ(X̂_{step-1})                          │
│      # 累积预测误差                                          │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 多路径集成（Stormer）或集合平均（AERIS/FourCastNet）           │
│    X_final = (1/M)Σ X̂^{(m)}                               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ PF-ODE求解（扩散模型AERIS）                                   │
│    10步DPM-Solver++积分，从x_{π/2}生成x_0                   │
└─────────────────────────────────────────────────────────────┘
```

**Stormer多路径推理详细流程**：

```
推理目标: T = 24h 预报
  ↓
┌─────────────────────────────────────┐
│ 路径1: [6h, 6h, 6h, 6h]            │
│   X_0 → X̂_6h → X̂_12h → X̂_18h → X̂_24h │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 路径2: [12h, 12h]                  │
│   X_0 → X̂_12h → X̂_24h             │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 路径3: [24h]                        │
│   X_0 → X̂_24h                      │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 集成平均                             │
│   X̂_24h_final = mean(X̂_24h^{(1,2,3)}) │
└─────────────────────────────────────┘
```

#### 【设计层】

**设计动机**：
- **单步确定性**：简单高效，梯度流稳定，适合中短期预报
- **残差预测**：变化量动态范围小，网络更易学习（尤其对大值变量如Z500）
- **概率预测**：自动学习变量权重，输出不确定性估计
- **多步微调**：直接优化长期目标，缓解train-test mismatch
- **扩散生成**：自然建模预测不确定性，长期稳定性强

**痛点解决（五流派对照）**：

| 问题 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **自回归误差累积** | 多步微调/Replay Buffer | 两步微调 | 自回归12步 | 扩散天然稳定 | ODE积分稳定 |
| **训练-推理不匹配** | 多步微调 | 两步微调 | 12步训练 | 扩散条件化 | 无（连续） |
| **时间尺度单一** | 随机Δt(Stormer) | 固定6h | 固定6h | 扩散无所谓 | 连续任意 |
| **不确定性量化** | NLL/扩散 | 初始场扰动 | 集合平均 | 天然概率分布 | 贝叶斯ODE |
| **长期稳定性** | 中（需微调） | 中（两步） | 中（12步限制） | 高（90天） | 高（数值稳定） |

**创新突破（五流派）**：

- **Transformer流派**：
  - **FengWu Replay Buffer**：首次在气象AI中应用，CPU缓存历史预测，GPU仅存batch，近似多步训练效果
  - **Stormer随机Δt**：数据增强+多时间尺度统一建模，同一样本构造多种训练目标
  - **AERIS扩散模型**：首个亿级参数扩散天气模型，10ExaFLOPS训练，90天S2S预报稳定
  - **Swin V2系统消融**：量化多步微调收益（8步微调后Z500 RMSE从700m降至600m，-14%）

- **Fourier/Operator流派**：
  - **FourCastNet两步微调**：最小自回归单元捕捉误差传播机制，仅需2倍显存
  - **SFNO**：球谐域算子学习，长期自回归预报稳定

- **Graph流派**：
  - **GraphCast双时刻输入**：输入 $[X_{t-\Delta t}, X_t]$，使模型学习趋势信息，提升预测稳定性
  - **GenCast集合扩散**：图Transformer + 扩散生成，50个成员的集合预报

- **Generative流派**：
  - **CorrDiff**：跨分辨率扩散，从ERA5 0.25°降尺度到2km区域高分辨率
  - **PHYS-Diff**：TC轨迹预测的条件扩散，融合TC属性、环境场和未来强迫

- **ODE流派**：
  - **ClimODE二阶ODE**：学习流速的时间导数，自动满足守恒约束
  - **AGODE自适应边权**：根据物理参数动态调整图边权重

#### 【对比层】

**五流派时间演化策略综合对比**：

| 维度 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **基础时间步长** | 6h（固定或随机） | 6h（固定） | 6h（固定） | 可变 | 连续 |
| **训练方式** | 单步 + 多步微调 | 单步 + 两步微调 | 自回归12步 | 扩散单步 | ODE函数学习 |
| **推理方式** | 自回归/多路径 | 自回归 | 自回归 | ODE求解 | RK4积分 |
| **概率建模** | NLL或扩散 | 初始场扰动 | 集合平均 | 天然概率 | 贝叶斯 |
| **长期稳定性** | 中（需微调） | 中（两步） | 中（12步限制） | 高（90天） | 高（数值稳定） |
| **不确定性量化** | ✅（NLL/扩散） | ✅（扰动集合） | ✅（集合） | ✅（天然） | ⚠️（贝叶斯） |
| **多时间尺度** | ✅（Stormer随机Δt） | ❌（固定） | ❌（固定） | ⚠️（扩散无所谓） | ✅（连续） |
| **训练-推理匹配** | 中（多步微调改善） | 中（两步微调） | 高（12步训练） | 高（条件化） | 完美（连续） |

**性能对比（10天Z500 RMSE）**：

| 模型 | 策略 | 10天RMSE | 相对IFS | 流派 |
|------|------|---------|---------|------|
| Stormer | 随机Δt + 8步微调 | ~600m | 优于 | Transformer |
| FengWu | 单步 + Replay Buffer | ~620m | 优于 | Transformer |
| AERIS | 扩散（集合平均） | ~630m | 相当 | Transformer/Generative |
| Swin V2 | 单步 + 8步微调 | ~650m | 优于 | Transformer |
| FourCastNet | 单步 + 两步微调 | **未说明** | 相当 | Fourier |
| GraphCast | 自回归12步 | **未说明** | 优于 | Graph |

**核心权衡矩阵**：

| 策略 | 训练成本 | 推理成本 | 长期精度 | 不确定性 | 适用场景 |
|------|---------|---------|---------|---------|---------|
| 单步确定性 | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ | 1-3天短期 |
| 多步微调(4-8步) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | 3-7天中期 |
| Replay Buffer | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | 高维输入+显存受限 |
| 随机Δt+多路径 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ | 最高长���技巧 |
| NLL概率输出 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | 自动权重学习 |
| 扩散生成 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅✅ | 90天S2S+集合 |
| 初始场扰动集合 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | 大规模集合(100+) |
| 连续时间ODE | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ | 物理建模+任意时刻 |

（注：⭐越多表示成本/效果越高）

**关键洞察**：

1. **训练-推理不匹配**是自回归模型的核心瓶颈。GraphCast通过12步训练部分缓解，Swin V2通过8步微调改善，Stormer通过随机Δt+多路径进一步提升。
2. **扩散模型**从根本上有望消除误差累积——AERIS在90天S2S预报中仍保持稳定，但代价是推理成本增加10倍以上。
3. **时间步长选择**是精度与稳定性的权衡点：6h是短期精度的最优选择，但对于长期预报，Stormer的随机Δt策略证明了更大步长的价值。
4. **概率建模的必要性**随预报时效增长而增加——3天以内确定性模型足够，7天以上需要集合或不确定性估计。

---

### 5.2 长时效训练策略范式

#### 【原理描述】

**核心定义**：长时效训练策略旨在解决自回归预报中误差逐级累积放大、训练分布与推理分布不一致的问题，通过多步微调、记忆回放、自适应步长等机制提升模型在中期（3-7天）和长期（7天以上）预报的技能。

**数学推导**：

##### 1. 多步微调（Multi-Step Fine-tuning）

阶段1（单步预训练）：
$$\mathcal{L}_1 = \mathbb{E}_{t \sim \mathcal{T}}\left[\left\|f_\theta(X_t) - X_{t+\Delta t}\right\|^2\right]$$

阶段2（K步微调）：
$$\mathcal{L}_K = \mathbb{E}_{t \sim \mathcal{T}}\left[\frac{1}{K}\sum_{k=1}^{K}\left\|f_\theta^{(k)}(X_t) - X_{t+k\Delta t}\right\|^2\right]$$

其中 $f_\theta^{(k)}$ 表示 $k$ 次自回归应用后的状态。

阶段3（可选，K'步微调，K'>K）：
$$\mathcal{L}_{K'} = \mathbb{E}_{t \sim \mathcal{T}}\left[\frac{1}{K'}\sum_{k=1}^{K'}\left\|f_\theta^{(k)}(X_t) - X_{t+k\Delta t}\right\|^2\right]$$

学习率调整：$\text{lr}_{\text{ft}} = \alpha \cdot \text{lr}_{\text{pretrain}}$，通常 $\alpha \in [0.1, 0.5]$。

##### 2. Replay Buffer（FengWu，Transformer流派）

Replay Buffer在CPU内存中维护一个历史预测缓冲区 $\mathcal{B}$，训练时以概率 $p_{\text{replay}}$ 采样缓冲区的预测，而非真实场：

$$X_{\text{input}} = \begin{cases} X_t \sim \mathcal{D}_{\text{real}} & \text{概率 } 1 - p_{\text{replay}} \\ \hat{X}_i \sim \mathcal{B} & \text{概率 } p_{\text{replay}} \end{cases}$$

Buffer更新（每隔 $T_{\text{update}}$ 步）：
$$\mathcal{B} \leftarrow \mathcal{B} \cup \left\{\hat{X}_t = \mu_\theta(X_t)\right\}$$

物理意义：让模型在训练阶段就接触到自回归推理中会遇到的"带误差的输入"，建立对预测误差的鲁棒性。

##### 3. 两步微调（FourCastNet，Fourier流派）

链式预测的损失函数：
$$\mathcal{L}_{\text{2-step}} = \left\|\mathbf{X}_1 - \mathbf{X}(t+\Delta t)\right\|^2 + \left\|\mathbf{X}_2 - \mathbf{X}(t+2\Delta t)\right\|^2$$

其中：
$$\mathbf{X}_1 = \mathcal{F}_\theta(\mathbf{X}(t)), \quad \mathbf{X}_2 = \mathcal{F}_\theta(\mathbf{X}_1)$$

这是最小化的多步训练（仅2步），平衡了训练成本与长期技巧。

##### 4. 自适应时间步长训练（Stormer）

$$\delta t \sim \mathcal{U}\{6\text{h}, 12\text{h}, 24\text{h}\}$$

$$\mathcal{L}_{\text{adapt}} = \mathbb{E}_{\delta t}\left[\left\|f_\theta(X_t, \delta t) - X_{t+\delta t}\right\|^2\right]$$

优势：同一时间窗口可构造多种训练目标（6h/12h/24h），数据利用效率高。

##### 5. 累积损失展开（Generalized）

$$f_\theta^{(k)}(X_t) = \underbrace{f_\theta \circ f_\theta \circ \cdots \circ f_\theta}_{k \text{ 次}}(X_t)$$

$$\mathcal{L}_{\text{rollout}} = \sum_{k=1}^{K} \lambda_k \cdot \mathbb{E}\left[\left\|f_\theta^{(k)}(X_t) - X_{t+k\Delta t}\right\|^2\right]$$

其中 $\lambda_k$ 为各步权重。

**理论依据**：
- 训练-推理不匹配（Train-Test Mismatch）：训练时输入真实场，推理时输入前一预测场，导致分布偏移
- 误差累积：每步预测的微小误差在自回归链中指数级放大（混沌敏感性）
- 梯度消失：多步反向传播时梯度指数衰减，深层步几乎无梯度更新

#### 【数据规格层】

**五流派训练阶段配置全览**：

| 模型 | 流派 | 阶段1 | 阶段2 | 阶段3 | 初始lr | 微调lr |
|------|------|-------|-------|-------|--------|--------|
| Swin V2 | Transformer | 单步70epoch | 4步微调15epoch | 8步微调15epoch | 1e-3 | 1e-4 |
| FengWu | Transformer | 单步+Replay Buffer | - | - | 5e-4 | N/A |
| Stormer | Transformer | 随机Δt单步 | 4步微调 | 8步微调 | 1e-3 | 1e-4 |
| AERIS | Transformer | 扩散单步 | - | - | **未说明** | N/A |
| FourCastNet | Fourier | 单步80epoch | 两步微调50epoch | - | 5e-4 | 1e-4 |
| GraphCast | Graph | 自回归12步 | - | - | **未说明** | N/A |
| TelePiT | Transformer | 直接预测周平均 | - | - | **未说明** | N/A |
| ClimODE | ODE | 连续ODE函数 | - | - | **未说明** | N/A |

**显存与计算成本对比**：

| 策略 | 显存需求（相对单步） | 计算量（相对单步） | 训练时间（相对单步） |
|------|-------------------|-----------------|-------------------|
| 单步 | 1× | 1× | 1× |
| 2步微调 | ~2× | ~2× | ~1.5× |
| 4步微调 | ~4× | ~4× | ~2× |
| 8步微调 | ~8× | ~8× | ~3× |
| Replay Buffer | ~1× | ~1× | ~1.5× |

#### 【架构层】

**计算流程图（通用长时效训练范式）**：

```
【多阶段训练流程】

阶段1: 单步预训练
  for epoch in 1..E1:
    X_t → f_θ(X_t) → X̂_{t+Δt}
    Loss = ||X̂ - X_{t+Δt}||²
    Optimizer: Adam(lr=1e-3)
    # 快速收敛，学习基本动力学
  ↓
阶段2: K步微调 (K=4示例)
  加载阶段1权重
  lr ← lr × 0.1
  for epoch in 1..E2:
    X_t → X̂_1 = f_θ(X_t)
    X̂_2 = f_θ(X̂_1)
    X̂_3 = f_θ(X̂_2)
    X̂_4 = f_θ(X̂_3)
    Loss = (1/4)Σ ||X̂_k - X_{t+kΔt}||²
    # 梯度检查点节省显存
  ↓
阶段3: K'步微调 (K'=8, 可选)
  加载阶段2权重
  lr ← lr × 0.1
  for epoch in 1..E3:
    X_t → 8步自回归 → X̂_8Δt
    Loss = (1/8)Σ ||X̂_k - X_{t+kΔt}||²
    # 直接优化长期目标
```

**Replay Buffer详细流程（FengWu）**：

```
初始化:
  Buffer = deque(maxlen=B)  # CPU内存
  p_replay = 0.5  # 回放概率

训练循环:
  for iteration:
    ┌─────────────────────────────────────┐
    │ 1. 采样输入                          │
    │   if random() < p_replay:           │
    │     X_input = random.choice(Buffer)  │
    │   else:                              │
    │     X_input = next(batch)  # 真实场 │
    └─────────────────────────────────────┘
      ↓
    ┌─────────────────────────────────────┐
    │ 2. 前向预测                          │
    │   μ, σ = f_θ(X_input)               │
    │   Loss = NLL(μ, σ, X_true)          │
    └─────────────────────────────────────┘
      ↓
    ┌─────────────────────────────────────┐
    │ 3. 反向传播 + 优化                   │
    │   loss.backward()                    │
    │   optimizer.step()                   │
    └─────────────────────────────────────┘
      ↓
    ┌─────────────────────────────────────┐
    │ 4. 更新Buffer（非训练步骤）           │
    │   with torch.no_grad():             │
    │     X_pred = μ.detach().cpu()        │
    │     Buffer.append(X_pred)            │
    └─────────────────────────────────────┘

Buffer大小: 通常1000-5000个样本（取决于CPU内存）
更新频率: 每10-50次迭代更新一次Buffer
```

**梯度检查点（Gradient Checkpointing）**：

多步微调中，中间激活需存储导致显存爆炸。梯度检查点策略仅保存每隔一段的激活，以额外计算换取显存节省。

#### 【设计层】

**设计动机**：
- **暴露于误差**：让模型在训练时就见到带误差的输入，建立对预测偏差的鲁棒性
- **直接优化长期目标**：多步损失直接惩罚长期偏离，避免逐级误差放大
- **内存效率**：Replay Buffer将历史预测存CPU，GPU仅存当前batch；梯度检查点以计算换显存
- **渐进式解冻**：从单步到多步、从大lr到小lr，训练更稳定

**痛点解决（五流派对照）**：

| 问题 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **显存爆炸** | 梯度检查点 | 两步（天然省显存） | 12步需检查点 | 扩散天然稳定 | ODE检查点 |
| **梯度消失** | 分阶段lr递减 | 两步 | 多步 | 无（单步扩散） | 连续梯度 |
| **误差累积** | 多步微调+多路径 | 两步微调 | 12步训练 | 扩散解耦 | 数值积分 |
| **训练-推理不匹配** | Replay Buffer | 两步微调 | Teacher Forcing | 条件化 | 完美匹配 |

**创新突破**：
- **Swin V2系统消融**：首次量化多步微调的收益——8步微调后Z500 RMSE从700m降至600m（-14%），5天以上收益尤其显著
- **FengWu Replay Buffer**：首次在气象AI中应用，显存友好（GPU占用与单步相同），近似多步训练效果
- **Stormer多路径推理**：训练随机Δt使模型具备多时间尺度泛化能力，推理时通过多路径集成进一步提升

#### 【对比层】

**长时效训练策略五流派综合对比**：

| 策略 | 显存效率 | 训练成本 | 长期精度 | 实现复杂度 | 主要采用 |
|------|---------|---------|---------|---------|---------|
| 多步微调(4-8步) | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中（需检查点） | Transformer |
| 两步微调 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 低 | Fourier |
| Replay Buffer | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 中（需Buffer管理） | Transformer |
| 扩散模型 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高（调参扩散） | Generative |
| 连续ODE | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 高（数值稳定性） | ODE |
| 直接预测 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 低 | TelePiT |
| 自回归12步 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 低 | Graph |

**多步微调收益量化**：

| 模型 | 基线（单步） | +多步策略 | 改善幅度 | 配置 |
|------|------------|---------|---------|------|
| Swin V2 | Z500 RMSE 700m | 600m | **-14%** | 8步微调 |
| Stormer | **未说明** | SOTA | 最优 | 随机Δt+8步 |
| FourCastNet | **未说明** | 显著提升3天+ | 显著 | 两步微调 |
| FengWu | **未说明** | 优于GraphCast | 显著 | Replay Buffer |

**关键权衡**：

| 策略 | ✅ 优点 | ❌ 缺点 |
|------|--------|--------|
| 多步微调(4-8步) | 直接优化长期目标，收益显著（RMSE降低10-15%） | 显存需求高（8步需8倍），训练时间长 |
| 两步微调 | 显存友好（仅2倍），实现简单，显著提升3天+ | 覆盖误差模式有限，7天以上改善不如8步 |
| Replay Buffer | 显存极友好（GPU同单步），近似多步效果 | 采样策略需调参，Buffer质量依赖模型早期性能 |
| 扩散模型 | 天然稳定（90天不发散），自然概率分布 | 推理成本高（10+次前向），训练复杂（需调噪调度） |
| 连续时间ODE | 任意时刻查询，训练-推理完美匹配，物理一致 | 数值稳定性需关注，训练可能收敛慢 |
| 直接周预测 | 避免逐日累积误差，过滤高频噪声 | 丢失逐日变率，仅适用于S2S周平均预报 |

**适用场景决策树**：

```
预报时效?
  ├─ < 3天 → 单步确定性（Swin V2, FourCastNet）
  ├─ 3-7天 → 多步微调(4步) 或 两步微调
  ├─ 7-14天 → 多步微调(8步) 或 Replay Buffer
  ├─ > 14天(S2S) → 扩散模型（AERIS）或 连续时间ODE
  └─ 概率集合 → 扩散生成（AERIS）或 初始场扰动（FourCastNet）
```

---

### 5.3 多任务与不确定性建模范式

#### 【原理描述】

**核心定义**：多任务与不确定性建模将气象预报扩展为联合预测多个物理量+量化预测不确定性的复合任务，通过不确定性感知损失、集合预报生成、变量权重学习等机制提升模型的整体预报质量与可靠性。

**数学推导**：

##### 1. 不确定性感知损失（FengWu，Transformer流派）

模型同时输出预测均值 $\mu_\theta(X_t)$ 和预测方差 $\sigma_\theta^2(X_t)$：

$$\mathcal{L}_{\text{NLL}} = \frac{1}{CHW}\sum_{c,i,j}\left[\frac{\left(X_{t+\Delta t,cij} - \mu_{cij}\right)^2}{2\sigma_{cij}^2} + \frac{1}{2}\log\left(\sigma_{cij}^2\right)\right]$$

物理机制：$\sigma_{cij}$ 大的位置（高不确定性），损失项自动被压缩，从而自动学习变量权重和区域权重。

##### 2. 多任务权重学习（Transformer流派-EWMoE）

$$\mathcal{L}_{\text{multi-task}} = \sum_{v \in \mathcal{V}} \alpha_v \cdot f(v) \cdot \mathcal{L}_v$$

其中 $f(v)$ 为可学习的变量重要性函数，$\alpha_v$ 为固定的气象先验权重。

##### 3. 扩散集合生成（AERIS，Transformer/Generative流派）

推理（PF-ODE，DPM-Solver++，10步）生成 $M$ 个独立样本后取集合平均或保留完整分布。

##### 4. 初始场扰动集合（FourCastNet，Fourier流派）

$$\mathbf{X}^{(e)}(0) = \mathbf{X}_{\text{true}}(0) + \sigma \cdot \boldsymbol{\xi}^{(e)}, \quad \boldsymbol{\xi}^{(e)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

其中 $\sigma = 0.3$（标准化空间），$e = 1, \ldots, E$。

##### 5. 降水诊断与对数变换（FourCastNet）

$$\widetilde{\text{TP}} = \log\left(1 + \frac{\text{TP}}{\epsilon}\right), \quad \epsilon = 10^{-5}$$

训练目标：$\tilde{y} = \log(1 + y/\epsilon)$，推理时指数还原。

##### 6. 扩散条件化策略（PreDiff，Generative流派）

条件 $c$ 包括：历史轨迹，大尺度环境场，静态地形等，通过Cross-Attention注入去噪网络。

**理论依据**：
- 不同变量预测难度差异巨大（如湿度场 vs 位势高度）
- 大气混沌性要求概率预报量化不确定性
- WMO业务标准要求集合预报的spread-skill ratio接近1

#### 【数据规格层】

**不确定性输出规格对比**：

| 模型 | 流派 | 输出类型 | 集合大小 | 校准质量 |
|------|------|---------|---------|---------|
| FengWu | Transformer | 均值+方差 | 可采样 | 中等 |
| AERIS | Transformer/Generative | 扩散样本 | 10-50 | 高 |
| FourCastNet | Fourier | 初始场扰动 | 100-1000 | 中等 |
| GenCast | Graph | 图扩散样本 | 50 | 高 |
| GraphCast | Graph | 确定性 | N/A | N/A |
| Swin V2 | Transformer | 确定性 | N/A | N/A |

#### 【架构层】

**计算流程图（五流派不确定性建模）**：

```
【不确定性感知训练（FengWu）】

输入 X_t [B, 189, H, W]
  ↓ 多模态编码（6个模态）→ 跨模态融合Transformer
  ↓ 双头解码: μ [B,189,H,W], σ = softplus(Decoder_σ(...))
  ↓ NLL损失: 0.5*log(σ²) + (X_true-μ)²/(2σ²)

【集合生成】

① 扩散采样（AERIS）:
  初始噪声 x_{π/2} ~ N(0, I)
    → PF-ODE 10步求解 → x_0^{(m)}
  M个独立样本 → 集合

② 初始场扰动（FourCastNet）:
  X^{(e)}(0) = X_true(0) + 0.3*ξ^{(e)}
    → 自回归预报E次
  E个成员 → 集合

③ NLL采样（FengWu）:
  μ, σ = f_θ(X_t)
    → X^{(m)} ~ N(μ, σ²)
  M个样本 → 集合
```

#### 【设计层】

**痛点解决（五流派对照）**：

| 问题 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **变量权重调参** | NLL自动学习 | 逐通道标准化 | 固定气压加权 | 端到端学习 | 无 |
| **集合生成** | 扩散采样 | 初始场扰动 | 图扩散 | 天然概率分布 | 贝叶斯 |
| **不确定性校准** | NLL校准 | 集合统计 | 扩散校准 | 天然校准 | 贝叶斯 |
| **极端降水** | 直接预测 | 对数变换诊断 | 直接预测 | 扩散生成 | 直接预测 |
| **spread-skill比** | ~0.8（AERIS） | 略欠发散 | ~1（GenCast） | ~0.9 | **未说明** |

**创新突破**：
- **FengWu NLL损失**：首次在气象AI中用不确定性损失替代手工权重，$\sigma$ 自动学习不同变量/区域的预测难度
- **AERIS扩散集合**：首个亿级参数扩散天气模型，spread-skill比接近IFS ENS
- **FourCastNet超大集合**：极高推理速度（7秒/100成员）使1000+成员集合预报成为可能
- **PreDiff条件扩散**：首次系统性将历史轨迹、环境场、强迫场作为条件注入扩散模型
- **CorrDiff跨分辨率扩散**：从0.25° ERA5到2km高分辨率的跨尺度扩散

#### 【对比层】

**五流派不确定性建模综合对比**：

| 维度 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|------|-----------|-----------------|-------|-----------|-----|
| **不确定性来源** | NLL学习的σ | 初始场高斯扰动 | 扩散采样 | 噪声采样天然 | 贝叶斯先验 |
| **集合大小** | 可采样（低M） | 100-1000+ | 50 | 10-50 | 贝叶斯积分 |
| **推理成本（相对单步）** | ~1× | E×（批量） | ~10×（扩散） | ~10× | 贝叶斯积分 |
| **校准质量** | 中等 | 中等 | 高 | 高 | 取决于先验 |
| **spread-skill比** | ~0.8 | <0.8 | ~1.0 | ~0.9 | **未说明** |
| **多任务权重** | ✅自动(NLL) | ❌固定 | ❌固定 | ⚠️端到端 | ❌ |

**评估指标对比**：

| 模型 | 3天CRPS | 7天CRPS | 集合大小 | 推理成本/100成员 |
|------|--------|---------|---------|------------------|
| IFS ENS | 基准 | 基准 | 51 | 数小时 |
| AERIS | 优于IFS | 与IFS相当 | 10-50 | 高（10×前向） |
| FourCastNet | **未说明** | **未说明** | 100-1000 | ~7秒 |
| FengWu | **未说明** | **未说明** | 可采样 | 中等 |
| GraphCast | **未说明** | **未说明** | 1（确定性） | 单次推理 |

**极端降水RQE对比（FourCastNet）**：

| 分位数 | RQE值 | 含义 |
|--------|-------|------|
| q=0.90 | -20% | 轻度低估 |
| q=0.95 | -25% | 中度低估 |
| q=0.99 | -35% | 严重低估 |
| q=0.999 | -40% | 极端低估 |

负值表示模型低估极端降水强度。扩散模型和高分辨率CorrDiff在此方面可能有改善。

**关键权衡**：

| 方法 | ✅ 优点 | ❌ 缺点 |
|------|--------|--------|
| NLL概率输出 | 自动权重学习，无需调参，输出置信区间 | 校准质量中等，推理同单步成本 |
| 扩散集合 | 高质量概率分布，长期稳定，校准好 | 推理成本高（10+倍），训练复杂 |
| 初始场扰动 | 实现极简，可超大集合（1000+），推理快 | 简单高斯扰动可能低估不确定性，缺模型不确定性 |
| 确定性模型 | 训练推理极简，效率最高 | 无不确定性量化 |
| 贝叶斯ODE | 理论优雅，量化参数不确定性 | 计算开销大，训练复杂 |

---

## 【第六部分：评估与验证范式】

### 6.1 评估指标范式

#### 【原理描述】

**核心定义**：气象预报评估指标体系是衡量模型技能的核心标准，需同时覆盖确定性精度（RMSE/ACC）、概率校准（CRPS）、物理一致性（能量谱）和极端事件能力（RQE），并与WMO/ECMWF等国际业务标准对齐。

**数学推导**：

##### 1. 纬度加权RMSE（全球模型通用基础）

$$\text{RMSE}(v, \tau) = \sqrt{\frac{\sum_{i,j} L(\phi_i) \cdot \left(X_{ij}^{\text{pred}} - X_{ij}^{\text{true}}\right)^2}{\sum_{i,j} L(\phi_i)}}$$

其中 $L(\phi_i) = \cos(\phi_i) / Z$ 为归一化纬度权重，$v$ 为变量，$\tau$ 为预报时效。

##### 2. 异常相关系数ACC（模式预测核心指标）

$$\text{ACC}(v, \tau) = \frac{\sum_{i,j} L(\phi_i) \cdot \left(X_{ij}^{\text{pred}} - C_{ij}\right) \cdot \left(X_{ij}^{\text{true}} - C_{ij}\right)}{\sqrt{\sum L \cdot \left(X^{\text{pred}} - C\right)^2} \cdot \sqrt{\sum L \cdot \left(X^{\text{true}} - C\right)^2}}$$

其中 $C_{ij}$ 为气候态（同期多年平均场）。

**技能阈值**：$\text{ACC}(Z500) > 0.6$ 视为有技巧预报。
- GraphCast/Pangu：10天达到0.6
- FengWu：10.75天达到0.6（首个AI模型突破10天）
- Stormer：超过10.75天

##### 3. 连续排序概率评分CRPS（概率预报综合指标）

$$\text{CRPS}(F, x) = \int_{-\infty}^{\infty} \left[F(y) - \mathbb{1}(y \geq x)\right]^2 dy$$

CRPS越低越好，CRPS=0表示完美概率预报。

##### 4. 相对分位数误差RQE（极端事件专用）

$$\text{RQE}(l) = \sum_{q \in Q} \frac{\mathbf{X}_{\text{pred}}^q(l) - \mathbf{X}_{\text{true}}^q(l)}{\mathbf{X}_{\text{true}}^q(l)}$$

其中 $q \in \{0.90, 0.95, 0.99, 0.999\}$ 为目标分位数。RQE负值表示低估极端事件。

##### 5. 能量谱（物理一致性检验）

$$E(k) = \frac{1}{2}\sum_{|\mathbf{k}'|=k} \left|\hat{u}(\mathbf{k}')\right|^2 + \left|\hat{v}(\mathbf{k}')\right|^2$$

用途：检测模型预测是否保留合理的高波数能量（避免过度平滑）。

##### 6. Spread-Skill Ratio（集合预报可靠性）

$$\text{SSR} = \frac{\overline{\text{Ensemble Spread}}}{\text{Ensemble RMSE}}$$

理想值 $\text{SSR} \approx 1$。

##### 7. Critical Success Index（CSI，降水验证）

$$\text{CSI} = \frac{\text{Hits}}{\text{Hits} + \text{Misses} + \text{False Alarms}}$$

**理论依据**：
- RMSE：L2范数，对大误差敏感，关注绝对精度
- ACC：Pearson相关，关注空间模式
- CRPS：概率预报综合评分，兼顾精度和离散度
- RQE：极端事件检测，不对称损失（低估危害更大）
- 能量谱：物理一致性，防止统计优化导致非物理平滑
- SSR：集合可靠性，spread与skill的一致性

#### 【数据规格层】

**WeatherBench2标准评估配置**：

| 维度 | 配置 |
|------|------|
| 训练数据 | 1979-01-01 至 2017-12-31 (ERA5再分析) |
| 验证数据 | 2018-01-01 至 2018-12-31 |
| 测试数据 | **部分模型使用2018，部分使用2020** |
| 空间分辨率 | 0.25° (721×1440 Lat-Lon网格) |
| 垂直层 | 13层（50, 100, 200, ..., 1000 hPa） |
| 核心评估变量 | Z500, T500, T850, T2m, U10, V10 |

**五流派评估变量与规模对比**：

| 模型 | 核心变量 | 总目标数 | 极端事件评估 | 流派 |
|------|---------|---------|------------|------|
| FengWu | Z500, T850, T2m, U/V风 | 880 | **未详细说明** | Transformer |
| Swin V2 | Z500, T2m, U10m | **未说明** | 能量谱+spread | Transformer |
| Stormer | Z500, T2m, T850, U10 | **未说明** | 极端事件案例 | Transformer |
| AERIS | Z500, T2m, U10, V10, T850 | WeatherBench2标准 | MJO/ENSO/SSR | Transformer |
| FourCastNet | Z500, T2m, U10, V10等 | 20变量全面评估 | RQE（极端降水） | Fourier |
| GraphCast | Z500, T850, T2m, U/V850, U/V10 | 880目标 | **未说明** | Graph |
| GenCast | 同GraphCast | 同GraphCast | 集合SSR | Graph |
| CorrDiff | T2m, U10, V10, 雷达反射率 | 4变量 | CSI | Generative |
| PHYS-Diff | TC位置、强度 | TC轨迹 | TC案例 | Generative |
| ClimODE | t2m, t, z, u10, v10 | 5变量 | **未说明** | ODE |

#### 【架构层】

**计算流程图（WeatherBench2标准评估管线）**：

```
【评估数据准备】

测试期 ERA5数据 [T_test, C, H, W]
气候态数据 [365_days, C, H, W]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. 模型批量推理                                              │
│    for t in T_test:                                          │
│      X̂_t = model(X_{t-Δt}, X_{t-2Δt}, ...)                 │
│    输出: predictions[τ, C, H, W]                             │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 逐时效逐变量计算                                          │
│    for τ in [1d, 3d, 5d, 7d, 10d]:                          │
│      for v in variables:                                      │
│        pred_v, true_v, clim_v                                │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 确定性指标                                                │
│    RMSE_v(τ) = sqrt(mean(L(φ) * (pred-true)²))              │
│    ACC_v(τ) = corr(L(φ)*(pred-clim), L(φ)*(true-clim))     │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 概率指标（集合模型）                                       │
│    if has_ensemble:                                          │
│      CRPS_v(τ) = mean(CRPS(ensemble, true))                 │
│      SSR_v(τ) = mean(spread) / RMSE_ensemble                 │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 能量谱分析（可选）                                         │
│    E_ratio(k) = E_pred(k) / E_true(k)                       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. 聚合与可视化                                              │
│    Global_Metric = mean_over_vars(RMSE)                      │
└─────────────────────────────────────────────────────────────┘
```

#### 【设计层】

**设计动机**：
- **业务对齐**：与ECMWF/NCEP/WMO等国际业务中心标准一致
- **多维度评估**：单一指标无法全面刻画模型技能——RMSE关注精度，ACC关注模式，CRPS关注概率
- **物理一致性**：能量谱防止深度学习为优化RMSE而过度平滑
- **极端事件**：RQE、CSI等专门评估暴雨、台风等高影响事件

**创新突破**：
- **Swin V2系统消融**：首次系统评估lagged ensemble的spread-skill比
- **AERIS MJO/ENSO评估**：首个在90天S2S预报中系统性评估MJO和ENSO指数
- **FourCastNet RQE**：首次在0.25°分辨率下系统性评估极端降水预测能力
- **WeatherBench2标准化**：构建了行业标准的评估基准

#### 【对比层】

**性能基准对比表（3天Z500 ACC）**：

| 模型 | 流派 | 3天ACC | 相对IFS HRES | 训练数据年份 |
|------|------|--------|------------|------------|
| IFS HRES | 物理模式 | ~0.98 | 基准 | N/A |
| Stormer | Transformer | ~0.985 | 优于 | 1979-2018 |
| Swin V2 | Transformer | ~0.98 | 相当/略优 | 1979-2015 |
| FengWu | Transformer | ~0.98 | 相当 | 1979-2015 |
| AERIS | Transformer/Generative | ~0.97 | 略逊（集合平均） | 1979-2018 |
| FourCastNet | Fourier | ~0.98 | 相当 | 1979-2015 |
| GraphCast | Graph | ~0.98 | 相当 | 1979-2017 |
| GenCast | Graph | ~0.98+ | 优于 | 1979-2018 |

**时效技能衰减对比（Z500 ACC随预报时效变化）**：

| 模型 | 1天 | 3天 | 5天 | 7天 | 10天 | 14天 | 30天 |
|------|-----|-----|-----|-----|------|------|------|
| IFS HRES | 0.99 | 0.98 | 0.95 | 0.90 | 0.80 | 0.65 | **未说明** |
| GraphCast | 0.99 | 0.98 | 0.95 | 0.90 | 0.80 | **未说明** | **未说明** |
| FengWu | 0.99 | 0.98 | 0.96 | 0.92 | **0.82** | **未说明** | **未说明** |
| AERIS | 0.99 | 0.98 | 0.95 | 0.90 | 0.80 | 0.70 | ~0.50 |
| FourCastNet | 0.99 | 0.98 | 0.94 | 0.88 | **未说明** | **未说明** | **未说明** |

**关键洞察**：
1. AI模型在3天以内的ACC与IFS相当或略优，但7天以上逐渐拉开差距
2. **FengWu是首个突破10天ACC>0.6的AI模型**（10.75天），Stormer进一步延伸
3. **AERIS在90天仍保持~0.5的ACC**，证明扩散模型在长期S2S预报中的稳定性优势
4. **FourCastNet的RQE显示所有深度学习模型都系统性低估极端降水**（q=0.99时RQE≈-35%）

---

### 6.2 下游任务验证范式

#### 【原理描述】

**核心定义**：下游任务验证通过实际应用场景（台风追踪、极端降水预警，风能评估、局地订正等）评估模型超越标准统计指标的实际业务价值。

**验证任务类型（五流派全览）**：

##### 1. 热带气旋（TC）追踪与强度预测

**方法（FourCastNet）**：MSLP局部最小值追踪台风眼，分析中心气压下降速率。

**代表案例**：Super Typhoon Mangkhut (96h预报)、Hurricane Michael (快速增强)。

##### 2. 极端降水与大气河事件

**方法（FourCastNet, CorrDiff）**：6小时累计降水场分析，阈值筛选极端降水，大气河诊断。

**代表案例**：Pineapple Express大气河事件 (36-72h预报准确刻画水汽带形态和登陆位置)。

##### 3. 局地订正与站点级预测

**方法（Local Off-Grid，Transformer流派）**：异质图（358个MADIS站点 + ERA5/HRRR格点），Delaunay三角剖分建立站点间连接。

**结果**：风误差降低80%，T2m误差降低25%。

##### 4. 长期气候信号评估

**方法（AERIS）**：MJO指数（RMM1, RMM2）、ENSO指数（Nino3.4区SST异常）。

**结果**：AERIS在90天MJO相关性与IFS ENS相当。

##### 5. 集合预报离散度评估

| 模型 | SSR | 含义 |
|------|-----|------|
| IFS ENS | ~1.0 | 完美校准 |
| AERIS | ~0.8 | 略欠发散（可接受） |
| FourCastNet | <0.8 | 欠发散 |
| GenCast | ~1.0 | 良好校准 |

##### 6. 跨分辨率降尺度验证

**方法（CorrDiff, FuXi-TC）**：雷达反射率 / 高密度站点观测对比，空间细节保留评估。

**结果**：CorrDiff实现ERA5到2km区域的12.5倍上采样，CSI评分显著提升。

**理论依据**：
- 统计指标可能掩盖模型在特定现象上的不足
- 下游任务直接关联业务需求和社会经济影响
- 下游验证是连接"学术性能"与"业务价值"的桥梁

#### 【数据规格层】

| 验证任务 | 数据集 | 关键指标 |
|---------|--------|---------|
| TC追踪 | IBTrACS + ERA5 | 路径误差(km), 强度偏差(hPa) |
| 极端降水 | GPM/IMERG雷达 | RQE, CSI, 命中率 |
| 局地订正 | MADIS站点观测 | 订正前后RMSE对比 |
| 长期信号 | OLR + SST | MJO ACC, Nino3.4相关 |
| 大气河 | IVT通量产品 | 登陆位置误差 |
| 降水降尺度 | 雷达回波 | CSI, bias |

#### 【架构层】

**TC追踪详细流程（FourCastNet）**：

```
预报序列: MSLP(t) ∈ [T=28, H=721, W=1440]
  ↓ 局部极值搜索（台风眼定位）
台风中心: (lat, lon, pressure) for t=6h, 12h, ..., 168h
  ↓
路径误差: haversine(pred, true)
强度偏差: p_pred - p_true
登陆时间: 首次跨越海岸线的时刻差
```

**局地订正详细流程（Local Off-Grid）**：

```
全球预报 → 双线性插值到站点
  → 异质图消息传递（站点←站点 + 站点←格点）
  → 站点级预报
  ↓
评估改善: RMSE_before - RMSE_after
```

#### 【设计层】

**创新突破**：
- **FourCastNet TC追踪**：首次用深度学习模型成功追踪台风快速增强过程，96h路径预报与ERA5高度吻合
- **Local Off-Grid局地订正**：首次系统评估图神经网络的站点级订正能力，风误差降低80%
- **AERIS 90天MJO/ENSO**：扩散模型在延伸期预报中首次保持MJO信号相关性
- **Stormer极端事件案例**：5天提前量准确预报极端天气事件
- **CorrDiff km级降尺度**：80M参数UNet实现ERA5到2km区域的12.5倍上采样

#### 【对比层】

**下游任务性能综合对比表**：

| 下游任务 | 传统基准 | Transformer | Fourier/Operator | Graph | Generative | 改善幅度 |
|---------|---------|------------|-----------------|-------|------------|---------|
| **TC追踪(96h)** | IFS | **未说明** | FourCastNet（与ERA5吻合） | **未说明** | FuXi-TC（条件扩散） | 接近ERA5 |
| **极端降水RQE** | IFS | **未说明** | -35%（q=0.99，低估） | **未说明** | CorrDiff（CSI高） | CSI提升 |
| **局地订正(风)** | 线性插值 | Local Off-Grid（-80%） | **未说明** | **未说明** | **未说明** | 风-80%, T2m-25% |
| **90天MJO** | IFS ENS | AERIS（相关性相当） | **未说明** | **未说明** | **未说明** | 与IFS ENS相当 |
| **集合SSR** | IFS ENS(~1.0) | AERIS(~0.8) | FourCastNet(<0.8) | GenCast(~1.0) | AERIS(~0.9) | 接近IFS |
| **大气河登陆** | IFS | **未说明** | FourCastNet(36-72h准确) | **未说明** | **未说明** | 登陆位置准确 |
| **降水降尺度** | 统计降尺度 | **未说明** | **未说明** | **未说明** | CorrDiff（2km） | 12.5×上采样 |

**应用场景适配指南**：

| 应用场景 | 推荐模型 | 下游验证指标 | 核心优势 | 流派 |
|---------|---------|------------|---------|------|
| **业务中期预报** | Stormer, FengWu | RMSE(5-10d), ACC(10d) | 确定性精度最高 | Transformer |
| **集合概率预报** | AERIS, GenCast | CRPS, SSR, 极端事件RQE | 自然概率分布+校准 | Transformer/Graph |
| **快速推理（应急）** | FourCastNet | 1-3天RMSE, TC追踪 | 7秒/100成员推理 | Fourier |
| **局地精细化** | Local Off-Grid | 站点RMSE, 风/T2m改善 | 站点级订正 | Transformer |
| **长期S2S** | AERIS | MJO ACC(90d), ENSO | 90天稳定，MJO保持 | Transformer |
| **极端事件预警** | FourCastNet, Stormer | RQE, TC路径误差 | 极端降水+TC案例验证 | Fourier/Transformer |
| **降水降尺度** | CorrDiff, FuXi-TC | CSI, 雷达对比 | km级高分辨率 | Generative |
| **TC强度预报** | FuXi-TC | DWT命中率, 强度偏差 | 条件扩散+WRF | Generative |
| **风能评估** | FourCastNet | 陆地/海洋ACC | 大集合+快速 | Fourier |
| **物理气候建模** | ClimODE | 变量守恒, 物理解耦 | 二阶ODE守恒 | ODE |

**关键洞察**：

1. **扩散模型（AERIS）**在长期S2S和概率预报中展现出独特优势，90天MJO相关性与IFS ENS相当——意味着AI在季节内尺度上已具备实用价值。

2. **局地订正（Local Off-Grid）**揭示了图神经网络在处理不规则观测网络上的天然优势——风误差降低80%具有直接业务意义。

3. **极端降水的系统性低估**是所有AI模型的共同瓶颈（FourCastNet RQE=-35% @ q=0.99），CorrDiff通过km级降尺度有望改善。

4. **FourCastNet的极速推理**（7秒/100成员）使超大集合预报成为可能，可能弥补其不确定性量化能力的不足。

5. **下游验证揭示的标准指标盲点**：GraphCast在RMSE上与IFS相当，但在台风快速增强预报上仍存在不足；AERIS集合平均的ACC略低于IFS，但spread信息本身具有决策价值。

---

## 【第七部分：动态拓展范式】

### 7.1 基础模型与迁移学习范式

#### 【原理描述】

**核心定义**：基础模型（Foundation Model）通过大规模预训练学习通用气象表示，在单一模型上实现多任务泛化，通过微调或提示学习适配下游任务，是气象AI从"单一模型单一任务"向"通用智能"演进的核心范式。

**数学推导**：

##### 1. ClimaX变量无关tokenization

$$\mathbf{t}_{i,j}^{(v)} = \text{Linear}_v\left(\text{flatten}\left(\mathbf{X}^{(v)}[i,j]\right)\right)$$

每个位置的 $V$ 个变量值被展平为 $V$ 维向量，然后通过线性投影得到token。

##### 2. 任务token嵌入

$$\mathbf{t}_{\text{task}} = \text{Embed}(\text{task\_id}) \in \mathbb{R}^D$$

任务ID可以是：变量标识、时效标识、数据集来源标识。

##### 3. 预训练-微调范式

预训练（大规模）：
$$\theta^* = \arg\min_\theta \sum_{(x, y) \in \mathcal{D}_{\text{pretrain}}} \mathcal{L}_{\text{pretrain}}(f_\theta(x), y)$$

微调（下游任务）：
$$\theta_{\text{ft}} = \arg\min_\theta \sum_{(x, y) \in \mathcal{D}_{\text{ft}}} \mathcal{L}_{\text{ft}}(f_{\theta^* + \delta\theta}(x), y)$$

##### 4. NeuralGCM端到端统一建模

$$p_\theta(\mathbf{x}_{t+1} \mid \mathbf{x}_t, \mathbf{f}) = \mathcal{N}\left(\mu_\theta(\mathbf{x}_t, \mathbf{f}), \sigma_\theta^2(\mathbf{x}_t, \mathbf{f})\right)$$

端到端建模将参数化过程直接融入网络。

**理论依据**：
- 气象AI的核心挑战是训练数据有限（ERA5仅40+年），预训练提供先验知识
- 气候模式的物理约束可通过微调迁移
- 基础模型减少为每个下游任务从头训练的计算开销

#### 【架构层】

**计算流程图（ClimaX通用基础模型）**：

```
【预训练阶段】

多源异构数据集:
  ├─ ERA5 (1979-2017)
  ├─ Climate Model Output (CMIP6)
  └─ Remote Sensing (卫星图像)
  ↓
1. 变量无关tokenization: 每个变量独立embedding
2. 任务/变量token注入: Embed(task_id)
3. Vision Transformer主干: 标准ViT，无气象特定修改
4. 多任务解码: 每个下游任务有专属解码头
  ↓ 预训练损失: L = Σ_task Σ_v L_task

【微调阶段】

下游任务数据（远小于预训练数据）
  ↓ 加载预训练权重 → 可选: 冻结主干仅微调解码头
  ↓ 下游任务微调: L_ft = L_task(f_{θ_ft}(x), y)
```

#### 【设计层】

**创新突破**：
- **ClimaX**：首个气象领域基础模型，通过变量无关tokenization和任务token实现跨数据集、跨任务的通用表示学习
- **NeuralGCM**：将神经网络参数化嵌入传统大气模式框架，实现端到端可微分的天气-气候建模

#### 【对比层】

**五流派基础模型能力对比**：

| 维度 | ClimaX | NeuralGCM | GenEPS | 单任务模型 |
|------|--------|-----------|--------|---------|
| **预训练任务** | 多任务（变量无关） | 天气预报+参数化 | 多模型集成 | 单一预报 |
| **泛化能力** | 跨数据集、跨任务 | 天气→气候 | 跨模型 | 仅限指定任务 |
| **物理一致性** | 弱（纯数据驱动） | 强（嵌入物理方程） | 中（各模型混合） | 取决于模型 |
| **下游适配** | 微调/提示学习 | 端到端微调 | 无需微调 | 无需微调 |

---

### 7.2 多模型集成与后处理范式

#### 【原理描述】

**核心定义**：多模型集成通过组合多个异质预报模型的输出，利用各模型优势互补，提升整体预报质量与可靠性。后处理则对模型原始输出进行统计/机器学习修正。

**数学推导**：

##### 1. 多模型集成平均

$$X_{\text{ensemble}} = \frac{1}{M}\sum_{m=1}^{M} X^{(m)}$$

##### 2. 扩散先验集成（GenEPS）

$$\hat{X}_0^{(e)} \sim p_\theta\left(X_0 \mid \mathcal{M} = \{X^{\text{Pangu}}, X^{\text{FengWu}}, X^{\text{FuXi}}, X^{\text{NeuralGCM}}\}\right)$$

通过扩散模型学习多模型输出的统一概率分布。

##### 3. 超级集合加权

$$X_{\text{super}} = \sum_{m=1}^{M} w_m \cdot X^{(m)}$$

其中权重 $w_m$ 通过回归优化确定。

##### 4. 统计后处理

$$X_{\text{bias-corrected}} = \alpha \cdot X_{\text{model}} + \beta$$

**创新突破**：
- **GenEPS扩散先验**：首次将扩散生成模型作为多AI预报集成的统一框架
- **多尺度集成**：Global（GraphCast）+ Regional（CorrDiff）+ Climate（NeuralGCM）的跨尺度集成

#### 【对比层】

**集成策略对比**：

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 简单平均 | 实现简单，无参数 | 不考虑模型质量差异 | 快速原型 |
| 加权平均 | 考虑模型质量 | 需优化权重，权重可能随时间变 | 业务部署 |
| 扩散先验 | 学习非线性集成，概率校准 | 计算昂贵，训练复杂 | 研究级 |
| 统计后处理 | 修正系统性偏差 | 依赖历史数据，可能过拟合 | 业务后处理 |

---

### 7.3 实时-滚动更新范式

#### 【原理描述】

**核心定义**：实时-滚动更新范式指模型在部署后持续接收最新观测数据，通过在线学习、滚动预测更新或数据同化集成，不断适应大气状态的演变与气候趋势的变化。

**数学推导**：

##### 1. 滚动预测更新

$$\hat{X}_{t+\Delta t}^{(n+1)} = f_\theta\left(\hat{X}_t^{(n)}, I_t\right)$$

##### 2. 数据同化集成

$$X_t^{\text{analysis}} = X_t^{\text{background}} + \mathbf{K}\left(Y_t^{\text{obs}} - H(X_t^{\text{background}})\right)$$

深度学习模型（如FourCastNet）可作为背景场，与传统数据同化框架结合。

##### 3. 在线学习更新

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}\left(f_{\theta_t}(X_t), X_t^{\text{obs}}\right)$$

**创新突破**：
- **IFS-HD**：ECMWF将FourCastNet作为高分辨率背景场，与传统4D-Var数据同化框架结合
- **AERIS强迫场**：在S2S预报中实时注入大尺度强迫（海温、SST异常），提升延伸期信号保持

**这一范式代表气象AI的未来发展方向——从"静态训练-部署"向"动态持续适应"的演进。**

---

### 7.4 分辨率连续拓展范式

#### 【原理描述】

**核心定义**：分辨率连续拓展指气象AI模型在从全球（0.25°）到区域（公里级）再到局地（百米级）的连续分辨率尺度上建立统一的预报能力。

**代表模型**：

| 模型 | 流派 | 分辨率 | 区域 | 关键技术 |
|------|------|--------|------|---------|
| CorrDiff | Generative | 0.25°→2km | 台湾 | 条件扩散UNet |
| FuXi-TC | Generative | 0.25°→~3km | TC区域 | 条件扩散 |
| ClimaX | Transformer | 可变 | 全球 | 变量无关tokenization |
| NeuralGCM | Neural-Physical | 0.7°-2.8° | 全球 | 端到端物理框架 |

**创新突破**：
- **CorrDiff**：80M参数UNet实现ERA5到2km区域的12.5倍上采样，是气象AI分辨率拓展的里程碑
- **FuXi-TC**：专门针对台风区域的跨分辨率扩散，融合FuXi大尺度预报和WRF高分辨率物理约束
- **NeuralGCM多分辨率**：在同一框架下支持0.7°~2.8°的分辨率变化

**关键洞察**：

1. **扩散模型是分辨率拓展的关键使能技术**——CorrDiff和FuXi-TC都依赖条件扩散生成实现跨分辨率映射
2. **物理约束在高分辨率下更关键**——地形强迫、海陆对比等局地物理过程主导，扩散模型需要有效的条件化
3. **计算成本仍是瓶颈**——CorrDiff推理需要A100 GPU，限制业务化部署；需要蒸馏或降采样策略

---

## 【第五至第七部分：五流派终极大对比】

### 五流派训练-评估全维度综合对比表

| 对比维度 | Transformer | Fourier/Operator | Graph | Generative | ODE |
|---------|------------|-----------------|-------|-----------|-----|
| **训练策略** | 单步+多步微调/Replay | 单步+两步微调 | 自回归12步 | 扩散单步 | ODE函数学习 |
| **时间步长** | 6h固定或随机{6,12,24}h | 6h固定 | 6h固定 | 可变（扩散无所谓） | 连续 |
| **长时效机制** | 多步微调/Replay Buffer | 两步微调 | 自回归限制 | 扩散天然稳定 | ODE积分 |
| **最长期效** | 14-90天 | 7-10天 | 10天 | 90天(S2S) | 任意时刻 |
| **不确定性** | NLL/扩散 | 初始场扰动 | 集合扩散 | 天然概率 | 贝叶斯 |
| **多任务权重** | ✅自动(NLL) | ❌固定 | ❌固定 | ✅端到端 | ❌ |
| **极端降水处理** | 直接预测 | 对数变换诊断 | 直接预测 | 扩散生成 | 直接预测 |
| **核心评估指标** | RMSE+ACC | RMSE+ACC+RQE | RMSE+ACC | CRPS+SSR | 守恒误差 |
| **下游验证强项** | 局地订正/长期MJO | TC追踪/极端降水 | 集合校准 | 降尺度/概率 | 物理建模 |
| **训练数据规模** | 37-39年ERA5 | 37年ERA5 | 39年ERA5 | 39年ERA5 | 有限 |
| **推理速度** | 中等 | 极高(7秒/100成员) | 快 | 慢(10+步) | 快 |
| **扩散/集合规模** | 10-50成员 | 100-1000+成员 | 50成员 | 10-50成员 | 贝叶斯积分 |
| **spread-skill比** | ~0.8 | <0.8（欠发散） | ~1.0 | ~0.9 | **未说明** |
| **极端降水RQE** | -25%~-35% (q=0.99) | -35% (q=0.99) | **未说明** | 依赖分辨率 | **未说明** |
| **基础模型能力** | ClimaX | **未说明** | **未说明** | **未说明** | NeuralGCM |
| **多模型集成** | GenEPS成员 | **未说明** | **未说明** | GenEPS(主导) | NeuralGCM |
| **实时更新** | 强迫场注入 | **未说明** | **未说明** | **未说明** | **未说明** |
| **分辨率拓展** | ClimaX(可变) | **未说明** | **未说明** | CorrDiff(km级) | NeuralGCM(多尺度) |
| **核心优势** | 全局建模+灵活架构+长期 | 极高效率+超大集合 | 球面均匀+极区友好 | 概率量化+长期稳定 | 物理一致+连续外推 |
| **核心劣势** | 计算成本高+极区冗余 | 平移等变假设+极端降水低估 | 图构造复杂+预处理繁 | 推理成本高+训练复杂 | 数值稳定性+训练慢 |

---

*（本文件涵盖【序言】【第一部分】【第二部分】【第三部分】【第四部分】【第五部分】【第六部分】【第七部分】，共计5大流派、21篇代表性论文的完整系统性跨流派融合分析。）*
