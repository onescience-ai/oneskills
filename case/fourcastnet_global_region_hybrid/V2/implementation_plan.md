# FourCastNet Global Region Hybrid 实现方案

## 1. 任务目标

构建一个**全球+区域混合天气预报系统**：
- 使用 FourCastNet 官方开源权重作为基座，提取最后一层全球特征
- 训练一个新的 FourCastNet 变体，融合全球特征进行高分辨率区域预测
- 引入地理位置范围参数，实现区域匹配与特征融合

## 2. 可行性分析结论

### 2.1 在现有 OneScience 代码中的可行性

| 评估项 | 结论 | 说明 |
|--------|------|------|
| 是否可直接复用现有组件 | ✅ 是 | FourCastNet 所有组件均可复用 |
| 是否需要修改现有文件 | ❌ 否 | 所有新增代码可放在 case 目录下 |
| 最小改造路径 | ✅ 清晰 | 只需新增 1 个融合模块 + 1 个新模型类 |
| 架构兼容性 | ✅ 良好 | FourCastNet 的 patch-grid 结构便于特征对齐 |

### 2.2 核心架构洞察

FourCastNet 的关键结构特点：

```
Input (720, 1440)
  -> Patch Embed (8x8)
  -> Token Seq (16200, 768)
  -> Reshape to Grid (90, 180, 768)  <-- 全球特征在此层级提取
  -> AFNO Trunk (12 layers)
  -> Head -> Output (720, 1440)
```

**关键发现**：
- 全球模型最后一层输出 shape 为 `(Batch, 90, 180, 768)`（patch grid 层级）
- 通过插值/crop 可将全球特征对齐到任意区域 patch grid
- 区域模型可以复用完全相同的组件，只需在 trunk 中间插入融合模块

## 3. 推荐架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Global + Region Hybrid Model                  │
├─────────────────────────────────────────────────────────────────┤
│  Global Branch (Frozen Pretrained)                              │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐│
│  │   Input     │ -> │   Embedding  │ -> │  Trunk (12 layers)  ││
│  │  (720,1440) │    │  (90x180,768)│    │   Last Layer Feature││
│  └─────────────┘    └──────────────┘    │   (90,180,768)      ││
│                                         └─────────────────────┘│
│                                                    │             │
│                                                    ▼             │
│  Region Branch (Trainable)                        Global Feature  │
│  ┌─────────────┐    ┌──────────────┐              Extraction    │
│  │   Input     │ -> │   Embedding  │ ─┐         ┌─────────────┐│
│  │  (H_r, W_r) │    │ (H_p, W_p, D)│  │         │  Interpolate││
│  └──────────────┘   └──────────────┘  │    ┌───>│  or Crop to ││
│                              │        │    │    │  Region Grid ││
│                              ▼        │    │    └─────────────┘ │
│                    ┌──────────────────┴────┤            │        │
│                    │  Fusion Block (New)  │<───────────┘        │
│                    │  - Global Feature    │                     │
│                    │    Injection         │                     │
│                    └─────────────────────┘                     │
│                              │                                  │
│                              ▼                                  │
│                    ┌──────────────────┐                         │
│                    │  Region Trunk    │                         │
│                    │  (N layers AFNO)│                         │
│                    └──────────────────┘                         │
│                              │                                  │
│                              ▼                                  │
│                    ┌──────────────────┐                         │
│                    │  Head (Linear)  │                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 融合策略（推荐：晚期融合）

```
Region Embedding -> [Fuser x 3] -> [Fusion Block] -> [Fuser x 3] -> Head
                                      ↑
                              Global Feature (interpolated)
```

**选择理由**：
1. 保持 FourCastNet 主干结构不变
2. 只需新增一个 Fusion Block 组件
3. 全球特征作为外部条件注入，不影响区域模型的基础能力
4. 便于调试和消融实验

### 3.3 全球特征对齐方案

```python
# 全球特征: (Batch, 90, 180, 768) - 对应全球 720x1440 范围
# 区域 patch grid: (H_pr, W_pr) - 对应区域 H_r x W_r 范围

# 步骤 1: 将全球特征插值到区域分辨率
aligned_global = F.interpolate(
    global_feature.permute(0, 3, 1, 2),  # (B, 768, 90, 180)
    size=(H_pr, W_pr),
    mode='bilinear',
    align_corners=False
).permute(0, 2, 3, 1)  # (B, H_pr, W_pr, 768)

# 步骤 2: 与区域特征融合
fused_feature = FusionBlock(region_feature, aligned_global, position_embed)
```

## 4. 复用组件清单

| 组件 | 注册名 | 来源 | 用途 |
|------|--------|------|------|
| `FourCastNetEmbedding` | `style="FourCastNetEmbedding"` | OneScience | 2D patch embedding |
| `FourCastNetFuser` | `style="FourCastNetFuser"` | OneScience | 主干特征融合 |
| `FourCastNetAFNO2D` | `style="FourCastNetAFNO2D"` | OneScience | 频域混合 |
| `FourCastNetFC` | `style="FourCastNetFC"` | OneScience | 逐位置通道混合 |

## 5. 需要新增的组件

### 5.1 GlobalRegionFusionBlock

**职责**：将全球特征与区域特征融合

**输入**：
- `region_feat`: `(Batch, H, W, dim_region)` - 区域特征
- `global_feat`: `(Batch, H, W, dim_global)` - 对齐后的全球特征
- `position_embed`: `(Batch, H, W, dim_pos)` - 位置编码（可选）

**输出**：
- `fused_feat`: `(Batch, H, W, dim_region)` - 融合后的区域特征

**融合方式（推荐：门控融合）**：
```python
gate = σ(Linear([region_feat, global_feat]))
fused = gate * region_feat + (1 - gate) * Linear(global_feat)
```

### 5.2 FourCastNetGlobalRegionHybrid（主模型）

**新增参数**：
- `region_img_size`: 区域输入尺寸 `(H_r, W_r)`
- `region_patch_size`: 区域 patch 尺寸（可与全球不同）
- `fusion_layer_idx`: 融合层位置（默认第 4 层）
- `global_checkpoint_path`: 全球模型权重路径

## 6. Shape 变化详解

### 6.1 全球分支（冻结，仅推理）

```
Input:        (Batch, 19, 720, 1440)
              ↓
Embedding:    (Batch, 16200, 768)
              ↓ reshape
Patch Grid:   (Batch, 90, 180, 768)
              ↓
Trunk x 12:   (Batch, 90, 180, 768)
              ↓ [提取最后一层特征]
Global Feat:  (Batch, 90, 180, 768)
```

### 6.2 区域分支（训练）

```
Input:              (Batch, in_chans_region, H_r, W_r)
                    ↓
Embedding:          (Batch, N_patches_r, embed_dim)
                    ↓ reshape
Patch Grid:         (Batch, H_pr, W_pr, embed_dim)
                    ↓
Pre-fusion Blocks:  (Batch, H_pr, W_pr, embed_dim)
                    ↓ [Fusion Block]
Fused Feature:      (Batch, H_pr, W_pr, embed_dim)
                    ↓
Post-fusion Blocks: (Batch, H_pr, W_pr, embed_dim)
                    ↓
Head:               (Batch, H_pr, W_pr, out_chans * p1 * p2)
                    ↓ rearrange
Output:             (Batch, out_chans, H_r, W_r)
```

### 6.3 关键维度关系

| 参数 | 全球模型 | 区域模型（示例 360x720） |
|------|----------|--------------------------|
| 输入分辨率 | 720 x 1440 | 360 x 720 |
| Patch size | 8 x 8 | 8 x 8 |
| Patch grid | 90 x 180 | 45 x 90 |
| Embed dim | 768 | 768 |
| Trunk depth | 12 | 6 (可配置) |

## 7. 文件组织结构

```
oneskills/codex/case/fourcastnet_global_region_hybrid/V2/
├── implementation_plan.md          # 本文档
├── query.md                       # 用户原始需求
│
├── models/
│   └── fourcastnet_global_region_hybrid.py  # 主模型实现
│
├── components/
│   ├── __init__.py
│   └── global_region_fusion.py     # 融合模块
│
├── utils/
│   ├── __init__.py
│   ├── synthetic_data.py           # 虚拟数据生成
│   └── checkpoint_utils.py          # 快速保存checkpoint
│
└── train_base.py                   # 基座模型快速训练脚本
```

## 8. 待确认事项

在进入代码生成阶段前，请确认以下问题：

| # | 问题 | 选项 |
|---|------|------|
| 1 | **区域尺寸**：期望的区域分辨率？ | A: 360x720 / B: 720x1440 / C: 其他 |
| 2 | **融合方式**：全球特征如何融合？ | A: 门控融合(推荐) / B: 直接拼接 / C: Cross-attention |
| 3 | **位置编码**：地理位置如何编码？ | A: 额外通道 / B: 可学习嵌入 / C: 正弦编码 |
| 4 | **基座权重**：checkpoint 路径？ | 提供路径或使用 HuggingFace 加载 |

## 9. 代码生成清单

确认后将生成以下文件：

1. **models/fourcastnet_global_region_hybrid.py**
   - `FourCastNetGlobalRegionHybrid` 主模型类
   - 支持全球特征注入
   - 晚期融合策略

2. **components/global_region_fusion.py**
   - `GlobalRegionFusionBlock` 融合模块
   - 门控融合实现

3. **utils/synthetic_data.py**
   - `generate_synthetic_weather_data()` 虚拟数据生成
   - 支持全球场和区域场生成

4. **utils/checkpoint_utils.py**
   - `save_checkpoint_early()` 快速保存工具
   - `load_global_pretrained()` 加载基座权重

5. **train_base.py**
   - 基座模型快速训练脚本
   - 支持训练中断保存 checkpoint

## 10. 风险与注意事项

### 10.1 潜在风险

| 风险 | 缓解措施 |
|------|----------|
| 全球特征与区域特征维度不匹配 | 使用 Linear 投影对齐维度 |
| 插值带来的精度损失 | 可尝试 crop 而非插值 |
| 融合层位置选择不当 | 提供可配置参数，便于调优 |

### 10.2 依赖项

- PyTorch >= 1.10
- einops
- timm
- numpy

所有依赖均已在 OneScience 环境中配置。
