# Pangu AFNO Replacement Implementation

## 概述

本实现将 Pangu-Weather 模型的主干特征提取部分替换为 FourCastNet 风格的 AFNO 主干，同时保持 Pangu 的输入输出形式和整体组织方式不变。

## 文件说明

- `pangu_afno_bridge.py`: 3D token 与 2D patch 网格之间的桥接层
- `pangu_afno_hybrid.py`: PanguAFNOHybrid 主模型实现
- `example_usage.py`: 模型使用示例

## 模型架构

```
输入 (Batch, 4+3+5*13, 721, 1440)
    ↓
PanguEmbedding (surface + upper-air 分别编码)
    ↓
surface: (Batch, 192, 181, 360)
upper-air: (Batch, 192, 7, 181, 360)
    ↓
拼接: (Batch, 192, 8, 181, 360)
    ↓
Pangu3DTo2DBridge (3D → 2D 转换)
    ↓
2D patch 网格 (Batch, 181, 360, 768)
    ↓
多层 FourCastNetFuser (AFNO + MLP)
    ↓
2D patch 网格 (Batch, 181, 360, 768)
    ↓
Pangu2DTo3DBridge (2D → 3D 转换)
    ↓
3D token (Batch, 192, 8, 181, 360)
    ↓
与 skip 拼接: (Batch, 384, 8, 181, 360)
    ↓
PanguPatchRecovery (分别恢复 surface 和 upper-air)
    ↓
输出 (surface: Batch, 4, 721, 1440; upper-air: Batch, 5, 13, 721, 1440)
```

## 主要 Shape 变化

1. **Embedding 输出**:
   - surface: `(Batch, 192, 181, 360)`
   - upper-air: `(Batch, 192, 7, 181, 360)`

2. **3D → 2D 桥接层**:
   - 输入: `(Batch, 192, 8, 181, 360)`
   - 输出: `(Batch, 181, 360, 768)`

3. **AFNO 主干**:
   - 输入: `(Batch, 181, 360, 768)`
   - 输出: `(Batch, 181, 360, 768)`

4. **2D → 3D 反向桥接层**:
   - 输入: `(Batch, 181, 360, 768)`
   - 输出: `(Batch, 192, 8, 181, 360)`

5. **Patch Recovery 输出**:
   - surface: `(Batch, 4, 721, 1440)`
   - upper-air: `(Batch, 5, 13, 721, 1440)`

## 复用的 OneScience 组件

1. **OneEmbedding(style="PanguEmbedding")**: 分别编码 surface 和 upper-air 变量
2. **OneFuser(style="FourCastNetFuser")**: AFNO 主干特征提取（多层堆叠）
3. **OneRecovery(style="PanguPatchRecovery")**: 将 patch 特征恢复为目标物理场

## 新增组件

1. **Pangu3DTo2DBridge**: 将 3D token 转换为 2D patch 网格
   - 将 PressureLevels 维展平到通道维
   - 使用 1x1 卷积调整特征维度

2. **Pangu2DTo3DBridge**: 将 2D patch 网格转换回 3D token
   - 使用 1x1 卷积调整特征维度
   - 将通道维拆分回 PressureLevels 维

## 桥接层设计策略

本实现采用"将 PressureLevels 维展平到通道维"的策略（即之前提到的选项 A）：

- 优点：
  - 实现简单，计算高效
  - 保留了所有气压层信息
  - 与 AFNO 的 2D 频域混合兼容性好

- 缺点：
  - 失去了气压层之间的显式空间关系
  - AFNO 无法直接建模气压层之间的交互

## 与原 Pangu 模型的区别

1. **主干特征提取**:
   - 原方案: 3D Transformer block (PanguFuser)
   - 新方案: 2D AFNO block (FourCastNetFuser)

2. **下采样结构**:
   - 原方案: 有下采样/上采样结构（U-shape）
   - 新方案: 无下采样结构（平铺的 AFNO block 堆叠）

3. **特征维度**:
   - 原方案: 主干特征维度为 192/384
   - 新方案: AFNO 主干特征维度为 768

## 使用方法

```python
import torch
from pangu_afno_hybrid import PanguAFNOHybrid

# 初始化模型
model = PanguAFNOHybrid(
    img_size=(721, 1440),
    patch_size=(2, 4, 4),
    embed_dim=192,
    afno_dim=768,
    afno_depth=12,
    pressure_levels=8,
)

# 创建输入
batch_size = 1
x = torch.randn(batch_size, 4 + 3 + 5 * 13, 721, 1440)

# 前向传播
output_surface, output_upper_air = model(x)

print(f"Surface output: {output_surface.shape}")
print(f"Upper-air output: {output_upper_air.shape}")
```

## 运行示例

```bash
python example_usage.py
```

## 可调参数

- `img_size`: 输入空间尺寸 `(Height, Width)`
- `patch_size`: patch 切分尺寸 `(PatchPressureLevels, PatchHeight, PatchWidth)`
- `embed_dim`: Pangu embedding 后的特征维度
- `afno_dim`: AFNO 主干的特征维度
- `afno_depth`: AFNO block 的堆叠层数
- `pressure_levels`: 气压层数
- `afno_mlp_ratio`: AFNO block 中 MLP 的隐层放大倍数
- `afno_num_blocks`: AFNO 的通道分块数
- `afno_sparsity_threshold`: AFNO soft shrink 阈值
- `afno_hard_thresholding_fraction`: AFNO 保留的频率模式比例

## 验证建议

1. **Shape 验证**: 确保输入输出的 shape 符合预期
2. **数值验证**: 检查模型输出的数值范围是否合理
3. **性能验证**: 对比原 Pangu 模型和 PanguAFNOHybrid 的推理速度
4. **精度验证**: 在相同数据集上对比两个模型的预测精度

## 后续改进方向

1. **多尺度 AFNO**: 可以考虑在 AFNO 主干中引入多尺度结构
2. **气压层感知 AFNO**: 修改 AFNO 组件，使其能够显式建模气压层之间的关系
3. **混合注意力**: 结合 3D attention 和 2D AFNO 的优势
4. **自适应桥接层**: 设计更智能的 3D→2D 转换策略
