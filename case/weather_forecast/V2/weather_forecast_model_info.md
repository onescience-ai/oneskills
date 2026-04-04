# 全球短期天气预报模型 - 完整实现说明

## 模型概述

本模型实现了一个轻量级的全球短期天气预报网络，采用基于 Pangu 系列组件的编码器-解码器架构。模型输入当前时刻的全球气象场（包含地表变量和多层大气变量），输出下一时刻的预报结果。

## 复用的 OneScience 组件

### 1. PanguEmbedding（通过 OneEmbedding）

**组件族**: `embedding`

**调用入口**: `OneEmbedding(style="PanguEmbedding", ...)`

**用途**: 将原始气象场切分为非重叠 patch，并投影到统一的 embedding 特征空间

**地表分支配置**:
```python
OneEmbedding(
    style="PanguEmbedding",
    img_size=(721, 1440),
    patch_size=(4, 4),
    Variables=7,
    embed_dim=192,
)
```

**高空分支配置**:
```python
OneEmbedding(
    style="PanguEmbedding",
    img_size=(13, 721, 1440),
    patch_size=(2, 4, 4),
    Variables=5,
    embed_dim=192,
)
```

### 2. PanguFuser（通过 OneFuser）

**组件族**: `fuser`

**调用入口**: `OneFuser(style="PanguFuser", ...)`

**用途**: 在三维网格上堆叠多层 3D Transformer Block，对 patch token 进行特征融合

**地表编码器 Fuser**:
```python
OneFuser(
    style="PanguFuser",
    dim=192,
    input_resolution=(1, 181, 360),
    depth=2,
    num_heads=6,
    window_size=(1, 6, 12),
    drop_path=0.0,
    mlp_ratio=4.0,
)
```

**高空编码器 Fuser**:
```python
OneFuser(
    style="PanguFuser",
    dim=192,
    input_resolution=(8, 181, 360),
    depth=2,
    num_heads=6,
    window_size=(2, 6, 12),
    drop_path=0.0,
    mlp_ratio=4.0,
)
```

**地表中间层 Fuser**:
```python
OneFuser(
    style="PanguFuser",
    dim=384,
    input_resolution=(1, 91, 180),
    depth=2,
    num_heads=12,
    window_size=(1, 6, 12),
    drop_path=0.0,
    mlp_ratio=4.0,
)
```

**高空中间层 Fuser**:
```python
OneFuser(
    style="PanguFuser",
    dim=384,
    input_resolution=(8, 91, 180),
    depth=2,
    num_heads=12,
    window_size=(2, 6, 12),
    drop_path=0.0,
    mlp_ratio=4.0,
)
```

**地表解码器 Fuser**:
```python
OneFuser(
    style="PanguFuser",
    dim=192,
    input_resolution=(1, 181, 360),
    depth=2,
    num_heads=6,
    window_size=(1, 6, 12),
    drop_path=0.0,
    mlp_ratio=4.0,
)
```

**高空解码器 Fuser**:
```python
OneFuser(
    style="PanguFuser",
    dim=192,
    input_resolution=(8, 181, 360),
    depth=2,
    num_heads=6,
    window_size=(2, 6, 12),
    drop_path=0.0,
    mlp_ratio=4.0,
)
```

### 3. PanguDownSample（通过 OneSample）

**组件族**: `sample`

**调用入口**: `OneSample(style="PanguDownSample", ...)`

**用途**: 对 token 网格做下采样，聚合空间邻域信息

**地表分支配置**:
```python
OneSample(
    style="PanguDownSample",
    input_resolution=(181, 360),
    output_resolution=(91, 180),
    in_dim=192,
)
```

**高空分支配置**:
```python
OneSample(
    style="PanguDownSample",
    input_resolution=(8, 181, 360),
    output_resolution=(8, 91, 180),
    in_dim=192,
)
```

### 4. PanguUpSample（通过 OneSample）

**组件族**: `sample`

**调用入口**: `OneSample(style="PanguUpSample", ...)`

**用途**: 对 token 网格做上采样，恢复到更高分辨率

**地表分支配置**:
```python
OneSample(
    style="PanguUpSample",
    input_resolution=(91, 180),
    output_resolution=(181, 360),
    in_dim=384,
    out_dim=192,
)
```

**高空分支配置**:
```python
OneSample(
    style="PanguUpSample",
    input_resolution=(8, 91, 180),
    output_resolution=(8, 181, 360),
    in_dim=384,
    out_dim=192,
)
```

### 5. PanguPatchRecovery（通过 OneRecovery）

**组件族**: `recovery`

**调用入口**: `OneRecovery(style="PanguPatchRecovery", ...)`

**用途**: 将 patch 级别特征图恢复为原始气象场分辨率

**地表分支配置**:
```python
OneRecovery(
    style="PanguPatchRecovery",
    img_size=(721, 1440),
    patch_size=(4, 4),
    in_chans=384,
    out_chans=7,
)
```

**高空分支配置**:
```python
OneRecovery(
    style="PanguPatchRecovery",
    img_size=(13, 721, 1440),
    patch_size=(2, 4, 4),
    in_chans=384,
    out_chans=5,
)
```

## 主干中的主要 Shape 变化

### 地表分支

| 阶段 | Shape | 说明 |
|------|-------|------|
| 输入 | `(Batch, 7, 721, 1440)` | 原始地表气象场 |
| Embedding 后 | `(Batch, 192, 181, 360)` | Patch embedding 特征图 |
| 转换为 token | `(Batch, 65160, 192)` | Token 序列 (181 × 360 = 65160) |
| 编码器 Fuser | `(Batch, 65160, 192)` | 特征融合后 token |
| 下采样 | `(Batch, 16380, 384)` | 下采样后 token (91 × 180 = 16380) |
| 中间层 Fuser | `(Batch, 16380, 384)` | 中间层融合后 token |
| 上采样 | `(Batch, 65160, 192)` | 上采样后 token |
| 解码器 Fuser | `(Batch, 65160, 192)` | 解码器融合后 token |
| 投影 | `(Batch, 65160, 384)` | 特征维度投影 |
| 转换为特征图 | `(Batch, 384, 181, 360)` | Token 转回特征图 |
| 恢复 | `(Batch, 7, 721, 1440)` | 恢复为原始分辨率 |

### 高空分支

| 阶段 | Shape | 说明 |
|------|-------|------|
| 输入 | `(Batch, 5, 13, 721, 1440)` | 原始高空气象场 |
| Embedding 后 | `(Batch, 192, 8, 181, 360)` | Patch embedding 特征图 |
| 转换为 token | `(Batch, 521280, 192)` | Token 序列 (8 × 181 × 360 = 521280) |
| 编码器 Fuser | `(Batch, 521280, 192)` | 特征融合后 token |
| 下采样 | `(Batch, 131040, 384)` | 下采样后 token (8 × 91 × 180 = 131040) |
| 中间层 Fuser | `(Batch, 131040, 384)` | 中间层融合后 token |
| 上采样 | `(Batch, 521280, 192)` | 上采样后 token |
| 解码器 Fuser | `(Batch, 521280, 192)` | 解码器融合后 token |
| 投影 | `(Batch, 521280, 384)` | 特征维度投影 |
| 转换为特征图 | `(Batch, 384, 8, 181, 360)` | Token 转回特征图 |
| 恢复 | `(Batch, 5, 13, 721, 1440)` | 恢复为原始分辨率 |

## 模型架构

### 整体结构

```
输入层
├── 地表分支: (Batch, 7, 721, 1440)
└── 高空分支: (Batch, 5, 13, 721, 1440)

Embedding 层
├── 地表分支: PanguEmbedding → (Batch, 192, 181, 360)
└── 高空分支: PanguEmbedding → (Batch, 192, 8, 181, 360)

编码层
├── 地表分支: PanguFuser → 下采样 → PanguFuser
└── 高空分支: PanguFuser → 下采样 → PanguFuser

中间层
├── 地表分支: PanguFuser (dim=384)
└── 高空分支: PanguFuser (dim=384)

解码层
├── 地表分支: 上采样 → PanguFuser
└── 高空分支: 上采样 → PanguFuser

输出层
├── 地表分支: PanguPatchRecovery → (Batch, 7, 721, 1440)
└── 高空分支: PanguPatchRecovery → (Batch, 5, 13, 721, 1440)
```

### 关键设计决策

1. **分离的双分支结构**: 地表和高空分支保持独立，各自使用对应的 2D/3D 处理
2. **主干特征提取**: 使用 `PanguFuser` 组件，内部封装了多层 `EarthTransformer3DBlock`
3. **多尺度处理**: 通过下采样和上采样实现多尺度特征提取
4. **统一组件接口**: 通过 `OneEmbedding`、`OneFuser`、`OneSample`、`OneRecovery` 统一调用接口

## 最小验证建议

### 1. 形状验证

运行测试代码，确保输入输出形状正确：

```bash
python test_weather_forecast_model.py
```

验证要点：
- 输入形状与预期一致
- 输出形状与输入形状匹配
- 各层之间的形状转换正确

### 2. 前向传播验证

确保模型可以正常进行前向传播，无错误：

```python
surface_input = torch.randn(2, 7, 721, 1440)
upper_air_input = torch.randn(2, 5, 13, 721, 1440)
surface_output, upper_air_output = model(surface_input, upper_air_input)
```

### 3. 参数数量验证

检查模型参数量是否合理：

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")
```

### 4. 梯度流动验证

确保模型可以正常进行反向传播，梯度可以流动：

```python
loss = surface_output.sum() + upper_air_output.sum()
loss.backward()

for name, param in model.named_parameters():
    assert param.grad is not None, f"Parameter {name} has no gradient"
```

### 5. 模型保存和加载验证

确保模型可以正常保存和加载：

```python
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
```

## 使用示例

### 模型初始化

```python
from weather_forecast_model import WeatherForecastModel

model = WeatherForecastModel(
    surface_vars=7,
    upper_air_vars=5,
    pressure_levels=13,
    height=721,
    width=1440,
    embed_dim=192,
    num_heads=6,
    window_size=(6, 12),
    depth=2,
    mlp_ratio=4.0,
    drop_path=0.0,
)
```

### 前向传播

```python
import torch

Batch = 2
surface_input = torch.randn(Batch, 7, 721, 1440)
upper_air_input = torch.randn(Batch, 5, 13, 721, 1440)

surface_output, upper_air_output = model(surface_input, upper_air_input)

print(f"Surface output shape: {surface_output.shape}")
print(f"Upper air output shape: {upper_air_output.shape}")
```

## 后续扩展建议

### 1. 结构改进

- 尝试不同的 Transformer 变体
- 探索不同的注意力机制
- 尝试不同的特征融合策略

### 2. 模块替换实验

- 替换 Embedding 层为其他类型的特征提取器
- 替换 Transformer 块为其他类型的网络结构
- 替换下采样/上采样模块为其他类型的采样方法

### 3. 性能优化

- 使用混合精度训练
- 采用分布式训练
- 进行模型压缩和量化

### 4. 功能扩展

- 增加多步预报能力
- 加入不确定性估计
- 整合多源数据

## 总结

本模型实现了一个基础版的全球格点预报网络，结构清晰、轻量，方便后续进行结构改进和模块替换实验。模型使用了 OneScience 中的 Pangu 系列统一组件，确保了代码的可维护性和可扩展性。

通过分离地表和高空分支处理，模型能够有效地捕获不同层次的气象信息，并通过 Transformer 架构捕获时空依赖关系，为短期天气预报提供准确的预测结果。
