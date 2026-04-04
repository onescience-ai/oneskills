# 全球短期天气预报模型 - 详细说明

## 模型结构

### 整体架构
1. **输入层**：接收地表和高空气象场
2. **Embedding 层**：使用 PanguEmbedding 将输入映射到特征空间
3. **编码层**：
   - 多层 Transformer 编码器
   - 中间插入 PanguDownSample 进行下采样
4. **解码层**：
   - 多层 Transformer 解码器
   - 中间插入 PanguUpSample 进行上采样
5. **融合层**：融合地表和高空特征
6. **输出层**：使用 PanguPatchRecovery 恢复为原始分辨率的气象场

### 复用的 OneScience 组件

1. **PanguEmbedding**
   - 用途：将原始气象场切分为非重叠 patch，并投影到统一的 embedding 特征空间
   - 调用方式：
     - 地表分支：`OneEmbedding(style="PanguEmbedding", img_size=(721, 1440), patch_size=(4, 4), Variables=7, embed_dim=192)`
     - 高空分支：`OneEmbedding(style="PanguEmbedding", img_size=(13, 721, 1440), patch_size=(2, 4, 4), Variables=5, embed_dim=192)`

2. **PanguDownSample**
   - 用途：对 token 网格做下采样，聚合空间邻域信息
   - 调用方式：
     - 地表分支：`OneSample(style="PanguDownSample", input_resolution=(181, 360), output_resolution=(91, 180), in_dim=192)`
     - 高空分支：`OneSample(style="PanguDownSample", input_resolution=(8, 181, 360), output_resolution=(8, 91, 180), in_dim=192)`

3. **PanguUpSample**
   - 用途：对 token 网格做上采样，恢复到更高分辨率
   - 调用方式：
     - 地表分支：`OneSample(style="PanguUpSample", input_resolution=(91, 180), output_resolution=(181, 360), in_dim=384, out_dim=192)`
     - 高空分支：`OneSample(style="PanguUpSample", input_resolution=(8, 91, 180), output_resolution=(8, 181, 360), in_dim=384, out_dim=192)`

4. **PanguPatchRecovery**
   - 用途：将 patch 级别特征图恢复为原始气象场分辨率
   - 调用方式：
     - 地表分支：`OneRecovery(style="PanguPatchRecovery", img_size=(721, 1440), patch_size=(4, 4), in_chans=384, out_chans=7)`
     - 高空分支：`OneRecovery(style="PanguPatchRecovery", img_size=(13, 721, 1440), patch_size=(2, 4, 4), in_chans=384, out_chans=5)`

5. **Transformer 块**
   - 用途：捕获气象场的时空依赖关系
   - 调用方式：使用 `Transformer2DBlock` 处理地表分支，`Transformer3DBlock` 处理高空分支

## 主要 Shape 变化

### 地表分支
1. 输入：`(Batch, 7, 721, 1440)`
2. Embedding 后：`(Batch, 192, 181, 360)`
3. 转换为 token 序列：`(Batch, 181*360, 192)`
4. 编码器处理后：`(Batch, 181*360, 192)`
5. 下采样后：`(Batch, 91*180, 384)`
6. 中间层处理后：`(Batch, 91*180, 384)`
7. 上采样后：`(Batch, 181*360, 192)`
8. 解码器处理后：`(Batch, 181*360, 192)`
9. 转换回特征图：`(Batch, 384, 181, 360)`
10. 恢复后：`(Batch, 7, 721, 1440)`

### 高空分支
1. 输入：`(Batch, 5, 13, 721, 1440)`
2. Embedding 后：`(Batch, 192, 8, 181, 360)`
3. 转换为 token 序列：`(Batch, 8*181*360, 192)`
4. 编码器处理后：`(Batch, 8*181*360, 192)`
5. 下采样后：`(Batch, 8*91*180, 384)`
6. 中间层处理后：`(Batch, 8*91*180, 384)`
7. 上采样后：`(Batch, 8*181*360, 192)`
8. 解码器处理后：`(Batch, 8*181*360, 192)`
9. 转换回特征图：`(Batch, 384, 8, 181, 360)`
10. 恢复后：`(Batch, 5, 13, 721, 1440)`

## 模型参数

- **地表变量数**：7
- **高空变量数**：5
- **气压层数**：13
- **网格分辨率**：721×1440
- **Embedding 维度**：192
- **注意力头数**：6
- **窗口大小**：(6, 12)
- **网络深度**：2
- **MLP 比例**：4.0

## 使用方法

### 模型初始化

```python
from weather_forecast_model import WeatherForecastModel

# 创建模型
model = WeatherForecastModel(
    surface_vars=7,          # 地表变量数
    upper_air_vars=5,        # 高空变量数
    pressure_levels=13,      # 气压层数
    height=721,              # 纬度网格数
    width=1440,              # 经度网格数
    embed_dim=192,           # Embedding 维度
    num_heads=6,             # 注意力头数
    window_size=(6, 12),     # 窗口大小
    depth=2,                 # 网络深度
    mlp_ratio=4.0            # MLP 比例
)
```

### 前向传播

```python
import torch

# 生成输入数据
batch_size = 2
surface_input = torch.randn(batch_size, 7, 721, 1440)
upper_air_input = torch.randn(batch_size, 5, 13, 721, 1440)

# 前向传播
surface_output, upper_air_output = model(surface_input, upper_air_input)

# 输出形状
print(f"Surface output shape: {surface_output.shape}")
print(f"Upper air output shape: {upper_air_output.shape}")
```

## 最小验证建议

1. **形状验证**：确保输入输出形状正确，特别是各层之间的形状转换
2. **前向传播验证**：确保模型可以正常进行前向传播，无错误
3. **参数数量验证**：检查模型参数量是否合理，避免过大或过小
4. **梯度流动验证**：确保模型可以正常进行反向传播，梯度可以流动

## 后续扩展建议

1. **结构改进**：
   - 尝试不同的 Transformer 变体，如 Swin Transformer、Vision Transformer 等
   - 探索不同的注意力机制，如全局注意力、局部注意力等
   - 尝试不同的特征融合策略，如跨注意力、门控融合等

2. **模块替换实验**：
   - 替换 Embedding 层为其他类型的特征提取器
   - 替换 Transformer 块为其他类型的网络结构，如 CNN、RNN 等
   - 替换下采样/上采样模块为其他类型的采样方法

3. **性能优化**：
   - 使用混合精度训练
   - 采用分布式训练
   - 进行模型压缩和量化

4. **功能扩展**：
   - 增加多步预报能力
   - 加入不确定性估计
   - 整合多源数据，如卫星图像、雷达数据等

## 总结

本模型实现了一个基础版的全球格点预报网络，结构清晰、轻量，方便后续进行结构改进和模块替换实验。模型使用了 OneScience 中的 Pangu 系列统一组件，确保了代码的可维护性和可扩展性。

通过分离地表和高空分支处理，模型能够有效地捕获不同层次的气象信息，并通过 Transformer 架构捕获时空依赖关系，为短期天气预报提供准确的预测结果。