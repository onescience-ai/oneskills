# 全球短期天气预报模型 - 详细执行信息

## 任务目标

实现一个基础版的全球格点预报网络，输入当前时刻的全球气象场，输出下一时刻的预报结果。该模型应结构清晰、轻量，方便后续进行结构改进和模块替换实验。

## 输入定义

### 输入数据类型
- **全球格点气象场**：包含地表变量和多层大气变量

### 输入变量组织

#### 地表变量（Surface Variables）
- 形状：`(Batch, SurfaceVars, Height, Width)`
- 典型变量：地表气压、海平面气压、2米温度、2米湿度、10米风速、10米风向等
- 推荐变量数：7

#### 多层大气变量（Upper-air Variables）
- 形状：`(Batch, UpperAirVars, PressureLevels, Height, Width)`
- 典型变量：位势高度、温度、湿度、u风、v风等
- 推荐变量数：5
- 推荐气压层数：13

### 输入分辨率
- 全球格点分辨率：`(721, 1440)`（纬度×经度）

## 输出定义

### 输出数据类型
- **下一时刻的全球格点气象场**：与输入格式相同

### 输出变量组织

#### 地表变量（Surface Variables）
- 形状：`(Batch, SurfaceVars, Height, Width)`
- 变量与输入相同

#### 多层大气变量（Upper-air Variables）
- 形状：`(Batch, UpperAirVars, PressureLevels, Height, Width)`
- 变量与输入相同

## 变量组织方式

### 数据结构
- 采用分离的地表和高空分支处理
- 统一使用 `Batch`、`PressureLevels`、`Height`、`Width` 命名

### 数据预处理
- 归一化：对输入变量进行标准化处理
- 数据增强：可选的随机旋转、翻转等操作

## 推荐复用的 OneScience 组件

### 1. PanguEmbedding
- **用途**：将原始气象场切分为非重叠 patch，并投影到统一的 embedding 特征空间
- **调用方式**：
  - 地表分支：`OneEmbedding(style="PanguEmbedding", img_size=(721, 1440), patch_size=(4, 4), Variables=7, embed_dim=192)`
  - 高空分支：`OneEmbedding(style="PanguEmbedding", img_size=(13, 721, 1440), patch_size=(2, 4, 4), Variables=5, embed_dim=192)`

### 2. PanguDownSample
- **用途**：对 token 网格做下采样，聚合空间邻域信息
- **调用方式**：
  - 地表分支：`OneSample(style="PanguDownSample", input_resolution=(181, 360), output_resolution=(91, 180), in_dim=192)`
  - 高空分支：`OneSample(style="PanguDownSample", input_resolution=(8, 181, 360), output_resolution=(8, 91, 180), in_dim=192)`

### 3. PanguUpSample
- **用途**：对 token 网格做上采样，恢复到更高分辨率
- **调用方式**：
  - 地表分支：`OneSample(style="PanguUpSample", input_resolution=(91, 180), output_resolution=(181, 360), in_dim=384, out_dim=192)`
  - 高空分支：`OneSample(style="PanguUpSample", input_resolution=(8, 91, 180), output_resolution=(8, 181, 360), in_dim=384, out_dim=192)`

### 4. PanguPatchRecovery
- **用途**：将 patch 级别特征图恢复为原始气象场分辨率
- **调用方式**：
  - 地表分支：`OneRecovery(style="PanguPatchRecovery", img_size=(721, 1440), patch_size=(4, 4), in_chans=384, out_chans=7)`
  - 高空分支：`OneRecovery(style="PanguPatchRecovery", img_size=(13, 721, 1440), patch_size=(2, 4, 4), in_chans=384, out_chans=5)`

## 主干结构建议

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

### 关键模块
- **Transformer 编码器/解码器**：用于捕获气象场的时空依赖关系
- **特征融合模块**：用于融合地表和高空特征信息
- **位置编码**：添加空间位置信息

## 主要 Shape 变化

### 地表分支
1. 输入：`(Batch, 7, 721, 1440)`
2. Embedding 后：`(Batch, 192, 181, 360)`
3. 下采样后：`(Batch, 384, 91, 180)`
4. 上采样后：`(Batch, 192, 181, 360)`
5. 恢复后：`(Batch, 7, 721, 1440)`

### 高空分支
1. 输入：`(Batch, 5, 13, 721, 1440)`
2. Embedding 后：`(Batch, 192, 8, 181, 360)`
3. 下采样后：`(Batch, 384, 8, 91, 180)`
4. 上采样后：`(Batch, 192, 8, 181, 360)`
5. 恢复后：`(Batch, 5, 13, 721, 1440)`

## 需要确认的关键点

1. **输入变量具体包含哪些**：请确认地表和高空变量的具体种类和数量
2. **预报时间步长**：请确认预报的时间步长（如 6 小时、12 小时等）
3. **模型复杂度**：请确认是否需要调整模型深度和宽度以平衡性能和速度
4. **数据来源**：请确认训练数据的来源和格式
5. **评估指标**：请确认模型评估的具体指标（如 RMSE、MAE 等）

## 待确认摘要

- 模型架构：基于 Transformer 的编码器-解码器结构，使用 Pangu 系列组件
- 输入输出：当前时刻的全球气象场 → 下一时刻的全球气象场
- 变量组织：地表变量和多层大气变量分离处理
- 组件复用：优先使用 OneScience 中的 Pangu 系列统一组件
- 后续扩展：结构设计便于后续进行结构改进和模块替换实验

## 自检结果

- [x] 任务目标明确：实现基础版全球格点预报网络
- [x] 输入输出定义清晰：当前时刻气象场 → 下一时刻预报结果
- [x] 变量组织合理：地表和高空变量分离处理
- [x] 组件复用策略：优先使用 OneScience 中的现有模块
- [x] 主干结构完整：编码器-解码器架构，包含必要的组件
- [x] Shape 变化明确：各层输入输出形状清晰
- [x] 待确认点已列出：需要用户确认的关键参数

## 用户操作

请确认上述详细执行信息是否符合您的需求。如果确认，我将继续生成完整的代码实现。如果有任何修改建议，请在确认时提出。