# Datapipe: <DatapipeName>

## 基本信息

- Datapipe 名：`<DatapipeName>`
- 数据类型：`<climate_or_other>`
- 主要任务：`<forecast_or_reconstruction_or_other>`
- 数据组织方式：`<per_year_h5_or_per_timestamp_h5_or_other>`

## Datapipe 职责

一句话说明该 datapipe 负责把什么原始数据组织成什么样的训练样本。

补充说明：

- 数据存储目录的基本结构
- 负责哪些预处理步骤
- 是否同时负责 `Dataset` 与 `DataLoader`

## 输入配置

- `<arg_1>`
  - `<说明>`
- `<arg_2>`
  - `<说明>`
- `<arg_3>`
  - `<说明>`

建议优先记录：

- 数据目录
- 年份选择
- 变量选择
- `input_steps`
- `output_steps`
- 归一化或缺失值处理方式
- `batch_size`
- `num_workers`

## 数据存储约定

- 主数据路径：`<path_pattern>`
- 统计量路径：`<stats_pattern>`
- 元数据来源：`<metadata_or_attrs>`

若有额外约定，也写清楚：

- `<constraint_1>`
- `<constraint_2>`

## 样本构造方式

- 输入样本：`<input_shape_and_semantics>`
- 输出样本：`<output_shape_and_semantics>`
- 附加返回项：`<extra_outputs>`

建议说明：

- 时间步如何切片
- 变量如何索引
- 是否裁剪空间尺寸
- 是否额外生成物理先验量，例如太阳天顶角

## DataLoader 约定

- 训练阶段：`<train_loader_behavior>`
- 验证阶段：`<val_loader_behavior>`
- 测试阶段：`<test_loader_behavior>`

## 适合优先使用的场景

- `<scene_1>`
- `<scene_2>`
- `<scene_3>`

## 风险点

- `<risk_1>`
- `<risk_2>`
- `<risk_3>`

优先写：

- 数据目录结构是否固定
- train/val/test 划分由谁负责
- 是否假设每年样本数一致
- 是否会受闰年、缺测、NaN 或变量命名影响

## 源码锚点

- `./onescience/<path_to_primary_impl>.py`
- `./onescience/<path_to_related_utils>.py`

统一约定：

- 一律使用 `./onescience/...` 相对路径
- 不写本机绝对路径
- 最多放 2 到 4 个高价值锚点
