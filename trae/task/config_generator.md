---
name: config-generator
description: Generate or adapt a model config file for a new dataset by reusing the nearest OneScience example config and aligning it with the generated datapipe. Use when an agent must produce a runnable config in a user-specified location.
---

# Config Generator

## 任务目标

根据：

- 用户指定的目标模型或目标 example
- 已生成的 datapipe 文件
- 数据集 README
- 现有 example 的配置模板

生成一个新的 config 文件。

输出位置由用户指定，不默认写回 `onescience` 项目。

## 必须先读取的资源

1. 用户指定的目标 example 目录，或与目标模型最接近的 example
2. 目标 example 当前使用的 config 文件
3. 已生成的 `DatasetName.py`
4. 用户提供的数据集 README

如果目标 example 存在 `inference.py`，并且当前任务需要生成推理脚本，也应一并读取，确认 config 里哪些字段会被 inference 复用。

## 强制工作流

### 第一步：先复用目标 example 的配置结构

不要从零发明配置格式。

优先保留目标 example 里已有的：

- 模型配置块
- 优化器配置块
- scheduler 配置块
- 训练轮数、batch size、日志和保存策略

### 第二步：只修改数据适配必需的部分

优先调整：

- datapipe 或 dataset 名称
- 数据路径
- train/val/test split 相关字段
- 输入维度和输出维度
- 归一化统计路径
- batch size 或 num_workers 中与数据组织强相关的参数

### 第三步：用 datapipe 反推配置字段

必须检查 datapipe 暴露了什么：

- 返回 `PyG Data` 还是 `dict`
- 输入特征维度是多少
- 输出目标维度是多少
- 是否依赖 `surf`、`edge_index`、mask、法向量等字段

如果模型配置里存在 `space_dim`、`in_dim`、`out_dim`、`fun_dim` 等字段，必须与 datapipe 实际返回保持一致。

如果 datapipe 没有实现归一化统计，config 中也不要保留无用的：

- `stats_dir`
- `coef_norm`
- `mean/std` 路径
- 依赖这些统计量的开关

但如果训练日志或数据尺度分析表明初始 loss 异常偏高是由缺少预处理导致，则 config 中应补充与 datapipe 一致的预处理开关或统计量路径。

### 第四步：保持路径外置

配置中的数据路径、统计量路径、输出路径都应该可由用户修改，不要写死为 `onescience` 内部路径。

如果当前任务需要生成 `inference.py`，则 config 还应覆盖推理会直接使用的字段，例如：

- checkpoint 路径或 checkpoint 目录
- result 输出目录
- test split 名称
- 可选的推理样本数、导出路径或可视化路径

只保留目标 example 和新推理脚本真正会使用的字段，不要为了“看起来完整”而堆无关配置。

## 生成时禁止事项

- 禁止重写整套配置体系
- 禁止引入目标 example 原本没有的复杂字段
- 禁止把数据路径写死到 `onescience` 项目内部
- 禁止在未检查 datapipe 字段前随意填写输入输出维度
- 禁止在 datapipe 不使用统计量时仍然保留无用的归一化配置

## 生成后的自检

生成完成后至少检查：

- config 结构是否延续目标 example 风格
- 数据路径是否可由用户配置
- datapipe 名称是否与生成文件一致
- 输入输出维度是否与 datapipe 一致
- 是否只改了数据适配真正需要改的部分
- 如果 datapipe 不依赖归一化，config 中是否已去掉无用的统计量字段
- 如果 datapipe 增加了预处理，config 是否同步补齐了对应配置
- 如果需要 inference，config 是否已包含 checkpoint 和结果输出所需字段

## 这份 skill 的作用边界

这份 skill 只负责生成 config。

它不负责：

- 生成 datapipe
- 生成训练脚本
- 修改模型源码
