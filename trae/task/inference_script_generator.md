---
name: inference-script-generator
description: Generate or adapt an inference script for a new dataset by reusing the nearest OneScience example inference flow and connecting it to the generated datapipe, config, and training script. Use when an agent must produce a runnable inference.py in a user-specified location.
---

# Inference Script Generator

## 任务目标

根据：

- 用户指定的目标模型或目标 example
- 已生成的 datapipe 文件
- 已生成的 config 文件
- 已生成的 `train.py`
- 目标 example 的现有 `inference.py` 或最接近的推理模板

生成一个新的 `inference.py`。

输出位置由用户指定，不默认写回 `onescience` 项目。

## 必须先读取的资源

1. 目标 example 的 `inference.py`，如果存在则优先读取
2. 目标 example 的 `train.py`
3. 目标 example 的相关 config 文件
4. 已生成的 `DatasetName.py`
5. 已生成的新 config 文件
6. 已生成的新 `train.py`
7. `oneskills/trae/code_standard/code_standard_skill.md`

如果目标 example 没有独立的 `inference.py`，则必须根据 `train.py + config + datapipe` 反推一个推理脚本。

## 强制工作流

### 第一步：先确定推理目标

必须先判断用户想做哪一种推理验证：

- 用测试集做批量推理
- 加载最佳 checkpoint 做评估
- 生成预测文件或可视化结果
- 只做最小可运行验证

如果用户没有额外说明，默认生成“可加载 checkpoint、可跑 test dataloader、可保存预测或评估结果”的标准 `inference.py`。

### 第二步：优先复用目标 example 的推理结构

如果目标 example 本身有 `inference.py`，优先保留它的：

- 配置加载方式
- 模型构建方式
- checkpoint 加载方式
- test dataloader 调用方式
- 结果保存结构

只替换与新数据集直接相关的部分。

如果目标 example 没有 `inference.py`，则优先复用 `train.py` 中的：

- 模型初始化逻辑
- config 读取逻辑
- device 设置逻辑
- datapipe 初始化逻辑

再补上 `model.eval()`、`torch.no_grad()`、结果保存和推理循环。

### 第三步：保证与 datapipe、config、train 一致

必须保证：

- 使用 `DatasetNameDatapipe` 的 `test_dataloader()`
- 读取与训练脚本一致的模型配置
- 加载与训练脚本保存规则一致的 checkpoint
- 访问与 datapipe 返回结构一致的字段

如果 train 脚本把 checkpoint 保存到：

- `checkpoint_dir/<model_name>.pth`

那么 inference 脚本也应按同一规则加载，不要另起一套命名方式。

### 第四步：按需处理归一化和反归一化

如果 datapipe 或 config 中没有启用归一化，就不要在 inference 中假装存在：

- `coef_norm`
- `mean/std`
- 反归一化逻辑

如果 datapipe 的训练流程确实依赖归一化，则 inference 中必须与训练保持一致：

- 使用相同统计量
- 在需要的时候做反归一化
- 不要漏掉输出还原

### 第五步：处理数据集专有评估差异

如果目标 example 的 `inference.py` 强依赖旧数据集的专有字段、专有 metric 或专有后处理，而新数据集没有这些内容，则：

1. 保留通用推理主流程
2. 移除不兼容的专有评估逻辑
3. 降级为通用 loss、通用误差统计或通用预测导出
4. 在交付说明中明确指出降级原因

不要直接照搬旧 example 的专有评估逻辑。

## 生成时禁止事项

- 禁止无故重写整个推理脚本
- 禁止跳过 datapipe 和 config，直接硬编码路径
- 禁止加载与训练脚本保存规则不一致的 checkpoint
- 禁止在 `inference.py` 中保留不存在的数据字段访问
- 禁止忘记 `model.eval()` 或 `torch.no_grad()`
- 禁止保留新数据集根本没有的旧 example 专有评估逻辑

## 生成后的自检

生成完成后至少检查：

- import 的 datapipe 名称是否正确
- config 路径是否指向新生成的配置文件
- checkpoint 路径和命名是否与训练脚本一致
- 是否调用了 `test_dataloader()`
- 数据字段访问是否与 datapipe 一致
- 是否正确设置了 `model.eval()` 和 `torch.no_grad()`
- 如果需要反归一化，逻辑是否与训练一致
- 如果不需要归一化，是否没有残留无用的 `mean/std` 或 `coef_norm`
- 输出保存路径是否来自用户指定目录或 config

## 这份 skill 的作用边界

这份 skill 只负责生成 `inference.py`。

它不负责：

- 生成 datapipe
- 生成 config
- 生成训练脚本
- 修改模型源码
