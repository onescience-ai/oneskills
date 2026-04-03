---
name: train-script-generator
description: Generate or adapt a training script for a new dataset by reusing the nearest OneScience example training flow and connecting it to the generated datapipe and config. Use when an agent must produce a runnable train.py in a user-specified location.
---

# Train Script Generator

## 任务目标

根据：

- 用户指定的目标模型或目标 example
- 已生成的 datapipe 文件
- 已生成的 config 文件
- 目标 example 的现有 `train.py`

生成一个新的训练脚本。

输出位置由用户指定，不默认写回 `onescience` 项目。

## 必须先读取的资源

1. 目标 example 的 `train.py`
2. 目标 example 的相关 config 文件
3. 已生成的 `DatasetName.py`
4. 已生成的新 config 文件
5. `oneskills/trae/code_standard/code_standard_skill.md`

如果目标 example 有 `inference.py`、`common.py` 或评估辅助脚本，并且训练脚本依赖它们，也要一并读取。

## 强制工作流

### 第一步：先理解原始训练流程

先识别目标 example 中哪些部分必须保留：

- 模型构建方式
- optimizer 和 scheduler
- 训练循环
- 验证流程
- checkpoint 保存逻辑
- 日志记录逻辑

不要一上来就重写整个 `train.py`。

### 第二步：只替换与新数据集直接相关的部分

优先调整：

- datapipe import
- datapipe 初始化
- batch 或 `data` 的字段访问方式
- target 读取方式
- 与输出维度强相关的 loss 和 metric
- 与归一化统计强相关、但新 datapipe 并未提供的逻辑

### 第三步：保持模型侧风格不变

如果用户指定了 `Transolver`，就优先沿用 `Transolver` 示例中的模型构建和训练流程。

如果训练脚本里使用 `onescience.modules` 组件，遵循 `code_standard_skill.md`，不要绕过统一入口乱改导入风格。

如果当前任务后续还要生成 `inference.py`，则训练脚本里的 checkpoint 保存方式必须稳定、明确，并与 config 保持一致，方便 inference 直接复用。

### 第四步：处理评估逻辑差异

如果原 example 的评估逻辑强依赖旧数据集的专有字段，而新数据集 README 中没有这些字段，则：

1. 保留训练主流程
2. 把评估降级为通用 loss 或通用导出
3. 在结果说明里明确指出降级原因

不要假装旧数据集专有字段仍然存在。

### 第五步：生成后必须做脚本审查

训练脚本生成完成后，必须主动做一轮审查，至少检查：

- 是否存在未定义变量
- datapipe 返回结构和训练脚本访问方式是否一致
- `model` 输入张量或图数据字段是否与 datapipe 输出对齐
- `pred` 和 `target` 的 shape 是否可用于当前 loss
- config 中的 `in_dim/out_dim/space_dim` 是否与 datapipe 和模型一致
- 是否引用了旧 example 中新数据集并不存在的字段
- 如果初始训练日志显示 loss 异常高，是否检查了输入输出尺度和预处理缺失问题

如果发现问题，必须先修正，再交付。

### 第六步：如果 loss 异常高，要优先做数据尺度排查

如果用户提供了类似下面的现象：

- 训练刚开始 loss 就非常大
- loss 在前几个 iteration 就明显异常
- 很快出现 `nan/inf`

则优先排查：

1. datapipe 是否缺少输入标准化
2. target 是否需要单独做归一化
3. 当前 loss 是否受原始量纲放大
4. config 是否遗漏了与预处理对应的参数

只有在排除了这些问题后，再去怀疑模型结构或优化器超参数。

## 生成时禁止事项

- 禁止无故重写整个训练脚本
- 禁止跳过 datapipe 和 config，直接硬编码数据路径
- 禁止假设新数据集字段与旧 example 完全相同
- 禁止默认修改 `onescience` 仓库内部源码
- 禁止在未审查未定义变量和 shape 对齐前直接交付

## 生成后的自检

生成完成后至少检查：

- import 的 datapipe 名称是否正确
- config 路径是否指向新生成的配置文件
- 训练循环是否基本保留目标 example 风格
- 数据字段访问是否与 datapipe 一致
- loss 和输出维度是否与新数据集匹配
- 是否不存在明显未定义变量
- 是否完成了 `pred/target` shape 对齐检查
- 如果用户提供了异常 loss 日志，是否完成了数据尺度和预处理检查
- 如果需要 inference，checkpoint 保存路径和命名是否足够稳定且可复用

## 这份 skill 的作用边界

这份 skill 只负责生成训练脚本。

它不负责：

- 生成 datapipe
- 生成 config
- 修改模型源码
