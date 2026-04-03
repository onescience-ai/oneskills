---
name: generated-case-review
description: Review generated datapipe, config, training code, and inference code before delivery. Use when an agent has produced AI4S adaptation files and needs to catch undefined variables, unused normalization logic, checkpoint mismatches, field mismatches, and tensor shape alignment issues.
---

# Generated Case Review

## 任务目标

这份 skill 用于在代码生成完成后做最后一轮审查。

它不是可选步骤。只要生成了 datapipe、config、训练脚本或推理脚本中的任意一种，就必须执行一次审查。

## 审查顺序

按下面顺序检查：

1. 先审查 datapipe
2. 再审查 config
3. 最后审查训练脚本
4. 再审查 inference 脚本
5. 最后再做一次跨文件一致性检查

## 一、datapipe 审查清单

必须检查：

- 文件名、类名是否正确
- `DatasetNameDataset` 是否继承 `BaseDataset`
- `DatasetNameDatapipe` 是否真的提供了 dataloader 接口
- `__getitem__` 返回的是 `dict` 还是 `PyG Data`
- 返回字段名是否清晰、稳定、未拼错
- 是否引用了不存在的字段名
- 是否存在无用的 `mean/std` 计算

### 关于归一化统计

如果训练脚本、config、模型前后处理都没有使用统计量，则应该删除或不要生成：

- `_calculate_normalization`
- `coef_norm`
- `mean_in/std_in`
- `mean_out/std_out`
- 对应的缓存文件路径

不要因为模板里有就保留。

但如果用户已经提供了异常训练日志，或者从数据字段说明可以判断原始尺度差异很大，则必须反查是否需要补充预处理。

## 二、config 审查清单

必须检查：

- datapipe 名称是否和生成文件一致
- 数据路径是否来自用户指定位置
- `in_dim/out_dim/space_dim/fun_dim` 是否与 datapipe 返回一致
- 是否保留了 datapipe 并未提供的统计量路径
- 是否引用了旧 example 才有、新数据集没有的字段

## 三、训练脚本审查清单

必须检查：

- import 是否都能对应到真实文件或真实符号
- 是否存在未定义变量
- datapipe 初始化参数是否真实存在
- batch 访问方式是否与 datapipe 返回结构一致
- loss 中使用的 `pred` 和 `target` 是否 shape 对齐
- 模型输入字段是否与 datapipe 输出字段一致
- 是否误用了旧 example 的专有字段、专有 metric 或专有后处理
- 如果出现异常高的初始 loss，是否已经检查输入输出尺度和预处理缺失

## 四、inference 前置说明

在做跨文件一致性检查前，如果当前任务包含 `inference.py`，必须先完成推理脚本专项审查。

## 五、inference 脚本审查清单

必须检查：

- import 是否都能对应到真实文件或真实符号
- checkpoint 路径和命名是否与训练脚本一致
- 是否调用了 datapipe 的 `test_dataloader()`
- 是否使用了与训练一致的模型构建逻辑
- 是否正确设置了 `model.eval()` 和 `torch.no_grad()`
- 是否引用了 datapipe 中真实存在的字段
- 如果有归一化，是否与训练保持一致
- 如果没有归一化，是否没有残留无用的 `coef_norm/mean/std`
- 是否误用了旧 example 的专有评估逻辑

## 六、跨文件一致性审查

必须把 datapipe、config、train、inference 四者连起来检查：

1. datapipe 输出什么字段
2. config 声明了什么维度
3. train 里读取了什么字段
4. model 期望什么输入
5. inference 里读取了什么字段
6. loss 和推理输出期望什么 shape

只要其中任意一环对不上，就必须修正。

## 七、最小验证动作

如果本地环境允许，优先做这些检查：

1. 对生成的 Python 文件做语法检查
2. 逐个检查 import 是否可解析
3. 手工或静态追踪一次样本从 datapipe 到 model/loss 的 shape 流动
4. 如果有训练日志，结合 loss 量级检查是否需要补充预处理
5. 如果生成了 inference，静态检查 checkpoint 加载、eval/no_grad 和 test loader 调用

如果环境不允许运行，也必须完成手工审查，不能跳过。

## 八、交付前必须修正的问题

发现以下问题时，不要直接交付：

- 未定义变量
- 拼写错误导致的字段不一致
- `pred/target` shape 不对齐
- datapipe 返回 `dict`，训练脚本却按 `data.x` 访问
- datapipe 返回 `PyG Data`，训练脚本却按 `batch["x"]` 访问
- config 中维度与 datapipe 不一致
- inference 加载的 checkpoint 路径和训练保存规则不一致
- inference 访问的数据字段与 datapipe 不一致
- 无用的 `mean/std` 统计逻辑残留
- 明显的高 loss 风险已经出现，但仍未评估是否需要预处理

这些问题必须先修完，再交付给用户。

## 九、最终说明要求

审查完成后，结果说明里至少要包含：

- 是否发现未定义变量问题
- 是否发现 shape 对齐问题
- 是否删除了无用的 `mean/std` 计算
- 是否发现异常高 loss 对应的数据预处理需求
- 如果生成了 inference，checkpoint 和 test loader 是否已对齐
- 还剩哪些风险没有办法在当前环境中验证
