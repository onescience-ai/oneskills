---
name: ai4s-case-generator
description: Generate an AI4S dataset adaptation case from a dataset README and an optional target model or example. Use when an agent needs to create a new datapipe only, or create a full runnable case including datapipe, config, training script, and inference script in a user-specified location.
---

# AI4S Case Generator

## 任务目标

把这份 skill 当作总控入口。

当用户只说一句简短的话，例如：

- 基于 oneskills，为这个数据集生成 datapipe
- 基于 oneskills，用 Transolver 适配这个数据集

智能体也应能自动拆解任务，并按需生成：

- `DatasetName.py`
- `DatasetNameDataset`
- `DatasetNameDatapipe`
- config 文件
- `train.py`
- `inference.py`

所有输出都放到用户指定的位置，默认不要修改 `onescience` 仓库内部文件。

## 默认输入

用户通常只需要提供：

- 数据集名
- 数据集 README 路径
- 输出目录
- 可选：目标模型名或目标 example 路径

如果用户没有额外说明，按下面规则理解任务：

1. 用户只提“生成 datapipe”
   只生成数据集接口文件
2. 用户提“适配到某个模型”或“复用某个 example”
   默认生成 datapipe、config、训练脚本和推理脚本

## 必须读取的子 skill

开始执行时，必须按需读取下面的 skill：

1. `oneskills/trae/task/dataset_interface_generator.md`
2. `oneskills/trae/code_standard/datapipe_code_standard_skill.md`
3. `oneskills/component_knowledge/dataset/datapipe_generation_patterns.md`

如果用户要求生成 config，再额外读取：

4. `oneskills/trae/task/config_generator.md`

如果用户要求生成训练脚本，再额外读取：

5. `oneskills/trae/task/train_script_generator.md`
6. `oneskills/trae/code_standard/code_standard_skill.md`

如果用户要求生成推理验证脚本，或者任务属于完整案例生成，再额外读取：

7. `oneskills/trae/task/inference_script_generator.md`

无论是只生成 datapipe，还是生成完整案例，最后都要读取：

8. `oneskills/trae/task/generated_case_review.md`

## 执行顺序

### 第一步：判断任务范围

先判断这次任务属于哪一种：

- `datapipe-only`
- `datapipe + config`
- `datapipe + config + train`
- `datapipe + config + train + inference`

默认规则：

- 没提模型，只做 `datapipe-only`
- 提了模型名、案例名或 example 路径，默认做 `datapipe + config + train + inference`
- 如果用户明确说只生成训练脚本或只补 inference，再按用户限定范围执行

### 第二步：总是先生成 datapipe

先根据 README 和模板生成数据集接口文件。

datapipe 是后续 config 和训练脚本的基础，没有 datapipe 时，不要先写 config 或 train。

### 第三步：如果用户指定了模型或 example，再生成 config

必须优先参考目标 example 的现有配置文件风格，保留模型结构和训练超参数组织方式，只改数据相关字段和必要的输入输出维度。

### 第四步：最后生成训练脚本

必须优先复用目标 example 的训练流程，只替换：

- datapipe import
- datapipe 初始化
- 数据字段访问
- 与新数据集强相关的 loss、target、评估逻辑

不要无故重写整个训练流程。

### 第五步：如果任务需要，再生成 inference.py

必须优先参考目标 example 的现有 `inference.py`。

如果目标 example 没有独立的 `inference.py`，则根据新生成的 datapipe、config、train.py 推导一个可运行的推理验证脚本。

推理脚本至少应做到：

- 能加载训练产出的 checkpoint
- 能调用 datapipe 的 `test_dataloader()`
- 能在 `eval/no_grad` 模式下完成推理
- 能输出预测结果、评估结果或最小验证结果

### 第六步：统一做生成后审查

所有代码生成结束后，必须再做一轮统一审查。

审查重点包括：

- datapipe 中是否保留了无用的 `mean/std` 计算
- config 是否仍然依赖并不存在的统计量字段
- 训练脚本里是否有未定义变量
- inference 脚本里是否有未定义变量或 checkpoint/load 逻辑错误
- datapipe、config、train、inference 之间的数据字段和 shape 是否一致
- 如果用户提供了训练日志，异常高的初始 loss 是否已经触发数据预处理排查

审查不通过时，先修正再交付，不要把“可能能跑”版本直接交给用户。

## 输出规则

所有生成物默认写到用户指定目录，例如：

- `<output_dir>/RAE2822.py`
- `<output_dir>/conf/transolver_rae2822.yaml`
- `<output_dir>/train.py`
- `<output_dir>/inference.py`

除非用户明确要求，否则不要：

- 把文件直接写回 `onescience/examples/...`
- 修改 `onescience/src/...`
- 修改原始 example 的源码

## 最终交付要求

交付时必须明确说明：

- 这次生成了哪些文件
- 每个文件的输出路径
- 参考了哪些 `onescience` 模板文件
- 是否保留了归一化统计；如果没有保留，要说明原因
- 是否完成了生成后审查；如果发现并修复了问题，要说明修复点
- 如果出现高 loss 风险，是否判断并处理了预处理需求
- 如果生成了 inference，checkpoint 加载和测试入口是如何与 train/config 对齐的
- 如果只生成了 datapipe，为什么没有生成 config、train 或 inference

## 给下游智能体的最短触发方式

当用户只给一句简短指令时，按下面方式理解：

### 示例 1：只生成 datapipe

```text
基于 oneskills，为 RAE2822 生成 datapipe，README 在 dataset/README.md，输出到 ./generated
```

### 示例 2：生成完整适配案例

```text
基于 oneskills，用 Transolver 适配 RAE2822，README 在 dataset/README.md，输出到 ./generated
```

### 示例 3：只补 inference 脚本

```text
基于 oneskills，为已经适配好的 RAE2822 + Transolver 案例补一个 inference.py，README 在 dataset/README.md，输出到 ./generated
```

收到这类指令后，不要先要求用户补一大段提示词，直接按本 skill 执行。
