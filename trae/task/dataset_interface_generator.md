---
name: dataset-interface-generator
description: Generate a new dataset interface file from a dataset README and existing OneScience-style templates. Use when an agent must create DatasetName.py, DatasetNameDataset, and DatasetNameDatapipe in a user-specified location.
---

# Dataset Interface Generator

## 任务目标

让智能体在拿到：

- 一个新数据集的 README
- 当前 onescience 仓库中的 datapipe 代码作为参考模板
- 本 skill 和 datapipe 规范 skill
- 一个用户指定的输出位置
- 可选：一个下游训练脚本或调用样例

之后，直接生成：

- `DatasetName.py`
- `DatasetNameDataset`
- `DatasetNameDatapipe`

这份 task 的目标是生成 datapipe 文件本身，不默认生成 `train.py` 或 config。

## 必须先读取的资源

执行本任务时，必须先读取：

1. `oneskills/component_knowledge/dataset/datapipe_generation_patterns.md`
2. `oneskills/trae/code_standard/datapipe_code_standard_skill.md`
3. 用户提供的新数据集 README
4. 同领域下最接近的现有 datapipe 文件
5. 如果用户提供了下游脚本，再额外读取该脚本

## 强制工作流

### 第一步：先确认输入和输出边界

必须先确认：

- 数据集名是什么
- 输出文件要放到哪里
- 用户是否只要 datapipe 文件，还是还要求包级导出
- 是否提供了下游训练脚本或调用样例

如果用户没有要求别的内容，默认只生成 datapipe 文件本身。

### 第二步：如果提供了下游脚本，先反推 datapipe 契约

需要确认：

- 当前训练脚本 import 的 datapipe 是谁
- 训练循环里实际访问了哪些字段
- 返回结构是 `PyG Data` 还是 `dict`
- 是否依赖 `pos`、`x`、`y`、`surf`、`edge_index` 等字段

如果用户没有提供下游脚本，则跳过这一步，改为依赖 README 和最接近模板来设计接口。

### 第三步：从 README 抽取数据接口需要的信息

必须抽取：

- 文件组织方式
- 核心字段
- 坐标字段
- 条件字段
- 目标字段
- split 信息
- 是否存在几何边界、表面、mask

忽略与接口实现无关的论文背景和实验结论。

### 第四步：选择最近似的现有 datapipe 模板

按优先级选择：

1. 同领域、同调用契约
2. 同领域、相近数据格式
3. 不同领域但返回结构一致

不要直接从零发明一套新风格。

### 第五步：生成新的 datapipe 文件

生成时必须满足：

- 文件名等于数据集名
- 类名等于数据集名加固定后缀
- 输出到用户指定的位置
- `DatasetNameDataset` 继承 `BaseDataset`
- `__getitem__` 返回结构与 README 或目标调用脚本匹配
- 只有在训练流程真实依赖时才计算 `mean/std`

如果 README、样本字段说明或用户提供的训练日志已经表明原始数据尺度可能导致异常高的初始 loss，则在这一步必须额外判断是否需要加入预处理逻辑。

### 第六步：按需补齐导出

只有在用户明确要求包级导入时，才需要同步修改目标目录中的：

- `__init__.py`

如果用户没有这个要求，任务到 datapipe 文件生成完成即可结束。

## 默认输出模板

如果目标是图/点集式训练脚本，默认输出应为：

```python
class DatasetNameDataset(BaseDataset):
    def __init__(...):
        ...

    def _init_paths(self):
        ...

    def _load_metadata(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        return Data(...)


class DatasetNameDatapipe:
    def __init__(self, params, distributed: bool):
        ...

    def train_dataloader(self):
        ...

    def val_dataloader(self):
        ...

    def test_dataloader(self):
        ...
```

## 命名推导规则

用户给出数据集名后，智能体必须自动推导：

- 数据集名 `FooBar`
- 文件名 `FooBar.py`
- 数据集类 `FooBarDataset`
- datapipe 类 `FooBarDatapipe`

如果用户说数据集名是 `RAE2822`，则必须推导为：

- `RAE2822.py`
- `RAE2822Dataset`
- `RAE2822Datapipe`

这里的 `RAE2822` 只是命名示例，不表示 skill 内置了该数据集的专有知识。

## 生成后的自检清单

生成完成后必须检查：

- 文件路径是否落在用户指定的位置
- 是否存在 `DatasetNameDataset` 和 `DatasetNameDatapipe`
- `DatasetNameDataset` 是否正确继承 `BaseDataset`
- `DatasetNameDatapipe` 是否能构造 `train/val/test` loader
- 返回字段名是否与 README 或目标调用脚本一致
- 如果实现了归一化，统计量是否只在训练集计算
- 如果训练流程不依赖归一化，是否已经移除无用的 `mean/std` 计算
- 如果出现高 loss 风险，是否已经评估是否需要加入输入或输出预处理
- 如果用户要求包级导入，是否已经更新目标目录下的 `__init__.py`

## 这份 skill 的作用边界

这份 skill 只负责“生成数据集接口文件”。

它不负责：

- 把文件默认写入 `onescience` 项目
- 自动改模型结构
- 自动生成 `train.py`
- 自动生成 config 文件
- 自动做数据下载
- 自动做实验复现报告
