---
name: onescience-datapipe-code-standard
description: Generate new Dataset and Datapipe files in a user-specified location while following OneScience's existing naming, class structure, and loader API. Use when an agent needs to create a new data interface from a dataset README and current datapipes examples.
---

# OneScience Datapipe Code Standard

## 目标

根据新数据集 README 和现有仓库代码，生成一个符合 `onescience.datapipes` 风格的 datapipe 文件。

这份 skill 只约束代码标准，不把输出目录写死在 `onescience` 项目中。生成位置由用户或调用方指定。

## 强制命名规则

给定数据集名 `DatasetName`：

- 文件名必须是 `DatasetName.py`
- 数据集类必须是 `DatasetNameDataset`
- datapipe 类必须是 `DatasetNameDatapipe`

例如当用户说数据集名是 `RAE2822`，则产物必须是：

- `RAE2822.py`
- `RAE2822Dataset`
- `RAE2822Datapipe`

## 强制继承规则

生成的数据集类必须继承：

```python
from onescience.datapipes.core import BaseDataset
```

也就是说，核心类声明必须是：

```python
class DatasetNameDataset(BaseDataset):
    ...
```

## 强制路径规则

输出文件路径由用户指定，不要默认写成：

- `onescience/src/onescience/datapipes/...`
- `examples/`
- 任意固定仓库目录

如果用户指定的是一个 Python 包目录，并且希望后续通过包级导入使用 datapipe，那么再补充更新该目标目录下的 `__init__.py`。

如果用户只是要生成一个可复用的 datapipe 文件，则不强制修改 `__init__.py`。

## 强制结构规则

如果目标训练脚本消费 `PyG Data`，优先生成如下结构：

```python
class DatasetNameDataset(BaseDataset):
    ...


class DatasetNameDatapipe:
    ...
```

### Dataset 类至少包含

- `__init__`
- `_init_paths`
- `_load_metadata`
- `__len__`
- `__getitem__`

### Datapipe 类至少包含

- `train_dataloader`
- `val_dataloader`
- `test_dataloader`

## 调用契约反推规则

如果用户提供了目标 `train.py` 或其他调用脚本，写 datapipe 前必须先看它。

### 如果训练脚本出现

```python
data = data.to(device)
out = model(data)
targets = data.y
```

则 datapipe 应优先返回 `torch_geometric.data.Data`，并包含：

- `pos`
- `x`
- `y`
- `surf`
- `edge_index`

### 如果训练脚本出现

```python
batch["x"]
batch["y"]
```

则 datapipe 应优先返回 `dict`，不要强行改成 `PyG Data`。

如果用户没有提供调用脚本，则根据 README 的数据组织方式和最接近模板来决定返回结构。

## 归一化规则

如果同类 datapipe 已经使用 `coef_norm` 或统计量缓存，则优先沿用当前目录的已有命名和保存方式，不要发明新的命名体系。

优先复用：

- `self.coef_norm`
- `mean_in.npy`
- `std_in.npy`
- `mean_out.npy`
- `std_out.npy`

但这不代表必须计算这些统计量。

只有在目标训练流程、config 或模型前后处理真实依赖这些统计量时，才保留这套逻辑。

如果训练流程根本没有使用这些统计量，则应删除或不要生成：

- `_calculate_normalization`
- 无用的 `coef_norm` 初始化
- 无用的 `stats_dir` 路径依赖

但如果数据原始量级明显会导致训练初始 loss 异常大，则应重新判断是否需要加入预处理或归一化。

也就是说，这里的原则不是“永远不要算 mean/std”，而是“只在训练确实需要时才保留或补充这套逻辑”。

## 分布式和 DataLoader 规则

优先沿用当前目录已有风格：

- 使用 `DistributedManager`
- 使用 `DistributedSampler`
- 根据 `distributed` 参数决定 sampler

## 生成时禁止事项

- 禁止只生成一个函数，不生成 `DatasetNameDataset`
- 禁止只生成 `DatasetNameDataset`，不生成 `DatasetNameDatapipe`
- 禁止只写伪代码
- 禁止脱离现有 `onescience.datapipes` 风格另起一套接口
- 禁止默认生成 `train.py`
- 禁止默认生成 config 文件
- 禁止为了模仿旧模板而保留训练时完全用不上的 `mean/std` 计算

## 条件导出规则

只有在下面两种情况之一成立时，才需要同步更新目标目录下的 `__init__.py`：

1. 用户明确要求后续通过包级导入使用 datapipe
2. 目标目录本身就是一个现有 Python 包，并且该包当前已经用 `__init__.py` 管理导出

如果不满足这些条件，只生成 datapipe 文件本身即可。

## 完成判据

只有同时满足以下条件才算完成：

1. 文件名正确
2. 类名正确
3. `DatasetNameDataset` 正确继承 `BaseDataset`
4. `DatasetNameDatapipe` 提供 dataloader 接口
5. 返回结构与 README 或目标调用脚本兼容
6. 如果用户要求包级导入，目标目录的 `__init__.py` 已同步更新
7. 如果训练流程不依赖归一化，代码中没有无用的统计量计算逻辑
