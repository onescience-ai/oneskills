# OneScience Datapipe Generation Patterns

## 目标

这份文档用于指导智能体根据一个新数据集的 README，生成一个符合当前 `onescience.datapipes` 风格的数据集接口文件。

这份 skill 只约束代码风格、类结构和命名规则，不把输出位置写死在 `onescience` 项目里。生成文件时，目标目录由用户或调用方指定。

目标产物是：

- 一个以数据集名字命名的 Python 文件
- 一个 `DatasetNameDataset` 类
- 一个 `DatasetNameDatapipe` 类

例如当数据集名为 `RAE2822` 时，目标产物应为：

- `RAE2822.py`
- `RAE2822Dataset`
- `RAE2822Datapipe`

## 1. 先确认消费契约，再写 datapipe

如果用户额外提供了某个下游训练脚本、推理脚本或调用样例，必须先看它实际消费哪些字段。

重点检查：

- 下游脚本 import 的 datapipe 或 dataset 是谁
- `__getitem__` 最终应该返回 `PyG Data` 还是普通 `dict`
- 训练循环里是否直接使用 `data.pos`、`data.x`、`data.y`、`data.surf`
- 是否依赖 `edge_index`、mask、法向量、边界点等字段

如果存在这样的调用：

```python
data = data.to(device)
out = model(data)
targets = data.y
data.surf
data.pos
```

则新 datapipe 应优先返回 `torch_geometric.data.Data`，并包含对应字段。

如果下游是 `batch["x"]`、`batch["y"]` 这种访问方式，则应优先返回 `dict`。

如果用户没有提供任何下游脚本，则根据 README 的数据组织形式和现有模板，选择最自然的一种返回结构，不要凭空发明新接口。

## 2. 先选最近似的现有 datapipe 作为模板

不要只看 README 生写，必须先选模板。`onescience` 仓库中的 datapipe 文件在这里充当“参考风格来源”，不是硬编码输出路径。

### 2.1 图/点集模式

优先参考：

- `onescience/src/onescience/datapipes/cfd/AirfRANS.py`
- `onescience/src/onescience/datapipes/cfd/ShapeNetCar.py`

适用场景：

- 下游模型使用 `torch_geometric.data.Data`
- 样本是点云、网格点、图节点或可转为节点集合

### 2.2 规则张量模式

优先参考：

- `onescience/src/onescience/datapipes/cfd/deepcfd.py`
- `onescience/src/onescience/datapipes/cfd/cfdbench.py`

适用场景：

- 下游模型直接吃规则网格张量
- `__getitem__` 返回 `dict`

### 2.3 多任务/多配置模式

优先参考：

- `onescience/src/onescience/datapipes/cfd/PDENNEval.py`
- `onescience/src/onescience/datapipes/cfd/cfdbench.py`

适用场景：

- 一个数据集要支持多个 task、split 或 PDE 设置

## 3. 文件命名和类命名是硬规则

给定数据集名 `DatasetName`：

- 文件名必须是 `DatasetName.py`
- 数据集类必须是 `DatasetNameDataset`
- datapipe 类必须是 `DatasetNameDatapipe`

不要擅自改成：

- `dataset_name.py`
- `dataset_dataset.py`
- `load_dataset_name.py`

除非用户明确要求别的命名方式。默认优先跟随当前 `onescience.datapipes` 风格。

## 4. 典型 datapipe 的结构模板

无论输出目录在哪里，生成代码时优先保持下面的结构顺序：

1. 导入
2. 辅助函数
3. `DatasetNameDataset(BaseDataset)`
4. `DatasetNameDatapipe`

### 4.1 必须保留的基础导入

生成的数据集类应继承：

```python
from onescience.datapipes.core import BaseDataset
```

这条继承关系是硬约束。

### 4.2 Dataset 类中优先包含

- `DOMAIN`
- `TASK`
- `DATA_FORMATS`
- `__init__`
- `_init_paths`
- `_load_metadata`
- 可选：`_calculate_normalization`
- 可选：`_load_single_simulation` 或 `_build_full_sample`
- `__len__`
- `__getitem__`

### 4.3 推荐构造函数签名

```python
def __init__(
    self,
    config,
    mode: str = "train",
    coef_norm=None,
):
```

### 4.4 推荐成员字段

- `self.mode`
- `self._provided_coef_norm`
- `self.data_list_names`
- `self.coef_norm`
- `self.dist = DistributedManager()`

## 5. Datapipe 包装类的标准接口

`DatasetNameDatapipe` 对外优先暴露：

- `train_dataloader()`
- `val_dataloader()`
- `test_dataloader()`

推荐约定：

- `train_dataloader()` 返回 `(loader, sampler)`
- `val_dataloader()` 返回 `(loader, sampler)`
- `test_dataloader()` 返回 `loader`

图数据模式下优先使用：

- `torch_geometric.loader.DataLoader`

如果用户给出的目标调用方式不同，可以在不破坏整体风格的前提下调整，但不要移除 `DatasetNameDatapipe` 这个包装层。

## 6. 归一化统计不是默认必选项

不要因为参考模板里算了 `mean/std`，就机械地在新 datapipe 里也算一遍。

只有在下面至少一项成立时，才需要实现归一化统计：

1. 目标模型或训练脚本明确读取这些统计量
2. config 中存在并实际使用 `coef_norm`、`mean_in/std_in`、`mean_out/std_out`
3. 现有目标 example 的前向、loss 或后处理确实依赖归一化后的输入输出

如果训练流程里根本没有使用这些统计量，则不要额外生成：

- `mean_in.npy`
- `std_in.npy`
- `mean_out.npy`
- `std_out.npy`

也不要为了“和旧模板保持一致”而保留无用的 `_calculate_normalization` 逻辑。

如果确实需要归一化，规则是：

1. 只在训练集上计算
2. 验证集和测试集复用训练集统计量
3. 统计量保存路径由用户配置或目标工程配置决定，不要写死到 `onescience` 仓库

### 6.1 当初始 loss 异常高时，要反查是否缺少预处理

如果用户提供了训练日志，或者已有案例显示训练一开始就出现明显异常大的 loss，例如：

- 首轮 loss 远高于同类任务常见量级
- loss 很快爆炸、出现 `nan/inf`
- 还没进入稳定训练阶段就持续数值异常

则要主动判断问题是否来自数据预处理缺失，而不是只盯着模型代码。

优先检查：

1. 输入特征和目标变量的数值范围是否跨度过大
2. 不同物理量是否直接混合在同一个输入中，且量纲差异明显
3. 当前 loss 是否对原始数值尺度非常敏感，例如 `MSE`
4. 参考 example 是否本来依赖归一化或其他预处理，但新 datapipe 没有保留

如果这些检查表明原始尺度会显著放大训练误差，就应该补充必要的预处理，例如：

- 输入标准化
- 输出标准化
- 按字段分别归一化
- 对几何量、条件量、目标量分开处理

不要把“高 loss”直接当成模型结构错误，也不要在没有判断数据尺度前盲目调学习率。

## 7. 从 README 中只提取 datapipe 需要的信息

必须抽取：

- 数据文件名
- 样本组织方式
- 坐标字段
- 条件字段
- 目标字段
- split 说明
- 是否有边界、表面、mask 或几何信息

不要把论文背景、实验结论、方法综述搬进 datapipe。

## 8. 生成前的决策顺序

必须按下面顺序执行：

1. 确认数据集名
2. 确认输出文件的目标位置
3. 判断数据所属领域和返回结构
4. 如果提供了下游脚本，先反推它的数据契约
5. 选择最近似 datapipe 模板
6. 从 README 抽出字段和形状
7. 决定是否真的需要归一化
8. 决定是否需要 `edge_index`
9. 决定是否需要在目标包里补 `__init__.py` 导出

## 9. 生成后的最小自检

生成 datapipe 后，至少检查：

- 文件名是否等于数据集名
- 是否存在 `DatasetNameDataset`
- 是否存在 `DatasetNameDatapipe`
- 是否继承 `BaseDataset`
- `train_dataloader/val_dataloader/test_dataloader` 是否齐全
- `__getitem__` 返回结构是否与 README 或目标调用脚本匹配
- 如果实现了归一化，是否只在 train 上计算
- 如果训练流程不依赖归一化，是否已经去掉无用的 `mean/std` 计算和缓存逻辑
- 如果用户提供了异常训练日志，是否已经检查高 loss 是否由缺少预处理导致
- 如果用户要求包级导入，是否已同步更新目标目录下的 `__init__.py`

## 10. 这份 skill 的作用边界

这份文档只解决“如何生成 OneScience 风格的数据集接口”。

它不负责：

- 把文件强制写入 `onescience` 项目
- 自动生成 `train.py`
- 自动生成 config 文件
- 修改模型结构
- 设计新损失函数
