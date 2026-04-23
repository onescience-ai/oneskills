# Datapipe: DrivAerML_FigConvUnetDatapipe

## 基本信息

- Datapipe 名称：`DrivAerML_FigConvUnetDatapipe`
- 数据类型：`cfd`
- 主要任务：`car-surface pressure and shear-stress regression`
- 数据组织方式：`partitioned_dgl_bin_files_plus_global_stats_json`

## Datapipe 职责

`DrivAerML_FigConvUnetDatapipe` 负责读取 DrivAerML 的分片二进制图文件，把每个样本中的多分片节点拼接后再按 `num_points` 采样，最终输出适合 FigConvUNet 训练脚本消费的普通字典样本。

补充说明：

- 原始样本是 `dgl.load_graphs(...)` 读取出来的一组分片图，而不是单一整图。
- datapipe 负责读取归一化统计量，但当前 `__getitem__` 不会直接应用归一化，只暴露 `encode/decode` 方法。
- datapipe 输出的是点级张量字典，不直接返回 DGL 图对象。

## 输入配置

- `data.source.data_dir`
  - 数据根目录。
- `data.source.stats_filename`
  - 统计量文件名，默认 `global_stats.json`。
- `data.num_points`
  - 每个样本最终保留的点数；`0` 表示保留全部点。
- `train.batch_size`
  - 训练集 batch 大小。
- `train.dataloader.num_workers`
  - 训练 DataLoader 工作进程数。
- `train.dataloader.pin_memory`
  - 训练 DataLoader 的 `pin_memory`。
- `eval.batch_size`
  - 验证/测试集 batch 大小。
- `eval.dataloader.num_workers`
  - 验证/测试 DataLoader 工作进程数。
- `eval.dataloader.pin_memory`
  - 验证/测试 DataLoader 的 `pin_memory`。

## 数据存储约定

- 主数据路径：
  - `partitions/`
  - `validation_partitions/`
  - `test_partitions/`
- 统计量路径：`<data_dir>/<stats_filename>`
- 元数据来源：`.bin` 图文件里的 `ndata` 字段，以及 `global_stats.json`

额外约定：

- `global_stats.json` 必须包含 `mean` 和 `std_dev` 两组字典。
- 每个 `.bin` 文件可包含多个图分片，datapipe 会先拼接再采样。

## 样本构造方式

- 输入样本：`sample["coordinates"] -> [NumPoints, 3]` 或对应维度坐标
- 输出样本：
  - `sample["pressure"]`
  - `sample["shear_stress"]`
- 附加返回项：
  - `sample["design"]`
  - `sample["indices"]`

具体说明：

- 若 `num_points > 0` 且原始点数不足，会重复采样补齐到固定点数。
- `design` 由文件名 `graph_partitions_<design>.bin` 推断。
- 当前返回的是普通字典，适合直接喂给 FigConv 类模型或自定义点云回归模型。

## DataLoader 约定

- 训练阶段：`train_dataloader()` 返回 `DataLoader`，默认 `shuffle=True`
- 验证阶段：`val_dataloader()` 返回 `DataLoader`
- 测试阶段：`test_dataloader()` 返回 `DataLoader`

补充说明：

- 若开启分布式，内部会自动创建 `DistributedSampler`。
- train 使用 `config.train.*`，val/test 使用 `config.eval.*`，配置位置和很多其它 datapipe 不同。

## 适合优先使用的场景

- 目标是 FigConvUNet 或其它直接消费点采样表面张量的模型。
- 数据天然是车体表面压力/切应力回归，不需要额外图边。
- 想保留“多分片图文件 -> 统一点采样张量”的数据协议。

## 风险点

- 当前实现虽然依赖 DGL 读取 `.bin`，但最终不输出图结构，换成图模型时通常还要再桥接一层。
- 归一化统计量不会自动应用到 `__getitem__` 返回值，训练脚本需要自己决定是否使用 `encode/decode`。
- 该 datapipe 目前没有在 `cfd/__init__.py` 里统一 re-export，很多自动发现逻辑可能需要直接指向源码类名。
- `num_points=0` 时会保留全部点，不同样本点数可能不同，批处理前要确认模型是否支持。

## 源码锚点

- `./onescience/src/onescience/datapipes/cfd/DrivAerML_FigConvUnet.py`
- `./onescience/src/onescience/datapipes/core/base_dataset.py`
