# Datapipe: DeepCFDDatapipe

## 基本信息

- Datapipe 名称：`DeepCFDDatapipe`
- 数据类型：`cfd`
- 主要任务：`steady 2d flow-field surrogate regression`
- 数据组织方式：`paired_pickle_arrays_loaded_in_memory`

## Datapipe 职责

`DeepCFDDatapipe` 负责把 `dataX.pkl` 和 `dataY.pkl` 中按样本堆叠的规则网格数据读入内存，完成随机划分，并组织成可直接供 DeepCFD 训练脚本消费的字典样本与 `DataLoader`。

补充说明：

- 原始数据目录默认只要求一个根目录，里面至少包含一对输入/输出 pickle 文件
- datapipe 负责读文件、打乱索引、按 `split_ratio` 划分训练与测试，并计算输出通道的 loss 权重
- datapipe 同时实现了 `DeepCFDDataset` 与 `DeepCFDDatapipe` 两层接口

## 输入配置

- `source.data_dir`
  - DeepCFD 数据根目录
- `source.data_x_name`
  - 输入文件名，默认示例为 `dataX.pkl`
- `source.data_y_name`
  - 输出文件名，默认示例为 `dataY.pkl`
- `data.split_ratio`
  - 训练集比例；剩余样本同时被当前实现当作 `test/val`
- `data.seed`
  - 划分前随机打乱的种子
- `dataloader.batch_size`
  - 批大小
- `dataloader.num_workers`
  - `DataLoader` 工作进程数

## 数据存储约定

- 主数据路径：`<data_dir>/<data_x_name>` 与 `<data_dir>/<data_y_name>`
- 统计量路径：`none`
- 元数据来源：pickle 数组自身的 shape 与样本顺序

额外约定：

- `dataX.pkl` 与 `dataY.pkl` 的样本数必须完全一致
- 当前实现默认 `dataY.pkl` 的输出通道数为 3，loss 权重计算里写死按 3 个通道展开

## 样本构造方式

- 输入样本：`sample["x"] -> (ChannelsIn, Height, Width)`，DeepCFD README 示例中为 3 通道输入
- 输出样本：`sample["y"] -> (ChannelsOut, Height, Width)`，DeepCFD README 示例中为 3 通道输出 `[Ux, Uy, p]`
- 附加返回项：`none`

具体说明：

- 单个 dataset 样本对应一个稳态流场样本，而不是时间窗序列
- `__getitem__` 返回普通字典：`{"x": tensor, "y": tensor}`
- 训练脚本按图像/规则网格方式直接消费 batch，不经过 `PyG Data`
- datapipe 还会从完整 `y` 张量统计 RMS 风格的 `channel_weights`，供训练脚本构造加权损失

## DataLoader 约定

- 训练阶段：`train_dataloader()` 返回 `(DataLoader, sampler)`；非分布式时 `shuffle=True`
- 验证阶段：`none`，当前 datapipe 没有单独 `val_dataloader()`
- 测试阶段：`test_dataloader()` 返回 `(DataLoader, sampler)`；非分布式时 `shuffle=False`

补充说明：

- `DeepCFDDataset(mode="val")` 虽然可构造，但和 `mode="test"` 使用的是同一段剩余划分
- 示例工程通常把 `test_dataloader()` 既当评估集也当测试集

## 适合优先使用的场景

- 直接复用 `./onescience/examples/cfd/DeepCFD/` 中的 `UNet` / `UNetEx` 训练流程
- 新数据集本身就是规则网格上的 `X -> Y` 配对回归任务
- 输入输出更接近图像张量，而不是点云图或 `(pos, fx, y)` 三元组

## 风险点

- 当前实现没有真正独立的 `val` 划分；`val` 与 `test` 会落在同一批样本上
- 全量 `pickle` 会一次性读入内存，数据规模增大时容易出现内存压力
- `channel_weights` 的实现默认输出通道数为 3；如果新任务目标变量数变化，需要同步改动 datapipe 或训练脚本
- 它不能直接对接 `CFD_Benchmark` 这类要求 `(pos, fx, y)` 批格式的算子训练流程；若要比较 `FNO/U_FNO/U_NO/U_Net`，通常需要额外桥接一个 DeepCFD-to-operator adapter datapipe

## 源码锚点

- `./onescience/src/onescience/datapipes/cfd/deepcfd.py`
- `./onescience/examples/cfd/DeepCFD/train.py`
- `./onescience/examples/cfd/DeepCFD/conf/deepcfd.yaml`
