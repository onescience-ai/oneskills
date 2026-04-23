# Datapipe: BENODatapipe

## 基本信息

- Datapipe 名称：`BENODatapipe`
- 数据类型：`cfd`
- 主要任务：`elliptic pde surrogate on heterogeneous graphs`
- 数据组织方式：`paired_npy_arrays_with_optional_pt_cache`

## Datapipe 职责

`BENODatapipe` 负责把 `RHS/SOL/BC` 三组 `npy` 文件组织成可直接供 BENO 系列模型消费的 `PyG HeteroData` 样本，并在第一次预处理后缓存成 `.pt` 文件以减少重复构图开销。

补充说明：

- 原始数据按固定前缀文件名组织，不是按 case 子目录逐个存放。
- datapipe 负责读取原始数组、边界值归一化、输入场平滑与梯度构造、网格采样、异构图构建以及缓存。
- datapipe 同时实现了 `BENODataset` 和 `BENODatapipe` 两层接口。

## 输入配置

- `source.data_dir`
  - BENO 数据根目录。
- `source.file_prefix`
  - 文件名前缀，对应 `RHS_<prefix>_all.npy`、`SOL_<prefix>_all.npy`、`BC_<prefix>_all.npy`。
- `source.cache_dir`
  - 预处理后缓存 `.pt` 文件的目录。
- `data.ntrain`
  - 训练样本数。
- `data.ntest`
  - 测试样本数。
- `data.resolution`
  - 方形规则底网格分辨率，代码里默认按 `resolution x resolution` 解释。
- `data.ns`
  - 图构建时的邻域/采样相关参数，传给 `MeshGenerator.ball_connectivity(...)`。
- `dataloader.batch_size`
  - PyG `GeoDataLoader` 批大小。
- `dataloader.num_workers`
  - DataLoader 工作进程数。
- `dataloader.pin_memory`
  - DataLoader 的 `pin_memory`。

## 数据存储约定

- 主数据路径：`<data_dir>/RHS_<file_prefix>_all.npy`、`<data_dir>/SOL_<file_prefix>_all.npy`、`<data_dir>/BC_<file_prefix>_all.npy`
- 缓存路径：`<cache_dir>/cached_<file_prefix>_<mode>_<count>.pt`
- 元数据来源：三个 `npy` 文件本身的 shape 和固定字段布局

额外约定：

- `BC_*.npy` 当前实现默认边界点数是 `128`，因为内部直接 `reshape(-1, 128, 1)`。
- `RHS_*.npy` 默认第 0/1 列是坐标，第 2 列是输入场，第 3 列是 `cell_state`。
- `SOL_*.npy` 当前实现默认只取第 0 列作为监督目标。

## 样本构造方式

- 输入样本：`HeteroData`
  - `data["G1"].x -> [NumNodes, 10]`
  - `data["G2"].x -> [NumNodes, 10]`
- 输出样本：`data["G1+2"].y`
  - 采样节点上的标量目标。
- 附加返回项：
  - `data["G1"].boundary`
  - `data["G2"].boundary`
  - `data["G1"].edge_index`
  - `data["G1"].edge_features`
  - `data["G1"].sample_idx`
  - `data["G1"].cell_state`

具体说明：

- `G1.x` 由坐标、原场、平滑场、两个方向梯度以及到边界的距离特征拼接而成。
- `G2.x` 复用了相同图结构，但把中间 4 个场特征置零，保留边界条件分支。
- `boundary` 里保留边界坐标和边界值，其中 `G1` 分支会把边界值清零。
- 图边通过 `MeshGenerator.ball_connectivity(...)` 构建，边特征来自 `MeshGenerator.attributes(...)`。

## DataLoader 约定

- 训练阶段：`train_dataloader()` 返回 `(GeoDataLoader, sampler)`，无分布式时 `sampler=None`
- 验证阶段：`none`
- 测试阶段：`test_dataloader()` 返回 `(GeoDataLoader, sampler)`

## 适合优先使用的场景

- 目标模型本身就是 BENO 或兼容其异构图输入协议的模型。
- 新数据仍然是规则底网格上采样得到的椭圆 PDE 问题。
- 希望保留预处理缓存，避免每次重新构图。

## 风险点

- 当前实现硬编码了部分 shape 约束，尤其是 `128` 个边界点和方形底网格。
- 只有 `train/test` 两个 split，没有单独的 `val_dataloader()`。
- 训练集目标会经过 `u_normalizer.encode(...)`，测试集目标保持原尺度，复用训练脚本时要注意解码或损失定义。
- 预处理阶段会执行较重的平滑、梯度和图构建，大数据集首次初始化时间较长。

## 源码锚点

- `./onescience/src/onescience/datapipes/cfd/beno.py`
- `./onescience/src/onescience/utils/beno/utilities.py`
- `./onescience/src/onescience/datapipes/core/base_dataset.py`
