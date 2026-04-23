# Datapipe: PDEBench Datapipe Family

## 基本信息

- 文档覆盖范围：
  - `PDEBenchFNODatapipe`
  - `PDEBenchDeepONetDatapipe`
  - `PDEBenchMPNNDatapipe`
  - `PDEBenchUNetDatapipe`
  - `PDEBenchUNODatapipe`
  - `PDEBenchPINODatapipe`
- 数据类型：`cfd`
- 主要任务：`operator learning / pde surrogate learning`
- 数据组织方式：`single_or_multi_file_hdf5`

## Datapipe 职责

`PDENNEval.py` 不是一个单一 datapipe，而是一组 PDEBench 风格数据接口。它们共用同一份 HDF5/H5 数据源读取逻辑，但会根据目标模型族把样本整理成不同协议。

补充说明：

- 这组 datapipe 统一负责空间/时间降采样、train/val 划分、单文件或多文件读取以及模型特定的样本封装。
- 目标模型不同，返回协议也不同，不能只看文件名就假设接口一致。
- 某些分支仍保留了占位注释或未完全实现的路径，接入前应先读对应类。

## 输入配置

### 共同配置

- `datapipe.source.data_dir`
  - HDF5 数据根目录。
- `datapipe.source.file_name`
  - 主 HDF5/H5 文件名。
- `datapipe.data.single_file`
  - `true` 表示一次性读单文件，`false` 表示按 seed/group 懒加载。
- `datapipe.data.initial_step`
  - 作为输入历史的时间步数。
- `datapipe.data.reduced_resolution`
  - 空间降采样倍率。
- `datapipe.data.reduced_resolution_t`
  - 时间降采样倍率。
- `datapipe.data.reduced_batch`
  - 样本维降采样倍率。
- `datapipe.data.test_ratio`
  - train/val 划分比例。
- `datapipe.dataloader.batch_size`
  - batch 大小。
- `datapipe.dataloader.num_workers`
  - 工作进程数。
- `datapipe.dataloader.pin_memory`
  - `pin_memory`。

### 模型族特有配置

- `PDEBenchMPNNDatapipe`
  - 还需要 `pde_name`、`variables`、`temporal_domain`、`spatial_domain`、`resolution`、`resolution_t`、`neighbors`、`time_window`
- `PDEBenchPINODatapipe`
  - 还需要 `reduced_resolution_pde`、`reduced_resolution_pde_t`、`if_grid_norm`

## 数据存储约定

- 主数据路径：`<data_dir>/<file_name>`
- 常见数据键：
  - `tensor`
  - `density`
  - `pressure`
  - `Vx/Vy/Vz`
  - `nu`
  - `x-coordinate/y-coordinate/z-coordinate/t-coordinate`
- 多文件模式下的典型组结构：
  - `<seed>/data`
  - `<seed>/grid/x`
  - `<seed>/grid/y`
  - `<seed>/grid/z`
  - `<seed>/grid/t`
  - 可选 `<seed>/global_maximums`

## 样本构造方式

### `PDEBenchFNODatapipe`

- 样本协议：`(history, target, grid)`
- `history = data[..., :initial_step, :]`
- `target = full trajectory / full field tensor`
- `grid` 是坐标网格或某些分支下的 `global_maximums`

### `PDEBenchDeepONetDatapipe`

- 样本协议：`(history, target, grid)`
- 与 FNO 类似，但训练脚本通常还会使用 datapipe 暴露出的 `dx`、`dt`

### `PDEBenchMPNNDatapipe`

- 样本协议：`(flattened_datapoints, coordinates, variables)`
- datapipe 额外暴露：
  - `pde`
  - `graph_creator`
- 真正的图构造通常在训练阶段通过 `graph_creator.create_data/create_graph(...)` 完成

### `PDEBenchUNetDatapipe`

- 样本协议：`(history, target)`
- 不额外返回 `grid`

### `PDEBenchUNODatapipe`

- 样本协议：`(history, target, grid)`

### `PDEBenchPINODatapipe`

- 监督样本协议：`(history, target, grid)`
- 额外还有 `pde_dataloader()`，供 PDE loss 使用
- datapipe 会暴露：
  - `dx`
  - `dt`

## DataLoader 约定

- `PDEBenchFNODatapipe`
  - `train_dataloader()` / `val_dataloader()` 返回 `(DataLoader, sampler)`
- `PDEBenchDeepONetDatapipe`
  - `train_dataloader()` / `val_dataloader()` 返回 `(DataLoader, sampler)`
- `PDEBenchMPNNDatapipe`
  - `train_dataloader()` / `val_dataloader()` 返回 `(DataLoader, sampler)`
- `PDEBenchUNetDatapipe`
  - `train_dataloader()` / `val_dataloader()` 返回 `(DataLoader, sampler)`
- `PDEBenchUNODatapipe`
  - `train_dataloader()` / `val_dataloader()` 返回 `(DataLoader, sampler)`
- `PDEBenchPINODatapipe`
  - `train_dataloader()` / `pde_dataloader()` / `val_dataloader()` 返回 `(DataLoader, sampler)`

补充说明：

- 这组接口当前没有统一的 `test_dataloader()`。
- `UNet/UNO/PINO` 的验证集 loader 里很多分支使用了 `drop_last=True`，评估时要注意。

## 适合优先使用的场景

- 目标就是复用 PDEBench 路线的 FNO / DeepONet / UNO / PINO / MPNN / UNet 示例。
- 数据本身已经接近 PDEBench HDF5 组织格式。
- 希望把“数据协议差异”压在 datapipe 层，而不是把所有模型都硬接成同一种输入。

## 风险点

- 同一源码文件里的不同 datapipe 返回协议不同，写训练脚本前必须先确认。
- 多文件模式、3D 分支、`tensor`/`density` 双格式分支并不完全等价，有些路径仍需要源码复核。
- `PDEBenchMPNNDataset` 的多文件路径目前在源码里有占位实现，不适合直接假设完整可用。
- `PDEBenchPINODataset` 某些非标准分支仍有简化或占位逻辑，若要做正式基准请先补齐。

## 源码锚点

- `./onescience/src/onescience/datapipes/cfd/PDENNEval.py`
- `./onescience/examples/cfd/PDEBench/`
- `./onescience/src/onescience/datapipes/core/base_dataset.py`
