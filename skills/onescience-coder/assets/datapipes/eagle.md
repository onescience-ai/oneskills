# Datapipe: EagleDatapipe

## 基本信息

- Datapipe 名称：`EagleDatapipe`
- 数据类型：`cfd`
- 主要任务：`mesh-based temporal forecasting`
- 数据组织方式：`split_txt_plus_npz_trajectory_dirs`

## Datapipe 职责

`EagleDatapipe` 负责把 Eagle CFD 时序样本从 `split` 文件指向的仿真目录读入，并把不同节点数、边数、cluster 数的轨迹样本 padding 成统一 batch。

补充说明：

- 原始数据是“split 文本文件 + 每个仿真一个目录”的组织方式。
- datapipe 负责窗口裁剪、三角面转边、可选 one-hot 节点类型、可选归一化、可选 cell/cluster 读取以及 padding collate。
- datapipe 输出的是普通张量字典，不是图对象。

## 输入配置

- `source.data_dir`
  - 仿真样本根目录。
- `source.splits_dir`
  - `train.txt` / `valid.txt` / `test.txt` 所在目录。
- `source.cluster_dir`
  - 当 `n_cluster > 1` 时的 cluster 标签目录。
- `data.window_length_train`
  - 训练窗口长度。
- `data.window_length_val`
  - 验证窗口长度。
- `data.window_length_test`
  - 测试窗口长度。
- `data.n_cluster`
  - `-1/1/10/20/30/40` 之一。
- `data.type_as_onehot`
  - 是否把 `node_type` 转成 one-hot。
- `data.with_cells`
  - 是否保留 `cells`。
- `data.normalized`
  - 是否使用内置的压力/速度统计量做归一化。
- `dataloader.batch_size`
  - DataLoader batch 大小。
- `dataloader.num_workers`
  - DataLoader 工作进程数。

## 数据存储约定

- split 文件：`<splits_dir>/train.txt`、`valid.txt`、`test.txt`
- 每个样本目录：
  - `sim.npz`
  - `triangles.npy`
- cluster 文件：`<cluster_dir>/<relative_case_path>/constrained_kmeans_<n_cluster>.npy`
- 元数据来源：`sim.npz` 内的 `pointcloud/VX/VY/PS/PG/mask`

额外约定：

- `valid` split 文件名必须是 `valid.txt`，不是 `val.txt`。
- `sim.npz` 默认按最多 `990` 帧组织，训练时会随机抽窗，验证/测试会从固定起点取窗。

## 样本构造方式

- 输入样本：
  - `mesh_pos`
  - `edges`
  - `velocity`
  - `pressure`
  - `node_type`
- 输出样本：
  - datapipe 本身不拆输入/标签，通常由训练脚本再按时间维切分。
- 附加返回项：
  - `cells`
  - `cluster`
  - `mask`
  - `cluster_mask`

具体说明：

- `edges` 由三角面 `faces -> unique undirected edges -> 双向边` 转换得到。
- `velocity` 由 `VX/VY` 组成，`pressure` 由 `PS/PG` 组成。
- `collate(...)` 会按 batch 内最大节点数、边数、cluster 数做 padding，并生成 `mask` 和 `cluster_mask`。

## DataLoader 约定

- 训练阶段：`train_dataloader()` 返回 `(DataLoader, sampler)`
- 验证阶段：`val_dataloader()` 返回 `(DataLoader, sampler)`
- 测试阶段：`test_dataloader(batch_size=None)` 返回 `(DataLoader, sampler)`

## 适合优先使用的场景

- 时序 CFD 网格预测，且需要按时间窗采样。
- 批内样本节点数不一致，需要统一 padding。
- 训练脚本后续还要接 pooling / clustering 相关逻辑。

## 风险点

- `__getitem__` 出错时会返回空字典 `{}`，`collate(...)` 会过滤空样本；如果坏样本过多，可能出现空 batch。
- `n_cluster` 只能取源码中允许的离散值。
- 归一化统计量是硬编码常数，不来自数据目录里的实时统计文件。
- datapipe 只负责组织整段窗口，具体“输入多少步、预测多少步”仍由模型或训练脚本决定。

## 源码锚点

- `./onescience/src/onescience/datapipes/cfd/eagle.py`
- `./onescience/src/onescience/datapipes/core/base_dataset.py`
