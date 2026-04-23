# Datapipe: DeepMindLagrangianDatapipe

## 基本信息

- Datapipe 名称：`DeepMindLagrangianDatapipe`
- 数据类型：`cfd`
- 主要任务：`particle-based rollout forecasting`
- 数据组织方式：`metadata_json_plus_split_tfrecords`

## Datapipe 职责

`DeepMindLagrangianDatapipe` 负责把 DeepMind 风格的粒子轨迹 `TFRecord` 数据解析成 DGL 图样本，并按训练/验证/测试 split 构建 `GraphDataLoader`。

补充说明：

- 原始数据不是普通 `npy/h5`，而是 `metadata.json + <split>.tfrecord`。
- datapipe 负责 TFRecord 解析、历史速度窗口拼接、边界特征生成、图边重建和训练期随机游走噪声注入。
- 该 datapipe 直接输出 DGLGraph，而不是 PyG 图或普通张量字典。

## 输入配置

- `datapipe.source.data_dir`
  - 数据根目录，应包含 `metadata.json` 和对应 split 的 `.tfrecord`。
- `data.num_history`
  - 历史速度帧数，决定输入里保留多少个速度历史。
- `data.noise_std`
  - 训练期随机游走噪声强度。
- `data.num_node_types`
  - 粒子类型 one-hot 的类别数。
- `data.train.split` / `data.valid.split` / `data.test.split`
  - 分别对应要读取的 TFRecord split 名称。
- `data.train.num_sequences` / `data.valid.num_sequences` / `data.test.num_sequences`
  - 每个 split 最多读取多少条轨迹序列。
- `data.train.num_steps` / `data.valid.num_steps` / `data.test.num_steps`
  - 每条轨迹保留多少个时间步，未设置时使用 `metadata.json` 的全长。
- `datapipe.dataloader.train.*`
  - 训练 GraphDataLoader 参数。
- `datapipe.dataloader.valid.*`
  - 验证 GraphDataLoader 参数。
- `datapipe.dataloader.test.*`
  - 测试 GraphDataLoader 参数。

## 数据存储约定

- 主数据路径：`<data_dir>/metadata.json` 与 `<data_dir>/<split>.tfrecord`
- 统计量来源：`metadata.json` 中的 `vel_mean/vel_std/acc_mean/acc_std`
- 元数据来源：`metadata.json` 中的 `sequence_length`、`dt`、`default_connectivity_radius`、`bounds`、`dim`

额外约定：

- 依赖 `DGL` 和 `tensorflow.compat.v1`，缺任何一个都无法初始化。
- TFRecord 里的 `particle_type` 也是序列化 bytes，需要走自定义 parse 逻辑。

## 样本构造方式

- 输入样本：DGL `Graph`
  - `ndata["x"]`：`[position_t, velocity_history, boundary_features, node_type_onehot]`
- 输出样本：`ndata["y"]`
  - `[next_position, next_velocity, next_acceleration]`
- 附加返回项：
  - `ndata["pos"]`
  - `ndata["mask"]`
  - `ndata["t"]`
  - `edata["x"]`（由 `graph_update(...)` 写入的边特征）

具体说明：

- 一个 dataset 样本对应“某条轨迹上的一个时间窗口”，不是整条轨迹。
- 训练模式下会只对动态粒子注入随机游走噪声，运动学粒子由 `mask` 排除。
- 边是按当前位置重新用半径图构造的，因此 rollout 时可复用 `graph_update(...)` 逻辑更新邻接关系。

## DataLoader 约定

- 训练阶段：`train_dataloader()` 返回 `GraphDataLoader`
- 验证阶段：`val_dataloader()` 返回 `GraphDataLoader`
- 测试阶段：`test_dataloader()` 返回 `GraphDataLoader`

补充说明：

- 这里不返回 `(loader, sampler)` 二元组，而是直接返回 loader。
- 分布式训练通过 `GraphDataLoader(..., use_ddp=self.distributed)` 控制。

## 适合优先使用的场景

- MeshGraphNet / GNS 一类粒子或拉格朗日图网络任务。
- 需要长时间 rollout、动态重建图边、并在训练期加入噪声扰动。
- 新数据已经是 TFRecord 轨迹格式，或者可以比较自然地桥接到该格式。

## 风险点

- 图边通过 `torch.cdist` 全对全距离构建，粒子数很大时开销明显。
- 训练、验证、测试 split 的配置分散在 `cfg.data.*` 和 `cfg.datapipe.*` 两处，写生成脚本时容易漏字段。
- `valid` split 在代码里使用 `mode='valid'`，需要保证文件名和配置名一致。
- 若新数据不是粒子轨迹而是静态网格，通常不应直接复用这个 datapipe。

## 源码锚点

- `./onescience/src/onescience/datapipes/cfd/deepmind_lagrangian.py`
- `./onescience/examples/cfd/Vortex_shedding_mgn/`
- `./onescience/src/onescience/datapipes/core/base_dataset.py`
