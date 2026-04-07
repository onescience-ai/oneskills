# Datapipe: DeepMind_CylinderFlowDatapipe

## 基本信息

- Datapipe 名：`DeepMind_CylinderFlowDatapipe`
- 数据类型：`cfd`
- 主要任务：`transient graph-based cylinder-flow rollout regression`
- 数据组织方式：`meta_json_plus_train_valid_test_tfrecord`

## Datapipe 职责

`DeepMind_CylinderFlowDatapipe` 负责把 DeepMind 涡激数据集的 `TFRecord` 时序样本读成 DGL 图，构造归一化统计量，并封装成现有 `Vortex_shedding_mgn` 训练流程可直接消费的 `GraphDataLoader`。

补充说明：

- 原始数据目录至少要求一个 `meta.json`，以及 `train.tfrecord`、`valid.tfrecord`、`test.tfrecord`
- datapipe 负责 TFRecord 解析、边特征构造、节点特征与目标构造、训练噪声注入、统计量保存和 DGL 图装配
- datapipe 同时实现了 `DeepMind_CylinderFlowDataset` 与 `DeepMind_CylinderFlowDatapipe` 两层接口

## 输入配置

- `source.data_dir`
  - 数据根目录，内部需要有 `meta.json` 和各 split 的 `.tfrecord`
- `source.stats_dir`
  - 归一化统计量保存目录，内部会写 `edge_stats.json` 和 `node_stats.json`
- `data.train_samples`
  - 训练轨迹数量
- `data.train_steps`
  - 每条训练轨迹保留的时间步数
- `data.val_samples`
  - 验证轨迹数量
- `data.val_steps`
  - 每条验证轨迹保留的时间步数
- `data.test_samples`
  - 测试轨迹数量
- `data.test_steps`
  - 每条测试轨迹保留的时间步数
- `data.noise_std`
  - 训练阶段节点速度噪声标准差
- `dataloader.batch_size`
  - DGL `GraphDataLoader` 批大小
- `dataloader.num_workers`
  - DGL `GraphDataLoader` 工作进程数

## 数据存储约定

- 主数据路径：`<data_dir>/meta.json`、`<data_dir>/train.tfrecord`、`<data_dir>/valid.tfrecord`、`<data_dir>/test.tfrecord`
- 统计量路径：`<stats_dir>/edge_stats.json` 与 `<stats_dir>/node_stats.json`
- 元数据来源：`meta.json` 中的 `field_names` / `features` 描述，以及 TFRecord 中的轨迹字段

额外约定：

- `val` 模式在实现内部会映射到 `valid.tfrecord`
- datapipe 依赖 `tensorflow` 解析 TFRecord，同时依赖 `dgl`
- 非训练 split 要求事先已有训练阶段保存的统计量

## 样本构造方式

- 输入样本：训练阶段返回 `DGLGraph`，其中 `graph.ndata["x"] -> [NumNodes, 6]`，通常是 `2 维速度 + 4 维节点类型 one-hot`
- 输出样本：`graph.ndata["y"] -> [NumNodes, 3]`，通常是 `2 维目标速度项 + 1 维压力`
- 附加返回项：
  - 训练阶段：只返回 `graph`
  - 验证 / 测试阶段：返回 `(graph, cells, mask)`

具体说明：

- 一个 dataset item 对应“某条轨迹的某个时间步”，而不是整个轨迹
- `gidx = idx // (num_steps - 1)`，`tidx = idx % (num_steps - 1)`，因此单个轨迹会被拆成多个时序图样本
- 边特征默认来自节点间相对位移和距离范数
- 训练阶段会对部分节点速度特征加噪声，以增强鲁棒性

## DataLoader 约定

- 训练阶段：`train_dataloader()` 返回 `(GraphDataLoader, sampler)`；分布式时使用 `DistributedSampler(shuffle=True)`
- 验证阶段：`val_dataloader()` 返回 `(GraphDataLoader, sampler)`；分布式时使用 `DistributedSampler(shuffle=False)`
- 测试阶段：`test_dataloader()` 只返回 `GraphDataLoader`

补充说明：

- 训练 / 验证 / 测试的 batch 解包协议并不完全一致，训练脚本和推理脚本都依赖这个差异
- 这套 datapipe 是当前 `Vortex_shedding_mgn` 路线的重要约束，不是一个通用 DGL 图模板

## 适合优先使用的场景

- 希望沿用 `./onescience/examples/cfd/Vortex_shedding_mgn/` 中的 DGL + MeshGraphNet 训练流程
- 数据本身是时序轨迹数据，且每个样本天然有图拓扑和节点类型
- 任务更接近“图上的下一时刻状态预测 / rollout”，而不是静态单样本几何回归

## 风险点

- 依赖 `tensorflow` 与 `dgl`；如果环境没有这两者，现有 datapipe 路线无法直接工作
- 现有样本协议默认是时序 rollout 图，不适合直接套用到单个稳态 VTK 结果文件数据集
- 验证 / 测试阶段会返回 `(graph, cells, mask)`，如果新 datapipe 不保留这个协议，现有 MGN 推理脚本需要同步改
- 节点输入、目标和边特征维度都被当前示例配置显式假设；新数据集字段一旦变化，datapipe 与模型配置都要一起改

## 源码锚点

- `./onescience/src/onescience/datapipes/cfd/deepmind_cylinderflow.py`
- `./onescience/examples/cfd/Vortex_shedding_mgn/train.py`
- `./onescience/examples/cfd/Vortex_shedding_mgn/conf/mgn_cylinderflow.yaml`
