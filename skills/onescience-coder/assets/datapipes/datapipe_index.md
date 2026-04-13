# Datapipe Index

## 目标

本文档用于登记当前已经补充到 `oneskills` 的数据处理与 `DataLoader` 知识。
它和 `models/`、`contracts/` 的分工不同：

- `models/`
  - 解决整模型结构、输入输出组织和组件链路
- `contracts/`
  - 解决模块如何初始化、关键参数与 `shape` 约束
- `datapipes/`
  - 解决数据如何读取、如何构造样本，以及 train / val / test 的 `DataLoader` 约定

## 建议使用方式

推荐在以下情况优先读取本文档：

1. 用户要求设计数据读取流程或 `DataLoader`
2. 任务涉及样本切分、变量选择、网格组织或 batch 结构
3. 任务需要确认 datapipe 最终输出给模型的样本格式

推荐检索顺序：

1. 若任务主要是模型结构改造，先看 `models/` 和 `contracts/`
2. 若任务已经涉及数据组织，再看本文档
3. 先读具体 datapipe 卡片
4. 卡片信息不足时，再回到 `./onescience/` 中对应 datapipe 源码

## 当前已登记 Datapipe

| Datapipe | 数据类型 | 数据组织方式 | 样本粒度 | 主要配置 | 状态 | 文档 |
|---|---|---|---|---|---|---|
| ERA5Datapipe | 全球大气再分析数据 | 每年一个 HDF5 文件 | 连续时间窗样本 | `dataset_dir`, `used_years`, `used_variables`, `input_steps`, `output_steps`, `normalize` | `stable` | `./datapipes/era5.md` |
| CMEMSDatapipe | 全球海洋再分析 / 预报数据 | 按年份目录、按时刻 HDF5 文件 | 连续时间窗样本 | `data_dir`, `channels`, `train_ratio/val_ratio/test_ratio`, `input_steps`, `output_steps` | `stable` | `./datapipes/cmems.md` |
| AirfRANSDatapipe | 二维翼型 CFD 数据 | 每个工况一个目录，含 `manifest.json`、`.vtu` 与 `.vtp` | 单仿真图样本 | `source.data_dir`, `data.splits.*`, `data.sampling.*`, `data.subsampling`, `model_hparams.build_graph` | `stable` | `./datapipes/airfrans.md` |
| ShapeNetCarDatapipe | 三维车体表面 CFD 数据 | `param*/sample/` 目录下成对 `quadpress_smpl.vtk` / `hexvelo_smpl.vtk`，可选预处理缓存 | 单仿真图样本 | `source.data_dir`, `source.preprocessed_save_dir`, `data.splits.fold_id`, `model_hparams.cfd_mesh/r/max_neighbors` | `stable` | `./datapipes/shapenetcar.md` |
| DeepCFDDatapipe | 规则网格稳态流场配对数据 | 同目录下成对 `dataX.pkl` / `dataY.pkl` | 单样本图像张量 | `source.data_dir`, `source.data_x_name`, `source.data_y_name`, `data.split_ratio`, `data.seed`, `batch_size` | `stable` | `./datapipes/deepcfd.md` |
| DeepMind_CylinderFlowDatapipe | 非结构网格圆柱绕流时序图数据 | `meta.json + train/valid/test.tfrecord` | 单轨迹-单时间步 DGL 图样本 | `source.data_dir`, `source.stats_dir`, `data.train_steps/val_steps/test_steps`, `noise_std`, `batch_size` | `stable` | `./datapipes/deepmind_cylinderflow.md` |

## 未登记 Datapipe 的处理方式

如果用户提供的是 `oneskills` 尚未登记的新数据集，不要跳过数据层设计。推荐做法：

1. 先读取 `./task/new_dataset_workflow.md`
2. 根据数据组织方式，从当前已登记 datapipe 中选择最接近的参考模板
3. 先把新数据集映射成目标模型真正需要的样本格式，再决定是否需要新建 datapipe
4. 若任务还要求复用某个已有模型或 example，再继续读取对应模型卡和示例工程

若新数据集以 `vtk / vtu / vtp` 结果文件为主，优先比较：

1. `./datapipes/airfrans.md`
2. `./datapipes/shapenetcar.md`
3. 目标模型对应的数据协议是否更接近 `PyG Data` 还是 `DGLGraph`

## 跨流程复用时的兼容性检查

如果任务要求复用一个已有 datapipe 去对接另一套模型或训练流程，优先检查：

- datapipe 的样本返回格式是否兼容
- `DataLoader` 的 batch 结构是否兼容
- 训练脚本的 batch 解包方式是否兼容
- 模型要求的输入输出协议是否兼容
- 是否需要新增最小桥接层，例如 adapter datapipe

第一轮规格里应先把这些兼容性判断写清楚，再进入代码生成。

## Datapipe 层的核心问题

优先用 datapipe 卡片解决下面这些问题：

- 数据按什么粒度存储
- 样本如何切分
- 变量如何映射到通道索引
- datapipe 最终返回给模型的张量形状是什么
- 是否包含额外几何量、边信息或位置编码
- `DataLoader` 在 train / val / test 阶段分别怎样组织

## 新增 Datapipe 文档的维护建议

新增 datapipe 时，建议至少完成以下内容：

1. 复制 `./datapipes/TEMPLATE.md`
2. 填写数据目录结构、输入配置、样本构造方式、风险点和源码锚点
3. 在本文档中登记
4. 检查 `README.md`、`task/SKILL.md`、`DEVELOPER_MANUAL.md` 是否需要同步更新
