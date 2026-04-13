# Datapipe: AirfRANSDatapipe

## 基本信息

- Datapipe 名：`AirfRANSDatapipe`
- 数据类型：`cfd`
- 主要任务：`2d airfoil flow-field regression`
- 数据组织方式：`per_simulation_dir_with_vtu_vtp_and_manifest`

## Datapipe 职责

`AirfRANSDatapipe` 负责把按单个翼型工况目录存储的 AirfRANS 非结构网格数据组织成 PyTorch Geometric 可直接消费的图样本，并封装成训练、验证、测试阶段可用的 `DataLoader`。

补充说明：

- 原始数据以“根目录下一个 `manifest.json`，每个仿真样本一个子目录”的方式组织
- datapipe 负责样本划分、VTK 读取、可选裁剪、可选单元/边界采样、归一化、训练/验证阶段点级下采样、图构建和 `DataLoader` 组装
- datapipe 同时实现了 `AirfRANSDataset` 与 `AirfRANSDatapipe` 两层接口

## 输入配置

- `source.data_dir`
  - AirfRANS 数据根目录，目录下需要包含 `manifest.json` 与各仿真子目录
- `source.stats_dir`
  - 归一化统计量保存目录，内部会读写 `mean_in/std_in/mean_out/std_out`
- `data.splits.train_name`
  - `manifest.json` 中训练全集对应的键名，例如 `full_train`
- `data.splits.val_split_ratio`
  - 从训练全集尾部切出验证集的比例
- `data.splits.test_name`
  - `manifest.json` 中测试集对应的键名，例如 `full_test`
- `data.splits.task`
  - 任务标签，示例配置中会写 `full/scarce/reynolds/aoa`
- `data.sampling.sample_strategy`
  - 采样策略，支持 `null`、`uniform`、`mesh`
- `data.sampling.n_boot`
  - 体网格采样点数基准
- `data.sampling.surf_ratio`
  - 表面采样点数相对 `n_boot` 的比例
- `data.crop`
  - 可选裁剪框，格式为 `(xmin, xmax, ymin, ymax)`
- `data.subsampling`
  - 训练和验证阶段最终保留的点数上限
- `dataloader.batch_size`
  - PyG `DataLoader` 批大小
- `dataloader.num_workers`
  - PyG `DataLoader` 工作进程数
- `model_hparams.build_graph`
  - 是否基于坐标构图
- `model_hparams.r`
  - `radius_graph` 的半径
- `model_hparams.max_neighbors`
  - 每个点的最大邻居数

## 数据存储约定

- 主数据路径：`<data_dir>/manifest.json`，以及 `<data_dir>/<sim_name>/<sim_name>_internal.vtu` 和 `<data_dir>/<sim_name>/<sim_name>_aerofoil.vtp`
- 统计量路径：`<stats_dir>/mean_in.npy`、`std_in.npy`、`mean_out.npy`、`std_out.npy`
- 元数据来源：`manifest.json`、样本目录名中的工况编码，以及 VTK 文件中的 `point_data` / `cell_data`

额外约定：

- `manifest.json` 必须包含配置里引用到的键，例如 `full_train`、`full_test`
- 样本名默认满足 `s.split("_")[2]` 是来流速度 `Uinf`，`s.split("_")[3]` 是攻角角度值
- `internal.vtu` 需要提供 `U`、`p`、`nut`、`implicit_distance`，`aerofoil.vtp` 需要提供 `U`、`p`、`nut`、`Normals`

## 样本构造方式

- 输入样本：`Data.x` 形状为 `[NumPoints, 7]`，语义为 `[pos_x, pos_y, u_inf_x, u_inf_y, sdf, normal_x, normal_y]`
- 输出样本：`Data.y` 形状为 `[NumPoints, 4]`，语义为 `[v_x, v_y, p, nut]`
- 附加返回项：`Data.pos`、`Data.surf`、`Data.edge_index`

具体做法：

- 一个 dataset 样本对应一个仿真目录，而不是连续时间窗
- 当 `sample_strategy=null` 时，读取全量内部网格点；`surf` 通过 `internal.point_data["U"][:, 0] == 0` 判定，表面法向量通过 `reorganize(...)` 从 `aerofoil.vtp` 对齐回内部网格表面点
- 当 `sample_strategy="uniform"` 或 `"mesh"` 时，会分别对体单元和翼型表面线单元采样，再在单元内部做 2D / 1D 插值，最后拼成统一的点集
- `uniform` 按单元面积与边界线段长度加权抽样；`mesh` 按单元数量均匀抽样
- 若设置了 `data.crop`，会先对内部网格做 `clip_box`
- 归一化统计量来自训练集；若本地不存在统计文件，训练集初始化时会临时强制关闭采样，用全量网格计算均值和标准差
- 训练和验证阶段还会做一次随机点级下采样：若当前点数大于 `data.subsampling`，随机保留固定数量点；测试阶段不做这一步
- 若 `model_hparams.build_graph=True`，会基于 `Data.pos` 用 `radius_graph` 生成 `edge_index`；否则返回 `None`

## DataLoader 约定

- 训练阶段：`train_dataloader()` 返回 `(PyGDataLoader, sampler)`；分布式时使用 `DistributedSampler(shuffle=True)` 且 `drop_last=True`，非分布式时直接 `shuffle=True`
- 验证阶段：`val_dataloader()` 返回 `(PyGDataLoader, sampler)`；分布式时使用 `DistributedSampler(shuffle=False)`，否则不打乱
- 测试阶段：`test_dataloader()` 只返回 `PyGDataLoader`，不使用分布式采样器，`shuffle=False` 且 `drop_last=False`

## 适合优先使用的场景

- 使用 AirfRANS 数据训练二维翼型流场代理模型，预测速度、压力和湍流黏度
- 模型以点云图结构作为输入，例如 Transolver、GraphSAGE、GUNet 一类几何泛化模型
- 需要在全量网格读取和随机单元采样两种数据组织方式之间切换

## 风险点

- train / val 划分不是由 `manifest.json` 单独给出，而是对 `train_name` 对应全集按 `val_split_ratio` 直接切分
- `data.splits.task` 在当前实现里会被读取，但真正决定训练/测试集合的是 `train_name` 与 `test_name`
- 样本目录名被直接解析为来流速度和攻角，一旦命名规则变化，边界条件构造会失效
- 全量网格模式把 `U_x == 0` 当成表面点判据，并假设内部表面点能与 `aerofoil.vtp` 做一一坐标对齐
- 验证和测试集要求先拿到训练阶段生成或加载好的归一化统计量，否则会直接报错

## 源码锚点

- `./onescience/src/onescience/datapipes/cfd/AirfRANS.py`
- `./onescience/src/onescience/datapipes/core/base_dataset.py`
- `./onescience/src/onescience/utils/transolver/reorganize.py`
