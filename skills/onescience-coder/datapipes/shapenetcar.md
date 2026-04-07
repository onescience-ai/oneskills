# Datapipe: ShapeNetCarDatapipe

## 基本信息

- Datapipe 名：`ShapeNetCarDatapipe`
- 数据类型：`cfd`
- 主要任务：`3d surface flow / pressure regression`
- 数据组织方式：`per_simulation_dir_with_vtk_or_preprocessed_npy`

## Datapipe 职责

`ShapeNetCarDatapipe` 负责把按参数文件夹组织的汽车外流场 VTK 结果组织成 `PyG Data` 图样本，并封装成现有 `Transolver-Car-Design` 训练流程可直接消费的 `DataLoader`。

补充说明：

- 原始数据通常以 `param*/sample/` 目录组织，每个样本至少包含一对 `quadpress_smpl.vtk` 和 `hexvelo_smpl.vtk`
- datapipe 负责 fold 划分、VTK 读取、法向量与 SDF 构造、表面/外部点拼接、可选读取预处理缓存、归一化和图构建
- datapipe 同时实现了 `ShapeNetCarDataset` 与 `ShapeNetCarDatapipe` 两层接口

## 输入配置

- `source.data_dir`
  - 原始数据根目录
- `source.preprocessed_save_dir`
  - 预处理缓存目录；若存在 `.npy` 缓存可优先读取
- `source.stats_dir`
  - 归一化统计量目录，保存 `mean/std` 数组
- `source.preprocessed`
  - 是否优先读取预处理缓存
- `data.splits.fold_id`
  - 交叉验证折号；当前实现按 `param0` 到 `param8` 做 fold 划分
- `dataloader.batch_size`
  - PyG `DataLoader` 批大小
- `dataloader.num_workers`
  - PyG `DataLoader` 工作进程数
- `model_hparams.cfd_mesh`
  - 是否使用原始 CFD 网格边
- `model_hparams.r`
  - 若不使用原始网格，`radius_graph` 的半径
- `model_hparams.max_neighbors`
  - 半径图最大邻居数

## 数据存储约定

- 主数据路径：`<data_dir>/param*/<sample>/quadpress_smpl.vtk` 与 `<data_dir>/param*/<sample>/hexvelo_smpl.vtk`
- 预处理缓存：`<preprocessed_save_dir>/<param*>/<sample>/{x,y,pos,surf,edge_index}.npy`
- 统计量路径：`<stats_dir>/mean_in.npy`、`std_in.npy`、`mean_out.npy`、`std_out.npy`
- 元数据来源：目录分层、VTK 点坐标、VTK 中的速度/压力向量场以及内部构造的法向量与 SDF

额外约定：

- 当前实现默认存在 `param0` 到 `param8` 九个参数分组目录
- `quadpress_smpl.vtk` 与 `hexvelo_smpl.vtk` 必须能通过坐标关系拼成统一点集
- 若没有预处理缓存，首次读取时会直接从 VTK 计算并可选择保存为 `.npy`

## 样本构造方式

- 输入样本：`Data.x` 形状为 `[NumPoints, 7]`，语义通常是 `[pos_x, pos_y, pos_z, sdf, normal_x, normal_y, normal_z]`
- 输出样本：`Data.y` 形状为 `[NumPoints, 4]`，语义通常是 `[v_x, v_y, v_z, p]`
- 附加返回项：`Data.pos`、`Data.surf`、`Data.edge_index`

具体说明：

- 一个 dataset 样本对应一个汽车几何工况，而不是时间窗序列
- datapipe 会把外部速度场点和表面压力点拼成统一点集
- `surf` 用于区分表面点和非表面点，现有 `Transolver-Car-Design` 训练脚本会对表面压力损失单独加权
- 若 `model_hparams.cfd_mesh=False`，会基于 `Data.pos` 重新构建 `radius_graph`
- 若 `model_hparams.cfd_mesh=True`，会复用从 CFD 网格恢复出的 `edge_index`

## DataLoader 约定

- 训练阶段：`train_dataloader()` 返回 `(PyGDataLoader, sampler)`；分布式时使用 `DistributedSampler(shuffle=True)`
- 验证阶段：`val_dataloader()` 返回 `(PyGDataLoader, sampler)`；分布式时使用 `DistributedSampler(shuffle=False)`
- 测试阶段：`none`；当前 datapipe 类没有单独暴露 `test_dataloader()`

补充说明：

- 当前实现里 `mode="test"` 存在，但 `ShapeNetCarDatapipe` 默认只构造 train / val 两个 loader
- 现有 `Transolver-Car-Design` 示例主要沿用 fold 划分下的训练/验证逻辑

## 适合优先使用的场景

- 新数据集由一批 `vtk` 流场结果文件组成，目标是做三维表面或几何相关物理量回归
- 希望把样本组织成 `PyG Data(pos, x, y, surf, edge_index)` 供 Transolver 一类点云 / 图模型使用
- 用户希望沿用 `./onescience/examples/cfd/Transolver-Car-Design/` 这类几何回归训练流程

## 风险点

- 当前划分逻辑依赖 `param0` 到 `param8` 的目录结构；若新数据集没有这层分组，需要重写 split 规则
- `Transolver-Car-Design` 的损失函数对 `surf` 和最后一个压力通道有显式假设，新数据集目标变量变动时要同步检查 train / inference
- 若新数据集只有表面点而没有外部体点，或字段语义和 `quadpress/hexvelo` 不同，不能机械复用 `x/y` 组织方式
- datapipe 默认是 `PyG Data` 协议，不适合直接喂给 DGL 版 MeshGraphNet

## 源码锚点

- `./onescience/src/onescience/datapipes/cfd/ShapeNetCar.py`
- `./onescience/examples/cfd/Transolver-Car-Design/train.py`
- `./onescience/examples/cfd/Transolver-Car-Design/conf/transolver_car.yaml`
