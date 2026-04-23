# Datapipe Index

## 使用建议

- 先按任务的数据组织方式选最接近的 datapipe，再看模型是否兼容。
- 先确认样本返回协议：是 `dict`、`(x, y, grid)`、PyG `Data/HeteroData`，还是 DGL `Graph`。
- 若 README 没写清字段名、shape 或 split 规则，先补只读探测，再决定生成什么 datapipe。
- 若现有 datapipe 不能直接喂给目标模型，优先加一层最小 adapter，不要直接重写整条训练链。

## 已登记 Datapipe

| Datapipe | 领域 | 典型任务 / 数据组织 | 文档 |
| --- | --- | --- | --- |
| `ERA5Datapipe` | weather | 全球气象格点时序 | [era5.md](./era5.md) |
| `CMEMSDatapipe` | ocean | 海洋格点时序 / 变量切片 | [cmems.md](./cmems.md) |
| `TJDatapipe` | weather | 区域气象数据读取 | [tjweather.md](./tjweather.md) |
| `AirfRANSDatapipe` | cfd | 二维翼型非结构网格 / PyG 图样本 | [airfrans.md](./airfrans.md) |
| `ShapeNetCarDatapipe` | cfd | 车体表面 VTK / PyG 图样本 | [shapenetcar.md](./shapenetcar.md) |
| `DeepCFDDatapipe` | cfd | 规则网格 `pickle -> tensor` | [deepcfd.md](./deepcfd.md) |
| `DeepMind_CylinderFlowDatapipe` | cfd | 网格时序流场 / rollout | [deepmind_cylinderflow.md](./deepmind_cylinderflow.md) |
| `BENODatapipe` | cfd | 椭圆 PDE 异构图 / `npy -> HeteroData` | [beno.md](./beno.md) |
| `CFDBenchDatapipe` | cfd | Tube/Cavity/Cylinder/Dam 基准 / `dict` batch | [cfdbench.md](./cfdbench.md) |
| `DeepMindLagrangianDatapipe` | cfd | 粒子轨迹 TFRecord / DGL 图 | [deepmind_lagrangian.md](./deepmind_lagrangian.md) |
| `EagleDatapipe` | cfd | CFD 时序轨迹 / padding collate | [eagle.md](./eagle.md) |
| `DrivAerML_FigConvUnetDatapipe` | cfd | 车体表面分片二进制图 / 点采样张量 | [drivaerml_figconvunet.md](./drivaerml_figconvunet.md) |
| `PDEBenchFNODatapipe` | cfd / operator | HDF5 单文件或多文件 / `(history, target, grid)` | [pdenneval.md](./pdenneval.md) |
| `PDEBenchDeepONetDatapipe` | cfd / operator | HDF5 单文件或多文件 / `(history, target, grid)` | [pdenneval.md](./pdenneval.md) |
| `PDEBenchMPNNDatapipe` | cfd / graph operator | HDF5 -> 时空图训练前样本 | [pdenneval.md](./pdenneval.md) |
| `PDEBenchUNetDatapipe` | cfd / operator | HDF5 -> `(history, target)` | [pdenneval.md](./pdenneval.md) |
| `PDEBenchUNODatapipe` | cfd / operator | HDF5 -> `(history, target, grid)` | [pdenneval.md](./pdenneval.md) |
| `PDEBenchPINODatapipe` | cfd / physics-informed operator | HDF5 + supervised / PDE 双 loader | [pdenneval.md](./pdenneval.md) |

## 新数据集接入时优先参考

- 新数据是二维或三维非结构表面/体网格，并且最后要喂给 PyG 图模型：先看 `AirfRANSDatapipe`、`ShapeNetCarDatapipe`。
- 新数据是规则网格上的 `X -> Y` 回归：先看 `DeepCFDDatapipe`；若带 case 参数、边界条件或自回归时间步，再看 `CFDBenchDatapipe`。
- 新数据是粒子轨迹、需要 DGL 图和 rollout：先看 `DeepMindLagrangianDatapipe`。
- 新数据是时序 CFD 轨迹，并且需要 padding、cluster 或窗口切片：先看 `EagleDatapipe`。
- 新数据是车体表面压力/切应力回归，并且模型直接吃点采样张量：先看 `DrivAerML_FigConvUnetDatapipe`。
- 新数据本身就是 HDF5 PDEBench 风格，目标是 FNO / UNO / PINO / DeepONet / MPNN / UNet：先看 `pdenneval.md`。
- 新任务需要异构图或多节点类型建模：先看 `BENODatapipe`。

## 兼容性提醒

- `DeepCFDDatapipe` 和 `CFDBenchDatapipe` 都是规则网格数据，但 batch 协议完全不同，不能直接互换。
- `AirfRANSDatapipe`、`ShapeNetCarDatapipe`、`BENODatapipe` 都是图数据，但分别对应 `PyG Data`、`PyG HeteroData` 和不同字段约定。
- `DeepMindLagrangianDatapipe`、`EagleDatapipe` 都处理时序流场，但一个输出 DGL 图，一个输出 padding 后的普通张量字典。
- `PDEBench` family 看似共用一个源文件，但不同模型族的样本返回协议并不相同，写 train 脚本前必须先确认。
