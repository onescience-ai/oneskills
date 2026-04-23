# Model Card: GraphSAGE

## 基本信息

- 模型名：`GraphSAGE`
- 任务类型：`CFD / graph message passing baseline`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/GraphSAGE.py`

## 模型定位

GraphSAGE 是 `CFD_Benchmark` 里最直接的 PyG 图卷积基线，适合在显式 `edge_index` 上做点级场回归。

补充说明：

- 它直接依赖 `torch_geometric.nn.SAGEConv`。
- 输入仍沿用 `(x, fx, geo)` 协议，其中 `geo` 实际上是 `edge_index`。
- 如果任务已经能稳定构图，但不需要很复杂的层次化结构，它是很好的最小图基线。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)`，`geo -> edge_index`
- 输入变量组织：
  - `x + fx`
    - 拼接后送入编码器
  - `geo`
    - `PyG` 风格 `edge_index`

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 每个节点输出目标物理量

## 主干结构

- `OneMlp(style="StandardMLP")`
  - encoder
- 多层 `SAGEConv + BatchNorm + ReLU`
- `OneMlp(style="StandardMLP")`
  - decoder

## 主要依赖组件

- `OneMlp`
- `torch_geometric.nn.SAGEConv`

## 主要 Shape 变化

- 拼接后：`(NumPoints, space_dim + fun_dim)`
- 编码后：`(NumPoints, n_hidden)`
- 多层图卷积后保持：`(NumPoints, n_hidden)`
- 解码后：`(NumPoints, out_dim)`，最后再 `unsqueeze(0)`

## 默认关键参数

- `n_layers=<from args>`
- `n_hidden=<from args>`
- `bn_bool=True`（源码固定）

## 常见修改点

- 按 datapipe 输出同步修改 `space_dim / fun_dim / out_dim`
- 如果 batch>1 的图要拼接，优先在 datapipe 或训练脚本层处理图 batching
- 若图非常稀疏或非常稠密，可优先先调整构图而不是先改模型

## 风险点

- `geo` 必须是真实 `edge_index`，不能拿坐标张量直接顶替。
- 当前实现默认 batch 维往往是 1，并在前向里 `squeeze(0)`，批图训练前要先核实协议。
- 它只处理节点特征，不消费显式边特征。

## 推荐检索顺序

1. 先看本模型卡
2. 再看对应图 datapipe
3. 再看 `Graph_UNet`

## 组件契约入口

- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/GraphSAGE.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_airfoil.py`
