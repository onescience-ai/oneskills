# Model Card: Graph_UNet

## 基本信息

- 模型名：`Graph_UNet`
- 任务类型：`CFD / hierarchical graph baseline`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/Graph_UNet.py`

## 模型定位

Graph_UNet 是基于图下采样/上采样的层次化 PyG 基线，适合在显式图拓扑上做多尺度节点级回归。

补充说明：

- 下采样和上采样通过 `OneSample(style="SpatialGraphDownsample/SpatialGraphUpsample")` 完成。
- 主干卷积层可选 `SAGE` 或 `GAT` 风格。
- 与 GraphSAGE 相比，它更强调多尺度结构和 skip connection。

## 输入定义

- 输入 shape：`x -> (NumPoints, coords)`，`fx -> (NumPoints, fun_dim)`，`geo -> edge_index`
- 输入变量组织：
  - `x`
    - 节点坐标；代码里默认前两列可作为位置
  - `fx`
    - 节点输入特征
  - `geo`
    - `edge_index`

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 节点级目标变量

## 主干结构

- `OneMlp`
  - encoder
- 多层 `SpatialGraphDownsample + GraphConv`
- `SpatialGraphUpsample + skip connection + GraphConv`
- `OneMlp`
  - decoder

## 主要依赖组件

- `OneMlp`
- `OneSample`
- `torch_geometric.nn.SAGEConv / GATConv`

## 主要 Shape 变化

- 编码后：`(NumPoints, n_hidden)`
- 每次下采样后节点数减少、通道数增大
- 上采样阶段与 skip 特征拼接后再卷积
- 解码后回到：`(NumPoints, out_dim)`，最后 `unsqueeze(0)`

## 默认关键参数

- `scale=5`
- `pool="random"`
- `pool_ratio=[0.5, 0.5, 0.5, 0.5, 0.5]`
- `layer="SAGE"`

## 常见修改点

- 若换图卷积风格，先看 `layer="SAGE" / "GAT"` 相关通道变化
- 若节点过少，谨慎增大下采样层数或减小 `pool_ratio`
- 与 datapipe 一起确认 `x` 里是否真的包含可用于采样的坐标

## 风险点

- `geo` 缺失时模型会直接报错。
- 下采样/上采样强依赖位置和图拓扑；如果构图质量差，多尺度路径会比 GraphSAGE 更脆弱。
- `x[:, :2]` 被默认当作坐标，换数据集时要确认这一假设仍成立。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `GraphSAGE`
3. 再看图 datapipe 与 `OneSample`

## 组件契约入口

- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/Graph_UNet.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_airfoil.py`
