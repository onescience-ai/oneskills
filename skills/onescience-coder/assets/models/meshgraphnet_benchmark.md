# Model Card: MeshGraphNet (CFD_Benchmark)

## 基本信息

- 模型名：`MeshGraphNet (CFD_Benchmark)`
- 任务类型：`CFD / explicit graph encode-process-decode baseline`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/MeshGraphNet.py`

## 模型定位

这是 `cfd_benchmark` 目录下的 MeshGraphNet 版本，接口是显式的 `(node_features, edge_features, graph)`，和项目里另一套 `onescience.models.meshgraphnet.MeshGraphNet` 需要区分。

补充说明：

- 源码里 `__name__` 实际写成 `LSMMeshGraphNet`，但在 benchmark 工厂里仍按 `MeshGraphNet` 使用。
- 输入边特征维默认固定成 `4`。
- 适合已经有 DGL 图和显式边特征的数据接口。

## 输入定义

- 输入 shape：
  - `node_features -> (NumNodes, input_dim_nodes)`
  - `edge_features -> (NumEdges, input_dim_edges)`
  - `graph -> DGLGraph / CuGraphCSC`
- 输入变量组织：
  - `node_features`
    - 节点输入物理特征
  - `edge_features`
    - 边相对几何或其它边特征
  - `graph`
    - 图拓扑

## 输出定义

- 输出 shape：`(NumNodes, output_dim)`
- 输出变量组织：
  - 每个节点输出目标变量

## 主干结构

- `OneMlp(style="MeshGraphMLP")`
  - edge encoder
- `OneMlp(style="MeshGraphMLP")`
  - node encoder
- 多层 `OneEdge(style="MeshEdgeBlock") -> OneNode(style="MeshNodeBlock")`
- `OneMlp(style="MeshGraphMLP")`
  - node decoder

## 主要依赖组件

- `OneMlp`
- `OneEdge`
- `OneNode`

## 主要 Shape 变化

- 编码后：
  - `node_features -> (NumNodes, hidden_dim_processor)`
  - `edge_features -> (NumEdges, hidden_dim_processor)`
- processor 多层后保持隐藏维不变
- decoder 后：`(NumNodes, output_dim)`

## 默认关键参数

- `processor_size=15`
- `hidden_dim_processor=128`
- `aggregation="sum"`
- `input_dim_edges=4`（源码固定）

## 常见修改点

- 随 datapipe 修改 `input_dim_nodes / output_dim`
- 若边特征设计变化，必须同步改 `input_dim_edges`
- 需要大图训练时，可再看 checkpoint segment 相关参数

## 风险点

- 很容易和 `meshgraphnet.md` 指向的另一套实现混淆。
- 接口不是 `(x, fx, geo)`，而是显式三参 `(node_features, edge_features, graph)`。
- 图和边特征都要由 datapipe 先准备好，模型本身不负责构图。

## 推荐检索顺序

1. 先看本模型卡
2. 再看通用 `meshgraphnet.md`
3. 再看图 datapipe

## 组件契约入口

- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/MeshGraphNet.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/`
