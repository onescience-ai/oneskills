# Model Card: MeshGraphNet

## 基本信息

- 模型名：`MeshGraphNet`
- 任务类型：`CFD / 图网格物理场回归与时序 rollout`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/meshgraphnet/meshgraphnet.py`

## 模型定位

MeshGraphNet 是一个面向非结构网格与图结构物理场建模的消息传递网络，适合在显式图拓扑上做节点级物理量预测。

补充说明：

- 当前 `OneScience` 里至少有两套 MeshGraphNet 风格实现：`onescience.models.meshgraphnet.MeshGraphNet` 和 `cfd_benchmark/MeshGraphNet`
- 现有 `./onescience/examples/cfd/Vortex_shedding_mgn/` 使用的是 `onescience.models.meshgraphnet.MeshGraphNet` 这一套 DGL 版本
- 它最重要的结构特点是：先分别编码节点与边特征，再通过多层 `Edge -> Node` 消息传递块传播信息，最后直接输出每个节点的目标物理量

## 输入定义

- 输入 shape：`node_features -> (NumNodes, NodeInDim)`，`edge_features -> (NumEdges, EdgeInDim)`，`graph -> DGLGraph`
- 输入变量组织：
  - `graph.ndata["x"]`
    - 节点输入特征，例如速度、节点类型、边界条件、几何特征或其它点级状态
  - `graph.edata["x"]`
    - 边输入特征，例如相对坐标差、边长、几何关系量
  - `graph`
    - DGL 图拓扑本身

补充说明：

- 在现有 `Vortex_shedding_mgn` 案例中，节点输入默认是 `2 维速度 + 4 维节点类型 one-hot = 6 维`
- 边输入默认是 `dx, dy, ||d|| = 3 维`
- 模型本身不负责建图，图结构和边特征必须由 datapipe 先准备好

## 输出定义

- 输出 shape：`(NumNodes, output_dim)`
- 输出变量组织：
  - 每个节点直接输出对应目标变量
  - 在现有 `Vortex_shedding_mgn` 案例中，默认输出 `2 维目标速度项 + 1 维压力 = 3 维`

## 主干结构

- `OneMlp(style="MeshGraphMLP")`
  - 编码边特征
- `OneMlp(style="MeshGraphMLP")`
  - 编码节点特征
- 多层消息传递处理器
  - 每层由 `OneEdge(style="MeshEdgeBlock")` 和 `OneNode(style="MeshNodeBlock")` 组成
- `OneMlp(style="MeshGraphMLP")`
  - 将处理后的节点隐状态解码为目标物理量

建议把它理解为：

- datapipe 决定 `graph.ndata["x"]`、`graph.edata["x"]` 和 `graph.ndata["y"]`
- MeshGraphNet 只负责图上的 Encode-Process-Decode
- 现有示例中的“时序 rollout”逻辑主要在 datapipe 与训练脚本，不在模型类内部

## 主要依赖组件

- `OneMlp`
- `OneEdge`
- `OneNode`

## 主要 Shape 变化

- datapipe 输出：
  - `graph.ndata["x"] -> (NumNodes, NodeInDim)`
  - `graph.edata["x"] -> (NumEdges, EdgeInDim)`
- 编码后：
  - 节点与边都映射到隐藏维度 `hidden_dim_processor`
- Processor 多层消息传递：
  - 隐状态维度保持不变
- 解码后：
  - `(NumNodes, output_dim)`

## 默认关键参数

- `processor_size=15`
- `hidden_dim_processor=128`
- `num_layers_node_processor=2`
- `num_layers_edge_processor=2`
- `hidden_dim_node_encoder=128`
- `hidden_dim_edge_encoder=128`
- `hidden_dim_node_decoder=128`
- `aggregation="sum"`

说明：

- `input_dim_nodes`、`input_dim_edges`、`output_dim` 由具体 datapipe 和任务目标决定
- 现有 `Vortex_shedding_mgn` 示例还提供了 `BiStrideMeshGraphNet`，但默认基线是标准 `MeshGraphNet`

## 常见修改点

- 新数据集的节点特征字段变化时，同步修改 `input_dim_nodes`
- 新数据集的边特征设计变化时，同步修改 `input_dim_edges`
- 新数据集的目标变量变化时，同步修改 `output_dim`
- 若新数据集是稳态单样本图，而不是时序 rollout 数据，需要同步调整 datapipe 与训练脚本，不要只替换模型类
- 若新数据集来自 `VTK / VTU / VTP` 文件，通常要先确定是构造成 DGL 图、PyG 图还是点云回归协议，再决定是否走 MeshGraphNet 路线

## 风险点

- `OneScience` 中存在两套 MeshGraphNet 风格实现；若用户说 `MGN`，要先确认是 `onescience.models.meshgraphnet.MeshGraphNet` 还是 `cfd_benchmark/MeshGraphNet`
- 当前 `Vortex_shedding_mgn` 示例默认依赖 DGL 图、TFRecord 数据和时序 rollout，不代表所有新数据集都能直接复用这套训练流程
- 模型前向需要显式传入 `node_features, edge_features, graph`，不是 `PyG Data` 协议；不能直接拿 Transolver 的 datapipe 喂给它
- DGL 依赖是硬要求；若环境没有 DGL，现有 MGN 路线无法直接运行

## 推荐检索顺序

1. 先看本模型卡
2. 再看最接近的 MGN 示例工程
3. 再看对应 datapipe 卡
4. 若仍不足，再回到源码锚点

## 组件契约入口

- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/meshgraphnet/meshgraphnet.py`
- `./onescience/src/onescience/models/meshgraphnet/bsms_mgn.py`
- `./onescience/examples/cfd/Vortex_shedding_mgn/train.py`
- `./onescience/examples/cfd/Vortex_shedding_mgn/conf/mgn_cylinderflow.yaml`
