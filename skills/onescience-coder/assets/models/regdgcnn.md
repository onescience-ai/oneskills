# Model Card: RegDGCNN

## 基本信息

- 模型名：`RegDGCNN`
- 任务类型：`CFD / dynamic graph point-cloud baseline`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/RegDGCNN.py`

## 模型定位

RegDGCNN 是动态 kNN 图上的点级回归基线，适合在没有显式固定图拓扑时，用局部邻域重建的方式做点云物理场预测。

补充说明：

- 它不会消费 datapipe 传入的 `geo`，而是在前向里基于当前点特征自行做 kNN。
- 多层 `EdgeConv` 风格局部图特征提取后，直接做点级预测。
- 与 PointNet 相比，它更强调局部邻域结构。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x + fx`
    - 拼接后作为动态图构图与卷积输入

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 点级目标变量

## 主干结构

- 动态 `kNN`
- 4 层 EdgeConv 风格 `Conv2d + LeakyReLU`
- 特征拼接
- `Conv1d`
  - 生成 `emb_dims`
- `point_pred`
  - 点级线性回归头

## 主要依赖组件

- `knn`
- `get_graph_feature`
- `Conv2d / Conv1d`

## 主要 Shape 变化

- 拼接并转置后：`(Batch, Channels, NumPoints)`
- 构图特征后：`(Batch, 2*Channels, NumPoints, k)`
- 多层 EdgeConv 后提取 `x1/x2/x3/x4`
- 拼接后：`(Batch, 9 * n_hidden, NumPoints)`
- `conv5` 后：`(Batch, emb_dims, NumPoints)`
- 点级头后：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `k=40`
- `emb_dims=<from args>`
- `dropout` 只在旧版本全连接头里使用，当前点级头不显式用到

## 常见修改点

- 随输入输出变量修改 `space_dim / fun_dim / out_dim`
- 若点数很多，优先先评估 `k` 带来的构图开销
- 作为点云基线时，可与 PointNet、GraphSAGE、Graph_UNet 做并列比较

## 风险点

- 每次前向都要构动态图，点数大时计算与显存开销明显。
- `geo` 输入当前被忽略，因此如果用户以为自己传入的图结构会被用上，会产生误解。
- `k` 太大或太小时都可能显著影响局部建模表现。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `PointNet`
3. 再看 `GraphSAGE`

## 组件契约入口

- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/RegDGCNN.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_airfoil.py`
