# Model Card: DeepONet

## 基本信息

- 模型名：`DeepONet`
- 任务类型：`CFD / operator learning with branch-trunk decomposition`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/DeepONet.py`

## 模型定位

DeepONet 是 `CFD_Benchmark` 里最典型的 branch-trunk 算子基线，适合把“输入函数特征”与“评估坐标”分开编码后再做点级场值预测。

补充说明：

- `branch_net` 处理 `fx`，`trunk_net` 处理 `x`。
- 结构化与非结构化几何都走同一套点级接口，只是 `unified_pos` 时位置编码来源不同。
- 若任务强调“函数到函数”的算子学习而不是局部卷积或图消息传递，这个模型是直接可复用的起点。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)` 或 `None`
- 输入变量组织：
  - `x`
    - 评估点坐标或统一位置编码。
  - `fx`
    - 与评估点对齐的输入函数值、边界条件或其它物理特征。

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 每个点单独输出 `out_dim` 个目标变量。

## 主干结构

- `OneMlp(style="StandardMLP")`
  - 构造 `branch_net`
- `OneMlp(style="StandardMLP")`
  - 构造 `trunk_net`
- 逐点乘法融合 `branch_feat * trunk_feat`
- `Linear + bias`
  - 投影到 `out_dim`

## 主要依赖组件

- `OneMlp`
- `timestep_embedding`
- `unified_pos_embedding`

## 主要 Shape 变化

- `branch_net` 后：`(Batch, NumPoints, n_hidden)`
- `trunk_net` 后：`(Batch, NumPoints, n_hidden)`
- 点级融合后：`(Batch, NumPoints, n_hidden)`
- 输出层后：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `branch_depth=<from args>`
- `trunk_depth=<from args>`
- `n_hidden=<from args>`
- `time_input=False` 时不启用时间分支

## 常见修改点

- 随输入函数通道数变化同步修改 `fun_dim`
- 随目标变量数变化同步修改 `out_dim`
- 若要加入显式时间条件，打开 `time_input` 并保证训练脚本传入 `T`

## 风险点

- `fx` 是 branch 分支的核心输入；若 datapipe 只给坐标、不提供函数特征，这个模型的优势会明显下降。
- `unified_pos=True` 会改变 `trunk_net` 的输入维度，不能只改配置不改 shape 理解。
- 它是逐点评估风格，不负责规则网格 reshape，也不直接消费图结构。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `OneMlp`
3. 再看最接近的 operator datapipe

## 组件契约入口

- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/DeepONet.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
