# Model Card: ONO

## 基本信息

- 模型名：`ONO`
- 任务类型：`CFD / orthogonal neural operator`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/ONO.py`

## 模型定位

ONO 是 Orthogonal Neural Operator 路线的 Transformer 风格基线，适合在 token 级算子学习里强调“几何编码 + 物理编码”双路交互。

补充说明：

- `x` 与 `fx` 都经过单独的 MLP 预处理。
- block 使用 `OneTransformer(style="OrthogonalNeuralBlock")`。
- 输出直接来自 block 链路，不额外再接独立解码头。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x`
    - 坐标或统一位置编码
  - `fx`
    - 输入场特征

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - `fx` 分支在最后一层 block 后被映射为目标输出

## 主干结构

- `preprocess_x`
- `preprocess_z`
- 多层 `OneTransformer(style="OrthogonalNeuralBlock")`
- 末层 block 直接输出 `out_dim`

## 主要依赖组件

- `OneMlp`
- `OneTransformer(style="OrthogonalNeuralBlock")`

## 主要 Shape 变化

- 两路预处理后都变成：`(Batch, NumPoints, n_hidden)`
- 每层 block 后保持 token 形态
- 末层输出：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `psi_dim=<from args>`
- `attn_type=<from args>`
- `n_layers=<from args>`
- `n_heads=<from args>`

## 常见修改点

- 修改 `fun_dim / out_dim`
- 比较不同 `attn_type`
- 若不使用时间条件，可保持 `time_input=False`

## 风险点

- `preprocess_x` 与 `preprocess_z` 的输入维度都可能受 `unified_pos` 影响。
- 输出来自 block 末层，因此替换 block 风格时要检查 `last_layer/out_dim`。
- 如果 datapipe 只给单一路输入，这个模型的双分支设计会被弱化。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `OneTransformer`
3. 再看 `Transformer`

## 组件契约入口

- `./contracts/onetransformer.md`
- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/ONO.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
