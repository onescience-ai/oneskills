# Model Card: Factformer

## 基本信息

- 模型名：`Factformer`
- 任务类型：`CFD / structured-grid transformer operator`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/Factformer.py`

## 模型定位

Factformer 是 `CFD_Benchmark` 里的结构化网格 Transformer 基线，核心是 factorized attention 风格的 `Factformer_block`，适合在规则 2D/3D 网格上做全局建模。

补充说明：

- 当前实现明确不支持 `unstructured` 几何。
- 输入接口和 FNO/Transformer 家族保持一致，仍是 `(x, fx, T)`。
- 若用户要比较“纯注意力主干”和谱算子主干，这个模型很适合作为对照组。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x`
    - 规则网格坐标或统一位置编码。
  - `fx`
    - 与每个网格点对齐的输入场特征。

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 最后一层 `Factformer_block` 直接输出目标通道。

## 主干结构

- `OneMlp(style="StandardMLP")`
  - 点级预处理
- 多层 `OneTransformer(style="Factformer_block")`
- 末层 block 直接投影到 `out_dim`

## 主要依赖组件

- `OneMlp`
- `OneTransformer(style="Factformer_block")`
- `timestep_embedding`

## 主要 Shape 变化

- 预处理后：`(Batch, NumPoints, n_hidden)`
- 每层 block 后保持：`(Batch, NumPoints, n_hidden)` 或末层变为 `out_dim`
- 最终输出：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `n_layers=<from args>`
- `n_heads=<from args>`
- `mlp_ratio=<from args>`
- `dropout=<from args>`

## 常见修改点

- 依据网格尺寸设置 `shapelist`
- 依据输入输出变量数修改 `fun_dim / out_dim`
- 若加时间条件，保持 `time_input` 与训练脚本传入的 `T` 一致

## 风险点

- 只支持结构化几何；给它喂非结构网格会直接报错。
- 末层 block 负责输出投影，替换 block 风格时要一起核对 `out_dim`。
- 规则网格点数过大时，注意力主干的显存压力通常高于谱算子基线。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `OneTransformer`
3. 再看与之对照的 `Transformer` 或 `FNO`

## 组件契约入口

- `./contracts/onetransformer.md`
- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/Factformer.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
