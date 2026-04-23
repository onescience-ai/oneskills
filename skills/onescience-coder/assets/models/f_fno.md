# Model Card: F_FNO

## 基本信息

- 模型名：`F_FNO`
- 任务类型：`CFD / factorized Fourier operator baseline`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/F_FNO.py`

## 模型定位

F_FNO 是 Factorized Fourier Neural Operator 版本的谱算子基线，适合在保持 FNO 接口不变的前提下，用分解式谱卷积替代标准谱块。

补充说明：

- 结构化分支使用 `FFNOSpectralConv*d`。
- 非结构化分支通过 `GeoSpectralConv2d + IPHI` 先投影到潜在规则网格，再做 FFNO 主干。
- 如果你已经接受 FNO 的数据协议，但想比较不同谱卷积实现，这个模型是直接替换项。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x`
    - 坐标或统一位置编码。
  - `fx`
    - 点级物理特征。

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 每个点输出 `out_dim` 个目标量。

## 主干结构

- `StandardMLP`
  - 预处理
- 可选 `GeoSpectralConv2d`
  - 非结构化几何的入/出投影
- 多层 `OneFourier(style="FFNOSpectralConv*d")`
- `fc1 + fc2`

## 主要依赖组件

- `OneFourier`
- `OneMlp`
- `IPHI`

## 主要 Shape 变化

- 预处理后：`(Batch, NumPoints, n_hidden)`
- 规则网格 reshape 后：`(Batch, n_hidden, *grid_shape)`
- 谱层后再展平：`(Batch, NumPoints, n_hidden)`
- 投影后：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `modes=<from args>`
- `n_layers=<from args>`
- `n_hidden=<from args>`
- `s1=96, s2=96` 用于非结构化投影默认潜在网格

## 常见修改点

- 根据 datapipe 修改 `shapelist`
- 依据场变量修改 `fun_dim / out_dim`
- 非结构化几何下若点分布变化明显，重新检查 `s1/s2` 与投影质量

## 风险点

- 它和 FNO 的输入接口相同，但内部 reshape 仍强依赖 `shapelist` 正确。
- 非结构化分支本质上是“先投影再谱卷积”，不等同于直接在点云上做算子学习。
- 若 datapipe 只返回 `{"x": ..., "y": ...}` 图像式 batch，仍需先桥接到 `(pos, fx)` 协议。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `FNO`
3. 再看 `OneFourier`

## 组件契约入口

- `./contracts/onefourier.md`
- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/F_FNO.py`
- `./onescience/src/onescience/modules/fourier/onefourier.py`
