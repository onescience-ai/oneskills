# Model Card: FNO

## 基本信息

- 模型名：`FNO`
- 任务类型：`CFD / steady-or-dynamic operator learning on structured grids`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/FNO.py`

## 模型定位

FNO 是当前 `CFD_Benchmark` 体系里最基础的谱算子基线，适合在规则网格上做全局场到全局场的算子学习，也支持带几何投影的 2D 非结构分支。

补充说明：

- 结构化分支最常见于 `structured_2D` 稳态代理任务
- 输入不是图像张量 `(Batch, Channels, Height, Width)`，而是位置 `x` 与可选场特征 `fx` 的二元组织
- 它最重要的结构特点是：前面先用 MLP 做点级特征映射，中间用多层谱卷积做全局建模，最后再映射回目标变量

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)` 或 `None`
- 输入变量组织：
  - `x`
    - 位置或规则网格坐标
  - `fx`
    - 与每个位置对齐的输入物理特征；若 `fun_dim=0`，可为空

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 每个网格点或采样点对应 `out_dim` 个目标变量

## 主干结构

- `OneMlp(style="StandardMLP")`
  - 将 `x` 与 `fx` 拼接后的点级特征映射到 `n_hidden`
- 可选 `OneFourier(style="GeoSpectralConv2d")`
  - 在非结构 2D 分支里做点集到潜在规则网格的投影
- 4 层 `OneFourier(style="FNOSpectralConv*d") + 1x1 Conv` 残差块
- `fc1 + fc2`
  - 将隐藏特征映射到 `out_dim`

## 主要依赖组件

- `OneMlp`
- `OneFourier`

## 主要 Shape 变化

- 输入拼接后：`(Batch, NumPoints, space_dim + fun_dim)` 或统一位置编码后的对应维度
- 预处理后：`(Batch, NumPoints, n_hidden)`
- 结构网格分支重排后：`(Batch, n_hidden, *shapelist)`
- 谱卷积主干后再展平：`(Batch, NumPoints, n_hidden)`
- 输出投影后：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `geotype="structured_2D"` 或 `unstructured`
- `space_dim=2`
- `fun_dim=0` 到若干输入通道
- `out_dim=1` 到若干目标通道
- `n_hidden=32` 或 `64`
- `modes=12`
- `unified_pos=0`
- `ref=8`

说明：

- `shapelist` 不是模型内部固定值，而是 dataloader 返回给训练入口的网格尺寸
- 若把 DeepCFD 规则网格数据接到 FNO，常见适配是 `space_dim=2`、`fun_dim=3`、`out_dim=3`

## 常见修改点

- 根据 datapipe 返回的网格尺寸同步修改 `shapelist`
- 根据输入物理量数量同步修改 `fun_dim`
- 根据目标变量数量同步修改 `out_dim`
- 若从 `DeepCFDDatapipe` 迁移过来，需要新增 adapter，把图像张量改造成 `(pos, fx, y)` 三元组

## 风险点

- FNO 不能直接消费 `DeepCFDDatapipe` 返回的 `{"x": ..., "y": ...}` batch
- `shapelist` 一旦和真实规则网格尺寸不一致，`reshape` 会直接失败
- `unified_pos=1` 会改变预处理层输入维度，不能只改配置不改数据理解
- 当前非结构分支围绕 `GeoSpectralConv2d` 组织，默认更偏向 2D 几何任务

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `OneFourier`
3. 再看 `OneMlp`
4. 若仍不足，再回到 `CFD_Benchmark` 训练入口

## 组件契约入口

- `./contracts/onefourier.md`
- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/FNO.py`
- `./onescience/src/onescience/modules/fourier/onefourier.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
