# Model Card: GFNO

## 基本信息

- 模型名：`GFNO`
- 任务类型：`CFD / group-equivariant Fourier operator`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/GFNO.py`

## 模型定位

GFNO 是面向二维场的 group-equivariant Fourier baseline，适合在旋转/对称性比较重要的 2D 流场任务上，对比普通 FNO 与群等变谱模型。

补充说明：

- 当前实现本质上围绕 2D `Conv2d / SpectralConv2d` 组织。
- 结构化和非结构化两条分支都保留，但核心主干都是 GFNO stem。
- 与普通 FNO 相比，它更强调群等变卷积和 group-wise normalization。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, 2)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x`
    - 2D 坐标或统一位置编码
  - `fx`
    - 点级输入场

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 点级目标变量

## 主干结构

- `OneMlp(style="StandardMLP")`
- 可选 `GeoSpectralConv2d`
  - 非结构化入/出投影
- `OneEquivariant(style="GroupEquivariantConv2d")`
  - lifting
- 4 层 `GSpectralConv2d + GroupEquivariantMLP2d + residual 1x1 gconv`
- 结构化分支用 `GroupEquivariantMLP2d` 输出头
- 非结构化分支用点级 `point_q`

## 主要依赖组件

- `OneFourier`
- `OneEquivariant`
- `OneMlp`

## 主要 Shape 变化

- 预处理后：`(Batch, NumPoints, n_hidden)`
- reshape 后：`(Batch, n_hidden, H, W)`
- GFNO stem 内部会扩展到 group-aware 通道布局
- 最终回到：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `modes=<from args>`
- `n_hidden=<from args>`
- `reflection=False`（源码当前固定）
- `s1=96, s2=96` 用于非结构化投影

## 常见修改点

- 主要用于二维任务，先保证 `shapelist` 真的是 2D
- 随输入输出变量修改 `fun_dim / out_dim`
- 若要比较几何投影效果，可在同一任务下切换 structured 与 unstructured 路径

## 风险点

- 当前实现明显偏二维；不适合直接推广成 3D 卡片复用。
- 非结构化分支仍依赖 GeoFNO 风格投影，不是原生点云卷积。
- 群等变通道布局更复杂，自己改 stem 时容易把通道数和 group_size 搞错。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `FNO`
3. 再看 `OneEquivariant` 与 `OneFourier`

## 组件契约入口

- `./contracts/onefourier.md`
- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/GFNO.py`
- `./onescience/src/onescience/modules/fourier/onefourier.py`
