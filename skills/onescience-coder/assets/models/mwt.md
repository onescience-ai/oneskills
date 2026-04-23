# Model Card: MWT

## 基本信息

- 模型名：`MWT`
- 任务类型：`CFD / multiwavelet operator`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/MWT.py`

## 模型定位

MWT 是多小波变换算子基线，适合在规则网格上比较“小波域算子”与傅里叶算子、卷积型 U-Net 之间的差异。

补充说明：

- 结构化分支会把空间尺寸补到 2 的幂。
- 非结构化分支先通过 GeoFNO 投影到潜在规则网格，再做 MWT 主干。
- 这是比 FNO 更偏“多分辨率频域”路线的算子模型。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x`
    - 坐标或统一位置编码
  - `fx`
    - 点级输入场

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 点级目标变量

## 主干结构

- `OneMlp`
  - 预处理到 `WMT_dim`
- 可选 `GeoSpectralConv2d`
  - 非结构化投影
- 多层 `OneFourier(style="MultiWaveletTransform*d")`
- `fc1 + fc2`

## 主要依赖组件

- `OneMlp`
- `OneFourier(style="MultiWaveletTransform*d")`
- `OneFourier(style="GeoSpectralConv2d")`

## 主要 Shape 变化

- 预处理后：`(Batch, NumPoints, WMT_dim)`
- reshape 到增强分辨率网格后：`(Batch, *augmented_resolution, c, k^2 or k)`
- 多层小波变换后再还原成点级表示
- 输出后：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `mwt_k=<from args>`
- `alpha=2`
- `base="legendre"`
- `s1=128, s2=128`

## 常见修改点

- 先根据数据集真实网格设置 `shapelist`
- 调整 `mwt_k`、`n_layers`、`modes`
- 如果只做二维 structured 任务，优先先确认网格尺寸补齐后的显存是否可接受

## 风险点

- 结构化分支会自动把分辨率补到 2 的幂，训练和评估时都要理解这一隐式 padding。
- `WMT_dim` 依赖 `mwt_k` 和几何维度，改动时容易连锁影响 reshape。
- 非结构化路径仍然不是原生图/点卷积，而是投影到潜在规则网格后处理。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `FNO`
3. 再看 `OneFourier`

## 组件契约入口

- `./contracts/onefourier.md`
- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/MWT.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
