# Model Card: U_FNO

## 基本信息

- 模型名：`U_FNO`
- 任务类型：`CFD / steady-or-dynamic operator learning with spectral-plus-multiscale baseline`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/U_FNO.py`

## 模型定位

U_FNO 是在 FNO 主干中插入并联 U-Net 多尺度分支的增强基线，适合同时捕获全局谱特征和局部多尺度结构。

补充说明：

- 输入输出接口与 `FNO` 保持同一套 `(x, fx) -> y` 约定
- 它最关键的区别不是数据接口变化，而是第 3、4 个谱块里额外引入了 `U_Net.multiscale()` 分支
- 适合用户已经接受 FNO 接口，但希望局部细节表达更强的场景

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)` 或 `None`
- 输入变量组织：
  - `x`
    - 位置或规则网格坐标
  - `fx`
    - 与位置对齐的输入场特征

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 每个位置对应 `out_dim` 个目标变量

## 主干结构

- `OneMlp(style="StandardMLP")`
- 可选 `OneFourier(style="GeoSpectralConv2d")`
- 4 层 FNO 风格谱块
- 其中第 3、4 层额外并联 `U_Net.multiscale(x)`
- `fc1 + fc2`

## 主要依赖组件

- `OneMlp`
- `OneFourier`
- `U_Net (CFD_Benchmark)`

## 主要 Shape 变化

- 预处理前后与 `FNO` 相同
- 中间主干在结构网格分支会进入 `(Batch, n_hidden, *shapelist)` 形式
- `U_Net.multiscale()` 的输入与输出都保持该网格特征形态
- 最终仍会展平回 `(Batch, NumPoints, out_dim)`

## 默认关键参数

- `geotype="structured_2D"`
- `space_dim=2`
- `fun_dim=0` 到若干输入通道
- `out_dim=1` 到若干目标通道
- `n_hidden=32`
- `modes=12`
- `epochs=500`

## 常见修改点

- 跟随数据接口同步修改 `fun_dim / out_dim / shapelist`
- 若要复用 DeepCFD 规则网格数据，优先新增 operator adapter datapipe，而不是改模型 forward
- 若希望更公平地和 `FNO / U_NO / U_Net` 对比，尽量保持同一组数据划分、归一化和评估指标

## 风险点

- 数据接口风险与 `FNO` 相同，不能直接喂 `DeepCFDDatapipe` 的字典 batch
- U_FNO 依赖 `U_Net.multiscale()`，因此一旦 `U_Net` 的空间尺寸约束不满足，问题会传导到 U_FNO
- 当前实现的非结构分支同样主要围绕 2D Geo 投影写法组织

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `FNO`
3. 再看 `U_Net (CFD_Benchmark)`
4. 再看 `OneFourier` 与 `OneMlp`

## 组件契约入口

- `./contracts/onefourier.md`
- `./contracts/onemlp.md`
- `./contracts/oneencoder.md`
- `./contracts/onedecoder.md`
- `./contracts/onehead.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/U_FNO.py`
- `./onescience/src/onescience/models/cfd_benchmark/U_Net.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
