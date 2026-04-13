# Model Card: U_Net (CFD_Benchmark)

## 基本信息

- 模型名：`U_Net`
- 任务类型：`CFD / structured-or-geo-projected field surrogate baseline`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/U_Net.py`

## 模型定位

这是 `CFD_Benchmark` 体系下的 `U_Net` 基线，用统一的 `(x, fx)` 接口把位置与场特征映射到 U-shape 多尺度主干中，再输出目标场。

补充说明：

- 它和 `./onescience/src/onescience/models/deepcfd/UNet.py` 不是同一个模型族
- 当前这个 `U_Net` 走的是 `CFD_Benchmark` 训练接口，目标是和 `FNO / U_FNO / U_NO` 共享同一套 benchmark 风格输入
- 它最重要的结构特点是：用 `OneEncoder + OneDecoder + OneHead` 组织一个多尺度 U-shape 主干

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)` 或 `None`
- 输入变量组织：
  - `x`
    - 坐标或规则网格位置
  - `fx`
    - 与位置对齐的输入场特征

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 每个位置对应 `out_dim` 个目标变量

## 主干结构

- `OneMlp(style="StandardMLP")`
- 可选 `OneFourier(style="GeoSpectralConv2d")`
- `OneEncoder(style="UNetEncoder{dim}D")`
- `OneDecoder(style="UNetDecoder{dim}D")`
- `OneHead(style="UNetHead{dim}D")`
- `fc1 + fc2`

## 主要依赖组件

- `OneMlp`
- `OneFourier`
- `OneEncoder`
- `OneDecoder`
- `OneHead`

## 主要 Shape 变化

- 预处理后：`(Batch, NumPoints, n_hidden)`
- 结构分支重排后：`(Batch, n_hidden, *shapelist)`
- 经过 U-shape 多尺度主干后保持网格张量形式
- 展平后恢复为：`(Batch, NumPoints, n_hidden)`
- 输出投影后：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `task="steady"`
- `geotype="structured_2D"`
- `space_dim=2`
- `fun_dim=0` 到若干输入通道
- `out_dim=1` 到若干目标通道
- `n_hidden=32`
- `unified_pos=0`

## 常见修改点

- 明确当前要比较的是这个 `CFD_Benchmark U_Net`，还是 `DeepCFD UNet / UNetEx`
- 若接 DeepCFD 数据，优先新增 operator adapter datapipe，再复用这个 `U_Net`
- 若只想做 CNN 图像基线，通常更接近 `DeepCFD` 目录下那套 `UNet/UNetEx` 训练流，而不是当前模型

## 风险点

- “U-Net” 在 OneScience 里至少有两套实现；如果第一轮不区分，很容易选错训练入口
- 当前模型不能直接消费 `DeepCFDDatapipe` 的字典 batch
- 当前非结构分支同样主要围绕 2D Geo 投影组织，不要默认它天然适配任意 3D 非结构任务

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `OneEncoder`
3. 再看 `OneDecoder`
4. 再看 `OneHead`
5. 最后再看 `OneMlp / OneFourier`

## 组件契约入口

- `./contracts/oneencoder.md`
- `./contracts/onedecoder.md`
- `./contracts/onehead.md`
- `./contracts/onemlp.md`
- `./contracts/onefourier.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/U_Net.py`
- `./onescience/src/onescience/modules/encoder/oneencoder.py`
- `./onescience/src/onescience/modules/decoder/onedecoder.py`
- `./onescience/src/onescience/modules/head/onehead.py`
