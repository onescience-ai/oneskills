# Model Card: U_NO

## 基本信息

- 模型名：`U_NO`
- 任务类型：`CFD / steady-or-dynamic operator learning with U-shaped neural operator`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/U_NO.py`

## 模型定位

U_NO 把 U-Net 的多尺度下采样/上采样结构和 Fourier Neural Operator 结合在一起，适合作为比 `FNO` 更强、比纯 CNN 更偏算子建模的中间基线。

补充说明：

- 输入输出接口仍然遵循 `CFD_Benchmark` 的 `(x, fx) -> y`
- 它和 `U_FNO` 的区别是：`U_NO` 把谱算子更深地嵌入到了整条 U-shape 的下采样和上采样路径里
- 在稳态任务下默认更倾向使用 `bn`，动态任务下会切成 `in`

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)` 或 `None`
- 输入变量组织：
  - `x`
    - 位置或规则网格坐标
  - `fx`
    - 点级或网格级输入物理特征

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 每个位置对应 `out_dim` 个目标变量

## 主干结构

- `OneMlp(style="StandardMLP")`
- 可选 `OneFourier(style="GeoSpectralConv2d")`
- U-shape 编码器与解码器路径
- 每个尺度处插入 `FNOSpectralConv*d + residual`
- `fc1 + fc2`

## 主要依赖组件

- `OneMlp`
- `OneFourier`
- `unet_layer` 系列基础层

## 主要 Shape 变化

- 预处理后：`(Batch, NumPoints, n_hidden)`
- 结构分支重排后：`(Batch, n_hidden, *shapelist)`
- 下采样路径中空间分辨率逐层降低、通道数提升
- 上采样路径中再逐层恢复
- 最终展平回：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `task="steady"`
- `geotype="structured_2D"`
- `space_dim=2`
- `fun_dim=0` 到若干输入通道
- `out_dim=1` 到若干目标通道
- `n_hidden=32`
- `modes=12`

## 常见修改点

- 根据目标任务修改 `fun_dim / out_dim`
- 根据规则网格分辨率修改 `shapelist`
- 若要接 DeepCFD 数据，通常需要把图像输入展平为 `pos + fx`，而不是直接把 `(Batch, Channels, Height, Width)` 送进 `forward`
- 若比较多个模型，优先统一训练脚本和评价指标，再去调模型容量

## 风险点

- 相比 `FNO`，U_NO 对空间尺度链路更敏感，`shapelist` 与 padding 不一致时更容易在 U-shape 中间层报错
- 直接复用 DeepCFD datapipe 不可行，必须桥接批格式
- 当前实现的非结构分支仍主要围绕 2D Geo 投影

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `OneFourier`
3. 再看 `OneMlp`
4. 若需要确认下采样/上采样细节，再回到 `unet_layer`

## 组件契约入口

- `./contracts/onefourier.md`
- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/U_NO.py`
- `./onescience/src/onescience/modules/fourier/onefourier.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
