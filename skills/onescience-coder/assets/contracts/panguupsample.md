# Contract: PanguUpSample

## 基本信息

- 组件名：`PanguUpSample`
- 所属模块族：`sample`
- 统一入口：`OneSample`
- 注册名：`style="PanguUpSample"`

## 组件职责

对二维或三维 token 网格做统一上采样，将低分辨率 token 恢复到更高分辨率的空间网格中。

这是 Pangu 系列多尺度解码阶段的统一 token 上采样组件。

## 支持输入

- 2D 输入：`(Batch, Height * Width, in_dim)`
- 3D 输入：`(Batch, PressureLevels * Height * Width, in_dim)`

内部统一做法：

- 对 2D 输入补一个长度为 1 的伪 `PressureLevels`
- 统一按三维网格恢复
- 通过线性层把每个 token 扩展到适合 2×2 子像素重排的形式
- 只在 `Height` 和 `Width` 方向做 2 倍上采样
- 最后按 `output_resolution` 做中心裁剪

## 构造参数

- `input_resolution`
  - 2D: `(Height, Width)`
  - 3D: `(PressureLevels, Height, Width)`
- `output_resolution`
  - 2D: `(OutHeight, OutWidth)`
  - 3D: `(OutPressureLevels, OutHeight, OutWidth)`
- `in_dim`
  - 输入 token 特征维
- `out_dim`
  - 输出 token 特征维

## 输出约定

- 2D 输出：`(Batch, OutHeight * OutWidth, out_dim)`
- 3D 输出：`(Batch, OutPressureLevels * OutHeight * OutWidth, out_dim)`

默认约束：

- `OutHeight <= 2 * Height`
- `OutWidth <= 2 * Width`
- `OutPressureLevels <= PressureLevels`

## 典型调用位置

- Pangu 主模型 decoder 中 `layer3 -> upsample -> layer4`

## 典型参数

- Surface:
  - `input_resolution=(91, 180)`
  - `output_resolution=(181, 360)`
  - `in_dim=384`
  - `out_dim=192`
- Upper-air:
  - `input_resolution=(8, 91, 180)`
  - `output_resolution=(8, 181, 360)`
  - `in_dim=384`
  - `out_dim=192`

## 风险点

- 这是 token 级组件，不接收图像张量
- 输入 token 数必须与 `input_resolution` 完全匹配
- `out_dim` 不是自动推断值，调用层需要明确传入
- 不建议继续回到旧的 `PanguUpSample2D` / `PanguUpSample3D` 依赖方式

## 源码锚点

- `./onescience/src/onescience/modules/sample/panguupsample.py`
- `./onescience/src/onescience/modules/sample/onesample.py`
