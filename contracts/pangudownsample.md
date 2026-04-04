# Contract: PanguDownSample

## 基本信息

- 组件名：`PanguDownSample`
- 所属模块族：`sample`
- 统一入口：`OneSample`
- 注册名：`style="PanguDownSample"`

## 组件职责

对二维或三维 token 网格做统一下采样，将相邻 2×2 空间邻域聚合到更低分辨率的 token 表示中。

这是 Pangu 系列多尺度编码阶段的统一 token 下采样组件。

## 支持输入

- 2D 输入：`(Batch, Height * Width, in_dim)`
- 3D 输入：`(Batch, PressureLevels * Height * Width, in_dim)`

内部统一做法：

- 对 2D 输入补一个长度为 1 的伪 `PressureLevels`
- 统一按三维网格恢复
- 只在 `Height` 和 `Width` 方向做 2×2 聚合
- 必要时自动做空间 padding

## 构造参数

- `input_resolution`
  - 2D: `(Height, Width)`
  - 3D: `(PressureLevels, Height, Width)`
- `output_resolution`
  - 2D: `(OutHeight, OutWidth)`
  - 3D: `(OutPressureLevels, OutHeight, OutWidth)`
- `in_dim`
  - 输入 token 特征维

## 输出约定

- 2D 输出：`(Batch, OutHeight * OutWidth, 2 * in_dim)`
- 3D 输出：`(Batch, OutPressureLevels * OutHeight * OutWidth, 2 * in_dim)`

默认约束：

- 只对空间维做 2 倍下采样
- `PressureLevels` 维通常保持不变

## 典型调用位置

- Pangu 主模型 encoder 中 `layer1 -> downsample -> layer2`

## 典型参数

- Surface:
  - `input_resolution=(181, 360)`
  - `output_resolution=(91, 180)`
  - `in_dim=192`
- Upper-air:
  - `input_resolution=(8, 181, 360)`
  - `output_resolution=(8, 91, 180)`
  - `in_dim=192`

## 风险点

- 这是 token 级组件，不接收图像张量
- 输入 token 数必须与 `input_resolution` 完全匹配
- 输出特征维固定为 `2 * in_dim`

## 源码锚点

- `./onescience/src/onescience/modules/sample/pangudownsample.py`
- `./onescience/src/onescience/modules/sample/onesample.py`
