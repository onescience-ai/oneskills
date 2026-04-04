# Contract: EarthAttention3D

## 基本信息

- 组件名：`EarthAttention3D`
- 所属模块族：`attention`
- 统一入口：`OneAttention`
- 注册名：`style="EarthAttention3D"`

## 组件职责

在三维窗口化 token 上执行带地球位置偏置的多头自注意力。

补充说明：

- 该组件处理的是已经完成窗口划分后的三维 token 张量
- 经度方向窗口数量折叠进 batch 维
- `(PressureLevels, Height)` 方向窗口数量合并为一维
- 它是 `EarthTransformer3DBlock` 的底层注意力核心

## 支持输入

- 2D 输入：`not_applicable`
- 3D 输入：`(Batch * NumWidthWindows, NumPressureHeightWindows, WindowTokens, dim)`

内部统一做法：

- 对窗口内 token 做 QKV 投影
- 加入地球位置偏置
- 如有 shifted-window mask，则在 softmax 前叠加
- 输出 shape 与输入保持一致

## 构造参数

- `dim`
  - 输入与输出特征维度
- `input_resolution`
  - padding 后三维网格尺寸 `(PressureLevels, Height, Width)`
- `window_size`
  - 窗口大小 `(WindowPressureLevels, WindowHeight, WindowWidth)`
- `num_heads`
  - 多头注意力头数
- `qkv_bias`, `qk_scale`, `attn_drop`, `proj_drop`
  - 标准注意力配置项

## 输出约定

- 2D 输出：`not_applicable`
- 3D 输出：`(Batch * NumWidthWindows, NumPressureHeightWindows, WindowTokens, dim)`

如果有明确边界条件，也写在这里：

- `dim` 必须能被 `num_heads` 整除
- 输入必须已经完成窗口划分，不能直接传原始三维网格

## 典型调用位置

- EarthTransformer3DBlock

## 典型参数

- Pangu / FengWu 默认三维窗口注意力
  - `window_size=(2, 6, 12)`
  - `num_heads=6` 或 `12`

## 风险点

- 这是窗口化 attention，不是原始三维网格 attention
- 在 FengWu 里，第一维窗口不一定对应真实气压层，也可能对应变量分支
- `input_resolution` 应该使用 padding 后的三维网格尺寸

## 源码锚点

- `./onescience/src/onescience/modules/attention/earthattention3d.py`
- `./onescience/src/onescience/modules/attention/oneattention.py`
- `./onescience/src/onescience/modules/transformer/earthtransformer3Dblock.py`
