# Contract: EarthTransformer3DBlock

## 基本信息

- 组件名：`EarthTransformer3DBlock`
- 所属模块族：`transformer`
- 统一入口：`OneTransformer`
- 注册名：`style="EarthTransformer3DBlock"`

## 组件职责

在三维 patch 网格上执行带地球位置偏置的局部窗口 Transformer 计算。

补充说明：

- 该组件处理的是展平后的三维 token 序列
- 内部使用 `OneAttention(style="EarthAttention3D")`
- 它是 `PanguFuser` 与 `FengWuFuser` 的底层关键 block

## 支持输入

- 2D 输入：`not_applicable`
- 3D 输入：`(Batch, PressureLevels * Height * Width, dim)`

内部统一做法：

- 先把 token 序列还原为三维网格
- 对不能整除窗口大小的网格先做 padding
- 以普通窗口或 shifted window 执行 `EarthAttention3D`
- 最后 crop 回原网格并接 MLP 残差

## 构造参数

- `dim`
  - 输入与输出 token 特征维度
- `input_resolution`
  - 三维网格尺寸 `(PressureLevels, Height, Width)`
- `num_heads`
  - 多头注意力头数
- `window_size`
  - 局部窗口大小 `(WindowPressureLevels, WindowHeight, WindowWidth)`
- `shift_size`
  - 循环移位大小 `(ShiftPressureLevels, ShiftHeight, ShiftWidth)`
- `drop_path`
  - Stochastic Depth 比例
- `mlp_ratio`, `qkv_bias`, `qk_scale`, `drop`, `attn_drop`, `act_layer`, `norm_layer`
  - 标准 Transformer 配置项

## 输出约定

- 2D 输出：`not_applicable`
- 3D 输出：`(Batch, PressureLevels * Height * Width, dim)`

如果有明确边界条件，也写在这里：

- 输出 token 数与输入 token 数保持一致
- `input_resolution` 必须与输入 token 数严格对应

## 典型调用位置

- PanguFuser 的每一层 block
- FengWuFuser 的每一层 block

## 典型参数

- Pangu 主干
  - `window_size=(2, 6, 12)`
  - `num_heads=6` 或 `12`
- FengWu 三维变量融合
  - `window_size=(2, 6, 12)`
  - `num_heads=12`

## 风险点

- 输入必须是展平 token 序列，不是三维网格张量
- 在 FengWu 中，`input_resolution[0]` 表示变量分支数，不一定是物理气压层数
- 若调用层的 `input_resolution` 不正确，padding / reverse window 会错位

## 源码锚点

- `./onescience/src/onescience/modules/transformer/earthtransformer3Dblock.py`
- `./onescience/src/onescience/modules/transformer/onetransformer.py`
- `./onescience/src/onescience/modules/attention/earthattention3d.py`
