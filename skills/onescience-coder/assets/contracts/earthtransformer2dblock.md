# Contract: EarthTransformer2DBlock

## 基本信息

- 组件名：`EarthTransformer2DBlock`
- 所属模块族：`transformer`
- 统一入口：`OneTransformer`
- 注册名：`style="EarthTransformer2DBlock"`

## 组件职责

在二维 patch 网格上执行带地球位置偏置的局部窗口 Transformer 计算。

补充说明：

- 该组件处理的是展平后的二维 token 序列
- 内部使用 `OneAttention(style="EarthAttention2D")`
- 适合 FengWu encoder / decoder 这类二维 patch-grid block

## 支持输入

- 2D 输入：`(Batch, Height * Width, dim)`
- 3D 输入：`not_applicable`

内部统一做法：

- 先把 token 序列还原为二维网格
- 对不能整除窗口大小的网格先做 padding
- 以普通窗口或 shifted window 执行 `EarthAttention2D`
- 最后 crop 回原网格并接 MLP 残差

## 构造参数

- `dim`
  - 输入与输出 token 特征维度
- `input_resolution`
  - 二维网格尺寸 `(Height, Width)`
- `num_heads`
  - 多头注意力头数
- `window_size`
  - 局部窗口大小 `(WindowHeight, WindowWidth)`
- `shift_size`
  - 循环移位大小 `(ShiftHeight, ShiftWidth)`
- `drop_path`
  - Stochastic Depth 比例
- `mlp_ratio`, `qkv_bias`, `qk_scale`, `drop`, `attn_drop`, `act_layer`, `norm_layer`
  - 标准 Transformer 配置项

## 输出约定

- 2D 输出：`(Batch, Height * Width, dim)`
- 3D 输出：`not_applicable`

如果有明确边界条件，也写在这里：

- 输出 token 数与输入 token 数保持一致
- `input_resolution` 必须与输入 token 数严格对应

## 典型调用位置

- FengWuEncoder 的高分辨率阶段
- FengWuEncoder 的中分辨率阶段
- FengWuDecoder 的中分辨率阶段
- FengWuDecoder 的高分辨率阶段

## 典型参数

- FengWu 默认二维 block
  - `dim=192` 或 `384`
  - `window_size=(6, 12)`
  - `num_heads=6` 或 `12`

## 风险点

- 输入必须是展平 token 序列，不是 `(Batch, Height, Width, dim)` 网格
- `shift_size` 与 `window_size` 的语义不能混淆
- 若调用层的 `input_resolution` 不正确，padding / reverse window 都会错位

## 源码锚点

- `./onescience/src/onescience/modules/transformer/earthtransformer2Dblock.py`
- `./onescience/src/onescience/modules/transformer/onetransformer.py`
- `./onescience/src/onescience/modules/attention/earthattention2d.py`
