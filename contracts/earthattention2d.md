# Contract: EarthAttention2D

## 基本信息

- 组件名：`EarthAttention2D`
- 所属模块族：`attention`
- 统一入口：`OneAttention`
- 注册名：`style="EarthAttention2D"`

## 组件职责

在二维窗口化 token 上执行带地球位置偏置的多头自注意力。

补充说明：

- 该组件处理的是已经完成窗口划分后的 token 张量
- 经度方向窗口数量折叠进 batch 维
- 它是 `EarthTransformer2DBlock` 的底层注意力核心

## 支持输入

- 2D 输入：`(Batch * NumWidthWindows, NumHeightWindows, WindowTokens, dim)`
- 3D 输入：`not_applicable`

内部统一做法：

- 对窗口内 token 做 QKV 投影
- 加入地球位置偏置
- 如有 shifted-window mask，则在 softmax 前叠加
- 输出 shape 与输入保持一致

## 构造参数

- `dim`
  - 输入与输出特征维度
- `input_resolution`
  - padding 后二维网格尺寸 `(Height, Width)`
- `window_size`
  - 窗口大小 `(WindowHeight, WindowWidth)`
- `num_heads`
  - 多头注意力头数
- `qkv_bias`, `qk_scale`, `attn_drop`, `proj_drop`
  - 标准注意力配置项

## 输出约定

- 2D 输出：`(Batch * NumWidthWindows, NumHeightWindows, WindowTokens, dim)`
- 3D 输出：`not_applicable`

如果有明确边界条件，也写在这里：

- `dim` 必须能被 `num_heads` 整除
- 输入必须已经完成窗口划分，不能直接传原始二维网格

## 典型调用位置

- EarthTransformer2DBlock

## 典型参数

- FengWu 默认二维窗口注意力
  - `window_size=(6, 12)`
  - `num_heads=6` 或 `12`

## 风险点

- 这是窗口化 attention，不是原始网格 attention
- `mask` 的窗口布局必须与输入窗口布局完全一致
- `input_resolution` 应该使用 padding 后的网格尺寸

## 源码锚点

- `./onescience/src/onescience/modules/attention/earthattention2d.py`
- `./onescience/src/onescience/modules/attention/oneattention.py`
- `./onescience/src/onescience/modules/transformer/earthtransformer2Dblock.py`
