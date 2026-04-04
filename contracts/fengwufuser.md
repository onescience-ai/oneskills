# Contract: FengWuFuser

## 基本信息

- 组件名：`FengWuFuser`
- 所属模块族：`fuser`
- 统一入口：`OneFuser`
- 注册名：`style="FengWuFuser"`

## 组件职责

在三维网格 `(Variables, Height, Width)` 上融合多个变量分支的中分辨率特征。

补充说明：

- 该组件处理的是展平后的三维 token 序列
- 输入通常来自多个 `FengWuEncoder` 输出的中分辨率特征拼接
- 内部使用多层 `EarthTransformer3DBlock`

## 支持输入

- 2D 输入：`not_applicable`
- 3D 输入：`(Batch, Variables * Height * Width, dim)`

内部统一做法：

- 将输入视作三维网格对应的 token 序列
- 在 `(Variables, Height, Width)` 三维网格上堆叠多层 3D Transformer
- 输出 token 数与输入保持一致

## 构造参数

- `input_resolution`
  - 三维网格尺寸 `(Variables, Height, Width)`
- `dim`
  - 输入与输出 token 特征维度
- `depth`
  - 3D Transformer block 层数
- `num_heads`
  - 多头自注意力头数
- `window_size`
  - 三维窗口大小 `(VariablesWindow, HeightWindow, WidthWindow)`
- `drop_path`
  - 可为单值或按层提供的序列
- `mlp_ratio`, `qkv_bias`, `qk_scale`, `drop`, `attn_drop`, `norm_layer`
  - 标准 Transformer 配置项

## 输出约定

- 2D 输出：`not_applicable`
- 3D 输出：`(Batch, Variables * Height * Width, dim)`

明确约束：

- 输出 token 数与输入 token 数保持一致
- `input_resolution` 必须与输入 token 数严格对应

## 典型调用位置

- FengWu 主模型中的跨变量中分辨率融合阶段

## 典型参数

- FengWu 默认配置
  - `input_resolution=(6, 91, 180)`
  - `dim=384`
  - `depth=6`
  - `num_heads=12`
  - `window_size=(2, 6, 12)`

## 风险点

- 该组件输入是展平 token 序列，不是二维网格特征
- 第一维 `Variables` 指的是变量分支数，不是时间步或气压层
- `dim` 必须与 encoder 输出和 decoder 输入保持一致

## 下层依赖契约入口

- `./oneskills/codex/contracts/onefuser.md`
- `./oneskills/codex/contracts/onetransformer.md`
- `./oneskills/codex/contracts/earthtransformer3dblock.md`
- `./oneskills/codex/contracts/oneattention.md`
- `./oneskills/codex/contracts/earthattention3d.md`

## 源码锚点

- `./onescience/src/onescience/modules/fuser/fengwufuser.py`
- `./onescience/src/onescience/modules/fuser/onefuser.py`
- `./onescience/src/onescience/models/fengwu/fengwu.py`
