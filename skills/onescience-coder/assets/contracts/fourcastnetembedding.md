# Contract: FourCastNetEmbedding

## 基本信息

- 组件名：`FourCastNetEmbedding`
- 所属模块族：`embedding`
- 统一入口：`OneEmbedding`
- 注册名：`style="FourCastNetEmbedding"`

## 组件职责

将二维气象场切分为不重叠 patch，并投影到统一的 embedding 特征空间。

这是 FourCastNet 输入编码阶段的 patch embedding 组件。

补充说明：

- 该组件只处理二维输入
- 输出是展平后的 patch token 序列
- 调用层通常会再把它恢复成二维 patch 网格，送入 `FourCastNetFuser`

## 支持输入

- 2D 输入：`(Batch, Channels, Height, Width)`
- 3D 输入：`not_applicable`

内部统一做法：

- 使用 `Conv2d` 完成 patch 切分和线性投影
- 不做自动 padding
- 输入空间尺寸必须与构造时的 `img_size` 一致

## 构造参数

- `img_size`
  - 输入空间尺寸 `(Height, Width)`
- `patch_size`
  - patch 切分尺寸 `(PatchHeight, PatchWidth)`
- `in_chans`
  - 输入变量通道数
- `embed_dim`
  - 输出 token 特征维度

## 输出约定

- 2D 输出：`(Batch, NumPatches, embed_dim)`
- 3D 输出：`not_applicable`

其中：

- `PatchGridHeight = Height // PatchHeight`
- `PatchGridWidth = Width // PatchWidth`
- `NumPatches = PatchGridHeight * PatchGridWidth`

## 典型调用位置

- FourCastNet 主模型输入编码阶段

## 典型参数

- FourCastNet 默认配置
  - `img_size=(720, 1440)`
  - `patch_size=(8, 8)`
  - `in_chans=19`
  - `embed_dim=768`

## 风险点

- 该组件不支持三维 `PressureLevels` 输入
- `img_size` 与实际输入不一致会直接报错
- `patch_size` 会直接改变 token 数和后续位置编码长度

## 源码锚点

- `./onescience/src/onescience/modules/embedding/fourcastnetembedding.py`
- `./onescience/src/onescience/modules/embedding/oneembedding.py`
- `./onescience/src/onescience/models/fourcastnet/fourcastnet.py`
