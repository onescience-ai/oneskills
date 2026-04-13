# Contract: FengWuEncoder

## 基本信息

- 组件名：`FengWuEncoder`
- 所属模块族：`encoder`
- 统一入口：`OneEncoder`
- 注册名：`style="FengWuEncoder"`

## 组件职责

对单个二维变量分支做层次化编码，输出中分辨率 token 序列和高分辨率 skip 特征。

补充说明：

- 该组件处理的是单个变量分支，不负责跨变量融合
- 内部使用 `PanguEmbedding`、`EarthTransformer2DBlock` 和 `PanguDownSample`
- 输出将供 FengWu 主模型中的 `FengWuFuser` 和 `FengWuDecoder` 使用

## 支持输入

- 2D 输入：`(Batch, in_chans, Height, Width)`
- 3D 输入：`not_applicable`

内部统一做法：

- 先做二维 patch embedding
- 在高分辨率 patch 网格上做若干层 2D Transformer
- 保存高分辨率 skip 特征
- 下采样到中分辨率后继续编码

## 构造参数

- `input_resolution`
  - 高分辨率 patch 网格尺寸 `(Height, Width)`
- `middle_resolution`
  - 中分辨率 patch 网格尺寸 `(OutHeight, OutWidth)`
- `in_chans`
  - 输入变量通道数
- `img_size`
  - 原始输入场尺寸 `(Height, Width)`
- `patch_size`
  - patch 切分尺寸 `(PatchHeight, PatchWidth)`
- `dim`
  - 高分辨率特征维度
- `depth`
  - 高分辨率 Transformer block 层数
- `depth_middle`
  - 中分辨率 Transformer block 层数
- `num_heads`
  - 若为二元组，顺序为 `(HighResolutionHeads, MiddleResolutionHeads)`
- `window_size`
  - 二维窗口大小
- `drop_path`
  - 可为单值或按层提供的序列

## 输出约定

- 2D 输出：
  - `x`: `(Batch, middle_resolution[0] * middle_resolution[1], 2 * dim)`
  - `skip`: `(Batch, input_resolution[0], input_resolution[1], dim)`
- 3D 输出：`not_applicable`

## 典型调用位置

- FengWu 主模型中的各变量分支编码器

## 典型参数

- surface 分支
  - `in_chans=4`
  - `img_size=(721, 1440)`
  - `patch_size=(4, 4)`
  - `dim=192`
- 高空变量分支
  - `in_chans=37`
  - `img_size=(721, 1440)`
  - `patch_size=(4, 4)`
  - `dim=192`

## 风险点

- 该组件处理的是单个变量分支，不要直接把多变量拼接输入给它
- `num_heads` 二元组顺序是 `(高分辨率, 中分辨率)`
- 输出 `skip` 是二维网格特征，不是展平 token 序列

## 下层依赖契约入口

- `./contracts/oneencoder.md`
- `./contracts/oneembedding.md`
- `./contracts/panguembedding.md`
- `./contracts/onesample.md`
- `./contracts/pangudownsample.md`
- `./contracts/onetransformer.md`
- `./contracts/earthtransformer2dblock.md`
- `./contracts/oneattention.md`
- `./contracts/earthattention2d.md`

## 源码锚点

- `./onescience/src/onescience/modules/encoder/fengwuencoder.py`
- `./onescience/src/onescience/modules/encoder/oneencoder.py`
- `./onescience/src/onescience/models/fengwu/fengwu.py`
