# Contract: PanguPatchRecovery

## 基本信息

- 组件名：`PanguPatchRecovery`
- 所属模块族：`recovery`
- 统一入口：`OneRecovery`
- 注册名：`style="PanguPatchRecovery"`

## 组件职责

将 patch 级别特征图恢复为原始气象场分辨率，把高维特征映射回最终的二维或三维变量场。

这是 Pangu 系列输出解码阶段的统一 patch recovery 组件。

## 支持输入

- 2D 输入：`(Batch, in_chans, Height, Width)`
- 3D 输入：`(Batch, in_chans, PressureLevels, Height, Width)`

内部统一做法：

- 对 2D 输入补一个长度为 1 的伪 `PressureLevels`
- 统一走 `ConvTranspose3d`
- 恢复后按 `img_size` 做中心裁剪
- 若原始输入为 2D，最后再去掉伪三维层

## 构造参数

- `img_size`
  - 2D: `(Height, Width)`
  - 3D: `(PressureLevels, Height, Width)`
- `patch_size`
  - 2D: `(PatchHeight, PatchWidth)`
  - 3D: `(PatchPressureLevels, PatchHeight, PatchWidth)`
- `in_chans`
  - 输入特征通道数
- `out_chans`
  - 输出变量通道数

## 输出约定

- 2D 输出：`(Batch, out_chans, OutHeight, OutWidth)`
- 3D 输出：`(Batch, out_chans, OutPressureLevels, OutHeight, OutWidth)`

## 典型调用位置

- Pangu 主模型中从融合后的 patch 特征恢复 surface 变量
- Pangu 主模型中从融合后的 patch 特征恢复 upper-air 变量

## 典型参数

- Surface:
  - `img_size=(721, 1440)`
  - `patch_size=(4, 4)`
  - `in_chans=384`
  - `out_chans=7`
- Upper-air:
  - `img_size=(13, 721, 1440)`
  - `patch_size=(2, 4, 4)`
  - `in_chans=384`
  - `out_chans=5`

## 风险点

- 这是特征图级组件，不接收 token 序列
- 输入通道数必须与 `in_chans` 一致
- 若反卷积恢复后的尺寸小于 `img_size`，实现会直接报错
- 对已经统一的 Pangu 分支，优先使用这个统一 recovery，而不是继续区分 `2D/3D` 恢复类

## 源码锚点

- `./onescience/src/onescience/modules/recovery/pangupatchrecovery.py`
- `./onescience/src/onescience/modules/recovery/onerecovery.py`
