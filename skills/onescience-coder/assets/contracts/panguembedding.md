# Contract: PanguEmbedding

## 基本信息

- 组件名：`PanguEmbedding`
- 所属模块族：`embedding`
- 统一入口：`OneEmbedding`
- 注册名：`style="PanguEmbedding"`

## 组件职责

将二维或三维原始气象场切分为非重叠 patch，并投影到统一的 embedding 特征空间。

这是 Pangu 系列输入编码阶段的统一 patch embedding 组件。

## 支持输入

- 2D 输入：`(Batch, Variables, Height, Width)`
- 3D 输入：`(Batch, Variables, PressureLevels, Height, Width)`

内部统一做法：

- 对 2D 输入先补一个长度为 1 的伪 `PressureLevels`
- 统一走 `Conv3d`
- patch 不整除时自动做零填充

## 构造参数

- `img_size`
  - 2D: `(Height, Width)`
  - 3D: `(PressureLevels, Height, Width)`
- `patch_size`
  - 2D: `(PatchHeight, PatchWidth)`
  - 3D: `(PatchPressureLevels, PatchHeight, PatchWidth)`
- `Variables`
  - 输入变量数
- `embed_dim`
  - 输出特征维度
- `norm_layer`
  - 可选归一化层

## 输出约定

- 2D 输出：`(Batch, embed_dim, OutHeight, OutWidth)`
- 3D 输出：`(Batch, embed_dim, OutPressureLevels, OutHeight, OutWidth)`
 
其中：

- `OutPressureLevels = ceil(PressureLevels / PatchPressureLevels)`
- `OutHeight = ceil(Height / PatchHeight)`
- `OutWidth = ceil(Width / PatchWidth)`

## 典型调用位置

- Pangu 主模型中的地表分支 patch embedding
- Pangu 主模型中的高空分支 patch embedding

## 典型参数

- Surface:
  - `img_size=(721, 1440)`
  - `patch_size=(4, 4)`
  - `Variables=7`
  - `embed_dim=192`
- Upper-air:
  - `img_size=(13, 721, 1440)`
  - `patch_size=(2, 4, 4)`
  - `Variables=5`
  - `embed_dim=192`

## 风险点

- 不要再默认拆成独立 2D/3D embedding 组件
- 若调用层传入 2D 输入，shape 必须是四维张量
- `patch_size` 变化会直接改变输出 token 网格尺寸

## 源码锚点

- `./onescience/src/onescience/modules/embedding/panguembedding.py`
- `./onescience/src/onescience/modules/embedding/oneembedding.py`
