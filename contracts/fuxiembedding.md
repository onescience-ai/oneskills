# Contract: FuxiEmbedding

## 基本信息

- 组件名：`FuxiEmbedding`
- 所属模块族：`embedding`
- 统一入口：`OneEmbedding`
- 注册名：`style="FuxiEmbedding"`

## 组件职责

将多时间步二维气象场组成的三维时空块切分为三维 patch，并投影到统一的 embedding 特征空间。

这是 Fuxi 输入编码阶段的 patch embedding 组件。

补充说明：

- 该组件处理的是三维时空块 `(TimeSteps, Height, Width)`
- 输出是三维 patch 特征图，不是展平 token 序列
- 当前实现不做自动 padding

## 支持输入

- 2D 输入：`not_applicable`
- 3D 输入：`(Batch, in_chans, TimeSteps, Height, Width)`

内部统一做法：

- 使用 `Conv3d` 完成 patch 切分和线性投影
- 可选 `norm_layer` 作用在展平后的 patch token 上
- 最终再恢复回三维 patch 特征图

## 构造参数

- `img_size`
  - 输入空间尺寸 `(TimeSteps, Height, Width)`
- `patch_size`
  - patch 切分尺寸 `(PatchTimeSteps, PatchHeight, PatchWidth)`
- `in_chans`
  - 输入变量通道数
- `embed_dim`
  - 输出特征维度
- `norm_layer`
  - 可选归一化层

## 输出约定

- 2D 输出：`not_applicable`
- 3D 输出：`(Batch, embed_dim, OutTimeSteps, OutHeight, OutWidth)`

其中：

- `OutTimeSteps = TimeSteps // PatchTimeSteps`
- `OutHeight = Height // PatchHeight`
- `OutWidth = Width // PatchWidth`

## 典型调用位置

- Fuxi 主模型输入编码阶段

## 典型参数

- Fuxi 默认配置
  - `img_size=(2, 721, 1440)`
  - `patch_size=(2, 4, 4)`
  - `in_chans=70`
  - `embed_dim=1536`

## 风险点

- 该组件不支持二维输入
- 当前实现不做自动 padding，不能整除的尾部区域不会进入输出特征图
- Fuxi 当前调用层通常要求 `OutTimeSteps = 1`

## 源码锚点

- `./onescience/src/onescience/modules/embedding/fuxiembedding.py`
- `./onescience/src/onescience/modules/embedding/oneembedding.py`
- `./onescience/src/onescience/models/fuxi/fuxi.py`
