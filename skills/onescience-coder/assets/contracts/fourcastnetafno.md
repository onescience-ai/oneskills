# Contract: FourCastNetAFNO2D

## 基本信息

- 组件名：`FourCastNetAFNO2D`
- 所属模块族：`afno`
- 统一入口：`OneAFNO`
- 注册名：`style="FourCastNetAFNO2D"`

## 组件职责

在二维 patch 网格上执行 AFNO 频域混合，实现全局感受野的 token 交互。

补充说明：

- 该组件是 FourCastNetFuser 中替代自注意力的核心混合模块
- 它处理的是二维 patch 网格特征
- 输出会与输入做残差相加

## 支持输入

- 2D 输入：`(Batch, Height, Width, Channels)`
- 3D 输入：`not_applicable`

内部统一做法：

- 先做 `rfft2` 将特征映射到频域
- 对各通道块执行复数 MLP 混合
- 用 `softshrink` 做频域稀疏化
- 再做 `irfft2` 回到空间域

## 构造参数

- `hidden_size`
  - 输入与输出通道数
- `num_blocks`
  - 通道分块数
- `sparsity_threshold`
  - soft shrink 阈值
- `hard_thresholding_fraction`
  - 保留的频率模式比例
- `hidden_size_factor`
  - 频域 MLP 中间层扩展倍数

## 输出约定

- 2D 输出：`(Batch, Height, Width, Channels)`
- 3D 输出：`not_applicable`

明确约束：

- 输入最后一维必须等于 `hidden_size`
- `hidden_size` 必须能被 `num_blocks` 整除

## 典型调用位置

- `FourCastNetFuser` 内部的频域混合层

## 典型参数

- FourCastNet 默认 AFNO
  - `hidden_size=768`
  - `num_blocks=8`
  - `sparsity_threshold=0.01`
  - `hard_thresholding_fraction=1.0`
  - `hidden_size_factor=1`

## 风险点

- 这里的 `Height` 与 `Width` 指的是 patch 网格尺寸，不是原始图像分辨率
- 输入最后一维不匹配会直接报错
- `hard_thresholding_fraction` 会直接影响频域保留范围

## 源码锚点

- `./onescience/src/onescience/modules/afno/fourcastnetafno.py`
- `./onescience/src/onescience/modules/afno/oneafno.py`
- `./onescience/src/onescience/modules/fuser/fourcastnetfuser.py`
