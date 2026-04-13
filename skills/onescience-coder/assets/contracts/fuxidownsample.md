# Contract: FuxiDownSample

## 基本信息

- 组件名：`FuxiDownSample`
- 所属模块族：`sample`
- 统一入口：`OneSample`
- 注册名：`style="FuxiDownSample"`

## 组件职责

对二维特征图做 2 倍下采样，并通过残差卷积块细化特征。

补充说明：

- 该组件处理的是二维特征图
- 不是 token 序列
- 通常位于 `FuxiTransformer` 的开头

## 支持输入

- 2D 输入：`(Batch, in_chans, Height, Width)`
- 3D 输入：`not_applicable`

内部统一做法：

- 先用步长为 2 的卷积下采样
- 再通过若干残差卷积块细化特征
- 若输入为奇数尺寸，尾部多出的 1 行或 1 列会在输出中裁掉

## 构造参数

- `in_chans`
  - 输入特征通道数
- `out_chans`
  - 输出特征通道数
- `num_groups`
  - `GroupNorm` 分组数
- `num_residuals`
  - 残差卷积块数量

## 输出约定

- 2D 输出：`(Batch, out_chans, OutHeight, OutWidth)`
- 3D 输出：`not_applicable`

其中：

- `OutHeight = Height // 2`
- `OutWidth = Width // 2`

## 典型调用位置

- `FuxiTransformer` 内部的下采样阶段

## 典型参数

- Fuxi 默认配置
  - `in_chans=1536`
  - `out_chans=1536`
  - `num_groups=32`
  - `num_residuals=2`

## 风险点

- 该组件不处理 token 序列
- 奇数尺寸输入会被裁到偶数下采样结果
- `num_groups` 需要与 `out_chans` 匹配

## 源码锚点

- `./onescience/src/onescience/modules/sample/fuxidownsample.py`
- `./onescience/src/onescience/modules/sample/onesample.py`
- `./onescience/src/onescience/modules/transformer/fuxitransformer.py`
