# Contract: FuxiTransformer

## 基本信息

- 组件名：`FuxiTransformer`
- 所属模块族：`transformer`
- 统一入口：`OneTransformer`
- 注册名：`style="FuxiTransformer"`

## 组件职责

在二维特征图上执行 Fuxi 的 U 形 trunk 特征提取。

补充说明：

- 输入是二维特征图，不是展平 token 序列
- 内部结构为 `FuxiDownSample -> SwinTransformerV2Stage -> FuxiUpSample`
- 中间会通过 skip connection 将下采样输出与 trunk 输出拼接

## 支持输入

- 2D 输入：`(Batch, embed_dim, Height, Width)`
- 3D 输入：`not_applicable`

内部统一做法：

- 先下采样到较低分辨率网格
- 对不整除窗口大小的 trunk 网格先做 ZeroPad
- `SwinTransformerV2Stage` 输出后做中心 crop
- 与下采样输出拼接后再上采样回原分辨率

## 构造参数

- `embed_dim`
  - 输入与输出特征通道数
- `num_groups`
  - 采样模块 `GroupNorm` 分组数
- `input_resolution`
  - 下采样后 trunk 网格尺寸 `(Height, Width)`
- `num_heads`
  - Swin attention 头数
- `window_size`
  - 局部窗口大小
- `depth`
  - `SwinTransformerV2Stage` 的 block 层数

## 输出约定

- 2D 输出：`(Batch, embed_dim, Height, Width)`
- 3D 输出：`not_applicable`

明确约束：

- 输出 shape 与输入 shape 保持一致
- `input_resolution` 指的是下采样后的 trunk 网格尺寸，不是输入特征图尺寸

## 典型调用位置

- Fuxi 主模型 trunk 部分

## 典型参数

- Fuxi 默认配置
  - `embed_dim=1536`
  - `num_groups=32`
  - `input_resolution=(90, 180)`
  - `num_heads=8`
  - `window_size=7`
  - `depth=48`

## 风险点

- 该组件输入是二维特征图，不是 token 序列
- `input_resolution` 与真实下采样后 trunk 尺寸不一致会导致 padding / crop 逻辑出错
- 上下游通道数必须与 `embed_dim` 对齐

## 下层依赖契约入口

- `./oneskills/codex/contracts/onetransformer.md`
- `./oneskills/codex/contracts/onesample.md`
- `./oneskills/codex/contracts/fuxidownsample.md`
- `./oneskills/codex/contracts/fuxiupsample.md`

## 源码锚点

- `./onescience/src/onescience/modules/transformer/fuxitransformer.py`
- `./onescience/src/onescience/modules/transformer/onetransformer.py`
- `./onescience/src/onescience/models/fuxi/fuxi.py`
