# Model Card: Fuxi

## 基本信息

- 模型名：`Fuxi`
- 任务类型：`多时间步二维气象场预报`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/fuxi/fuxi.py`

## 模型定位

Fuxi 是一个面向多时间步二维气象场的时空 patch 模型，适合将多个时间步的二维变量场联合编码后做二维 trunk 建模。

补充说明：

- 输入沿 `TimeSteps` 维堆叠，但 trunk 本身只处理二维特征图
- 它不是 Pangu 那种三维 token 主干，也不是 FourCastNet 那种纯二维单时刻输入
- 它最重要的结构特点是：三维 patch embedding 先把时间维压缩到 1，再交给二维 U 形 trunk

## 输入定义

- 输入 shape：`(Batch, in_chans, TimeSteps, Height, Width)`
- 输入变量组织：
  - 单一路径输入
  - 多个时间步的二维气象场沿 `TimeSteps` 维堆叠

## 输出定义

- 输出 shape：`(Batch, out_chans, Height, Width)`
- 输出变量组织：
  - 输出为单时刻二维目标场
  - 最终通过 patch 重排恢复，再用双线性插值对齐到目标分辨率

## 主干结构

- `OneEmbedding(style="FuxiEmbedding")`
  - 将 `(TimeSteps, Height, Width)` 三维时空块映射为 patch 特征
- `squeeze(TimeSteps)`
  - 当前实现要求 embedding 后时间维为 1
- `OneTransformer(style="FuxiTransformer")`
  - `FuxiDownSample -> SwinTransformerV2Stage -> FuxiUpSample`
- `OneFC(style="FuxiFC")`
  - 生成 patch 级输出变量
- patch 重排恢复二维网格
- `F.interpolate(..., mode="bilinear")`
  - 对齐到目标 `(Height, Width)`

## 主要依赖组件

- `FuxiEmbedding`
- `FuxiTransformer`
- `FuxiDownSample`
- `FuxiUpSample`
- `FuxiFC`

## 主要 Shape 变化

- 输入：`(Batch, 70, 2, 721, 1440)`
- embedding 输出：`(Batch, 1536, 1, 180, 360)`
- squeeze 后 trunk 输入：`(Batch, 1536, 180, 360)`
- trunk 内下采样后：`(Batch, 1536, 90, 180)`
- trunk 输出：`(Batch, 1536, 180, 360)`
- FC 输出：`(Batch, 180, 360, 70 * 4 * 4)`
- patch 重排后：`(Batch, 70, 720, 1440)`
- 双线性插值后输出：`(Batch, 70, 721, 1440)`

## 默认关键参数

- `img_size=(2, 721, 1440)`
- `patch_size=(2, 4, 4)`
- `in_chans=70`
- `out_chans=70`
- `embed_dim=1536`
- `num_groups=32`
- `num_heads=8`
- `window_size=7`

## 常见修改点

- 修改 `img_size` 或 `patch_size` 时，要同步检查 embedding 后时间维是否仍为 1
- 修改 `embed_dim` 时，要同步检查 `FuxiTransformer` 与 `FuxiFC` 的维度对齐
- 修改空间 patch 尺寸时，要同步检查 patch 重排尺寸和最终插值逻辑

## 风险点

- 当前实现要求 `patch_size[0] == img_size[0]`，否则 embedding 后时间维不一定能被 `squeeze`
- `FuxiTransformer` 处理的是二维特征图，不是展平 token 序列
- patch 重排阶段使用的是 embedding 后的二维网格尺寸，不是原始输入分辨率

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `FuxiEmbedding` 和 `FuxiTransformer`
3. 若需要理解 trunk 内部，再看 `FuxiDownSample`、`FuxiUpSample`、`FuxiFC`
4. 若仍有疑问，再回到模型源码

## 组件契约入口

- `./oneskills/codex/contracts/fuxiembedding.md`
- `./oneskills/codex/contracts/fuxitransformer.md`
- `./oneskills/codex/contracts/fuxidownsample.md`
- `./oneskills/codex/contracts/fuxiupsample.md`
- `./oneskills/codex/contracts/fuxifc.md`

## 源码锚点

- `./onescience/src/onescience/models/fuxi/fuxi.py`
- `./onescience/src/onescience/modules/transformer/fuxitransformer.py`
- `./onescience/src/onescience/modules/embedding/fuxiembedding.py`
- `./onescience/src/onescience/modules/sample/fuxidownsample.py`
