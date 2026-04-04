# Component Index

## 目标

本文件用于登记当前推荐优先检索的组件。

这里不限定某个具体模型族，后续可以持续追加新的组件。

建议将它作为：

- 组件入口索引
- 当前推荐组件登记表
- 任务执行时的优先检索页

## 建议维护规则

新增组件时，建议至少补以下字段：

- 组件名
- 所属模块族
- 调用入口
- 注册名
- 输入形态摘要
- 当前状态
- 对应契约卡片

推荐状态值：

- `stable`
- `in_progress`
- `legacy_split`

推荐新增流程：

1. 先复制 `./oneskills/codex/contracts/TEMPLATE.md`
2. 按组件实际情况填写字段
3. 将新契约文件加入本表
4. 若该组件已经替代旧的分裂实现，在风险点中明确写出

## 当前已登记组件

| 组件 | 模块族 | 调用入口 | 注册名 | 输入形态摘要 | 状态 | 契约卡片 |
|---|---|---|---|---|---|---|
| OneEmbedding | embedding | `direct_import` | `style="<EmbeddingStyle>"` | 按 `style` 分发 | `stable` | `./oneskills/codex/contracts/oneembedding.md` |
| PanguEmbedding | embedding | `OneEmbedding` | `PanguEmbedding` | 2D 场 / 3D 场 | `stable` | `./oneskills/codex/contracts/panguembedding.md` |
| FourCastNetEmbedding | embedding | `OneEmbedding` | `FourCastNetEmbedding` | 2D 场 | `stable` | `./oneskills/codex/contracts/fourcastnetembedding.md` |
| FuxiEmbedding | embedding | `OneEmbedding` | `FuxiEmbedding` | 3D 时空块 | `stable` | `./oneskills/codex/contracts/fuxiembedding.md` |
| OneSample | sample | `direct_import` | `style="<SampleStyle>"` | 按 `style` 分发 | `stable` | `./oneskills/codex/contracts/onesample.md` |
| PanguDownSample | sample | `OneSample` | `PanguDownSample` | 2D token / 3D token | `stable` | `./oneskills/codex/contracts/pangudownsample.md` |
| PanguUpSample | sample | `OneSample` | `PanguUpSample` | 2D token / 3D token | `stable` | `./oneskills/codex/contracts/panguupsample.md` |
| FuxiDownSample | sample | `OneSample` | `FuxiDownSample` | 2D 特征图 | `stable` | `./oneskills/codex/contracts/fuxidownsample.md` |
| FuxiUpSample | sample | `OneSample` | `FuxiUpSample` | 2D 特征图 | `stable` | `./oneskills/codex/contracts/fuxiupsample.md` |
| OneRecovery | recovery | `direct_import` | `style="<RecoveryStyle>"` | 按 `style` 分发 | `stable` | `./oneskills/codex/contracts/onerecovery.md` |
| PanguPatchRecovery | recovery | `OneRecovery` | `PanguPatchRecovery` | 2D 特征图 / 3D 特征图 | `stable` | `./oneskills/codex/contracts/pangupatchrecovery.md` |
| OneFuser | fuser | `direct_import` | `style="<FuserStyle>"` | 按 `style` 分发 | `stable` | `./oneskills/codex/contracts/onefuser.md` |
| PanguFuser | fuser | `OneFuser` | `PanguFuser` | 3D token | `stable` | `./oneskills/codex/contracts/pangufuser.md` |
| FourCastNetFuser | fuser | `OneFuser` | `FourCastNetFuser` | 2D patch 网格特征 | `stable` | `./oneskills/codex/contracts/fourcastnetfuser.md` |
| FengWuFuser | fuser | `OneFuser` | `FengWuFuser` | 3D token | `stable` | `./oneskills/codex/contracts/fengwufuser.md` |
| OneEncoder | encoder | `direct_import` | `style="<EncoderStyle>"` | 按 `style` 分发 | `stable` | `./oneskills/codex/contracts/oneencoder.md` |
| FengWuEncoder | encoder | `OneEncoder` | `FengWuEncoder` | 2D 场 | `stable` | `./oneskills/codex/contracts/fengwuencoder.md` |
| OneDecoder | decoder | `direct_import` | `style="<DecoderStyle>"` | 按 `style` 分发 | `stable` | `./oneskills/codex/contracts/onedecoder.md` |
| FengWuDecoder | decoder | `OneDecoder` | `FengWuDecoder` | 中分辨率 token + 高分辨率 skip | `stable` | `./oneskills/codex/contracts/fengwudecoder.md` |
| OneTransformer | transformer | `direct_import` | `style="<TransformerStyle>"` | 按 `style` 分发 | `stable` | `./oneskills/codex/contracts/onetransformer.md` |
| EarthTransformer2DBlock | transformer | `OneTransformer` | `EarthTransformer2DBlock` | 2D token | `stable` | `./oneskills/codex/contracts/earthtransformer2dblock.md` |
| EarthTransformer3DBlock | transformer | `OneTransformer` | `EarthTransformer3DBlock` | 3D token | `stable` | `./oneskills/codex/contracts/earthtransformer3dblock.md` |
| FuxiTransformer | transformer | `OneTransformer` | `FuxiTransformer` | 2D 特征图 | `stable` | `./oneskills/codex/contracts/fuxitransformer.md` |
| OneAttention | attention | `direct_import` | `style="<AttentionStyle>"` | 按 `style` 分发 | `stable` | `./oneskills/codex/contracts/oneattention.md` |
| EarthAttention2D | attention | `OneAttention` | `EarthAttention2D` | 2D 窗口化 token | `stable` | `./oneskills/codex/contracts/earthattention2d.md` |
| EarthAttention3D | attention | `OneAttention` | `EarthAttention3D` | 3D 窗口化 token | `stable` | `./oneskills/codex/contracts/earthattention3d.md` |
| OneAFNO | afno | `direct_import` | `style="<AFNOStyle>"` | 按 `style` 分发 | `stable` | `./oneskills/codex/contracts/oneafno.md` |
| FourCastNetAFNO2D | afno | `OneAFNO` | `FourCastNetAFNO2D` | 2D patch 网格特征 | `stable` | `./oneskills/codex/contracts/fourcastnetafno.md` |
| OneFC | fc | `direct_import` | `style="<FCStyle>"` | 按 `style` 分发 | `stable` | `./oneskills/codex/contracts/onefc.md` |
| FourCastNetFC | fc | `OneFC` | `FourCastNetFC` | 任意前缀维度的特征张量 | `stable` | `./oneskills/codex/contracts/fourcastnetfc.md` |
| FuxiFC | fc | `OneFC` | `FuxiFC` | 任意前缀维度的特征张量 | `stable` | `./oneskills/codex/contracts/fuxifc.md` |

## 天气预测任务的优先检索提示

当任务描述中出现以下语义时，建议优先检查 `fuser` 模块族，而不是先查底层 block：

- 主干特征提取
- token 融合
- 中间层建模
- 全球天气 trunk
- surface 与 upper-air 联合建模
- 3D token 主干

对于天气预测任务，常见的推荐检索顺序是：

1. 先检查 `embedding` 模块族
2. 再检查 `fuser` 模块族
3. 再检查 `sample` 模块族
4. 最后检查 `recovery` 模块族

若上层组件卡片已经说明“内部依赖某个 block / attention / wrapper”，建议继续按下面顺序下钻：

1. 先看该组件自己的契约卡
2. 再看对应 `One*` 入口卡
3. 再看下层 block / attention / afno / fc 卡
4. 只有契约仍不足时，再回到源码

在某个模块族内部，再根据以下条件选择具体组件：

- 输入输出 shape 是否匹配
- 是否支持当前任务需要的 2D / 3D 形态
- 调用入口与注册名是否明确
- 契约卡片是否已覆盖当前任务的关键参数
- 是否已有相近任务案例可参考

补充约束：

- 若 `fuser` 契约已能覆盖主干需求，默认不要直接从 `transformer_layers` 拼 encoder/decoder 作为首选实现
- 若需要回到底层 block，应该先在规格中说明为什么当前 `fuser` 组件不适用

## 组件索引的作用

优先用它解决以下问题：

1. 某个组件该从哪个 `One*` 入口初始化
2. `style` 该写什么
3. 当前优先推荐看哪几张契约卡
4. 某个模块族下目前有哪些高频组件

## 契约模板

组件契约建议固定使用：

- `./oneskills/codex/contracts/TEMPLATE.md`
- `./oneskills/codex/contracts/naming_convention.md`

## 调用层建议

- 写新代码时，优先依赖当前推荐组件
- 改老代码时，优先替换调用层，而不是先改底层实现
- 如果用户只关心“怎么接起来”，优先看单组件契约卡片，不必先读源码

## 适合优先用契约解决的问题

- 某个组件该从哪个 `One*` 入口初始化
- `style` 该写什么
- 参数最小集合是什么
- 输入输出 shape 怎么对齐
- surface 分支和 upper-air 分支典型配置是什么

## 需要回到源码确认的问题

- 具体 padding 或 crop 细节
- 运行时边界条件和异常信息
- 某个模型文件中的精确调用顺序
- 最近一次重构后的实现差异

## 检索约定

源码锚点统一使用 `./onescience/...` 相对路径。

默认假设：

- `oneskills/`
- `onescience/`

位于同一工作目录下。
