# Component Index

## 目标

本文件是 `./contracts/` 的统一入口索引。

作用不是复述源码，而是帮助用户与智能体快速回答下面几个问题：

1. 当前任务应该先看哪个模块族
2. 某个组件应该从哪个 `One*` 入口初始化
3. `style` 应该写什么
4. 当前推荐优先使用哪些组件
5. 契约不足时，下一步应继续看哪一层

## 使用方式

建议按下面顺序检索：

1. 先根据任务语义，确定要查看的模块族
2. 先看高层业务组件，再决定是否继续下钻
3. 只有当高层组件不足以覆盖任务时，再看 `One*` 入口或底层模块
4. 契约仍不足时，再回到 `./onescience/` 对应源码锚点

默认不要一开始就直接检索底层 block 或 attention。

## 分层说明

当前索引按三层组织：

1. 统一入口层
   - `One*` 组件
   - 作用是统一调度、按 `style` 分发、提供调用入口
2. 高层业务组件层
   - 任务里最常直接拼装、替换、组合的组件
3. 底层通用模块层
   - block / attention / afno / fc 等下层实现单元

这三层在目录中保持平铺，在索引中进行逻辑分层。

## 统一入口层

| 组件 | 模块族 | 角色 | 注册方式 | 当前状态 | 契约卡片 |
|---|---|---|---|---|---|
| OneEmbedding | embedding | 统一 embedding 入口 | `style="<EmbeddingStyle>"` | `stable` | `./contracts/oneembedding.md` |
| OneSample | sample | 统一采样入口 | `style="<SampleStyle>"` | `stable` | `./contracts/onesample.md` |
| OneRecovery | recovery | 统一恢复入口 | `style="<RecoveryStyle>"` | `stable` | `./contracts/onerecovery.md` |
| OneFuser | fuser | 统一主干融合入口 | `style="<FuserStyle>"` | `stable` | `./contracts/onefuser.md` |
| OneEncoder | encoder | 统一 encoder 入口 | `style="<EncoderStyle>"` | `stable` | `./contracts/oneencoder.md` |
| OneDecoder | decoder | 统一 decoder 入口 | `style="<DecoderStyle>"` | `stable` | `./contracts/onedecoder.md` |
| OneTransformer | transformer | 统一 transformer 入口 | `style="<TransformerStyle>"` | `stable` | `./contracts/onetransformer.md` |
| OneAttention | attention | 统一 attention 入口 | `style="<AttentionStyle>"` | `stable` | `./contracts/oneattention.md` |
| OneAFNO | afno | 统一 AFNO 入口 | `style="<AFNOStyle>"` | `stable` | `./contracts/oneafno.md` |
| OneFC | fc | 统一前馈层入口 | `style="<FCStyle>"` | `stable` | `./contracts/onefc.md` |
| OneMlp | mlp | 统一 MLP 入口 | `style="<MlpStyle>"` | `stable` | `./contracts/onemlp.md` |
| OneFourier | fourier | 统一谱算子入口 | `style="<FourierStyle>"` | `stable` | `./contracts/onefourier.md` |
| OneHead | head | 统一预测头入口 | `style="<HeadStyle>"` | `stable` | `./contracts/onehead.md` |

## 高层业务组件层

这些组件更适合作为任务实现、模型拼装、模块替换时的优先检索对象。

| 组件 | 模块族 | 调用入口 | 注册名 | 输入形态摘要 | 当前状态 | 契约卡片 |
|---|---|---|---|---|---|---|
| PanguEmbedding | embedding | `OneEmbedding` | `PanguEmbedding` | 2D 场 / 3D 场 | `stable` | `./contracts/panguembedding.md` |
| FourCastNetEmbedding | embedding | `OneEmbedding` | `FourCastNetEmbedding` | 2D 场 | `stable` | `./contracts/fourcastnetembedding.md` |
| FuxiEmbedding | embedding | `OneEmbedding` | `FuxiEmbedding` | 3D 时空块 | `stable` | `./contracts/fuxiembedding.md` |
| PanguDownSample | sample | `OneSample` | `PanguDownSample` | 2D token / 3D token | `stable` | `./contracts/pangudownsample.md` |
| PanguUpSample | sample | `OneSample` | `PanguUpSample` | 2D token / 3D token | `stable` | `./contracts/panguupsample.md` |
| FuxiDownSample | sample | `OneSample` | `FuxiDownSample` | 2D 特征图 | `stable` | `./contracts/fuxidownsample.md` |
| FuxiUpSample | sample | `OneSample` | `FuxiUpSample` | 2D 特征图 | `stable` | `./contracts/fuxiupsample.md` |
| PanguPatchRecovery | recovery | `OneRecovery` | `PanguPatchRecovery` | 2D 特征图 / 3D 特征图 | `stable` | `./contracts/pangupatchrecovery.md` |
| PanguFuser | fuser | `OneFuser` | `PanguFuser` | 3D token | `stable` | `./contracts/pangufuser.md` |
| FourCastNetFuser | fuser | `OneFuser` | `FourCastNetFuser` | 2D patch 网格特征 | `stable` | `./contracts/fourcastnetfuser.md` |
| FengWuFuser | fuser | `OneFuser` | `FengWuFuser` | 3D token | `stable` | `./contracts/fengwufuser.md` |
| FengWuEncoder | encoder | `OneEncoder` | `FengWuEncoder` | 2D 场 | `stable` | `./contracts/fengwuencoder.md` |
| FengWuDecoder | decoder | `OneDecoder` | `FengWuDecoder` | 中分辨率 token + 高分辨率 skip | `stable` | `./contracts/fengwudecoder.md` |
| FuxiTransformer | transformer | `OneTransformer` | `FuxiTransformer` | 2D 特征图 | `stable` | `./contracts/fuxitransformer.md` |

## 底层通用模块层

这些组件通常作为继续下钻时的第二优先级，不建议替代高层业务组件成为默认起点。

| 组件 | 模块族 | 调用入口 | 注册名 | 输入形态摘要 | 当前状态 | 契约卡片 |
|---|---|---|---|---|---|---|
| EarthTransformer2DBlock | transformer | `OneTransformer` | `EarthTransformer2DBlock` | 2D token | `stable` | `./contracts/earthtransformer2dblock.md` |
| EarthTransformer3DBlock | transformer | `OneTransformer` | `EarthTransformer3DBlock` | 3D token | `stable` | `./contracts/earthtransformer3dblock.md` |
| EarthAttention2D | attention | `OneAttention` | `EarthAttention2D` | 2D 窗口化 token | `stable` | `./contracts/earthattention2d.md` |
| EarthAttention3D | attention | `OneAttention` | `EarthAttention3D` | 3D 窗口化 token | `stable` | `./contracts/earthattention3d.md` |
| FourCastNetAFNO2D | afno | `OneAFNO` | `FourCastNetAFNO2D` | 2D patch 网格特征 | `stable` | `./contracts/fourcastnetafno.md` |
| FourCastNetFC | fc | `OneFC` | `FourCastNetFC` | 任意前缀维度的特征张量 | `stable` | `./contracts/fourcastnetfc.md` |
| FuxiFC | fc | `OneFC` | `FuxiFC` | 任意前缀维度的特征张量 | `stable` | `./contracts/fuxifc.md` |

## 按模块族检索

### embedding

适用于：

- patch embedding
- 输入变量编码
- 2D 场或 3D 场的初始 token 化

优先顺序：

1. 先看高层 embedding 组件
2. 再看 `OneEmbedding`

### fuser

适用于：

- 主干特征提取
- token 融合
- trunk 建模
- surface 与 upper-air 联合建模
- 3D token 主干

优先顺序：

1. 先看高层 fuser 组件
2. 再看 `OneFuser`
3. 只有 `fuser` 不适用时，再继续看 transformer / attention

补充约束：

- 若 `fuser` 契约已能覆盖主干需求，默认不要直接从底层 transformer block 手工拼 encoder/decoder 作为首选实现
- 若需要回到底层 block，应先在规格中说明当前 `fuser` 为什么不适用

### sample

适用于：

- 下采样
- 上采样
- 特征分辨率调整
- U 形主干中的尺度变换

优先顺序：

1. 先看具体 downsample / upsample 组件
2. 再看 `OneSample`

### recovery

适用于：

- patch 级输出恢复
- token 到物理场的输出投影
- 2D / 3D 特征恢复到目标变量网格

优先顺序：

1. 先看具体 recovery 组件
2. 再看 `OneRecovery`

### encoder / decoder

适用于：

- 明确的编码器分支
- 明确的解码器分支
- 多分支结构中的局部表征抽取与恢复

优先顺序：

1. 先看具体 encoder / decoder 组件
2. 再看 `OneEncoder` / `OneDecoder`

### transformer / attention / afno / fc

适用于：

- 高层组件无法覆盖时的继续下钻
- 对 block 级或算子级实现进行替换
- 明确需要底层算子能力时

优先顺序：

1. 先看高层组件
2. 再看对应 `One*` 入口
3. 最后看底层模块

### mlp / fourier / head

适用于：

- 神经算子前的点级或 token 级特征映射
- `FNO / U_FNO / U_NO` 这类谱域主干
- `U_Net` 多尺度主干后的输出投影

优先顺序：

1. 先看模型卡，确认具体依赖的是哪一个 wrapper
2. 再看 `OneMlp` / `OneFourier` / `OneHead`
3. 只有 wrapper 契约仍不足时，再回到对应底层实现

## 典型检索顺序

对于天气预测、全球格点预报、surface 与 upper-air 联合建模这类任务，推荐优先顺序：

1. `embedding`
2. `fuser`
3. `sample`
4. `recovery`

在某个模块族内部，再根据以下条件筛选：

- 输入输出 shape 是否匹配
- 是否支持当前任务需要的 2D / 3D 形态
- 调用入口与注册名是否明确
- 契约卡片是否已覆盖当前任务关键参数
- 是否已有相近模型或案例可以参考

## 继续下钻的规则

若上层组件卡片中已经写明“内部依赖某个 block / attention / afno / fc / wrapper”，建议按下面顺序继续下钻：

1. 先看该高层组件自己的契约卡
2. 再看对应 `One*` 入口卡
3. 再看下层 block / attention / afno / fc 卡
4. 只有契约仍不足时，再回到源码

## 建议维护规则

新增组件时，建议至少补以下字段：

- 组件名
- 所属模块族
- 调用入口
- 注册名
- 输入形态摘要
- 当前状态
- 契约卡片

推荐状态值：

- `stable`
- `in_progress`
- `legacy_split`

推荐新增流程：

1. 先复制 `./contracts/TEMPLATE.md`
2. 按组件实际情况填写字段
3. 将新契约文件加入本索引
4. 若该组件已经替代旧实现，在风险点中明确说明

## 相关文档

- `./contracts/TEMPLATE.md`
- `./contracts/naming_convention.md`

## 检索约定

源码锚点统一使用 `./onescience/...` 相对路径。

默认假设：

- ``
- `onescience/`

位于同一工作目录下。
