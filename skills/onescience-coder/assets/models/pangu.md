# Model Card: Pangu

## 基本信息

- 模型名：`Pangu`
- 任务类型：`全球天气预报 / 多层大气与地表联合建模`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/pangu/pangu.py`

## 模型定位

Pangu 是一个面向全球天气预报的三维主干模型，适合同时建模地表变量和多气压层大气变量。

补充说明：

- 输入不是单一路径，而是 surface 与 upper-air 两类变量联合组织
- 主干不是独立的 surface / upper-air 双 encoder，而是统一的 3D token trunk
- 它最重要的结构特点是：surface patch token 先补一个 `PressureLevels=1`，再与 upper-air patch token 拼成统一三维网格

## 输入定义

- 输入 shape：`(Batch, 4 + 3 + 5 * 13, Height, Width)`
- 输入变量组织：
  - 前 4 个通道：待预测的 surface 变量
  - 接下来的 3 个通道：静态 mask
  - 剩余 `5 * 13` 个通道：upper-air 变量按气压层展平

## 输出定义

- 输出 shape：
  - surface: `(Batch, 4, Height, Width)`
  - upper-air: `(Batch, 5, 13, Height, Width)`
- 输出变量组织：
  - surface 输出只包含 4 个待预测 surface 变量
  - 3 个静态 mask 仅参与输入，不属于输出目标

## 主干结构

- `OneEmbedding(style="PanguEmbedding")`
  - 分别编码 surface 与 upper-air
- `surface.unsqueeze(2)` 与 upper-air 特征沿 `PressureLevels` 维拼接
- `OneFuser(style="PanguFuser")`
  - `layer1 -> downsample -> layer2 -> layer3 -> upsample -> layer4`
- `OneRecovery(style="PanguPatchRecovery")`
  - 分别恢复 surface 与 upper-air 目标场

## 主要依赖组件

- `PanguEmbedding`
- `PanguFuser`
- `PanguDownSample`
- `PanguUpSample`
- `PanguPatchRecovery`

## 主要 Shape 变化

- surface 输入：`(Batch, 7, 721, 1440)`
- upper-air 输入：`(Batch, 5, 13, 721, 1440)`
- surface embedding 输出：`(Batch, 192, 181, 360)`
- upper-air embedding 输出：`(Batch, 192, 7, 181, 360)`
- 拼接后主干输入：`(Batch, 192, 8, 181, 360)`
- 展平后 trunk token：`(Batch, 8 * 181 * 360, 192)`
- 下采样后 trunk token：`(Batch, 8 * 91 * 180, 384)`
- 上采样恢复后 trunk token：`(Batch, 8 * 181 * 360, 192)`
- 与 skip 拼接后恢复：
  - surface: `(Batch, 4, 721, 1440)`
  - upper-air: `(Batch, 5, 13, 721, 1440)`

## 默认关键参数

- `img_size=(721, 1440)`
- `patch_size=(2, 4, 4)`
- `embed_dim=192`
- `num_heads=(6, 12, 12, 6)`
- `window_size=(2, 6, 12)`

## 常见修改点

- 修改输入变量数时，要同步检查 surface / upper-air 的切分逻辑与 patch recovery 输出通道数
- 修改 `patch_size` 时，要同步检查 embedding、sample、patch recovery 和位置 shape 推导
- 修改主干深度或头数时，要同步检查 `PanguFuser` 的 `dim`、`num_heads` 和上下游维度匹配

## 风险点

- surface 输入 7 通道但输出 4 通道，这不是错误，而是因为静态 mask 只参与输入
- trunk 输入是统一三维 token 网格，不要误改成两个完全分离的 encoder/decoder 主干
- `PanguFuser` 处理的是展平 token 序列，`PanguPatchRecovery` 处理的是特征图，二者输入形态不同

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `PanguEmbedding` 和 `PanguFuser`
3. 再看 `PanguDownSample`、`PanguUpSample`、`PanguPatchRecovery`
4. 若仍有疑问，再回到模型源码

## 组件契约入口

- `./contracts/panguembedding.md`
- `./contracts/pangufuser.md`
- `./contracts/pangudownsample.md`
- `./contracts/panguupsample.md`
- `./contracts/pangupatchrecovery.md`

## 源码锚点

- `./onescience/src/onescience/models/pangu/pangu.py`
- `./onescience/src/onescience/modules/fuser/pangufuser.py`
- `./onescience/src/onescience/modules/sample/pangudownsample.py`
- `./onescience/src/onescience/modules/recovery/pangupatchrecovery.py`
