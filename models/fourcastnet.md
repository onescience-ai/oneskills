# Model Card: FourCastNet

## 基本信息

- 模型名：`FourCastNet`
- 任务类型：`全球天气预报 / 单时刻二维场建模`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/fourcastnet/fourcastnet.py`

## 模型定位

FourCastNet 是一个面向二维全球气象场的 patch-grid AFNO 模型，适合处理单时刻的二维变量场。

补充说明：

- 输入不包含 `PressureLevels` 维度
- 主干不是 Transformer attention，而是 AFNO 频域混合加逐位置前馈网络
- 它最重要的结构特点是：先做 2D patch embedding，再把 token 序列恢复成二维 patch 网格进入 trunk

## 输入定义

- 输入 shape：`(Batch, Channels, Height, Width)`
- 输入变量组织：
  - 单一路径二维气象场
  - 所有变量都直接作为二维通道输入

## 输出定义

- 输出 shape：`(Batch, out_chans, Height, Width)`
- 输出变量组织：
  - 输出变量数通常与输入变量数一致
  - 最终通过线性头按 patch 重排回原分辨率

## 主干结构

- `OneEmbedding(style="FourCastNetEmbedding")`
  - 将二维气象场映射成 patch token 序列
- 加上可学习位置编码
- token 序列恢复成二维 patch 网格
- 多层 `OneFuser(style="FourCastNetFuser")`
  - 每层内部是 `OneAFNO(style="FourCastNetAFNO2D") + OneFC(style="FourCastNetFC")`
- 通过线性 head 预测 patch 输出
- 用 `einops.rearrange` 恢复为 `(Batch, out_chans, Height, Width)`

## 主要依赖组件

- `FourCastNetEmbedding`
- `FourCastNetFuser`
- `FourCastNetAFNO2D`
- `FourCastNetFC`

## 主要 Shape 变化

- 输入：`(Batch, Channels, 720, 1440)`
- patch embedding 输出：`(Batch, 16200, 768)`
- 恢复成 patch 网格：`(Batch, 90, 180, 768)`
- trunk 输出：`(Batch, 90, 180, 768)`
- head 输出：`(Batch, 90, 180, out_chans * 8 * 8)`
- 重排后输出：`(Batch, out_chans, 720, 1440)`

## 默认关键参数

- `img_size=(720, 1440)`
- `patch_size=(8, 8)`
- `in_chans=19`
- `out_chans=19`
- `embed_dim=768`
- `depth=12`
- `mlp_ratio=4.0`
- `num_blocks=8`
- `sparsity_threshold=0.01`
- `hard_thresholding_fraction=1.0`

## 常见修改点

- 修改 `img_size` 或 `patch_size` 时，要同步检查 patch grid 大小、位置编码长度和最终重排逻辑
- 修改 `embed_dim` 时，要同步检查 `FourCastNetFuser`、`FourCastNetAFNO2D` 和 `FourCastNetFC` 的维度对齐
- 修改 `depth` 或 `drop_path_rate` 时，要同步检查 block 堆叠数和按层 drop path 分配

## 风险点

- `FourCastNetEmbedding` 输出是展平 token 序列，但 `FourCastNetFuser` 输入是二维 patch 网格，调用层必须做 reshape
- `FourCastNetAFNO2D` 中的 `Height` 和 `Width` 指的是 patch 网格尺寸，不是原始图像分辨率
- `embed_dim` 必须能被 AFNO 的 `num_blocks` 整除

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `FourCastNetEmbedding` 和 `FourCastNetFuser`
3. 若需要理解 trunk 内部，再看 `FourCastNetAFNO2D` 和 `FourCastNetFC`
4. 若仍有疑问，再回到模型源码

## 组件契约入口

- `./oneskills/codex/contracts/fourcastnetembedding.md`
- `./oneskills/codex/contracts/fourcastnetfuser.md`
- `./oneskills/codex/contracts/fourcastnetafno.md`
- `./oneskills/codex/contracts/fourcastnetfc.md`

## 源码锚点

- `./onescience/src/onescience/models/fourcastnet/fourcastnet.py`
- `./onescience/src/onescience/modules/fuser/fourcastnetfuser.py`
- `./onescience/src/onescience/modules/afno/fourcastnetafno.py`
- `./onescience/src/onescience/modules/embedding/fourcastnetembedding.py`
