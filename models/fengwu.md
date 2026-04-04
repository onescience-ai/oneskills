# Model Card: FengWu

## 基本信息

- 模型名：`FengWu`
- 任务类型：`多变量中期天气预报 / 多分支二维场联合建模`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/fengwu/fengwu.py`

## 模型定位

FengWu 是一个面向多变量全球天气预报的多分支模型，适合分别编码不同变量族，再在中分辨率层面做跨变量三维融合。

补充说明：

- 输入不是单一路径，而是 surface、Z、R、U、V、T 六个变量分支
- 每个分支先独立编码，再通过统一三维 fuser 做跨变量融合
- 它最重要的结构特点是：多分支 2D encoder/decoder，加一个中分辨率 3D fuser

## 输入定义

- 输入 shape：
  - surface: `(Batch, 4, Height, Width)`
  - z/r/u/v/t: `(Batch, pressure_level, Height, Width)`
- 输入变量组织：
  - surface 分支单独编码
  - 五个高空变量分支分别编码

## 输出定义

- 输出 shape：
  - surface: `(Batch, 4, Height, Width)`
  - z/r/u/v/t: `(Batch, pressure_level, Height, Width)`
- 输出变量组织：
  - 六个变量分支分别解码并返回

## 主干结构

- 6 个 `OneEncoder(style="FengWuEncoder")`
  - 分别编码 surface、Z、R、U、V、T
- 中分辨率特征按变量轴拼接成三维网格
- `OneFuser(style="FengWuFuser")`
  - 在 `(Variables, Height, Width)` 三维网格上融合
- 6 个 `OneDecoder(style="FengWuDecoder")`
  - 分别恢复各变量分支

## 主要依赖组件

- `FengWuEncoder`
- `FengWuFuser`
- `FengWuDecoder`
- `PanguEmbedding`
- `PanguDownSample`
- `PanguUpSample`
- `PanguPatchRecovery`

## 主要 Shape 变化

- surface 输入：`(Batch, 4, 721, 1440)`
- 高空变量输入：`(Batch, 37, 721, 1440)`
- 单分支 encoder 中间输出：`(Batch, 91 * 180, 384)`
- 单分支 skip 输出：`(Batch, 181, 360, 192)`
- 6 分支拼接后：`(Batch, 6, 91 * 180, 384)`
- 展平后 fuser 输入：`(Batch, 6 * 91 * 180, 384)`
- fuser 输出后再切回 6 个分支
- 单分支 decoder 输出：
  - surface: `(Batch, 4, 721, 1440)`
  - 高空变量: `(Batch, 37, 721, 1440)`

## 默认关键参数

- `img_size=(721, 1440)`
- `pressure_level=37`
- `embed_dim=192`
- `patch_size=(4, 4)`
- `num_heads=(6, 12, 12, 6)`
- `window_size=(2, 6, 12)`

## 常见修改点

- 修改 `pressure_level` 时，要同步检查 5 个高空分支的输入输出通道数
- 修改 `patch_size` 时，要同步检查 encoder / decoder 的 patch 网格尺寸和 fuser 输入分辨率
- 修改分支数时，要同步检查 `FengWuFuser` 的 `input_resolution[0]`

## 风险点

- `FengWuEncoder` / `FengWuDecoder` 处理的是单个变量分支，不要误改成所有变量共用单一路径
- `FengWuFuser` 的第一维 `Variables` 指的是分支数，不是时间步或气压层
- decoder 中 `num_heads` 的高低分辨率顺序需要与调用层保持一致

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `FengWuEncoder`、`FengWuFuser`、`FengWuDecoder`
3. 若需要理解 encoder / decoder 内部，再看 `PanguEmbedding`、`PanguDownSample`、`PanguUpSample`、`PanguPatchRecovery`
4. 若仍有疑问，再回到模型源码

## 组件契约入口

- `./oneskills/codex/contracts/fengwuencoder.md`
- `./oneskills/codex/contracts/fengwufuser.md`
- `./oneskills/codex/contracts/fengwudecoder.md`
- `./oneskills/codex/contracts/panguembedding.md`
- `./oneskills/codex/contracts/pangudownsample.md`
- `./oneskills/codex/contracts/panguupsample.md`
- `./oneskills/codex/contracts/pangupatchrecovery.md`

## 源码锚点

- `./onescience/src/onescience/models/fengwu/fengwu.py`
- `./onescience/src/onescience/modules/encoder/fengwuencoder.py`
- `./onescience/src/onescience/modules/fuser/fengwufuser.py`
- `./onescience/src/onescience/modules/decoder/fengwudecoder.py`
