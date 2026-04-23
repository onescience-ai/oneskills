# Model Card: Swin_Transformer

## 基本信息

- 模型名：`Swin_Transformer`
- 任务类型：`CFD / windowed transformer on structured 2D grids`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/Swin_Transformer.py`

## 模型定位

Swin_Transformer 是窗口化注意力的二维结构化网格基线，适合在规则 2D 网格上比较局部窗口注意力与全局 Transformer 或谱算子路线。

补充说明：

- 只支持 `structured_2D`。
- 会把网格尺寸补到 `window_size` 的整数倍。
- 每个 stage 由多个 `SwinTransformerBlock` 组成。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, 2)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x`
    - 2D 网格坐标或统一位置编码
  - `fx`
    - 点级场特征

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 点级目标变量

## 主干结构

- `OneMlp`
  - 点级预处理
- 多层 `BasicLayer`
  - 内部由 `OneTransformer(style="SwinTransformerBlock")` 堆叠
- `fc1 + fc2`

## 主要依赖组件

- `OneMlp`
- `OneTransformer(style="SwinTransformerBlock")`

## 主要 Shape 变化

- 预处理后：`(Batch, NumPoints, n_hidden)`
- reshape 到补齐后的 2D 网格
- 展平成窗口 token 后进入多个 `BasicLayer`
- 回到原始网格并裁剪 padding
- 输出：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `window_size=4`
- `depth=2`（每个 stage 固定）
- `n_layers=<from args>`
- `n_heads=<from args>`

## 常见修改点

- 首先确认 `shapelist` 是二维
- 若网格尺寸与窗口不整除，关注 padding 后的有效区域裁剪
- 与 `Factformer`、`Transformer` 做对比时可直接复用同一 structured datapipe

## 风险点

- 非 2D structured 任务会直接报错。
- `window_size` 变化会影响 padding、感受野和显存。
- 若 datapipe 返回的 `NumPoints` 与 `shapelist` 展平后不一致，reshape 会直接失败。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `Factformer`
3. 再看 `OneTransformer`

## 组件契约入口

- `./contracts/onetransformer.md`
- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/Swin_Transformer.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
