# Model Card: Galerkin_Transformer

## 基本信息

- 模型名：`Galerkin_Transformer`
- 任务类型：`CFD / linear-attention transformer baseline`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/Galerkin_Transformer.py`

## 模型定位

Galerkin_Transformer 是标准 Transformer 基线的线性注意力版本，适合比较 `FlashAttention` 风格自注意力与 `Galerkin` 风格注意力在物理场建模上的差异。

补充说明：

- 接口仍是 `(x, fx, T)`。
- 通过 `OneTransformer(style="Galerkin_Transformer_block")` 堆叠主干。
- 它比标准 Transformer 更接近“线性注意力算子”路线。

## 输入定义

- 输入 shape：`(Batch, NumPoints, *)`
- 输入变量组织：
  - `x`
    - 坐标或统一位置编码
  - `fx`
    - 输入场特征

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 末层 block 直接输出目标通道。

## 主干结构

- `OneMlp(style="StandardMLP")`
- 多层 `OneTransformer(style="Galerkin_Transformer_block")`
- 末层线性头在 block 内完成

## 主要依赖组件

- `OneMlp`
- `OneTransformer(style="Galerkin_Transformer_block")`

## 主要 Shape 变化

- 预处理后：`(Batch, NumPoints, n_hidden)`
- 多层 attention 后保持 token 形态
- 末层输出：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `n_layers=<from args>`
- `n_heads=<from args>`
- `mlp_ratio=<from args>`
- `dropout=<from args>`

## 常见修改点

- 与 `Transformer` 共用大部分数据协议，可直接在同一训练脚本里切换模型名做对比
- 依据任务修改 `fun_dim / out_dim`
- 若加时间条件，保持 `time_input` 与 datapipe 返回同步

## 风险点

- 仍然是 token 级全局交互，长序列显存虽优于标准注意力，但仍高于局部卷积模型。
- 末层投影在 block 内完成，更换 block 风格时要一起核查输出维度。
- 若 `unified_pos=True`，输入维度会随 `ref` 和 `shapelist` 变化。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `Transformer`
3. 再看 `OneTransformer`

## 组件契约入口

- `./contracts/onetransformer.md`
- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/Galerkin_Transformer.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
