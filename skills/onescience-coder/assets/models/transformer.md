# Model Card: Transformer

## 基本信息

- 模型名：`Transformer`
- 任务类型：`CFD / standard token transformer baseline`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/Transformer.py`

## 模型定位

Transformer 是 `CFD_Benchmark` 里最标准的 token 级自注意力基线，适合与 FNO、Factformer、Galerkin_Transformer、Swin 等模型做直接对比。

补充说明：

- 主干是普通多头自注意力加 MLP 的堆叠。
- 最后一层 block 直接负责输出投影。
- 结构化和非结构化几何都复用同一 token 接口。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x`
    - 坐标或统一位置编码
  - `fx`
    - 输入场特征

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 每个 token 对应一个点的目标变量

## 主干结构

- `OneMlp`
  - 预处理
- 多层 `Transformer_block`
  - `FlashAttention + MLP`
- 末层 block 直接输出 `out_dim`

## 主要依赖组件

- `OneMlp`
- `OneAttention(style="FlashAttention")`

## 主要 Shape 变化

- 预处理后：`(Batch, NumPoints, n_hidden)`
- 多层 block 后保持 token 形态
- 末层输出：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `n_layers=<from args>`
- `n_heads=<from args>`
- `mlp_ratio=<from args>`
- `dropout=<from args>`

## 常见修改点

- 修改 `fun_dim / out_dim`
- 与 `Galerkin_Transformer` 做公平对比时保持 datapipe 和训练配置不变
- 如果任务有时间条件，显式打开 `time_input`

## 风险点

- 长序列显存压力大于局部窗口注意力和多数卷积基线。
- 末层 block 直接输出目标维度，自己插额外头时要避免重复投影。
- `unified_pos` 与 `shapelist` 会共同影响输入维度解释。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `Factformer`
3. 再看 `Galerkin_Transformer`

## 组件契约入口

- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/Transformer.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
