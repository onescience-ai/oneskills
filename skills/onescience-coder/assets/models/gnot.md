# Model Card: GNOT

## 基本信息

- 模型名：`GNOT`
- 任务类型：`CFD / transformer-style neural operator with expert routing`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/GNOT.py`

## 模型定位

GNOT 是带 MoE 风格 block 的 neural operator Transformer，适合在标准 token 接口下比较“几何嵌入 + 物理嵌入 + 专家路由”这条路线。

补充说明：

- `x` 和 `fx` 分别走两套预处理分支。
- block 调用时会同时传入 `x` 的几何嵌入、`fx` 的物理嵌入和原始坐标 `pos`。
- 若任务想强调不同局部模式的专家分工，这个模型比普通 Transformer 更合适。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x`
    - 几何坐标
  - `fx`
    - 物理场输入特征

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 每个点输出 `out_dim` 个目标变量

## 主干结构

- `preprocess_x`
  - 编码几何分支
- `preprocess_z`
  - 编码物理分支
- 多层 `OneTransformer(style="GNOTTransformerBlock")`
- `fc1 + fc2`

## 主要依赖组件

- `OneMlp`
- `OneTransformer(style="GNOTTransformerBlock")`

## 主要 Shape 变化

- `preprocess_x` 后：`(Batch, NumPoints, n_hidden)`
- `preprocess_z` 后：`(Batch, NumPoints, n_hidden)`
- 每层 block 后保持 token 形态
- 输出头后：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `n_experts=3`
- `n_layers=<from args>`
- `n_heads=<from args>`
- `mlp_ratio=<from args>`

## 常见修改点

- 若输入字段变化，优先同时修改 `dim_x/dim_z` 对应的 `space_dim/fun_dim`
- 如果只做静态任务，可以保持 `time_input=False`
- 对比实验时可直接与 `Transformer`、`Factformer` 共用同一 datapipe

## 风险点

- 它依赖 block 内部对 `x/fx/pos` 三路信息的协同处理，不能随意把几何分支删掉。
- `unified_pos=True` 会同时改变 `preprocess_x` 与 `preprocess_z` 的输入维度。
- 专家路由的真实行为在 block 里，若卡片信息不够，需要回到 `GNOTTransformerBlock` 源码确认。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `OneTransformer`
3. 再看 `Transformer`

## 组件契约入口

- `./contracts/onetransformer.md`
- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/GNOT.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
