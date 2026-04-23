# Model Card: Transolver (CFD_Benchmark)

## 基本信息

- 模型名：`Transolver (CFD_Benchmark)`
- 任务类型：`CFD / token-based physics transformer baseline`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/Transolver.py`

## 模型定位

这是 `cfd_benchmark` 目录下的 benchmark 版 `Transolver`，它和项目里另一个直接吃 `PyG Data` 的 `Transolver2D/3D` 不是同一套接口。

补充说明：

- 这里的接口是标准 benchmark 路线：`(x, fx, T) -> y`。
- 主干通过多层 `OneTransformer(style="Transolver_block")` 实现 slicing 注意力。
- 如果用户说“沿用 CFD_Benchmark 的 Transolver 训练流程”，应优先看这张卡，而不是通用 `transolver.md`。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x`
    - 坐标或统一位置编码
  - `fx`
    - 与点对齐的输入场特征

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 每个点的目标物理量

## 主干结构

- `OneMlp`
  - 点级预处理
- 多层 `OneTransformer(style="Transolver_block")`
- 末层 block 直接输出 `out_dim`

## 主要依赖组件

- `OneMlp`
- `OneTransformer(style="Transolver_block")`

## 主要 Shape 变化

- 预处理后：`(Batch, NumPoints, n_hidden)`
- 多层 slicing attention 后保持 token 形态
- 末层输出：`(Batch, NumPoints, out_dim)`

## 默认关键参数

- `slice_num=<from args>`
- `n_layers=<from args>`
- `n_heads=<from args>`
- `mlp_ratio=<from args>`

## 常见修改点

- 随 datapipe 修改 `fun_dim / out_dim`
- 对比 benchmark 里的 FNO/Transformer 时，尽量保持同一 datapipe 协议
- 若几何为 structured，可用 `unified_pos`；若为 unstructured，直接走原坐标

## 风险点

- 很容易和 `./models/transolver.md` 指向的另一套实现混淆。
- 这是 token 接口，不直接消费 `PyG Data.x / Data.pos`。
- `slice_num` 会影响注意力切片行为，改动时通常需要联动训练稳定性观察。

## 推荐检索顺序

1. 先看本模型卡
2. 再看通用 `transolver.md` 以区分实现
3. 再看 `OneTransformer`

## 组件契约入口

- `./contracts/onetransformer.md`
- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/Transolver.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_airfoil.py`
