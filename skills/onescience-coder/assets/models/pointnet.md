# Model Card: PointNet

## 基本信息

- 模型名：`PointNet`
- 任务类型：`CFD / point-cloud baseline`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/PointNet.py`

## 模型定位

PointNet 是最简单的点云基线之一，适合在不显式使用图卷积的前提下，对非结构点集做点级场回归。

补充说明：

- 虽然前向签名保留了 `geo`，但当前实现并不真正使用图边。
- 核心思路是“点级 MLP + 全局最大池化 + 全局特征回灌”。
- 当你想先验证“只靠点集本身能做到什么”，它比图模型更直接。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x + fx`
    - 拼接为每个点的输入特征

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 每个点输出目标变量

## 主干结构

- `OneMlp`
  - encoder
- `in_block`
- `max_block`
  - 生成全局 pooled 特征
- `out_block`
  - 拼接局部与全局特征后再编码
- `decoder`

## 主要依赖组件

- `OneMlp`
- `global_max_pool`

## 主要 Shape 变化

- 编码后：`(NumPoints, n_hidden)`
- `in_block` 后：`(NumPoints, 2 * n_hidden)`
- `max_block` 后做全局池化：`(1, 32 * n_hidden)`
- 回灌拼接后：`(NumPoints, 34 * n_hidden)` 量级
- 输出后再 `unsqueeze(0)`

## 默认关键参数

- `n_hidden=<from args>`
- `fun_dim=<from args>`
- `out_dim=<from args>`

## 常见修改点

- 依据输入字段修改 `space_dim / fun_dim`
- 作为图模型对照组时，尽量沿用同一 datapipe 和同一指标
- 若任务更像全局标量回归，可在输出头而不是主干上改动

## 风险点

- 代码里会在 `geo is None` 时抛错，但实际上并不使用图边，这是一个接口遗留点。
- 不使用显式邻接关系，因此局部几何结构表达能力弱于图模型。
- 目前默认 batch 维通常较小，真正大 batch 点云训练前要先确认显存。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `GraphSAGE`
3. 再看对应点云 datapipe

## 组件契约入口

- `./contracts/onemlp.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/PointNet.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_airfoil.py`
