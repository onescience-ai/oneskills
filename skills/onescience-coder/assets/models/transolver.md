# Model Card: Transolver

## 基本信息

- 模型名：`Transolver`
- 任务类型：`CFD / 几何相关物理场点级回归`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/transolver/Transolver2D.py`

## 模型定位

Transolver 是一个面向 PDE 与 CFD 场景的几何感知点级回归模型，适合在非结构化网格、点云或几何相关采样点上直接预测物理量。

补充说明：

- 模型族同时提供二维和三维主干：`Transolver2D`、`Transolver3D`
- 增强版本 `Transolver2D_plus`、`Transolver3D_plus` 保持调用方式不变，只替换为更强的 `unstructured_plus` 物理注意力
- 它最重要的结构特点是：先把每个点的输入特征映射到隐空间，再通过多层物理切片注意力 block 在全局点集上做交互，最终直接输出每个点的目标变量

## 输入定义

- 输入 shape：`Data.x -> (NumPoints, InChannels)`，`Data.pos -> (NumPoints, 2 or 3)`
- 输入变量组织：
  - `Data.x`
    - 每个点的几何特征、边界条件特征、物理先验特征或其它点级输入特征
  - `Data.pos`
    - 与点一一对应的二维或三维坐标

补充说明：

- `space_dim` 对应的是 `Data.x` 的输入通道数，不只是坐标维度
- 若 `unified_pos=True`，模型内部会基于 `Data.pos` 额外构造参考网格距离编码，datapipe 不需要预先生成这部分特征

## 输出定义

- 输出 shape：`(NumPoints, out_dim)`
- 输出变量组织：
  - 每个点直接输出对应目标变量
  - 常见 CFD 场景中可对应速度、压力、湍流量等点级物理量

## 主干结构

- `OneMlp(style="StandardMLP")`
  - 将点级输入特征映射到隐藏维度
- 加入可学习 `placeholder` 偏置
- 多层 `OneTransformer(style="Transolver_block")`
  - 标准版使用 `geotype="unstructured"`
  - plus 版使用 `geotype="unstructured_plus"`
- 最后一层 block 直接输出 `out_dim`

建议把它理解为：

- datapipe 决定 `Data.x/Data.pos/Data.y`
- Transolver 负责点级特征映射和全局物理交互
- 训练脚本通常只需要根据数据维度切换 2D 或 3D 版本，而不改模型内部结构

## 主要依赖组件

- `OneTransformer`
- `Transolver_block`
- `OneMlp(style="StandardMLP")`

## 主要 Shape 变化

- datapipe 输出：`Data.x -> (NumPoints, InChannels)`
- 模型前向开始：扩成 `x -> (1, NumPoints, InChannels)`
- 若 `unified_pos=True`
  - 2D 版本拼接后变为 `(1, NumPoints, InChannels + ref^2)`
  - 3D 版本拼接后变为 `(1, NumPoints, InChannels + ref^3)`
- `preprocess` 后：`(1, NumPoints, n_hidden)`
- 最后一层 block 后：`(1, NumPoints, out_dim)`
- 输出前去掉 batch 维：`(NumPoints, out_dim)`

## 默认关键参数

- `n_hidden=256`
- `n_layers=8`
- `n_head=8`
- `mlp_ratio=2`
- `slice_num=32`
- `ref=8`
- `unified_pos=1`

说明：

- `space_dim` 和 `out_dim` 由具体 datapipe 与任务目标决定
- `Transolver2D` / `Transolver2D_plus` 通常与二维翼型、二维流场任务搭配
- `Transolver3D` / `Transolver3D_plus` 通常与三维车体、三维几何流场任务搭配

## 常见修改点

- 根据 `Data.pos` 的坐标维度在训练脚本中选择 2D 或 3D 版本
- 当新数据集的输入特征字段变化时，同步修改 `space_dim`
- 当新数据集的目标变量变化时，同步修改 `out_dim`
- 当复用已有案例训练流程时，优先保持模型类和训练主循环不变，只替换 datapipe、配置项和数据相关指标
- 若新数据集没有 `surf`、法向量、湍流黏度等字段，需要在 datapipe 里补占位或同步简化损失与评估逻辑

## 风险点

- `space_dim` 一旦和 `Data.x` 通道数不一致，模型会在第一层 MLP 直接报错
- `unified_pos=True` 会改变实际输入维度，不能在 datapipe 中重复构造同类参考网格距离特征
- plus 版与普通版的主要差异在 `geotype`，不是新的输入输出协议；不要误以为需要改 datapipe 返回格式
- 训练脚本里常把 Transolver 和 GraphSAGE / PointNet / GUNet 共用一套配置块，改 datapipe 时要确认这些备用模型分支不会被意外破坏

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `OneTransformer` 契约
3. 再看最接近的 Transolver 示例工程
4. 若仍不足，再回到模型源码和 `Transolver_block` 实现

## 组件契约入口

- `./contracts/onetransformer.md`

## 源码锚点

- `./onescience/src/onescience/models/transolver/Transolver2D.py`
- `./onescience/src/onescience/models/transolver/Transolver3D.py`
- `./onescience/src/onescience/models/transolver/Transolver2D_plus.py`
- `./onescience/src/onescience/modules/transformer/Transolver_block.py`
