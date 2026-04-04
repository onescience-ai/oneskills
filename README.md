# Codex OneScience Guide

`oneskills/codex` 是面向代码智能体的轻量知识入口。

目标不是复述 `./onescience/` 源码，而是让用户和智能体先通过稳定的组件契约完成大多数开发任务，只有在契约不足时才回到源码。

开发和维护 `codex` 时，优先参考：

- `./oneskills/codex/DEVELOPER_MANUAL.md`

## 目录结构

- `models/`
  - 模型知识卡
  - 面向智能体，描述整模型的输入输出、主干结构、组件组成链路和常见修改点
  - 用户明确提到模型名时，优先先看这里
- `contracts/`
  - 组件契约卡片
  - 面向智能体，描述组件职责、注册名、参数、输入输出 shape、调用位置和源码锚点
  - 新增组件时优先复用 `contracts/TEMPLATE.md`
  - 变量命名优先遵循 `contracts/naming_convention.md`
- `task/`
  - 面向智能体的执行工作流
  - 用于指导如何基于契约卡片完成替换、组合、改写等任务

## 当前覆盖范围

当前已覆盖 Pangu 相关的几个高频组件：

- `PanguEmbedding`
- `PanguDownSample`
- `PanguUpSample`
- `PanguPatchRecovery`
- `PanguFuser`

同时，已补充 FourCastNet 相关组件：

- `FourCastNetEmbedding`
- `FourCastNetFuser`
- `FourCastNetAFNO2D`
- `FourCastNetFC`

同时，已补充 Fuxi 相关组件：

- `FuxiEmbedding`
- `FuxiTransformer`
- `FuxiDownSample`
- `FuxiUpSample`
- `FuxiFC`

同时，已补充 FengWu 相关组件：

- `FengWuEncoder`
- `FengWuFuser`
- `FengWuDecoder`

其中 Pangu 相关组件适合二维/三维天气场的 patch 编码、采样、恢复和三维主干融合。

FourCastNet 相关组件则覆盖：

- 二维 patch embedding
- AFNO 主干 block
- 频域混合模块
- 逐位置前馈通道混合模块

Fuxi 相关组件则覆盖：

- 三维时空 patch embedding
- 二维 U 形 Swin trunk
- 二维特征图下采样与上采样
- patch 级输出线性投影

FengWu 相关组件则覆盖：

- 单变量分支二维 encoder
- 中分辨率跨变量 3D fuser
- 单变量分支 decoder

## 使用建议

如果用户只是要“让智能体写代码”，推荐优先使用下面顺序：

1. 读取 `task/SKILL.md`
2. 若用户明确提到模型名，先读取 `models/model_index.md`
3. 再读取 `contracts/component_index.md`
4. 读取 `contracts/naming_convention.md`
5. 按任务需要读取 `models/` 与 `contracts/` 下对应文档
6. 若模型卡和组件卡仍不足以支撑实现，再回到 `./onescience/` 对应源码锚点补充确认

对于天气预测、全球格点预报、surface 与 upper-air 联合建模这类任务，建议额外优先确认：

1. 主干特征提取是否应复用 `fuser` 组件
2. 是否应先组织成统一 3D token 主干
3. 具体组件应优先按模块族检索，再在模块族内部根据 shape、输入形态、注册名和契约完整度选择
4. 只有在 `fuser` 不适用时，才回到底层 Transformer block

组件总览建议优先看：

- `contracts/component_index.md`

模型总览建议优先看：

- `models/model_index.md`

## 设计原则

- 用户优先：尽量不要求用户直接阅读 OneScience 源码
- 契约优先：优先依赖组件契约，而不是长篇源码转写
- 源码兜底：文档和代码不一致时，以源码为准
- 推荐优先：优先记录当前推荐使用的组件，不再鼓励新增历史分裂实现的默认依赖
