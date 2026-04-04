# Naming Convention

## 目标

本文件定义 `oneskills/codex` 当前优先采用的变量命名约定。

目的只有一个：当多个组件表达相同语义时，尽量使用同一个名字，避免智能体在跨文件理解时把同义变量误判成不同概念。

## 优先命名

以下语义优先使用这些名字：

| 语义 | 推荐名字 |
|---|---|
| 批大小 | `Batch` |
| 输入变量数 | `Variables` |
| 输入时间步数 | `TimeSteps` |
| 气压层数 | `PressureLevels` |
| 高度方向网格数 | `Height` |
| 宽度方向网格数 | `Width` |
| 输出时间步数 | `OutTimeSteps` |
| 输出气压层数 | `OutPressureLevels` |
| 输出高度 | `OutHeight` |
| 输出宽度 | `OutWidth` |
| patch 时间步尺寸 | `PatchTimeSteps` |
| patch 气压层尺寸 | `PatchPressureLevels` |
| patch 高度尺寸 | `PatchHeight` |
| patch 宽度尺寸 | `PatchWidth` |
| 输入 token 特征维 | `in_dim` |
| 输出 token 特征维 | `out_dim` |
| 输入特征通道数 | `in_chans` |
| 输出特征通道数 | `out_chans` |

## 不推荐混用的写法

如果表达的是同一个语义，尽量不要混用：

- `B` / `Batch`
- `T` / `TimeSteps`
- `Pl` / `PressureLevels`
- `H` / `Height`
- `W` / `Width`
- `batch_size` / `Batch`
- `time` / `TimeSteps`
- `levels` / `PressureLevels`
- `lat` / `Height`
- `lon` / `Width`

## 使用范围

优先统一以下位置：

- 组件注释
- Example 示例
- shape 描述
- 合同契约文档
- 代码内部局部变量

## 例外

- 公共 API 参数名若已经稳定，不强制改名
- 第三方库接口字段名不强行覆盖
- 已经成为注册接口的一部分时，优先保持兼容
