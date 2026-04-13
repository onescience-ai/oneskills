# Model Card: <ModelName>

## 基本信息

- 模型名：`<ModelName>`
- 任务类型：`<task_type>`
- 当前状态：`<stable_or_other>`
- 主实现文件：`./onescience/<path_to_model>.py`

## 模型定位

一句话说明该模型适合解决什么问题。

补充说明：

- 该模型是二维还是三维主干
- 主要适用于哪类输入组织方式
- 它与其它模型相比最重要的结构特点是什么

## 输入定义

- 输入 shape：`<input_shape>`
- 输入变量组织：
  - `<branch_or_layout_1>`
  - `<branch_or_layout_2>`

## 输出定义

- 输出 shape：`<output_shape>`
- 输出变量组织：
  - `<branch_or_layout_1>`
  - `<branch_or_layout_2>`

## 主干结构

- `<stage_1>`
- `<stage_2>`
- `<stage_3>`

建议只写调用层真正需要知道的结构链路，不要把源码逐行展开。

## 主要依赖组件

- `<component_1>`
- `<component_2>`
- `<component_3>`

这里优先写当前推荐先读的组件契约。

## 主要 Shape 变化

- `<shape_step_1>`
- `<shape_step_2>`
- `<shape_step_3>`

## 默认关键参数

- `<arg_1>=<value>`
- `<arg_2>=<value>`
- `<arg_3>=<value>`

## 常见修改点

- `<edit_point_1>`
- `<edit_point_2>`
- `<edit_point_3>`

## 风险点

- `<risk_1>`
- `<risk_2>`
- `<risk_3>`

优先写：

- 输入输出变量数是否对称
- token / patch / 网格 shape 是否容易错
- 哪些修改会联动多个组件

## 推荐检索顺序

1. 先看本模型卡
2. 再看相关组件契约
3. 若契约不足，再回到源码锚点

## 组件契约入口

- `./contracts/<component_1>.md`
- `./contracts/<component_2>.md`

## 源码锚点

- `./onescience/<path_to_model>.py`
- `./onescience/<path_to_component_or_registry>.py`

统一约定：

- 一律使用 `./onescience/...` 相对路径
- 不写本机绝对路径
