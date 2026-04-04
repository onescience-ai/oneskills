# Contract: OneRecovery

## 基本信息

- 组件名：`OneRecovery`
- 所属模块族：`recovery`
- 统一入口：`direct_import`
- 注册名：`style="<RecoveryStyle>"`

## 组件职责

为恢复类组件提供统一注册入口。

补充说明：

- 调用层通过 `style` 选择具体 recovery 实现
- 当前天气相关模型最常用的是 `PanguPatchRecovery`
- wrapper 本身不定义固定 shape，真实规则以下层具体组件为准

## 支持输入

- 2D 输入：`depends_on_style`
- 3D 输入：`depends_on_style`

内部统一做法：

- 先检查 `style` 是否已注册
- 构造参数直接透传
- 前向时直接调用被选中的 recovery 模块

## 构造参数

- `style`
  - 具体 recovery 实现的注册名
- `**kwargs`
  - 直接透传给对应 recovery 实现

## 输出约定

- 2D 输出：`depends_on_style`
- 3D 输出：`depends_on_style`

如果有明确边界条件，也写在这里：

- 真实输入输出语义应以下层 recovery 契约为准
- 新增 recovery 组件后需同步注册到 `onerecovery.py`

## 典型调用位置

- Pangu 主模型
- FengWuDecoder

## 典型参数

- Patch 恢复
  - `style="PanguPatchRecovery"`

## 风险点

- `OneRecovery` 只是入口，不代表统一的恢复逻辑
- 若调用层只看 wrapper，容易忽略 2D / 3D patch 恢复差异
- 新 recovery 未注册时，`style` 会直接失败

## 源码锚点

- `./onescience/src/onescience/modules/recovery/onerecovery.py`
- `./onescience/src/onescience/modules/recovery/pangupatchrecovery.py`
