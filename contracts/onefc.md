# Contract: OneFC

## 基本信息

- 组件名：`OneFC`
- 所属模块族：`fc`
- 统一入口：`direct_import`
- 注册名：`style="<FCStyle>"`

## 组件职责

为逐位置全连接模块提供统一注册入口。

补充说明：

- 调用层通过 `style` 选择具体前馈实现
- 当前天气相关模型常通过它调用 `FourCastNetFC` 与 `FuxiFC`
- wrapper 本身不定义固定前缀维度，真实约束来自具体 FC 组件

## 支持输入

- 2D 输入：`depends_on_style`
- 3D 输入：`depends_on_style`

内部统一做法：

- 先检查 `style` 是否已注册
- 再将构造参数透传给具体 FC 模块
- 前向时对输入前缀维度不做额外处理

## 构造参数

- `style`
  - 具体 FC 实现的注册名
- `**kwargs`
  - 直接透传给对应 FC 实现

## 输出约定

- 2D 输出：`depends_on_style`
- 3D 输出：`depends_on_style`

如果有明确边界条件，也写在这里：

- 前缀维度是否保留，应以下层具体 FC 契约为准
- 新 FC 组件后需同步注册到 `onefc.py`

## 典型调用位置

- FourCastNetFuser
- Fuxi 主模型

## 典型参数

- FourCastNet 通道混合
  - `style="FourCastNetFC"`
- Fuxi patch 级输出映射
  - `style="FuxiFC"`

## 风险点

- `OneFC` 不会自动判断输入最后一维是否是特征维
- 同为 FC，`FourCastNetFC` 与 `FuxiFC` 的调用目的不同
- 只看 wrapper 无法判断输出维是否应等于输入维

## 源码锚点

- `./onescience/src/onescience/modules/fc/onefc.py`
- `./onescience/src/onescience/modules/fc/fourcastnetfc.py`
- `./onescience/src/onescience/modules/fc/fuxifc.py`
