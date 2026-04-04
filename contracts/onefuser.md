# Contract: OneFuser

## 基本信息

- 组件名：`OneFuser`
- 所属模块族：`fuser`
- 统一入口：`direct_import`
- 注册名：`style="<FuserStyle>"`

## 组件职责

为融合类组件提供统一注册入口。

补充说明：

- 调用层通过 `style` 选择具体 fuser 实现
- 当前天气相关模型常通过它调用 `PanguFuser`、`FengWuFuser`、`FourCastNetFuser`
- wrapper 本身不定义固定 shape，真实语义来自具体 fuser

## 支持输入

- 2D 输入：`depends_on_style`
- 3D 输入：`depends_on_style`

内部统一做法：

- 先校验 `style` 是否已注册
- 再将构造参数透传给具体 fuser
- 前向时不改写输入输出

## 构造参数

- `style`
  - 具体 fuser 实现的注册名
- `**kwargs`
  - 直接透传给对应 fuser 实现

## 输出约定

- 2D 输出：`depends_on_style`
- 3D 输出：`depends_on_style`

如果有明确边界条件，也写在这里：

- 真实 shape 约束以下层具体 fuser 契约为准
- 新 fuser 组件后需同步注册到 `onefuser.py`

## 典型调用位置

- Pangu 主模型
- FourCastNet 主模型
- FengWu 主模型

## 典型参数

- Pangu 三维 trunk
  - `style="PanguFuser"`
- FengWu 三维变量融合
  - `style="FengWuFuser"`
- FourCastNet 2D trunk block
  - `style="FourCastNetFuser"`

## 风险点

- 同为 fuser，三种实现处理的对象并不相同
- `PanguFuser` / `FengWuFuser` 是 3D token，`FourCastNetFuser` 是 2D 网格特征
- 若只看 wrapper，容易把“模块族相同”误解为“shape 可互换”

## 源码锚点

- `./onescience/src/onescience/modules/fuser/onefuser.py`
- `./onescience/src/onescience/modules/fuser/pangufuser.py`
- `./onescience/src/onescience/modules/fuser/fengwufuser.py`
- `./onescience/src/onescience/modules/fuser/fourcastnetfuser.py`
