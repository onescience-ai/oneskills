# Contract: OneAFNO

## 基本信息

- 组件名：`OneAFNO`
- 所属模块族：`afno`
- 统一入口：`direct_import`
- 注册名：`style="<AFNOStyle>"`

## 组件职责

为 AFNO 类组件提供统一注册入口。

补充说明：

- 调用层通过 `style` 选择具体 AFNO 实现
- 当前天气相关模型最常用的是 `FourCastNetAFNO2D`
- wrapper 本身不定义固定 shape，真实规则以下层具体 AFNO 组件为准

## 支持输入

- 2D 输入：`depends_on_style`
- 3D 输入：`depends_on_style`

内部统一做法：

- 先检查 `style` 是否已注册
- 再将构造参数透传给具体 AFNO 模块
- 前向时直接调用底层 AFNO

## 构造参数

- `style`
  - 具体 AFNO 实现的注册名
- `**kwargs`
  - 直接透传给对应 AFNO 实现

## 输出约定

- 2D 输出：`depends_on_style`
- 3D 输出：`depends_on_style`

如果有明确边界条件，也写在这里：

- 当前 weather 相关默认实现只覆盖 `FourCastNetAFNO2D`
- 新 AFNO 组件后需同步注册到 `oneafno.py`

## 典型调用位置

- FourCastNetFuser

## 典型参数

- FourCastNet 频域混合
  - `style="FourCastNetAFNO2D"`

## 风险点

- `OneAFNO` 不负责把 token 序列恢复成二维网格
- 只看 wrapper 无法知道输入通道数是否需要能被 `num_blocks` 整除
- AFNO 的真实频域约束要看下层具体实现

## 源码锚点

- `./onescience/src/onescience/modules/afno/oneafno.py`
- `./onescience/src/onescience/modules/afno/fourcastnetafno.py`
