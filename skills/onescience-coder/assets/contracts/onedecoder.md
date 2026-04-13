# Contract: OneDecoder

## 基本信息

- 组件名：`OneDecoder`
- 所属模块族：`decoder`
- 统一入口：`direct_import`
- 注册名：`style="<DecoderStyle>"`

## 组件职责

为解码器类组件提供统一注册入口。

补充说明：

- 调用层通过 `style` 选择具体 decoder 实现
- 当前天气相关模型最常用的是 `FengWuDecoder`
- wrapper 本身不定义固定 shape，真实规则来自具体 decoder

## 支持输入

- 2D 输入：`depends_on_style`
- 3D 输入：`depends_on_style`

内部统一做法：

- 先检查 `style` 是否已注册
- 再将构造参数透传给具体 decoder
- 前向时直接调用底层 decoder

## 构造参数

- `style`
  - 具体 decoder 实现的注册名
- `**kwargs`
  - 直接透传给对应 decoder 实现

## 输出约定

- 2D 输出：`depends_on_style`
- 3D 输出：`depends_on_style`

如果有明确边界条件，也写在这里：

- 真实 shape 约束以下层 decoder 契约为准
- 新 decoder 组件后需同步注册到 `onedecoder.py`

## 典型调用位置

- FengWu 主模型

## 典型参数

- FengWu 单分支解码
  - `style="FengWuDecoder"`

## 风险点

- `OneDecoder` 只是入口，不表示所有 decoder 可互换
- 只看 wrapper 无法判断输入是 token 序列还是带 skip 的组合输入
- `style` 未注册时会直接报错

## 源码锚点

- `./onescience/src/onescience/modules/decoder/onedecoder.py`
- `./onescience/src/onescience/modules/decoder/fengwudecoder.py`
