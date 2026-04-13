# Contract: OneHead

## 基本信息

- 组件名：`OneHead`
- 所属模块族：`head`
- 统一入口：`direct_import`
- 注册名：`style="<HeadStyle>"`

## 组件职责

为预测头类组件提供统一注册入口，常用于多尺度主干后的最终输出映射。

补充说明：

- 当前与 CFD 任务最相关的是 `UNetHead1D/2D/3D`
- `MaskedMSAHead` 适用于另一类掩码预测场景，不是当前稳态 CFD 比较任务的首选
- wrapper 本身不定义固定输入 shape，真实约束来自具体 head

## 支持输入

- 2D 输入：`depends_on_style`
- 3D 输入：`depends_on_style`

内部统一做法：

- 通过 `style` 选择具体 head
- 构造参数直接透传
- forward 时不做额外逻辑包装

## 构造参数

- `style`
  - 具体 head 的注册名
- `in_channels`
  - 输入通道数
- `out_channels`
  - 输出通道数
- `**kwargs`
  - 其余参数透传给对应 head 实现

## 输出约定

- 2D 输出：`depends_on_style`
- 3D 输出：`depends_on_style`

额外约束：

- `UNetHead*d` 通常保持空间维不变，只映射通道数
- 是否带额外激活、卷积核大小等，取决于具体 head 实现

## 典型调用位置

- `CFD_Benchmark` 中 `U_Net` 的多尺度输出头

## 典型参数

- 2D U-Net 输出头
  - `style="UNetHead2D"`
  - `in_channels=n_hidden`
  - `out_channels=n_hidden`
- 3D U-Net 输出头
  - `style="UNetHead3D"`

## 风险点

- `style` 未注册会直接报错
- 很多时候 head 只是最后一层映射，不负责补齐前面丢失的空间尺寸；如果 encoder/decoder 级联 shape 没对齐，head 无法兜底
- `OneHead` 只适合“已有主干输出，再做最后投影”的场景，不要把它当作完整 decoder 使用

## 源码锚点

- `./onescience/src/onescience/modules/head/onehead.py`
- `./onescience/src/onescience/modules/head/unet_head.py`
