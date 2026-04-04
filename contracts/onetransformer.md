# Contract: OneTransformer

## 基本信息

- 组件名：`OneTransformer`
- 所属模块族：`transformer`
- 统一入口：`direct_import`
- 注册名：`style="<TransformerStyle>"`

## 组件职责

为 transformer 类组件提供统一注册入口。

补充说明：

- 调用层通过 `style` 选择具体 transformer 实现
- 当前天气相关模型常通过它调用 `EarthTransformer2DBlock`、`EarthTransformer3DBlock`、`FuxiTransformer`
- wrapper 本身不规定固定 shape，真实规则来自具体实现

## 支持输入

- 2D 输入：`depends_on_style`
- 3D 输入：`depends_on_style`

内部统一做法：

- 先检查 `style` 是否已注册
- 再将构造参数透传给具体 transformer
- 前向时直接调用底层模块

## 构造参数

- `style`
  - 具体 transformer 实现的注册名
- `**kwargs`
  - 直接透传给对应 transformer 实现

## 输出约定

- 2D 输出：`depends_on_style`
- 3D 输出：`depends_on_style`

如果有明确边界条件，也写在这里：

- shape 约束以下层 transformer 契约为准
- 新 transformer 组件后需同步注册到 `onetransformer.py`

## 典型调用位置

- PanguFuser
- FengWuEncoder
- FengWuDecoder
- FengWuFuser
- Fuxi 主模型

## 典型参数

- 二维 Earth block
  - `style="EarthTransformer2DBlock"`
- 三维 Earth block
  - `style="EarthTransformer3DBlock"`
- Fuxi U-shape trunk
  - `style="FuxiTransformer"`

## 风险点

- `OneTransformer` 只是分发入口，不代表统一的 tensor 语义
- `EarthTransformer2DBlock` 与 `FuxiTransformer` 都属于 transformer，但输入形态完全不同
- 只看 wrapper 无法判断是否需要先展平 token 或先恢复网格

## 源码锚点

- `./onescience/src/onescience/modules/transformer/onetransformer.py`
- `./onescience/src/onescience/modules/transformer/earthtransformer2Dblock.py`
- `./onescience/src/onescience/modules/transformer/earthtransformer3Dblock.py`
- `./onescience/src/onescience/modules/transformer/fuxitransformer.py`
