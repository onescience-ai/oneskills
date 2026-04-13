# Contract: OneEmbedding

## 基本信息

- 组件名：`OneEmbedding`
- 所属模块族：`embedding`
- 统一入口：`direct_import`
- 注册名：`style="<EmbeddingStyle>"`

## 组件职责

为 embedding 类组件提供统一注册入口。

补充说明：

- 调用层通过 `style` 选择具体 embedding 实现
- 它本身不定义固定 shape 语义，真实约束来自被选中的具体组件
- 当前天气相关模型常通过它调用 `PanguEmbedding`、`FourCastNetEmbedding`、`FuxiEmbedding`

## 支持输入

- 2D 输入：`depends_on_style`
- 3D 输入：`depends_on_style`

内部统一做法：

- 先检查 `style` 是否已注册
- 再将构造参数透传给具体实现
- 前向时不改写输入输出 shape

## 构造参数

- `style`
  - 具体 embedding 实现的注册名
- `**kwargs`
  - 直接透传给对应 embedding 实现

## 输出约定

- 2D 输出：`depends_on_style`
- 3D 输出：`depends_on_style`

如果有明确边界条件，也写在这里：

- `style` 必须已在 `oneembedding.py` 注册
- shape 约束应以下层具体 embedding 契约为准

## 典型调用位置

- Pangu 主模型
- FourCastNet 主模型
- Fuxi 主模型
- FengWuEncoder

## 典型参数

- Pangu 2D / 3D patch embedding
  - `style="PanguEmbedding"`
- FourCastNet patch embedding
  - `style="FourCastNetEmbedding"`
- Fuxi 时空块 embedding
  - `style="FuxiEmbedding"`

## 风险点

- 不要把 `OneEmbedding` 当成具体实现本身
- 只看 wrapper 无法得到真实 shape 约束
- 新增 embedding 后若未注册，调用层即使写对构造参数也无法工作

## 源码锚点

- `./onescience/src/onescience/modules/embedding/oneembedding.py`
- `./onescience/src/onescience/modules/embedding/panguembedding.py`
- `./onescience/src/onescience/modules/embedding/fourcastnetembedding.py`
- `./onescience/src/onescience/modules/embedding/fuxiembedding.py`
