# Contract: OneSample

## 基本信息

- 组件名：`OneSample`
- 所属模块族：`sample`
- 统一入口：`direct_import`
- 注册名：`style="<SampleStyle>"`

## 组件职责

为采样类组件提供统一注册入口。

补充说明：

- 调用层通过 `style` 选择具体下采样或上采样实现
- 它本身不规定 token 或特征图语义
- 当前天气相关模型常通过它调用 `PanguDownSample`、`PanguUpSample`、`FuxiDownSample`、`FuxiUpSample`

## 支持输入

- 2D 输入：`depends_on_style`
- 3D 输入：`depends_on_style`

内部统一做法：

- 先检查 `style` 是否已注册
- 将构造参数透传给具体采样模块
- 前向时不额外改写输入输出

## 构造参数

- `style`
  - 具体采样实现的注册名
- `**kwargs`
  - 直接透传给对应采样实现

## 输出约定

- 2D 输出：`depends_on_style`
- 3D 输出：`depends_on_style`

如果有明确边界条件，也写在这里：

- shape 语义应以下层具体采样组件为准
- 新增采样组件后需同步注册到 `onesample.py`

## 典型调用位置

- Pangu 主模型
- FuxiTransformer
- FengWuEncoder
- FengWuDecoder

## 典型参数

- Pangu token 采样
  - `style="PanguDownSample"`
  - `style="PanguUpSample"`
- Fuxi 2D 特征图采样
  - `style="FuxiDownSample"`
  - `style="FuxiUpSample"`

## 风险点

- `OneSample` 不会自动检查调用层的 shape 语义是否正确
- 同为 sample，`Pangu*` 与 `Fuxi*` 处理对象完全不同，不要混用
- 只看 wrapper 无法判断输入是 token 序列还是二维特征图

## 源码锚点

- `./onescience/src/onescience/modules/sample/onesample.py`
- `./onescience/src/onescience/modules/sample/pangudownsample.py`
- `./onescience/src/onescience/modules/sample/panguupsample.py`
- `./onescience/src/onescience/modules/sample/fuxidownsample.py`
