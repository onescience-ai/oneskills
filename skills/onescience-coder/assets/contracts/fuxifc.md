# Contract: FuxiFC

## 基本信息

- 组件名：`FuxiFC`
- 所属模块族：`fc`
- 统一入口：`OneFC`
- 注册名：`style="FuxiFC"`

## 组件职责

在最后一个特征维度上执行线性映射，将 trunk 输出映射为 patch 级目标变量。

补充说明：

- 该组件通常位于 Fuxi 主模型末端
- 它不改变前面的 batch 或空间网格维度
- 本质上是一个作用在最后一维上的线性投影层

## 支持输入

- 2D 输入：`(..., in_channels)`
- 3D 输入：`(..., in_channels)`

内部统一做法：

- 仅对最后一个特征维度执行 `Linear`
- 保持前面的所有维度不变

## 构造参数

- `in_channels`
  - 输入特征维度
- `out_channels`
  - 输出特征维度

## 输出约定

- 2D 输出：`(..., out_channels)`
- 3D 输出：`(..., out_channels)`

明确约束：

- 该组件不依赖固定空间 shape
- 主要约束来自最后一维的特征维度匹配

## 典型调用位置

- Fuxi 主模型中的 patch 输出投影阶段

## 典型参数

- Fuxi 默认配置
  - `in_channels=1536`
  - `out_channels=70 * 4 * 4`

## 风险点

- 该组件不是空间算子，不能直接当作 embedding 或 transformer 使用
- 主要检查点是最后一维是否与调用层一致

## 源码锚点

- `./onescience/src/onescience/modules/fc/fuxifc.py`
- `./onescience/src/onescience/modules/fc/onefc.py`
- `./onescience/src/onescience/models/fuxi/fuxi.py`
