# Contract: FourCastNetFC

## 基本信息

- 组件名：`FourCastNetFC`
- 所属模块族：`fc`
- 统一入口：`OneFC`
- 注册名：`style="FourCastNetFC"`

## 组件职责

在最后一个特征维度上执行两层前馈映射，完成逐位置通道混合。

补充说明：

- 该组件通常位于 `FourCastNetFuser` 中的 AFNO 之后
- 它不关心前面的 batch 或空间维度
- 本质上是一个作用在最后一维上的 MLP

## 支持输入

- 2D 输入：`(..., in_features)`
- 3D 输入：`(..., in_features)`

内部统一做法：

- 对最后一个特征维度执行 `Linear -> Act -> Dropout -> Linear -> Dropout`
- 保持前面的所有维度不变

## 构造参数

- `in_features`
  - 输入特征维度
- `hidden_features`
  - 隐层特征维度
- `out_features`
  - 输出特征维度
- `act_layer`
  - 激活函数类型
- `drop`
  - dropout 比例

## 输出约定

- 2D 输出：`(..., out_features)`
- 3D 输出：`(..., out_features)`

明确约束：

- 该组件不依赖固定空间 shape
- 主要约束来自最后一维的特征维度匹配

## 典型调用位置

- `FourCastNetFuser` 内部的通道混合层

## 典型参数

- FourCastNet 默认配置
  - `in_features=768`
  - `hidden_features=3072`
  - `out_features=768`
  - `drop=0.0`

## 风险点

- 该组件不是空间算子，不能直接把它当作 embedding 或 fuser 使用
- 主要检查点是最后一维是否与调用层一致

## 源码锚点

- `./onescience/src/onescience/modules/fc/fourcastnetfc.py`
- `./onescience/src/onescience/modules/fc/onefc.py`
- `./onescience/src/onescience/modules/fuser/fourcastnetfuser.py`
