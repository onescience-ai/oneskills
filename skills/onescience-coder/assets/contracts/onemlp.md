# Contract: OneMlp

## 基本信息

- 组件名：`OneMlp`
- 所属模块族：`mlp`
- 统一入口：`direct_import`
- 注册名：`style="<MlpStyle>"`

## 组件职责

为多种 MLP 实现提供统一注册入口，常用于模型最前面的特征预处理、维度映射和末尾的小型投影层。

补充说明：

- 调用层通过 `style` 选择具体 MLP 实现
- 在当前 OneScience 里，`StandardMLP` 是最常见的通用选择
- wrapper 本身不限定前缀维度，通常只要求最后一维是 feature 维

## 支持输入

- 2D 输入：`(..., FeatureDim)`
- 3D 输入：`(..., FeatureDim)`

内部统一做法：

- 保持除最后一维之外的前缀维度不变
- 只对最后一维做逐位置特征映射
- 真实约束以具体 `style` 的实现为准

## 构造参数

- `style`
  - 具体 MLP 实现的注册名，例如 `StandardMLP`
- `input_dim`
  - 输入特征维度
- `hidden_dims`
  - 隐层维度列表
- `output_dim`
  - 输出特征维度
- `activation`
  - 激活函数名；常见为 `gelu`
- `**kwargs`
  - 其余参数直接透传给具体 MLP

## 输出约定

- 2D 输出：`(..., output_dim)`
- 3D 输出：`(..., output_dim)`

额外约束：

- 调用层必须保证最后一维和 `input_dim` 一致
- 不同 `style` 对 `n_layers`、`res`、`use_bias` 等附加参数的支持不完全相同

## 典型调用位置

- Transolver 的输入特征映射
- `CFD_Benchmark` 中 `FNO / U_FNO / U_NO / U_Net` 的预处理层

## 典型参数

- CFD_Benchmark 预处理
  - `style="StandardMLP"`
  - `input_dim=space_dim + fun_dim`
  - `hidden_dims=[n_hidden * 2]`
  - `output_dim=n_hidden`
- 点云或 token 特征升维
  - `style="StandardMLP"`
  - `activation="gelu"`

## 风险点

- `style` 未注册时会直接报错
- 很多调用位置默认只变换最后一维；如果把通道维和空间维排错，shape 会在第一层就不匹配
- 不同模型对 `StandardMLP` 传入的附加参数略有差别，跨模型复用时不要假设所有 kwargs 都被所有 style 支持

## 源码锚点

- `./onescience/src/onescience/modules/mlp/onemlp.py`
- `./onescience/src/onescience/modules/mlp/MLP.py`
