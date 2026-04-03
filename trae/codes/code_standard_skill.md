***

name: onescience组件使用规范
description: 定义 OneScience 框架下 AI4S 深度学习组件的标准化使用规范。强制要求所有组件通过统一注册机制调用，以消除引用路径错误，确保代码的可维护性与框架兼容性。
category: Code Generation / AI4S Framework
version: 1.0.0
--------------

## 能力描述

本技能用于规范 OneScience 框架下 AI4S 深度学习组件的实例化与调用流程。核心原则是：组件必须通过统一的导入入口直接使用，不得读取其内部实现代码后再重新生成**等价实现**。此举确保代码风格一致性，避免因重复实现导致的维护冗余、版本偏差或路径引用错误。

## 使用场景
代码生成时，引导代码正确生成

## 核心规则

### 1. 导入规则（强制）

所有组件（包括模型层、算子、嵌入层等）**必须**通过 `onescience.modules` 进行统一导入。严禁以下行为：
- 直接引用 onescience 下的子模块内部路径
- 读取组件源码后，在当前文件中重新编写**等价的类或函数实现**

```python
# ✅ 正确示例：直接 import 并使用
from onescience.modules import PanguEmbedding2D, PanguEmbedding3D, PanguDownSample3D

embedding = PanguEmbedding2D(img_size=(721, 1440), patch_size=(4, 4), embed_dim=192, in_chans=7)

# ❌ 错误示例：读取源码后重新生成等价实现（禁止）
# 以下代码不允许出现 —— 即使功能等价，也属于重复实现
class PanguEmbedding2D:
    def __init__(self, img_size, patch_size, embed_dim, in_chans):
        # 自行编写的实现...
        pass
```

### 2. 参数规范（强制）

初始化组件时，必须根据组件定义传入正确的参数，并必须在代码中或注释中明确说明各参数的含义与用途。禁止在不提供参数说明的情况下直接调用组件。

参数说明要求：

- 每个参数需标注其数据类型
- 简要说明参数的作用
- 若参数有默认值，可选择性标注

```python
# ✅ 正确示例：提供参数说明
embedding_layer = PanguEmbedding2D(
    img_size=(721, 1440),      # tuple: 输入图像尺寸 (高度, 宽度)
    patch_size=(4, 4),         # tuple: 图像块划分尺寸
    embed_dim=192,             # int: 嵌入向量维度
    in_chans=4+3,              # int: 输入通道数 (地表变量数 + 气压层数)
    norm_layer=None            # nn.Module: 归一化层类型，None 表示不使用
)

# ❌ 错误示例：参数无注释说明
embedding_layer = PanguEmbedding2D(721, 1440, 4, 4, 192, 7, None)
```

## 典型应用场景

- 场景 A：构建 Pangu 风格嵌入层
- 用户需求：创建一个适用于 Pangu 气象模型的 2D 嵌入层。

实现方式：

```python
from onescience.modules import PanguEmbedding2D

# 实例化组件
embedding_layer = PanguEmbedding2D(
                    img_size=(721, 1440),
                    patch_size=(4, 4),
                    embed_dim=192,
                    in_chans = 4+3,
                    norm_layer=None)

# 前向计算
output = embedding_layer(input_tensor)
```

- 场景 B：3D 下采样操作
- 用户需求：实现一个 3D 特征图的下采样功能。

实现方式：

```python
from onescience.modules import PanguDownSample3D

# 配置采样倍率
downsampler = PanguDownSample3D(
                input_resolution=(13, 128, 256),
                output_resolution=(13, 64, 128),
                in_dim=192)        
# 执行采样
output = downsampler(input_tensor)
```

