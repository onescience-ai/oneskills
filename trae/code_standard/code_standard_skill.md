---
name: onescience组件使用规范
description: onescience组件使用规范技能，用于构建和实例化 OneScience 框架下的 AI4S 深度学习组件。确保所有组件通过标准注册机制调用，避免直接导入底层实现类。
category: Code Generation / AI4S Framework
---

## 能力描述
此技能能够：
- 解析用户对深度学习组件的功能需求（如"下采样"、"Pangu风格的嵌入"）
- 通过两层匹配机制确定组件：注册组件类型 → 功能性组件
- 生成符合规范的导入、实例化和调用代码

## 核心规则

### 1. 导入规则（强制）
```python
✅ 必须使用：from onescience.modules import OneEmbedding, OneSample, ...
❌ 禁止使用：from onescience.modules.embeddings import PanguEmbedding2D
❌ 禁止使用：from onescience.modules import PanguEmbedding2D
❌ 禁止使用：from onescience.models.module import Module
```

### 2. 实例化规则
- 必须通过工厂模式：module = ComponentName(style="<SpecificStyle>", **kwargs)
- style 参数必须指定，值为功能性组件的类名（如 "PanguEmbedding2D"）
- kwargs 仅包含用户显式指定或与默认值不同的参数

### 3. 可用注册组件白名单
```python
["OneEmbedding", "OneFuser", "OneSample", "OneRecovery", "OneAttention", 
 "OneMlp", "OneFourier", "OneEncoder", "OneDecoder", "OneHead", "OnePooling",
 "OneTransformer", "OneEdge", "OneNode", "OneProcessor", "OneEquivariant",
 "OneFC", "OneAFNO", "OneLinear", "OneDiffusion", "OneMSA", "OnePairformer"]
```

| 功能关键词 | 注册组件 |
|-----------|---------|
| 嵌入、embedding | OneEmbedding |
| 下采样、上采样、sample | OneSample |
| 融合、fuse、merge | OneFuser |
| 恢复、recovery、patch recovery | OneRecovery |
| 注意力、attention | OneAttention |
| MLP、全连接 | OneMlp |
| 傅里叶、fourier | OneFourier |
| 编码器、encoder | OneEncoder |
| 解码器、decoder | OneDecoder |

## 执行流程 (Execution Workflow)

### 1. 意图识别
从用户需求中提取核心信息：
- **功能类型**：嵌入、采样、注意力、融合等
- **风格归属**：Pangu、FourCastNet、ViT 等
- **维度特性**：2D、3D 或其他特殊要求
- **超参数**：用户显式指定的配置参数

### 2. 两层组件匹配

#### 第一层：确定注册组件类型
- 根据功能类型从白名单筛选候选注册组件
  - 示例：嵌入 → `OneEmbedding`，下采样 → `OneSample`
- 若意图模糊，遍历白名单检查各注册组件目录下的功能性组件是否匹配

#### 第二层：确定功能性组件（style 参数）
- 进入候选注册组件对应目录，遍历功能性组件及其描述
- 将用户意图与功能性组件描述匹配，选择最符合的实现
- 记录该功能性组件的类名作为 `style` 值（如 `"PanguDownSample3D"`）
- 若多个组件符合，按优先级选择（最具体 > 最常用）或提供选项
- 分析所选组件的 `__init__` 参数，结合用户超参数构建 `kwargs`

> **关键约束**：注册组件与功能性组件必须正确对应，`style` 值必须是功能性组件的类名

### 3. 代码生成
生成符合规范的代码片段：

```python
# 导入（仅导入注册组件，不可动态添加）
from onescience.modules import <注册组件类>

# 实例化（仅传递用户指定或非默认值参数）
module = <注册组件类>(style="<功能性组件类名>", **kwargs)

# 前向调用（可选）
output = module(input_tensor)
```

### 典型场景示例:
- 场景 A: 构建 Pangu 风格的 Embedding 层
 - User: "我需要创建一个适合 Pangu 模型的 2D 嵌入层。"
 - Agent Action:
```python
from onescience.modules import OneEmbedding

# 使用 Pangu 特定的 style
embedding_layer = OneEmbedding(style="PanguEmbedding2D", embed_dim=512)
output = embedding_layer(input_tensor)
```

- 场景 B: 3D 采样操作 (上采样/下采样)
 - User: "我需要做一个 3D 的下采样操作。"
 - Agent Action:
```python
from onescience.modules import OneSample

downsampler = OneSample(style="PanguDownSample3D", scale_factor=2)
downsampled_data = downsampler(input_tensor)
```
