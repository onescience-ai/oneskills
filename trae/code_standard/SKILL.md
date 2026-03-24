---
name: onescience组件使用规范
description: onescience组件使用规范技能，专门用于构建和实例化 OneScience 框架下的 AI4S（AI for Science）深度学习组件。该技能确保所有组件均通过标准的 onescience.modules 注册机制调用，避免直接导入底层实现类，保证代码的兼容性与规范性。
category: Code Generation / AI4S Framework
---

## 能力描述 (Capabilities)
此技能能够：
 - 解析用户对深度学习组件的功能需求（如“下采样”、“Pangu风格的嵌入”）。
 - 通过逐层匹配机制，先从白名单中确定注册组件类型（如 OneSample），再从其对应目录中匹配具体的功能性组件（如 PanguDownSample3D），提取 style 参数。
 - 生成包含标准导入、工厂实例化及前向调用的代码片段。。

## 核心规则与约束 (Core Rules & Constraints)
Agent 在执行此技能时必须严格遵守以下规则：
1. 注册机制强制原则
✅ 必须 使用 from onescience.modules import <ComponentName> 进行导入。
❌ 禁止从子目录导入，例如 from onescience.modules.embeddings import PanguEmbedding2D, PanguEmbedding3D 等
❌ 禁止直接导入具体实现类，例如 from onescience.modules import PanguPatchRecovery2D, PanguPatchRecovery3D, PanguDownSample3D, PanguUpSample3D, PanguEmbedding2D, PanguEmbedding3D 等。
❌ 禁止出现 from onescience.models.module import Module

2. 实例化模式
- 所有组件必须通过工厂模式实例化：module = ComponentName(style="<SpecificStyle>", **kwargs)。
- style 参数必须指定，其值为对应目录下功能性组件的类名（如 "PanguEmbedding2D"）。
- 分析所选功能性组件的初始化参数及其默认值，结合用户提供的超参数，将这些参数以 **kwargs 形式传递给工厂实例化。
3. 可用注册组件白名单
以下为允许导入的注册组件类，每个类对应 onescience.modules 下的一个子目录（例如 OneEmbedding 对应 embeddings/ 目录），目录内包含多个功能性组件的具体实现。
```python
[ "OneEmbedding", "OneFuser", "OneSample", "OneRecovery", "OneAttention", 
    "OneMlp", "OneFourier", "OneEncoder", "OneDecoder", "OneHead", "OnePooling",
    "OneTransformer", "OneEdge", "OneNode", "OneProcessor", "OneEquivariant",
    "OneFC", "OneAFNO", "OneLinear", "OneDiffusion", "OneMSA", "OnePairformer"]
```

## 执行逻辑 (Execution Logic)
1. 意图识别
解析用户自然语言需求，提取关键信息：功能类型（如嵌入、采样、注意力）、风格/模型归属（如Pangu、FourCastNet）、维度/特殊要求（如2D、3D）。
2. 组件匹配（两层匹配）
- 第一层：匹配注册组件类型
  - 根据意图中的功能类型，从白名单中筛选出候选注册组件。
  - 例如：意图包含“嵌入”→候选为 OneEmbedding；意图包含“下采样”→候选为 OneSample。
  - 若意图模糊（如“我需要一个处理特征的操作”），可依次遍历白名单，检查每个注册组件目录下的功能性组件是否可能满足。

- 第二层：匹配功能性组件 → 确定 style
  - 进入候选注册组件对应的目录（知识库），遍历其中的功能性组件及其描述。
  - 将用户意图与每个功能性组件的描述进行匹配，选择最符合的一个。
  - 匹配成功后，记录该功能性组件的类名作为 style 参数值（如 "PanguDownSample3D"）。
  - 若多个功能性组件均符合，可提供选项或根据优先级（如最具体、最常用）选择。
  - 分析所选功能性组件的初始化参数，并结合用户提供的超参数，将这些参数作为 kwargs 传递给工厂实例化。
> 注意：注册组件与功能性组件必须正确对应，且 style 值必须是功能性组件的类名。。
3. 代码生成：
- 导入语句：from onescience.modules import <注册组件类>。注意，导入语句不可在代码中动态添加。
- 实例化：module = <注册组件类>(style="<功能性组件类名>", **kwargs)，其中 kwargs 是根据用户提供的超参数以及所选功能性组件的初始化参数，有选择地填充的键值对（即仅传递用户显式指定或与默认值不同的参数）。
- 前向调用示例（可选）：output = module(input_tensor)
4. 合规性检查：在输出前自我审查，确保没有违反“严禁直接导入底层类”的规则，所有导入均来自 onescience.modules，且 style 参数值准确无误

## 知识库与示例 (Knowledge Base & Examples)
### 通用模板:
```python
from onescience.modules import <ComponentName>

# 实例化
module = <ComponentName>(style="<SpecificStyle>", **custom_kwargs)

# 调用
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

