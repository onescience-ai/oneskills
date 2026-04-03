# Project Context
> **重要**: 在开始任何任务前，必须先阅读本文件和 architecture.md

This repository belongs to the OneScience scientific computing platform.

> 详细架构信息请参考 [architecture.md](.claude/architecture.md)

## 可复用组件

平台包含 38+ 模型和丰富的基础模块：

**核心模型库** (`src/onescience/models/`):
- 地球科学: afno, graphcast, fourcastnet, fengwu, fuxi, nowcastnet
- 计算流体: meshgraphnet, deepcfd, beno, cfdbench
- 生物信息: evo2, alphafold3, protenix, rfdiffusion
- 材料化学: UMA, mace

**基础模块** (`src/onescience/modules/`):
- attention, embedding, encoder/decoder
- fourier/fft, diffusion, equivariant
- loss, layer, linear/fc

## 工程规范

1. **注册机制**: 所有模型/模块使用前必须注册
2. **维度匹配**: dataset → model → loss 维度必须一致
3. **配置驱动**: 超参数通过 config 文件定义
4. **分布式支持**: 训练脚本需支持多 GPU/多节点

## 代码模式

**模型注册示例**:
```python
from onescience.registry import MODELS

@MODELS.register_module()
class MyModel(nn.Module):
    pass
```

**模块复用示例**:
```python
from onescience.modules.attention import MultiHeadAttention
from onescience.modules.embedding import PatchEmbedding
```

## 编码原则

- 优先复用现有模块，避免重复实现
- 遵循项目的注册机制
- 保持与现有代码风格一致

## 禁止原则

- 生成新项目时禁止使用src/onescience/datapipes/climate/era5_hdf5.py文件及该文件内的类