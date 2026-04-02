

# <div align="center"><strong>oneskills</strong></div>
### <div align="center"></div>



OneSkills 是基于 OneScience 构建的一个开源知识与 Skills（能力）库，旨在为智能体（Agents）开发提供可复用、可组合、可扩展的能力模块，优化多种智能体AI4S领域代码生成效果。

> 🚀 OneSkills = Knowledge × Skills × Agents
>

## 📖 项目简介

在智能体时代，单一模型能力已无法满足复杂任务需求。OneSkills 提供：

- 📚 **结构化知识库（Knowledge Units）**
- 🧠 **可执行能力模块（Skills）**
- 🔗 **面向智能体的标准接口（Agent-ready APIs）**

帮助开发者更好的使用具备专业能力的智能体系统。

## 🔗 与 OneScience 的关系

- **OneScience**：科学知识与智能体基础框架。用于代码运行环境。
- **OneSkills**：能力层（Skills & Knowledge），基于OneScience构建，用户开发环境。

## 🎯 设计目标

- **模块化（Modular）**：每个 模块可独立开发、测试和复用；
- **标准化（Standardized）**：统一 Skill 接口与描述规范；
- **可组合（Composable）**：支持多个 Skills 组合形成复杂能力；
- **领域扩展（Domain Scalable）**：覆盖气象、生信、材料、流体等多领域；
- **AI 原生（AI-native）**：专为 LLM / Agent 设计；



## ⚙️ 快速开始



1. ### 安装

   oneskills 是嵌入在当前项目开发环境中的支持工具。在已有项目目录或者新建一个项目目录，例如 mymodel.

   ```shell
   cd mymodel
   git clone https://github.com/onescience-ai/oneskills.git
   #若有必要
   # git clone https://github.com/onescience-ai/OneScience.git
   
   ```

   oneskills项目，用于赋能智能体开发，所以需要在开发环境中集成相关智能体，例如Claude code/ Trae等。详细见操作手册。

   ### 2. 使用一个 Skill

   ​    如下使用trae/task/react_prompt.md 中的skill：

   ```
   基于./oneskills/trae/task/react_prompt.md中的工作流执行以下任务：
   将Pangu-Weather模型中的Swin-Transformer Fuser模块替换为FourCastNet的AFNO（Adaptive Fourier Neural Operator）模块，并将改造完成的完整模型代码保存至当前项目hybrid_pangu_fourcastnet.py文件中。
   ```

   将上述的prompt复制到Trae的对话框中，点击执行按钮，即可开始执行任务。 执行完成后，将在当前项目下生成hybrid_pangu_fourcastnet.py文件。

   ### 3. 开发一个 Skill

   当标准技能无法满足您的需求时，您可以创建自定义技能来扩展 onescience 框架的能力。

   自定义技能文件可以放在当前项目目录下（对目录没有强制要求），技能文件内容通常包括以下部分：

   - 名称（Name）
   - 描述（Description）
   - 输入输出定义
   - 实现逻辑（代码或调用接口）

   以气象模型组合适配的场景为例，编写skills并展示使用过程。

```
---
name: model_adapter
description: 通用气象AI模型组合适配器，用于统一不同模型模块的维度、变量和网格，支持模块组合、替换和特征融合。
tags:
  - 气象AI
  - 模型组合
  - 适配器
---

# 气象AI模型组合适配器

## 1. 技能目标
提供一种通用方式，将不同气象AI模型（如 Fuxi、Pangu、FourCastNet、GraphCast）模块进行组合。核心目标：
- **维度适配**：统一各模型输入/输出张量维度、变量集合、时间和空间结构。
- **模块连接**：通过可插拔适配器实现异构模块无缝组合。
- **保形与保真**：确保数值稳定性、可微分性及计算效率。

## 2. 适用场景
- 异构编码器-适配器-解码器组合（如 Fuxi 编码器 + 适配器 + Pangu 解码器）。
- 模块替换（如 FourCastNet 的 AFNO 层替换为 Pangu 的 3D Swin Transformer）。
- 多模型特征融合和预训练模型拼接。

## 3. 输入与输出
| 参数 | 类型 | 描述 |
|------|------|------|
| 输入张量 | torch.Tensor | 待组合模块的输出，包含 `(B, T, V, H, W)` 或类似布局 |
| 输出张量 | torch.Tensor | 适配器处理后的张量，可直接作为下游模块输入 |

## 4. 通用适配流程
1. **接口分析**：提取输入/输出维度、变量、时间/空间结构及归一化参数。
2. **维度对齐**：
   - 变量差异 → 特征投影（`nn.Linear` 或 `1x1 Conv`）
   - 空间网格差异 → 插值（`grid_sample` / `xarray interp`）
   - 时间步差异 → 重塑或聚合
   - 张量顺序不同 → 使用 `einops.rearrange` 或 `torch.permute`
3. **适配器设计**：
   - 独立 `nn.Module`，支持梯度回传
   - 可选可学习参数，用于优化变量映射
   - 内部记录每一步变换，便于调试
4. **验证与测试**：
   - 输出形状检查
   - 数值范围检查
   - 单元和集成测试（前向传播无误）

## 5. 各模型典型特征
- **Fuxi**：高斯网格 `(H=721, W=1440)`，约 70 个变量。
- **Pangu**：高空+地面变量，使用高斯网格，变量布局略不同。
- **FourCastNet**：规则网格 `(H=720, W=1440)`，20 个变量，无显式垂直维。
- **GraphCast**：基于 GNN，均匀经纬网格，垂直层 L=37。

> 适配要点：统一空间网格、变量维度、张量顺序，并根据目标模块位置编码要求调整。

## 6. 输出内容
- **接口分析表**：模块名称、输入/输出形状、变量列表、网格类型。
- **适配流程图**：展示数据流、适配器位置和操作。
- **验证清单**：形状检查点、数值测试和单元测试建议。

```

如上完成了 ./self_skills/earth_skill.md，下面我们在下述任务中使用该skills

```
基于 ./oneskills/trae/task/react_prompt.md 中定义的工作流执行以下任务：

## 任务目标
对 Pangu-Weather 模型进行结构改造，将其中的 Swin-Transformer Fuser 模块替换为 FourCastNet 中的 AFNO（Adaptive Fourier Neural Operator）模块，并生成完整模型实现代码。

## 执行约束
- 必须加载并使用 ./self_skills/earth_skill.md 中定义的技能，以辅助模型结构理解、气象数据处理或模块实现
```

和使用skill一样，将上述prompt提交到Trae对话框中，即可提交运行

## 🌍 应用场景

- 📊 数据分析助手
- 🧑‍💻 开发辅助 Agent
- 🧠 AI Copilot 系统



## 🤝 社区

欢迎加入我们一起构建 AI-native 能力生态：

- 提交 Issue
- 发起 PR
- 分享你的 Skill

------

## ⭐ Star History

如果这个项目对你有帮助，欢迎 ⭐️ 支持！
