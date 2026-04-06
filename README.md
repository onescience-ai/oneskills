# <div align="center"><strong>oneskills</strong></div>

### <div align="center">OneScience Skills Library for AI-native Scientific Research</div>

> 🚀 OneSkills = Knowledge × Skills × Agents

---

## 📖 项目简介

OneSkills 是基于 OneScience 构建的开源知识与能力库（Skills Library），专为智能体（Agents）开发设计，提供可复用、可组合、可扩展的能力模块，显著优化 AI4S（AI for Science）领域的代码生成效果。

在智能体时代，单一模型能力已无法满足复杂科学计算任务需求。OneSkills 通过结构化知识库与可执行能力模块，帮助开发者构建专业级 AI-native 科学研究系统。

### 核心特性

- 📚 **结构化知识库**：基于 OneScience 的科学知识与模型组件文档
- 🧠 **可执行能力模块**：标准化的 Skills 定义与工作流
- 🔗 **Agent-ready 接口**：专为 Claude Code、Trae 等智能体设计
- 🌐 **多领域覆盖**：气象、生信、材料、流体等科学领域

---

## 🔗 与 OneScience 的关系

| 项目 | 定位 | 用途 |
|------|------|------|
| **OneScience** | 科学知识与智能体基础框架 | 代码运行环境、模型训练与推理 |
| **OneSkills** | 能力层（Skills & Knowledge） | 用户开发环境、智能体能力扩展 |

**关系说明**：
- OneScience 提供底层科学计算框架与模型实现
- OneSkills 基于 OneScience 构建，提供能力模块与开发指导
- 两者协同工作，实现从模型训练到智能体开发的完整链路

---

## ⚙️ 快速开始

### 安装使用

OneSkills 是嵌入式开发工具库，无需独立安装。将其克隆到项目目录后，复制 skills 到对应智能体的配置目录：

```bash
git clone https://github.com/onescience-ai/oneskills.git

# 复制 skills 到你的项目（选择你使用的工具）
cp -r onescills/skills /your/project/.claude/skills      # Claude Code
cp -r onescills/skills /your/project/.cursor/skills      # Cursor
cp -r onescills/skills /your/project/.codex/skills       # Codex CLI
cp -r onescills/skills /your/project/.kiro/steering      # Kiro
cp -r onescills/skills /your/project/skills/custom       # DeerFlow 2.0
cp -r onescills/skills /your/project/.trae/skills         # Trae
cp -r onescills/skills /your/project/.antigravity        # Antigravity
cp -r onescills/skills /your/project/.github/superpowers # VS Code (Copilot)
cp -r onescills/skills /your/project/skills              # OpenClaw
cp -r onescills/skills /your/project/.windsurf/skills   # Windsurf
cp -r onescills/skills /your/project/.gemini/skills     # Gemini CLI
cp -r onescills/skills /your/project/.aider/skills      # Aider
cp -r onescills/skills /your/project/.opencode/skills   # OpenCode
cp -r onescills/skills /your/project/.qwen/skills       # Qwen Code
```

### 智能体配置文件说明

| 工具 | 配置文件 | 路径 | 说明 |
|------|----------|------|------|
| **Claude Code** | `CLAUDE.md` | 项目根目录 | - |
| **Kiro** | `.kiro/steering/*.md` | 项目根目录 | 支持 always/globs/手动三种模式 |
| **DeerFlow 2.0** | `skills/custom/*/SKILL.md` | 项目根目录 | 字节跳动开源 SuperAgent，自动发现自定义 skills |
| **Trae** | `.trae/skills/*/*.md` | 项目级规则 | - |
| **Antigravity** | `GEMINI.md` 或 `AGENTS.md` | 项目根目录 | - |
| **VS Code** | `.github/copilot-instructions.md` | 项目根目录 | Copilot 自定义指令 |
| **Cursor** | `.cursor/rules/*.md` | 项目级规则目录 | - |
| **OpenClaw** | `skills/*/SKILL.md` | 工作区级 skills 目录 | 自动发现 |
| **Windsurf** | `.windsurf/skills/*/SKILL.md` | 项目级 skills 目录 | - |
| **Gemini CLI** | `.gemini/skills/*/SKILL.md` | 项目级 skills 目录 | - |
| **Aider** | `.aider/skills/*/SKILL.md` | 项目级 skills 目录 | - |
| **OpenCode** | `.opencode/skills/*/SKILL.md` | 项目级 skills 目录 | - |
| **Qwen Code** | `.qwen/skills/*/SKILL.md` | 项目级 skills 目录 | - |

### 使用 Skills

Skills 通过自然语言提示词触发。以下是一个模型改造任务示例：

```
基于 ./oneskills/trae/task/react_prompt.md 中的工作流执行以下任务：

## 任务目标
将 Pangu-Weather 模型中的 Swin-Transformer Fuser 模块替换为 FourCastNet 的 AFNO（Adaptive Fourier Neural Operator）模块，并将改造完成的完整模型代码保存至当前项目 hybrid_pangu_fourcastnet.py 文件中。

## 执行约束
- 优先复用 ./oneskills/models/ 中的模型卡
- 使用 ./oneskills/contracts/ 中的组件契约
- 生成代码需包含完整的 forward 实现
```

将上述提示提交到支持 Skills 的智能体（如 Trae）对话框，即可自动执行任务并生成代码。

### 开发 Skills

当标准 Skills 无法满足需求时，可创建自定义 Skills：

**Skills 文件结构**（YAML 元数据 + Markdown 正文）：

```markdown
---
name: skill_name
description: 技能描述，简明扼要
tags:
  - 领域标签
  - 功能标签
---

# 技能标题

## 1. 技能目标
## 2. 适用场景
## 3. 输入与输出
## 4. 实现逻辑
## 5. 验证与测试
```

**完整示例**：参考 `./skills/` 目录下的现有 Skills 文件。

---

## 📁 项目结构

```
oneskills/
├── skills/                    # 可复用能力模块
│   ├── onescience-installer/  # OneScience 安装助手
│   ├── onescience-coder/      # OneScience 代码生成
│   └── ...                    # 其他领域 Skills
├── models/                    # 模型卡（Model Cards）
│   ├── fengwu.md
│   ├── fourcastnet.md
│   ├── fuxi.md
│   ├── pangu.md
│   └── model_index.md
├── contracts/                 # 组件契约（Component Contracts）
│   ├── component_index.md
│   └── naming_convention.md
├── datapipes/                 # 数据卡（DataPipe Cards）
│   ├── cmems.md
│   └── datapipe_index.md
├── trae/                      # Trae 智能体集成
│   └── task/
│       └── react_prompt.md
└── self_skills/               # 用户自定义 Skills 目录
```

---

## 🌍 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| 📊 **数据分析助手** | 科学数据处理与可视化 | 气象数据读取、网格插值、结果可视化 |
| 🧑‍💻 **开发辅助 Agent** | 模型代码生成与改造 | 模块替换、特征融合、架构优化 |
| 🧠 **AI Copilot 系统** | 交互式智能开发 | 自然语言编程、错误诊断、性能优化 |
| 🔬 **科研工作流** | 复杂科学计算任务 | 多模型组合、跨领域适配、自动化流程 |

---

## 🤝 社区与贡献

欢迎加入 OneSkills 社区，共同构建 AI-native 科学研究生态：

### 参与方式

- **提交 Issue**：报告问题、提出功能需求
- **发起 PR**：贡献新 Skills、修复 Bug、优化文档
- **分享 Skill**：分享您的自定义 Skills 至社区

### 贡献指南

1. Fork 项目并创建分支
2. 添加/修改 Skills 文件
3. 确保符合项目规范与格式
4. 提交 Pull Request 并描述变更

### 社区交流

- GitHub Issues：[issues](https://github.com/onescience-ai/oneskills/issues)
- Discussion：[discussions](https://github.com/onescience-ai/oneskills/discussions)

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## ⭐ Star History

如果这个项目对你有帮助，欢迎 ⭐️ 支持！

---

<div align="center">
  <strong>Built with OneScience for AI-native Scientific Research</strong>
</div>
