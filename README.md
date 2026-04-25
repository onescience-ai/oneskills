# <div align="center">OneSkills</div>

<p align="center">
  A skills library for AI-native scientific research built around <strong>OneScience</strong>.
</p>

<p align="center">
  Reusable skills for <strong>coding</strong>, <strong>debugging</strong>, <strong>runtime submission</strong>, and <strong>environment setup</strong>.
</p>

<p align="center">
  Works with <strong>Trae</strong>, <strong>Claude Code</strong>, <strong>Codex CLI</strong>, and other skill-based agents.
</p>

---

## What is OneSkills?

`OneSkills` 是面向 AI4S（AI for Science）场景的技能仓库。

它把 OneScience 相关的研发经验整理成可复用的 `SKILL.md` 能力模块，让智能体在科学计算任务中能按统一流程完成：

- 数据读取与分析
- 模型与组件改造
- OneScience 工程代码生成
- SLURM / DCU 环境任务提交
- 日志排查与结果调试
- 远程环境安装与初始化

如果你希望智能体不只是“写一点代码”，而是围绕 `数据 → 模型 → 运行 → 调试` 给出更稳定的工程化执行路径，这个仓库就是为此准备的。

---

## Included skills

当前仓库内置 4 个核心技能，加上 1 个技能编排入口：

| Skill | 作用 | 典型场景 |
| --- | --- | --- |
| `onescience-skill` | 统一任务识别与技能编排 | 用户只描述需求，由技能自动选择执行链路 |
| `onescience-coder` | OneScience 代码分析、方案设计、代码改造 | 数据管道接入、模型替换、组件复用 |
| `onescience-debug` | 面向 OneScience 的测试识别与排障编排 | DataPipe 测试、模型测试、端到端链路排查 |
| `onescience-runtime` | 基于 `onescience.json` 和 `tpl.slurm` 的运行提交 | SLURM / DCU 集群任务生成与提交 |
| `onescience-installer` | 远程 DCU 环境安装助手 | conda、uv、OneScience 远程安装初始化 |

---

## How it works

OneSkills 的基本使用方式很简单：

1. 把 `skills/` 复制到你的智能体工作目录
2. 在对话里显式触发某个 skill，或直接调用 `onescience-skill`
3. 智能体读取对应的 `SKILL.md`
4. 按技能里定义的流程生成代码、分析方案、提交任务或执行调试

一个典型示例：

```text
使用 onescience-skill，生成读取部分 ERA5 数据集的代码，并给出运行方案
```

对于支持技能系统的智能体，这类提示会触发技能编排，再路由到合适的 OneScience 专项技能。

---

## Install

### Trae

```bash
git clone https://github.com/onescience-ai/oneskills.git
mkdir -p /your/project/.trae
cp -r oneskills/skills /your/project/.trae/skills
```

### Claude Code

```bash
git clone https://github.com/onescience-ai/oneskills.git
mkdir -p /your/project/.claude
cp -r oneskills/skills /your/project/.claude/skills
```

### Codex CLI

```bash
git clone https://github.com/onescience-ai/oneskills.git
mkdir -p /your/project/.codex
cp -r oneskills/skills /your/project/.codex/skills
```

### Other agents

你也可以把 `skills/` 复制到其它支持技能目录约定的智能体环境中，例如：

```bash
cp -r oneskills/skills /your/project/.opencode/skills
```

> 如果你希望智能体结合 OneScience 代码库进行源码分析，建议同时准备 `onescience` 仓库作为工作区上下文。

---

## Recommended files for runtime tasks

如果你希望技能不仅生成代码，还能继续提交到集群运行，建议把下面两个文件一并放到你的工程根目录：

- `onescience.json`
- `tpl.slurm`

它们分别用于：

- 定义运行模式、资源规格、模块环境、conda 环境和脚本入口
- 作为固定的 SLURM 作业模板生成提交脚本

示例：

```bash
cp oneskills/onescience.json /your/project/
cp oneskills/tpl.slurm /your/project/
```

---

## Usage examples

### 1. 代码生成

```text
使用 onescience-coder，基于 OneScience 实现 ERA5 数据读取 DataPipe
```

### 2. 自动技能编排

```text
使用 onescience-skill，帮我把海洋数据读取、训练脚本和运行流程串起来
```

### 3. 调试与测试

```text
使用 onescience-debug，检查这个 OneScience 模型改造是否属于模型测试还是端到端测试
```

### 4. 运行提交

```text
使用 onescience-runtime，读取当前工程的 onescience.json 并提交到 slurm
```

### 5. 远程安装

```text
使用 onescience-installer，在 DCU 环境安装 earth 领域的 OneScience 依赖
```

---

## Repository layout

```text
oneskills/
├── README.md
├── skills/
│   ├── SKILL.md                    # onescience-skill，技能编排入口
│   ├── onescience-coder/SKILL.md
│   ├── onescience-debug/SKILL.md
│   ├── onescience-installer/SKILL.md
│   └── onescience-runtime/SKILL.md
├── onescience.json                # 运行时配置示例
├── tpl.slurm                      # SLURM 模板
├── onescience-agent.md            # OneScience Agent 总体提示词
├── claude/                        # Claude 侧补充上下文和检查规则
└── assets/                        # README 资源
```

---

## Skill design principles

这些技能不是“提示词片段”，而是带有明确职责边界的执行模块。当前设计重点包括：

- **流程化**：优先把任务拆成稳定的科研工程步骤
- **可复用**：面向 OneScience 常见研发场景沉淀固定套路
- **可路由**：通过 `onescience-skill` 自动选择合适技能链路
- **面向运行**：不仅生成代码，也考虑集群提交与日志排障
- **面向真实场景**：优先支持 AI4S 中常见的数据、训练、调试闭环

---

## Create your own skills

当内置技能不完全匹配你的流程时，可以在 `skills/` 下继续扩展自定义技能。

一个最小技能通常包含：

```markdown
---
name: my-skill
description: 简要描述该技能解决什么问题
---

# Skill Title

## 角色
## 输入
## 处理流程
## 输出要求
## 限制条件
```

建议保持：

- 名称清晰，能直接体现职责
- `description` 足够具体，便于智能体正确路由
- 流程步骤可执行，而不是宽泛描述
- 对输入、输出、边界条件给出明确要求

你可以直接参考 `skills/` 下现有技能的组织方式。

---

## Screenshot

下面是 OneSkills 在 Trae 中的一个使用示例：

![OneSkills usage](./assets/trae_usage.png)

---

## Contributing

欢迎通过以下方式参与共建：

- 提交 Issue，报告问题或提出新技能需求
- 提交 PR，补充技能、优化文档或修复流程
- 分享你的领域化 Skill 模板

建议贡献内容包括：

- 新的 AI4S 场景技能
- 更清晰的技能路由逻辑
- 更稳定的运行配置模板
- 更完整的示例提示词与文档

---

## Related files

- `skills/SKILL.md`: 技能总入口与路由规则
- `skills/onescience-coder/SKILL.md`: 代码分析与设计技能
- `skills/onescience-debug/SKILL.md`: 测试识别与排障技能
- `skills/onescience-runtime/SKILL.md`: 运行提交技能
- `skills/onescience-installer/SKILL.md`: 远程安装技能
- `onescience.json`: 运行配置示例
- `tpl.slurm`: SLURM 模板
- `onescience-agent.md`: 通用 Agent 提示词

---

## Acknowledgement

本 README 的组织方式参考了 `huggingface/skills` 这类技能仓库首页的呈现思路：突出“技能是什么、怎么安装、怎么触发、包含哪些能力”。

参考仓库：

- https://github.com/huggingface/skills

---

## License

本仓库当前未包含单独的 `LICENSE` 文件；如需开源发布，建议补充明确许可证。
