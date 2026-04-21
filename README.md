# <div align="center"><strong>oneskills</strong></div>

### <div align="center">OneScience Skills Library for AI-native Scientific Research</div>

> 🚀 OneSkills = Knowledge × Skills × Agents

***

## 📖 项目简介

OneSkills 是基于 OneScience 构建的开源知识与能力库（Skills Library），专为智能体（Agents）开发设计，提供可复用、可组合、可扩展的能力模块，聚焦 AI4S（AI for Science）科学智能科研领域，实现从**环境安装、科学代码生成、自动提交、代码测试**的全流程自动化科研能力，显著提升科学研究效率与代码生成效果。

### 核心特性

- 📚 结构化知识库：基于 OneScience 的科学知识与模型组件文档
- 🧠 可执行能力模块：标准化的 Skills 定义与工作流
- 🔗 Agent-ready 接口：专为 Claude Code、Trae 等智能体设计
- 🌐 多领域覆盖：气象、生信、材料、流体等科学领域

**核心 Skills**：

| 目录                              | 作用                    |
| ------------------------------- | --------------------- 
| onescience-installer/       | **基于DCU的开发环境安装智能安装**             |
| onescience-coder/          | **基于成熟模型的代码生成与改造**  |
| onescience-runtime/         | **基于DCU的智能化作业提交**          |
| onescience-skill/           | **需求拆解与管理**             |
| onescience-test/            | **代码测试与验证**             |

<br />


## ⚙️ 快速开始

### 安装使用

OneSkills 是嵌入式开发工具库，无需独立安装。为获得更完整的功能支持，处理复杂任务时建议配合 **OneScience** 使用。 将两者克隆到同一项目目录后，复制 skills 到对应智能体的配置目录:

```bash
git clone https://github.com/onescience-ai/onescience.git # 可选
git clone https://github.com/onescience-ai/oneskills.git


cp -r oneskills/skills /your/project/.trae/skills         # Trae

# 其他智能体
cp -r oneskills/skills /your/project/.claude/skills      # Claude Code
cp -r oneskills/skills /your/project/.codex/skills       # Codex CLI
cp -r oneskills/skills /your/project/.opencode/skills   # OpenCode

```

### Trae 远程连接 SSH 配置

在 Trae 中配置远程连接 SSH 的步骤如下：

1. 打开 Trae IDE，点击左侧边栏的 "远程连接" 图标
2. 添加新连接，在scnet下载ssh-key登录，配置ssh远程连接
3. 测试连接，确保能够成功连接到远程服务器
4. 配置工作目录，设置为你的项目路径

### 使用 OneSkills开始开发

Skills 通过自然语言提示词触发。以下是一个数据读取分析任务示例：

```
生成读取部分ERA5数据集代码， 并使用slurm提交运行

```

将上述提示提交到 `OneScience科研智能体` 对话框，即可自动生成代码，并提交运行。
如图：

![usage_screen](./assets/trae_usage.png)


### 配置文件说明

#### onescience.json

`onescience.json` 是 OneScience 的运行时配置文件，用于定义作业提交的参数和环境设置。需要复制到用户当前工程的根目录。

**主要配置项**：
- `runtime.mode`：运行模式，支持 `slurm` 等
- `runtime.cluster`：集群配置，包括分区、节点数、GPU 数量等
- `runtime.modules`：需要加载的环境模块
- `runtime.conda`：conda 环境配置
- `runtime.script`：作业脚本配置

**示例配置**：
```json
{
  "runtime": {
    "mode": "slurm",
    "cluster": {
      "partition": "hpctest02",
      "nodes": 1,
      "gpus_per_node": 1,
      "cpus_per_task": 8,
      "memory": "64GB",
      "time_limit": "02:00:00",
      "gpu_type": "dcu",
      "ntasks_per_node": 1
    },
    "modules": [
      "sghpc-mpi-gcc/26.3",
      "sghpcdas/25.6"
    ],
    "conda": {
      "enabled": true,
      "env_name": "onescience311",
      "activate_script": "source ~/.bashrc && conda activate onescience311"
    },
    "script": {
      "path": "slurm_submit.sh",
      "generate": true,
      "template": "default",
      "job_name": "era5_dataloader",
      "code_path": "era5_reader.py",
      "env_vars": {
        "ONESCIENCE_DATASETS_DIR": "/public/share/sugonhpcapp01/onestore/onedatasets/",
        "ONESCIENCE_MODELS_DIR": "/public/share/sugonhpcapp01/onestore/onemodels/"
      },
      "work_dir": "."
    }
  }
}
```

#### tpl.slurm

`tpl.slurm` 是 Slurm 作业提交模板文件，用于生成实际的作业提交脚本。需要复制到用户当前工程的根目录。

**主要功能**：
- 定义作业的资源需求（CPU、GPU、内存等）
- 配置环境变量和模块加载
- 设置 Conda 环境激活
- 执行用户指定的代码

**使用方法**：
1. 将 `tpl.slurm` 复制到项目根目录
2. 根据需要修改 `onescience.json` 中的配置




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

***

## 🌍 应用场景

| 场景                   | 描述         | 示例                |
| -------------------- | ---------- | ----------------- |
| 📊 **数据分析助手**        | 科学数据处理与可视化 | 气象数据读取、网格插值、结果可视化 |
| 🧑‍💻 **开发辅助 Agent** | 模型代码生成与改造  | 模块替换、特征融合、架构优化    |
| 🧠 **AI Copilot 系统** | 交互式智能开发    | 自然语言编程、错误诊断、性能优化  |
| 🔬 **科研工作流**         | 复杂科学计算任务   | 多模型组合、跨领域适配、自动化流程 |

***

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

***

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

***

## ⭐ Star History

如果这个项目对你有帮助，欢迎 ⭐️ 支持！

***

<div align="center">
  <strong>Built with OneScience for AI-native Scientific Research</strong>
</div>
