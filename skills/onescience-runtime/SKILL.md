---
name: onescience-runtime
description: 在 SLURM 环境中提交 AI4S (AI for Science) 相关代码。自动从 onescience.json 读取配置，分析用户代码情况，基于模板生成或融合 SLURM 提交脚本。除非用户指定，否则无需手动编写 SLURM 脚本。
---
# OneScience Runtime Skill

## 描述

OneScience Runtime 技能专注于在 SLURM 环境中提交 AI4S (AI for Science) 相关代码。该技能的核心功能是：

1. **自动配置读取**：从 `.trae/skills/onescience.json` 读取 SLURM 配置
2. **项目配置生成**：根据配置生成项目级别的配置文件
3. **智能脚本生成**：根据配置和代码分析，自动生成 SLURM 脚本
4. **环境配置**：自动配置 Python 运行环境（Conda/虚拟环境）
5. **脚本融合**：如果用户代码中已有 SLURM 脚本，会生成融合版本
6. **灵活提交**：支持用户指定自定义 SLURM 脚本

## 核心特性

### SLURM 模板

技能使用 `./tpl.slurm` 作为默认模板，该模板包含以下特性：

**DCU 环境配置**：
- 加载 DCU 相关模块
- 设置 DCU 可见设备
- 加载 DCU 环境变量

**Conda 环境配置**：
- 初始化 Conda
- 激活 Conda 环境
- 支持自定义 Conda 环境名

**onescience 环境配置**：
- 设置数据集目录：`ONESCIENCE_DATASETS_DIR`
- 设置模型目录：`ONESCIENCE_MODELS_DIR`
- 自动加载环境变量

**分布式训练支持**：
- 自动获取节点列表
- 设置 MASTER_ADDR 和 MASTER_PORT
- 支持多节点多卡训练
- 使用 srun 启动分布式任务

**日志和监控**：
- 输出日志到 `logs/` 目录
- 显示启动时间
- 显示 Python 和 hipcc 路径
- 显示节点和任务信息

### 脚本融合

- **检测现有脚本**：自动检测用户代码中的 SLURM 脚本
- **融合生成**：将用户脚本与模板融合，生成新脚本
- **保留特性**：保留用户脚本的特殊配置
- **环境适配**：保留模板中的 DCU、Conda 和 onescience 环境配置

### 灵活提交

- **用户指定**：用户可指定使用自定义 SLURM 脚本
- **自动提交**：生成脚本后自动提交到 SLURM
- **状态监控**：支持查询作业状态

## 配置文件

### tpl.slurm 模板说明

技能使用 `./tpl.slurm` 作为默认 SLURM 脚本模板，该模板包含完整的 DCU、Conda 和 onescience 环境配置。

**模板结构**：

```bash
#!/bin/bash
# SLURM 配置部分
#SBATCH -p {cluster.partition}
#SBATCH -N {cluster.nodes}
#SBATCH --gres={cluster.gpu_type}:{cluster.gpus_per_node}
#SBATCH --cpus-per-task={cluster.cpus_per_task}
#SBATCH --ntasks-per-node={cluster.ntasks_per_node}
#SBATCH -J {job_name}
#SBATCH --time={cluster.time_limit}
#SBATCH -o logs/%j.out
#SBATCH --exclusive

# 环境初始化
echo "START TIME: $(date)"
module purge

##### always load modules: Launch DCU ENV #####
module load sghpc-mpi-gcc/25.8
module load sghpcdas/25.6

##### python always Launch Conda ENV #####
source ~/.bashrc
conda activate {conda.env_name}

##### onescience datasets and models Launch env #####
export ONESCIENCE_DATASETS_DIR="{env_vars.ONESCIENCE_DATASETS_DIR}"
export ONESCIENCE_MODELS_DIR="{env_vars.ONESCIENCE_MODELS_DIR}"

##### Show env #####
which python

##### Set DCU #####
export HIP_VISIBLE_DEVICES={hip_visible_devices}

export OMP_NUM_THREADS={cluster.cpus_per_task}
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# 第一个节点的地址
export MASTER_ADDR=$(hostname)

# 在每个节点上启动 torchrun
echo SLURM_NNODES=$SLURM_NNODES
echo "Nodes: ${nodes_array[*]}"
echo SLURM_NTASKS=$SLURM_NTASKS

python {script.code_path}
```

**模板变量说明**：
- `{cluster.partition}`：SLURM 队列名称，从配置文件 `cluster.partition` 读取
- `{cluster.nodes}`：节点数，从配置文件 `cluster.nodes` 读取
- `{cluster.gpus_per_node}`：每节点 GPU 数，从配置文件 `cluster.gpus_per_node` 读取
- `{cluster.cpus_per_task}`：每任务 CPU 核心数，从配置文件 `cluster.cpus_per_task` 读取
- `{cluster.ntasks_per_node}`：每节点任务数，从配置文件 `cluster.ntasks_per_node` 读取
- `{cluster.time_limit}`：时间限制，从配置文件 `cluster.time_limit` 读取
- `{cluster.gpu_type}`：GPU 类型，从配置文件 `cluster.gpu_type` 读取
- `{conda.env_name}`：Conda 环境名，从配置文件 `conda.env_name` 读取
- `{env_vars.ONESCIENCE_DATASETS_DIR}`：数据集目录，从配置文件 `script.env_vars.ONESCIENCE_DATASETS_DIR` 读取
- `{env_vars.ONESCIENCE_MODELS_DIR}`：模型目录，从配置文件 `script.env_vars.ONESCIENCE_MODELS_DIR` 读取
- `{script.code_path}`：代码路径，从配置文件 `script.code_path` 读取
- `{hip_visible_devices}`：DCU 设备可见性，根据 `gpus_per_node` 动态设置

**重要说明**：
- 模板中的变量使用 `{key.path}` 格式，表示从配置文件中读取嵌套值
- 所有配置项均来自 `.trae/skills/onescience.json`，**不应硬编码具体值**
- 模板提供的是结构参考，实际运行时会根据配置文件动态替换变量

### onescience.json 配置格式

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
      "env_name": "earth310",
      "activate_script": "source ~/.bashrc && conda activate earth310"
    },
    "resources": {
      "gpu_type": "dcu",
      "cpu_memory_ratio": 8,
      "disk_space": "100GB"
    },
    "script": {
      "path": "slurm_submit.sh",
      "generate": true,
      "template": "default",
      "env_vars": {
        "ONESCIENCE_DATASETS_DIR": "/public/share/sugonhpcapp01/onestore/onedatasets/",
        "ONESCIENCE_MODELS_DIR": "/public/share/sugonhpcapp01/onestore/onemodels/"
      },
      "work_dir": ".",
      "code_path": "train.py"
    }
  }
}
```

**配置说明**：
- `mode`: 提交模式（固定为 "slurm"）
- `cluster.partition`: SLURM 队列名称（如 "hpctest02", "gpu", "largedev"）
- `cluster.nodes`: 节点数（正整数）
- `cluster.gpus_per_node`: 每节点 GPU 数（正整数）
- `cluster.cpus_per_task`: 每任务 CPU 核心数（正整数）
- `cluster.memory`: 内存大小（如 "64GB"）
- `cluster.time_limit`: 时间限制（格式 "HH:MM:SS"）
- `cluster.gpu_type`: GPU 类型（如 "dcu", "rtx3090", "a100"）
- `cluster.ntasks_per_node`: 每节点任务数（正整数）
- `modules`: 需要加载的模块列表
- `conda.enabled`: 是否启用 Conda 环境
- `conda.env_name`: Conda 环境名称
- `conda.activate_script`: Conda 激活脚本命令
- `script.env_vars`: 环境变量字典，支持自定义变量
- `script.work_dir`: 工作目录路径
- `script.code_path`: 主程序代码路径

**重要说明**：
- **不要硬编码配置值**：配置文件中的值应根据实际环境调整
- **路径配置**：`ONESCIENCE_DATASETS_DIR` 和 `ONESCIENCE_MODELS_DIR` 应指向实际的数据和模型存储路径
- **GPU 类型**：`gpu_type` 应与集群实际使用的 GPU 类型匹配
- **队列选择**：`partition` 应根据作业类型选择合适的队列

## 工作流程

### 1. 自动读取配置

技能首先尝试读取 `.trae/skills/onescience.json` 配置文件，包括：
- SLURM 集群配置（partition, nodes, gpus_per_node 等）
- Conda 环境配置（env_name, activate_script）
- 资源配置（gpu_type, cpu_memory_ratio, disk_space）
- 脚本配置（path, generate, template, env_vars, work_dir, code_path）

### 2. 项目配置生成

根据读取的配置，技能会：
- 生成项目配置文件（如 `project_config.yaml`）
- 包含完整的运行环境和资源配置
- 作为项目级别的配置参考

### 3. 分析用户代码

- 检测代码类型（训练/推理/评估）
- 检测是否需要分布式训练
- 检测是否有现有的 SLURM 脚本
- 分析代码依赖和环境要求
- 分析 Python 环境需求（版本、依赖库）

### 4. 生成或融合 SLURM 脚本

- **无现有脚本**：基于 `./tpl.slurm` 模板和配置生成新脚本
- **有现有脚本**：融合用户脚本和模板，生成新脚本
- **用户指定**：使用用户指定的脚本
- **环境配置**：根据 `conda.activate_script` 配置自动添加 Python 环境激活命令
- **模板变量替换**：将模板中的 `{cluster.partition}` 等变量替换为配置文件中的值
- **DCU 环境适配**：保留模板中的 DCU 模块加载和环境变量设置
- **onescience 环境适配**：保留模板中的 ONESCIENCE_DATASETS_DIR 和 ONESCIENCE_MODELS_DIR 环境变量

### 5. 提交作业

- 生成 SLURM 提交脚本
- 提交作业到 SLURM
- 返回作业信息

## 输入参数

### 必需参数

- `code_path`: 要运行的代码路径（如 `train.py`）

### 可选参数

- `config_path`: 配置文件路径（如 `conf/config.yaml`）
- `data_path`: 数据路径
- `output_path`: 输出路径
- `slurm_script`: 用户指定的 SLURM 脚本路径（可选，如果指定则直接使用）
- `partition`: 队列名称（覆盖配置文件）
- `nodes`: 节点数（覆盖配置文件）
- `gpus_per_node`: 每节点 GPU 数（覆盖配置文件）
- `cpus_per_task`: CPU 核心数（覆盖配置文件）
- `memory`: 内存大小（覆盖配置文件）
- `time_limit`: 时间限制（覆盖配置文件）
- `conda.env_name`: Conda 环境名（覆盖配置文件）
- `conda.activate_script`: Conda 激活脚本（覆盖配置文件）
- `work_dir`: 工作目录（覆盖配置文件）

### 覆盖配置参数

- `partition`: 队列名称（覆盖配置文件）
- `nodes`: 节点数（覆盖配置文件）
- `gpus_per_node`: 每节点 GPU 数（覆盖配置文件）
- `cpus_per_task`: CPU 核心数（覆盖配置文件）
- `memory`: 内存大小（覆盖配置文件）
- `time_limit`: 时间限制（覆盖配置文件）
- `conda.env_name`: Conda 环境名（覆盖配置文件）
- `conda.activate_script`: Conda 激活脚本（覆盖配置文件）
- `work_dir`: 工作目录（覆盖配置文件）
- `code_path`: Python 代码路径（覆盖配置文件）

## 输出结果

运行完成后返回：

1. **作业信息**
   - 作业 ID
   - 作业状态
   - 提交时间
   - 脚本路径

2. **运行结果**
   - 输出文件路径
   - 日志文件路径
   - 错误信息（如有）

3. **性能指标**
   - 运行时间
   - 资源使用情况
   - 作业队列等待时间

## 使用场景

### 场景 1：简单训练任务

用户只需提供代码路径，技能会自动读取 `.trae/skills/onescience.json` 配置，生成并提交 SLURM 脚本。

```bash
# 用户只需提供代码
code_path: fuxi_train.py
```

技能会：
1. 读取 `.trae/skills/onescience.json` 配置文件
2. 分析代码依赖和环境需求
3. 读取 `./tpl.slurm` 模板（包含 DCU、Conda 和 onescience 环境配置）
4. 生成项目配置文件（基于模板和用户配置）
5. 生成 SLURM 脚本（包含 Python 环境配置）
6. 提交作业到指定队列

### 场景 2：使用现有 SLURM 脚本

用户代码中已有 SLURM 脚本，技能会融合生成新脚本。

```bash
code_path: train.py
# 检测到同目录下有 slurm.sh
```

技能会：
1. 读取 `.trae/skills/onescience.json` 配置
2. 读取 `./tpl.slurm` 模板（包含 DCU、Conda 和 onescience 环境配置）
3. 读取现有脚本
4. 融合配置和脚本
5. 生成新脚本（包含 Python 环境配置）
6. 提交作业到指定队列

### 场景 3：指定自定义 SLURM 脚本

用户指定使用特定的 SLURM 脚本。

```bash
code_path: train.py
slurm_script: custom_slurm.sh
```

技能会：
1. 读取 `.trae/skills/onescience.json` 配置用于环境验证
2. 使用用户指定的脚本
3. 提交作业

## 工作流程示例

### 示例 1：从零生成

**输入**：
```json
{
  "code_path": "fuxi_train.py",
  "config_path": "conf/config.yaml"
}
```

**处理**：
1. 读取 `.trae/skills/onescience.json` 配置文件
2. 分析 `fuxi_train.py` 代码依赖
3. 读取 `./tpl.slurm` 模板（包含 DCU、Conda 和 onescience 环境配置）
4. 检测到使用 `torch.distributed.run`
5. 基于 `./tpl.slurm` 模板和配置生成脚本
6. 生成项目配置文件（包含 Python 环境配置）
7. 提交作业到指定队列

**生成的脚本**（示例，实际值根据配置文件动态生成）：
```bash
#!/bin/bash
#SBATCH -p hpctest02
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH -J fuxi_train
#SBATCH --time=02:00:00
#SBATCH -o logs/%j.out
#SBATCH --exclusive

echo "START TIME: $(date)"
module purge

##### always load modules: Launch DCU ENV #####
module load sghpc-mpi-gcc/25.8
module load sghpcdas/25.6

##### python always Launch Conda ENV #####
source ~/.bashrc
conda activate earth310

##### onescience datasets and models Launch env #####
export ONESCIENCE_DATASETS_DIR="/public/share/sugonhpcapp01/onestore/onedatasets/"
export ONESCIENCE_MODELS_DIR="/public/share/sugonhpcapp01/onestore/onemodels/"

##### Show env #####
which python

##### Set DCU #####
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export OMP_NUM_THREADS=8
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

export MASTER_ADDR=$(hostname)

echo SLURM_NNODES=$SLURM_NNODES
echo "Nodes: ${nodes_array[*]}"
echo SLURM_NTASKS=$SLURM_NTASKS

python fuxi_train.py
```

### 示例 2：融合现有脚本

**输入**：
```json
{
  "code_path": "train.py"
}
```

**现有脚本** (`slurm.sh`)：
```bash
#!/bin/bash
#SBATCH -p hpctest02
#SBATCH -N 4
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH -J GraphCast
#SBATCH --time=72:00:00
#SBATCH -o logs/%j.out
#SBATCH --exclusive

# ... 用户自定义配置 ...
python train.py
```

**处理**：
1. 读取 `.trae/skills/onescience.json` 配置（节点数、GPU 等）
2. 读取现有脚本
3. 融合配置和脚本
4. 生成新脚本（包含 Python 环境配置）
5. 提交作业到指定队列

**生成的新脚本**（示例，实际值根据配置文件动态生成）：
```bash
#!/bin/bash
#SBATCH -p hpctest02
#SBATCH -N 4
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH -J GraphCast
#SBATCH --time=72:00:00
#SBATCH -o logs/%j.out
#SBATCH --exclusive

echo "START TIME: $(date)"
module purge

##### if DCU: Launch DCU ENV #####
module load sghpc-mpi-gcc/25.8
module load sghpcdas/25.6

##### python always Launch Conda ENV #####
source ~/.bashrc
conda activate earth310

##### onescience datasets and models Launch env #####
export ONESCIENCE_DATASETS_DIR="/public/share/sugonhpcapp01/onestore/onedatasets/"
export ONESCIENCE_MODELS_DIR="/public/share/sugonhpcapp01/onestore/onemodels/"

##### Show env #####
which python

##### Set DCU #####
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export OMP_NUM_THREADS=16
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

export MASTER_ADDR=$(hostname)

echo SLURM_NNODES=$SLURM_NNODES
echo "Nodes: ${nodes_array[*]}"
echo SLURM_NTASKS=$SLURM_NTASKS

python train.py
```

### 示例 3：指定自定义脚本

**输入**：
```json
{
  "code_path": "train.py",
  "slurm_script": "my_slurm.sh"
}
```

**处理**：
1. 读取用户指定的 `my_slurm.sh`
2. 读取 `.trae/skills/onescience.json` 配置用于环境验证
3. 直接提交该脚本

## 生成逻辑

### 1. 配置读取

- **读取配置文件**：从 `.trae/skills/onescience.json` 读取 SLURM 配置
- **解析环境配置**：解析 Conda 环境、Python 版本等
- **合并用户参数**：合并命令行参数和配置文件

### 2. 代码分析

- **检测分布式训练**：检查是否使用 `torch.distributed.run` 或 `torchrun`
- **检测训练类型**：检查代码中的训练逻辑
- **检测依赖**：检查代码中的 import 语句
- **分析 Python 环境需求**：分析代码对 Python 版本和库的依赖

### 3. 模板选择

- **默认模板**：使用 `./tpl.slurm`
- **自定义模板**：根据配置选择

### 4. 脚本生成

- **读取模板**：从 `./tpl.slurm` 读取模板（包含 DCU、Conda 和 onescience 环境配置）
- **替换变量**：根据配置替换模板中的变量（partition, nodes, memory 等）
- **添加环境配置**：根据 `conda.activate_script` 配置添加 Python 环境激活命令
- **设置环境变量**：设置 WORLD_SIZE、MASTER_ADDR 等分布式训练环境变量
- **设置工作目录**：设置正确的 cd 命令
- **生成项目配置文件**：基于模板生成项目级别的配置文件（如 `project_config.yaml`）
- **保留模板特性**：保留模板中的 DCU 模块加载、onescience 环境变量等配置

### 5. 脚本融合

- **保留用户配置**：保留用户脚本中的特殊配置
- **应用新配置**：应用新的资源配置（从 `.trae/skills/onescience.json` 读取）
- **合并环境设置**：合并环境设置（Conda 激活命令等）
- **生成项目配置**：生成或更新项目级别的配置文件
- **模板特性保留**：保留模板中的 DCU、Conda 和 onescience 环境配置

## 配置优先级

1. **用户指定参数**（最高优先级）
2. **命令行参数**（覆盖配置文件）
3. **onescience.json 配置文件**
4. **默认值**（最低优先级）

## 最佳实践

1. **配置文件**：使用 `.trae/skills/onescience.json` 存储常用配置
2. **代码分析**：确保代码结构清晰，便于分析
3. **模板定制**：根据集群特点定制 `./tpl.slurm`，保留 DCU、Conda 和 onescience 环境配置
4. **脚本测试**：在小规模上测试生成的脚本
5. **日志监控**：定期检查作业日志
6. **环境配置**：确保 `conda.activate_script` 配置正确，支持自定义 Python 环境
7. **项目配置**：生成 `project_config.yaml` 作为项目级别的配置参考
8. **资源管理**：合理设置 `memory` 和 `cpus_per_task`，避免资源请求过高
9. **模板变量**：确保模板中的 `{cluster.partition}` 等变量在配置文件中正确设置

## 注意事项

- **不要硬编码配置值**：配置文件中的值应根据实际环境调整
- **确保路径正确**：`ONESCIENCE_DATASETS_DIR` 和 `ONESCIENCE_MODELS_DIR` 应指向实际的存储路径
- **确保 Conda 环境已正确安装并可激活**
- **注意资源请求合理性**：避免 `--mem` 请求过高导致提交失败
- **DCU 环境配置**：HIP_VISIBLE_DEVICES、模块加载会自动保留
- **onescience 环境变量**：ONESCIENCE_DATASETS_DIR、ONESCIENCE_MODELS_DIR 会自动保留
- **模板变量格式**：使用 `{key.path}` 格式表示从配置文件读取嵌套值，而非硬编码值

## 示例配置

### 单卡训练配置

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
      "env_name": "earth310",
      "activate_script": "source ~/.bashrc && conda activate earth310"
    },
    "script": {
      "env_vars": {
        "ONESCIENCE_DATASETS_DIR": "/your/datasets/path/",
        "ONESCIENCE_MODELS_DIR": "/your/models/path/"
      },
      "code_path": "train.py"
    }
  }
}
```

### 多卡训练配置

```json
{
  "runtime": {
    "mode": "slurm",
    "cluster": {
      "partition": "gpu",
      "nodes": 1,
      "gpus_per_node": 4,
      "cpus_per_task": 32,
      "memory": "128GB",
      "time_limit": "24:00:00",
      "gpu_type": "a100",
      "ntasks_per_node": 4
    },
    "conda": {
      "enabled": true,
      "env_name": "pytorch"
    },
    "script": {
      "env_vars": {
        "ONESCIENCE_DATASETS_DIR": "/your/datasets/path/",
        "ONESCIENCE_MODELS_DIR": "/your/models/path/"
      },
      "code_path": "train.py"
    }
  }
}
```

### 多节点训练配置

```json
{
  "runtime": {
    "mode": "slurm",
    "cluster": {
      "partition": "largedev",
      "nodes": 8,
      "gpus_per_node": 8,
      "cpus_per_task": 16,
      "memory": "128GB",
      "time_limit": "72:00:00",
      "gpu_type": "dcu",
      "ntasks_per_node": 8
    },
    "conda": {
      "enabled": true,
      "env_name": "earth310"
    },
    "script": {
      "env_vars": {
        "ONESCIENCE_DATASETS_DIR": "/your/datasets/path/",
        "ONESCIENCE_MODELS_DIR": "/your/models/path/"
      },
      "code_path": "train.py"
    }
  }
}
```

**重要说明**：
- 所有示例中的路径和配置值应根据实际环境调整
- **不要直接使用示例中的具体路径**，应替换为实际的存储路径
- `gpu_type` 应与集群实际使用的 GPU 类型匹配
- `partition` 应根据作业类型选择合适的队列
