---
name: onescience-runtime
description: 在用户使用slurm提交运行情况下
---

***

name: onescience-runtime
description: 在 SLURM 环境中自动提交代码。读取用户手动配置的 onescience.json 配置，基于 tpl.slurm 模板生成 SLURM 脚本并提交。
------------------------------------------------------------------------------------------

# OneScience Runtime Skill

## 功能

在 SLURM 环境中自动提交 AI4S 代码：

- 读取 `onescience.json` 配置
- 基于 `tpl.slurm` 模板生成 SLURM 脚本（**模板内容固定，禁止修改**）
- 仅使用模板中的变量替换机制，不添加额外内容
- 自动提交作业到 SLURM
- 支持 DCU 环境配置和分布式训练

## 用户配置文件路径

`onescience.json`

## 配置项说明

### 核心配置

| 配置项                               | 描述         | 示例值                                      |
| --------------------------------- | ---------- | ---------------------------------------- |
| `runtime.mode`                    | 运行模式       | "slurm"                                  |
| `runtime.cluster.partition`       | SLURM 分区   | "hpctest02"                              |
| `runtime.cluster.nodes`           | 节点数        | 1                                        |
| `runtime.cluster.gpus_per_node`   | 每节点 GPU 数  | 1                                        |
| `runtime.cluster.cpus_per_task`   | 每任务 CPU 数  | 8                                        |
| `runtime.cluster.time_limit`      | 时间限制       | "02:00:00"                               |
| `runtime.cluster.gpu_type`        | GPU 类型     | "dcu"                                    |
| `runtime.cluster.ntasks_per_node` | 每节点任务数     | 1                                        |
| `runtime.modules`                 | 模块列表       | \["sghpc-mpi-gcc/26.3", "sghpcdas/25.6"] |
| `runtime.conda.env_name`          | Conda 环境名称 | "onescience311"                          |
| `runtime.script.job_name`         | 作业名称       | "onescience\_job"                        |
| `runtime.script.code_path`        | 代码路径       | "train.py"                               |
| `runtime.script.env_vars`         | 环境变量       | 包含 ONESCIENCE\_DATASETS\_DIR 等           |

## SLURM 模板详解

**重要：模板文件路径为** **`tpl.slurm`，模板内容固定，禁止修改。**

模板结构如下：

### 1. SLURM 作业配置（必须保留）

```bash
#!/bin/bash
#SBATCH -p {cluster.partition}        # 分区设置
#SBATCH -N {cluster.nodes}            # 节点数
#SBATCH --gres={cluster.gpu_type}:{cluster.gpus_per_node}  # GPU资源
#SBATCH --cpus-per-task={cluster.cpus_per_task}  # CPU资源
#SBATCH --ntasks-per-node={cluster.ntasks_per_node}  # 每节点任务数
#SBATCH -J {job_name}                 # 作业名称
#SBATCH --time={cluster.time_limit}   # 时间限制
#SBATCH -o logs/%j.out                # 输出日志
#SBATCH --exclusive                   # 独占模式
```

### 2. 环境初始化（必须保留）

```bash
echo "START TIME: $(date)"
module purge

source /etc/profile
source /etc/profile.d/modules.sh
module use /work2/share/sghpc_sdk/modulefiles/
```

### 3. DCU 环境配置（必须保留）

```bash
##### if DCU: Launch DCU ENV #####
{modules_config}
```

### 4. Python Conda 环境（必须保留）

```bash
##### python always Launch Conda ENV #####
source ~/.bashrc
conda activate {conda.env_name}

source $ROCM_PATH/cuda/env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```

### 5. OneScience 环境变量（必须保留）

```bash
##### onescience datasets and models Launch env #####
export ONESCIENCE_DATASETS_DIR="{env_vars.ONESCIENCE_DATASETS_DIR}"
export ONESCIENCE_MODELS_DIR="{env_vars.ONESCIENCE_MODELS_DIR}"
```

### 6. 运行环境信息（必须保留）

```bash
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
```

### 7. 代码执行（必须保留）

```bash
python {script.code_path}
```

## 模板变量说明

模板中的变量会被 `onescience.json` 中的配置自动替换：

| 变量                                   | 来源                                                | 描述            |
| ------------------------------------ | ------------------------------------------------- | ------------- |
| `{cluster.partition}`                | `runtime.cluster.partition`                       | SLURM 分区名称    |
| `{cluster.nodes}`                    | `runtime.cluster.nodes`                           | 节点数量          |
| `{cluster.gpu_type}`                 | `runtime.cluster.gpu_type`                        | GPU 类型（如 dcu） |
| `{cluster.gpus_per_node}`            | `runtime.cluster.gpus_per_node`                   | 每节点 GPU 数量    |
| `{cluster.cpus_per_task}`            | `runtime.cluster.cpus_per_task`                   | 每任务 CPU 数量    |
| `{cluster.ntasks_per_node}`          | `runtime.cluster.ntasks_per_node`                 | 每节点任务数        |
| `{cluster.time_limit}`               | `runtime.cluster.time_limit`                      | 时间限制          |
| `{job_name}`                         | `runtime.script.job_name`                         | 作业名称          |
| `{conda.env_name}`                   | `runtime.conda.env_name`                          | Conda 环境名称    |
| `{script.code_path}`                 | `runtime.script.code_path`                        | 要执行的代码路径      |
| `{env_vars.ONESCIENCE_DATASETS_DIR}` | `runtime.script.env_vars.ONESCIENCE_DATASETS_DIR` | 数据集目录         |
| `{env_vars.ONESCIENCE_MODELS_DIR}`   | `runtime.script.env_vars.ONESCIENCE_MODELS_DIR`   | 模型目录          |
| `{modules_config}`                   | `runtime.modules`                                 | 自动生成的模块加载命令   |
| `{hip_visible_devices}`              | `runtime.cluster.gpus_per_node`                   | GPU 设备可见性配置   |

## 模板使用规则

### 1. 模板固定性

- **模板文件** **`tpl.slurm`** **内容固定**
- **禁止修改模板文件**
- **禁止在生成的 SLURM 脚本中添加模板之外的内容**
- **只能使用模板中定义的变量替换机制**

### 2. 变量替换规则

- 所有 `{variable}` 格式的变量都会被 `onescience.json` 中的对应配置替换
- 变量不存在时，保持原样（不替换）
- 不允许在模板之外添加新的配置项

### 3. 生成脚本结构

生成的 SLURM 脚本必须严格遵循模板结构：

```
#!/bin/bash
#SBATCH 配置行
...
环境初始化
...
DCU 环境配置
...
Conda 环境
...
OneScience 环境变量
...
运行环境信息
...
代码执行
```

## 输出

- **作业信息**：作业 ID、状态、提交时间
- **日志文件**：`logs/%j.out`（其中 %j 为作业 ID）
- **运行时间**：作业开始和结束时间
- **环境信息**：Python 路径、节点列表、SLURM 环境变量

## 最佳实践

1. **日志目录**：确保 `logs` 目录存在，否则作业会失败
2. **资源规划**：根据任务类型合理设置节点数和 GPU 数量
3. **时间限制**：为作业设置合理的时间限制，避免资源浪费
4. **模块依赖**：确保在 `modules` 中指定了所有必要的模块
5. **环境变量**：正确设置 `ONESCIENCE_DATASETS_DIR` 和 `ONESCIENCE_MODELS_DIR` 路径
6. **模板固定**：模板内容固定，禁止修改，只能通过变量替换进行配置

## 故障排查

### 常见问题

1. **作业提交失败**
   - 检查 SLURM 分区是否存在
   - 验证资源请求是否合理
   - 确保 `logs` 目录存在
   - 检查模板文件是否存在且未被修改
2. **环境激活失败**
   - 检查 Conda 环境是否存在
   - 验证 `.bashrc` 文件是否正确
   - 确认模块路径是否正确
3. **数据路径错误**
   - 确认 `ONESCIENCE_DATASETS_DIR` 环境变量设置正确
   - 检查数据集是否存在于指定路径
4. **模块加载失败**
   - 验证模块名称是否正确
   - 检查模块是否在当前环境中可用
   - 确认模块路径是否已正确设置
5. **模板相关错误**
   - 确认模板文件路径：`tpl.slurm`
   - 确认模板文件未被修改
   - 检查模板变量是否正确配置

## 注意事项

- **模板固定性**：模板文件内容固定，禁止修改，只能通过变量替换进行配置
- **配置文件只读**：`onescience.json` 为只读文件，禁止自动修改
- **日志路径固定**：日志文件固定输出到 `logs/%j.out`
- **DCU 环境**：模板默认配置为 DCU 环境，不可修改
- **分布式训练**：模板已配置分布式训练所需的环境变量
- **独占模式**：默认使用 `--exclusive` 模式，确保作业获得完整节点资源
- **变量替换**：只允许使用模板中定义的变量，不允许添加新变量
- **脚本生成**：生成的 SLURM 脚本必须严格遵循模板结构，不允许添加额外内容