# OneScience Runtime Skill

## 描述

OneScience Runtime 技能用于在 OneScience 环境中运行 AI4S (AI for Science) 相关代码。该技能支持两种提交方式：

1. **SLURM 方式**：提交到集群，需要检查或新生成提交脚本
2. **SCNET MCP 接口方式**：调用 SCNET MCP 接口提交

运行完成后会返回结果。

## 功能特性

### SLURM 提交方式

- 检查现有提交脚本或新生成提交脚本
- 支持配置集群资源（CPU/GPU、内存、时间等）
- 提交作业到 SLURM 集群
- 监控作业状态
- 收集运行结果

### SCNET MCP 接口提交方式

- 调用 SCNET MCP 接口提交作业
- 支持配置作业参数
- 异步提交并获取作业 ID
- 查询作业状态
- 返回运行结果

## 使用场景

- 运行 AI4S 模型训练代码
- 执行科学计算任务
- 提交大规模数据处理作业
- 运行推理和评估脚本

## 提交方式选择

### 使用 SLURM 方式

适用于：
- 本地或私有集群环境
- 需要完全控制提交脚本
- 需要检查或修改提交参数

**配置示例**：
```yaml
runtime:
  mode: "slurm"
  cluster:
    partition: "gpu"
    nodes: 1
    gpus_per_node: 4
    cpus_per_gpu: 8
    memory: "64GB"
    time_limit: "48:00:00"
  script:
    path: "/path/to/submit.sh"
    generate: true  # 是否新生成脚本
```

### 使用 SCNET MCP 方式

适用于：
- 使用 SCNET 平台
- 需要与 SCNET 平台集成
- 通过 MCP 接口统一管理作业

**配置示例**：
```yaml
runtime:
  mode: "scnet_mcp"
  mcp_endpoint: "https://mcp.scnet.org/api/v1"
  mcp_token: "${SCNET_MCP_TOKEN}"
  resource:
    gpu_type: "A100"
    gpu_count: 4
    cpu_count: 16
    memory: "64GB"
```

## 输入参数

### 基础参数

- `code_path`: 要运行的代码路径
- `config_path`: 配置文件路径（可选）
- `data_path`: 数据路径（可选）
- `output_path`: 输出路径（可选）

### SLURM 特定参数

- `partition`: 集群分区
- `nodes`: 节点数
- `gpus_per_node`: 每节点 GPU 数
- `cpus_per_gpu`: 每 GPU CPU 数
- `memory`: 内存大小
- `time_limit`: 时间限制
- `script_path`: 提交脚本路径
- `generate_script`: 是否生成新脚本

### SCNET MCP 特定参数

- `mcp_endpoint`: MCP 接口地址
- `mcp_token`: MCP 认证令牌
- `gpu_type`: GPU 类型
- `gpu_count`: GPU 数量
- `cpu_count`: CPU 数量
- `memory`: 内存大小

## 输出结果

运行完成后返回：

1. **作业信息**
   - 作业 ID
   - 作业状态
   - 提交时间
   - 完成时间

2. **运行结果**
   - 输出文件路径
   - 日志文件路径
   - 错误信息（如有）

3. **性能指标**
   - 运行时间
   - 资源使用情况
   - 作业队列等待时间

## 示例

### SLURM 方式示例

```python
from onescience.runtime import submit_to_slurm

result = submit_to_slurm(
    code_path="/path/to/train.py",
    config_path="/path/to/config.yaml",
    partition="gpu",
    nodes=1,
    gpus_per_node=4,
    generate_script=True
)

print(f"作业 ID: {result.job_id}")
print(f"作业状态: {result.status}")
print(f"输出路径: {result.output_path}")
```

### SCNET MCP 方式示例

```python
from onescience.runtime import submit_to_scnet_mcp

result = submit_to_scnet_mcp(
    code_path="/path/to/train.py",
    config_path="/path/to/config.yaml",
    gpu_type="A100",
    gpu_count=4,
    cpu_count=16
)

print(f"作业 ID: {result.job_id}")
print(f"作业状态: {result.status}")
```

## 错误处理

- 作业提交失败：返回错误信息和可能的解决方案
- 作业运行失败：返回日志文件路径和错误详情
- 资源不足：提示调整资源配置
- 超时：返回超时信息和建议

## 最佳实践

1. **资源预估**：根据任务需求合理预估资源
2. **脚本检查**：使用 SLURM 方式时检查提交脚本
3. **结果保存**：确保输出路径配置正确
4. **日志监控**：定期检查作业日志
5. **错误处理**：处理可能的错误情况

## 集成建议

- 与 OneScience Model Skill 集成
- 与 OneScience Data Skill 集成
- 与 OneScience Pipeline Skill 集成

## 注意事项

- 确保集群环境配置正确
- SCNET MCP 方式需要有效的认证令牌
- 大型任务建议使用 SCNET MCP 方式
- 小型任务可以使用 SLURM 方式
