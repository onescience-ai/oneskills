---
name: AI4S Pipeline Master
description: AI for Science 研究工作流主技能，统筹整个 AI4S 研究 pipeline，协调各子技能完成复杂任务,包括模型复现、数据加载分析、模型训练、模型诊断等
---

# Skill: AI4S Pipeline Master

## 能力描述

本技能作为 AI4S 研究工作流的主控技能，负责统筹整个研究 pipeline，协调各个子技能（Data Engineer、Model Architect、Trainer、Infra Optimizer、Debugger）完成复杂的科学研究任务。基于 OneScience 组件库和构建的知识体系，提供从数据接入到模型诊断的全流程支持。

## 核心能力

### 1. Pipeline 统筹能力

本技能负责整个 AI4S Pipeline 的协调和调度：

```
AI4S Pipeline 
├── Data Layer
│   ├── Ingestion (数据接入)
│   ├── Processing (数据处理)
│   ├── Fusion (多源融合)
│   └── Benchmark (基准构建)
│
├── Model Layer
│   ├── Build (模型搭建)
│   ├── Modify (模型修改)
│   ├── Innovate (模型创新)
│   └── Reproduce (论文复现)
│
├── Training Layer
│   ├── Strategy (训练策略)
│   └── Pipeline (训练流程)
│
├── System Layer
│   ├── Optimization (计算优化)
│   ├── Parallelism (并行策略)
│   └── Platform (平台适配)
│
├── Optimization Layer
│   └── Compression (模型压缩)
│
├── Evaluation Layer
│   └── Analysis (结果分析)
│
└── Debug Layer
    └── Diagnosis (模型诊断)
```

### 2. 子技能协调

根据任务需求，自动调用合适的子技能：

| 子技能 | 职责 | 调用时机 |
|-------|------|---------|
| data-engineer | 数据接入、处理、融合 | 数据相关任务 |
| model-architect | 模型搭建、修改、创新 | 模型架构任务 |
| trainer | 训练策略、流程配置 | 训练相关任务 |
| infra-optimizer | 系统优化、并行、平台适配 | 工程优化任务 |
| debugger | 诊断、调试、问题定位 | 异常和调试任务 |

### 3. 任务分解能力

将复杂任务分解为子任务，并分配给合适的子技能：

```python
# 任务分解示例
def decompose_task(user_request):
    # 识别主任务类型
    task_type = identify_task_type(user_request)
    
    # 分解为子任务
    sub_tasks = []
    
    if "数据" in user_request:
        sub_tasks.append({
            "skill": "data-engineer",
            "task": "data_ingestion"
        })
    
    if "模型" in user_request:
        sub_tasks.append({
            "skill": "model-architect",
            "task": "model_build"
        })
    
    if "训练" in user_request:
        sub_tasks.append({
            "skill": "trainer",
            "task": "training_pipeline"
        })
    
    if "优化" in user_request:
        sub_tasks.append({
            "skill": "infra-optimizer",
            "task": "system_optimization"
        })
    
    return sub_tasks
```

## 使用方法

### 场景 1：完整论文复现任务

**用户需求**：复现 GraphCast 论文，包括数据处理、模型搭建、训练和优化。

**Prompt**：
```
基于./oneskills/trae/task/react_prompt.md中的工作流执行以下任务：
完整复现 GraphCast 论文，要求：
1. 数据层：处理 ERA5 气象数据，构建训练/验证集
2. 模型层：搭建 GraphCast 模型架构
3. 训练层：配置训练流程，启用混合精度和梯度检查点
4. 系统层：配置多GPU并行训练
5. 诊断层：监控训练过程，确保稳定性
6. 保存至 graphcast_complete.py
```

**执行流程**：
1. 主技能解析任务，分解为多个子任务
2. 调用 data-engineer 处理数据
3. 调用 model-architect 搭建模型
4. 调用 trainer 配置训练流程
5. 调用 infra-optimizer 配置系统优化
6. 调用 debugger 监控训练过程
7. 汇总结果，生成完整代码

### 场景 2：模型架构创新任务

**用户需求**：在 GraphCast 基础上创新设计多尺度融合模型。

**Prompt**：
```
基于./oneskills/trae/task/react_prompt.md中的工作流执行以下任务：
在 GraphCast 基础上创新设计多尺度融合模型，要求：
1. 数据层：支持全球-区域多源数据融合
2. 模型层：在 GraphCast 架构上创新设计多尺度融合模块
3. 训练层：配置微调训练流程
4. 保存至 multiscale_graphcast.py
```

**执行流程**：
1. 主技能解析任务，识别为模型创新任务
2. 调用 data-engineer 处理多源数据融合
3. 调用 model-architect 进行架构创新
4. 调用 trainer 配置微调训练
5. 汇总结果，生成创新模型代码

### 场景 3：模型优化和压缩任务

**用户需求**：优化 Pangu-Weather 模型，降低参数量同时保持性能。

**Prompt**：
```
基于./oneskills/trae/task/react_prompt.md中的工作流执行以下任务：
优化 Pangu-Weather 模型，要求：
1. 系统层：启用混合精度训练和梯度检查点
2. 优化层：实现模型压缩（剪枝/量化/蒸馏）
3. 诊断层：验证压缩后性能
4. 保存至 optimized_pangu.py
```

**执行流程**：
1. 主技能解析任务，识别为优化任务
2. 调用 infra-optimizer 进行系统优化
3. 调用 model-architect 进行模型压缩
4. 调用 debugger 验证压缩效果
5. 汇总结果，生成优化代码

## 核心规则

### 1. 任务分解规则（强制）

所有复杂任务必须分解为子任务：

```python
# 任务分解规则
def decompose_task(user_request):
    # 1. 识别主任务类型
    task_type = identify_task_type(user_request)
    
    # 2. 分解为子任务
    sub_tasks = []
    
    # 3. 为每个子任务分配技能
    for sub_task in sub_tasks:
        skill = assign_skill(sub_task)
        sub_task["skill"] = skill
    
    return sub_tasks
```

### 2. 子技能调用规则（强制）

子技能调用必须遵循以下流程：

```python
# 子技能调用流程
def call_sub_skill(skill_name, task_description):
    # 1. 验证技能存在
    if not skill_exists(skill_name):
        raise SkillNotFoundError(f"Skill {skill_name} not found")
    
    # 2. 调用技能
    result = execute_skill(skill_name, task_description)
    
    # 3. 验证结果
    if not validate_result(result):
        raise ValidationError(f"Invalid result from {skill_name}")
    
    return result
```

### 3. 结果汇规则（强制）

所有子技能结果必须汇总：

```python
# 结果汇总规则
def aggregate_results(sub_results):
    # 1. 验证所有结果
    for result in sub_results:
        if not validate(result):
            raise ValidationError(f"Invalid result: {result}")
    
    # 2. 汇总结果
    aggregated = {
        "code": merge_code(sub_results),
        "config": merge_config(sub_results),
        "logs": collect_logs(sub_results)
    }
    
    return aggregated
```

## 子技能接口

### 1. data-engineer 接口

```python
def data_engineer_task(task_description):
    """
    数据层任务处理
    
    任务类型：
    - data_ingestion: 数据接入
    - data_processing: 数据处理
    - data_fusion: 多源融合
    - benchmark: Benchmark 构建
    """
    pass
```

### 2. model-architect 接口

```python
def model_architect_task(task_description):
    """
    模型层任务处理
    
    任务类型：
    - model_build: 模型搭建
    - model_modify: 模型修改
    - model_innovate: 模型创新
    - model_reproduce: 论文复现
    """
    pass
```

### 3. trainer 接口

```python
def trainer_task(task_description):
    """
    训练层任务处理
    
    任务类型：
    - training_strategy: 训练策略
    - training_pipeline: 训练流程
    """
    pass
```

### 4. infra-optimizer 接口

```python
def infra_optimizer_task(task_description):
    """
    系统层任务处理
    
    任务类型：
    - system_optimization: 系统优化
    - parallelism: 并行策略
    - platform_adaptation: 平台适配
    """
    pass
```

### 5. debugger 接口

```python
def debugger_task(task_description):
    """
    诊断层任务处理
    
    任务类型：
    - model_diagnosis: 模型诊断
    - training_debug: 训练调试
    - performance_analysis: 性能分析
    """
    pass
```

## 典型应用流程

### 步骤 1：任务解析

扫描资源：
- 组件知识：`./oneskills/component_knowledge/modules/*.md`
- 代码规范：`./oneskills/trae/code_standard/*.md`
- 领域知识：`./oneskills/trae/knowledge/*.md`

解析任务：
- 识别主任务类型
- 分解为子任务
- 分配子技能

### 步骤 2：子任务执行

按顺序执行子任务：
1. 数据层任务（data-engineer）
2. 模型层任务（model-architect）
3. 训练层任务（trainer）
4. 系统层任务（infra-optimizer）
5. 诊断层任务（debugger）

### 步骤 3：结果汇总

汇总所有子技能结果：
- 合并代码
- 合并配置
- 收集日志
- 验证完整性

### 步骤 4：最终验证

验证最终结果：
- 组件导入正确性
- 参数匹配性
- 可运行性

## 输出格式

### 主技能输出格式

```python
"""
AI4S Pipeline Master - {任务名称}

任务描述：{任务描述}
执行流程：{执行流程}
子技能调用：{子技能调用列表}
"""

# 导入必要的组件
from onescience.modules import Component1, Component2, Component3

# 主模型定义
class MainModel(nn.Module):
    """
    主模型描述
    
    架构：
    - Encoder: ...
    - Processor: ...
    - Decoder: ...
    
    子技能调用：
    - data-engineer: ...
    - model-architect: ...
    - trainer: ...
    - infra-optimizer: ...
    - debugger: ...
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Encoder
        self.encoder = Component1(**encoder_config)
        
        # Processor
        self.processor = Component2(**processor_config)
        
        # Decoder
        self.decoder = Component3(**decoder_config)
    
    def forward(self, x):
        # 前向传播逻辑
        return output


if __name__ == '__main__':
    # 测试代码
    model = MainModel(**model_config)
    x = torch.randn(**input_shape)
    output = model(x)
    print(f'Output shape: {output.shape}')
```

### 子技能调用记录

```python
# 子技能调用记录
sub_skill_calls = [
    {
        "skill": "data-engineer",
        "task": "data_ingestion",
        "description": "处理 ERA5 气象数据",
        "output": "data/era5_processed/"
    },
    {
        "skill": "model-architect",
        "task": "model_build",
        "description": "搭建 GraphCast 模型",
        "output": "models/graphcast.py"
    },
    {
        "skill": "trainer",
        "task": "training_pipeline",
        "description": "配置训练流程",
        "output": "configs/training_config.yaml"
    },
    {
        "skill": "infra-optimizer",
        "task": "system_optimization",
        "description": "配置系统优化",
        "output": "configs/system_config.yaml"
    },
    {
        "skill": "debugger",
        "task": "model_diagnosis",
        "description": "监控训练过程",
        "output": "logs/training_logs/"
    }
]
```

## 最佳实践

### 1. 任务分解最佳实践

将复杂任务分解为最小可执行单元：

```python
# 任务分解最佳实践
def decompose_to_minimal_tasks(user_request):
    # 1. 识别主任务
    main_task = identify_main_task(user_request)
    
    # 2. 分解为子任务
    sub_tasks = []
    
    # 3. 继续分解直到最小单元
    for sub_task in sub_tasks:
        if is_complex(sub_task):
            sub_tasks.extend(decompose_to_minimal_tasks(sub_task))
    
    return sub_tasks
```

### 2. 子技能调用最佳实践

确保子技能调用的正确性：

```python
# 子技能调用最佳实践
def call_sub_skill_with_validation(skill_name, task_description):
    # 1. 验证技能存在
    if not skill_exists(skill_name):
        raise SkillNotFoundError(f"Skill {skill_name} not found")
    
    # 2. 验证任务描述
    if not validate_task_description(task_description):
        raise TaskDescriptionError(f"Invalid task description: {task_description}")
    
    # 3. 调用技能
    result = execute_skill(skill_name, task_description)
    
    # 4. 验证结果
    if not validate_result(result):
        raise ValidationError(f"Invalid result from {skill_name}")
    
    return result
```

### 3. 结果汇总最佳实践

确保结果的完整性和一致性：

```python
# 结果汇总最佳实践
def aggregate_with_validation(sub_results):
    # 1. 验证所有结果
    for result in sub_results:
        if not validate(result):
            raise ValidationError(f"Invalid result: {result}")
    
    # 2. 检查依赖关系
    check_dependencies(sub_results)
    
    # 3. 汇总结果
    aggregated = {
        "code": merge_code(sub_results),
        "config": merge_config(sub_results),
        "logs": collect_logs(sub_results)
    }
    
    # 4. 验证汇总结果
    if not validate_aggregated(aggregated):
        raise ValidationError("Invalid aggregated result")
    
    return aggregated
```

## 错误处理

### 常见错误 1：任务分解不完整

**错误**：任务分解不完整，遗漏某些子任务

**解决**：使用多级分解策略

```python
# 多级分解策略
def multi_level_decomposition(user_request):
    # 第一级：粗粒度分解
    coarse_tasks = coarse_grained_decomposition(user_request)
    
    # 第二级：细粒度分解
    fine_tasks = []
    for task in coarse_tasks:
        if is_complex(task):
            fine_tasks.extend(fine_grained_decomposition(task))
        else:
            fine_tasks.append(task)
    
    return fine_tasks
```

### 常见错误 2：子技能调用失败

**错误**：子技能调用失败或返回错误结果

**解决**：实现重试机制和错误回退

```python
# 重试机制和错误回退
def call_sub_skill_with_retry(skill_name, task_description, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = execute_skill(skill_name, task_description)
            if validate_result(result):
                return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # 回退到上一级任务
            task_description = fallback_task(task_description, e)
    
    raise RetryExhaustedError(f"Failed to execute {skill_name} after {max_retries} attempts")
```

### 常见错误 3：结果不一致

**错误**：子技能结果不一致，无法合并

**解决**：实现结果一致性检查

```python
# 结果一致性检查
def check_consistency(sub_results):
    # 1. 检查组件兼容性
    check_component_compatibility(sub_results)
    
    # 2. 检查参数一致性
    check_parameter_consistency(sub_results)
    
    # 3. 检查维度匹配
    check_dimension_matching(sub_results)
    
    # 4. 检查数据格式
    check_data_format_consistency(sub_results)
```

## 扩展性

### 添加新子技能

1. 在 `oneskills/trae/skills/` 下创建新技能文件
2. 定义技能接口和功能
3. 在主技能中注册新技能

### 添加新的任务类型

1. 在任务类型枚举中添加新类型
2. 实现任务处理逻辑
3. 更新子技能分配规则

## 性能优化

### 1. 并行子任务执行

```python
# 并行执行不相关的子任务
def execute_in_parallel(sub_tasks):
    parallel_tasks = []
    sequential_tasks = []
    
    # 分离可并行和必须顺序执行的任务
    for task in sub_tasks:
        if is_parallelizable(task):
            parallel_tasks.append(task)
        else:
            sequential_tasks.append(task)
    
    # 并行执行
    results = parallel_execute(parallel_tasks)
    
    # 顺序执行
    for task in sequential_tasks:
        results.append(execute_skill(task["skill"], task["task"]))
    
    return results
```

### 2. 结果缓存

```python
# 结果缓存
def cache_results(task_description, result):
    # 缓存结果
    cache_key = generate_cache_key(task_description)
    cache(cache_key, result)
    
    return result

def get_cached_result(task_description):
    # 获取缓存结果
    cache_key = generate_cache_key(task_description)
    if is_cached(cache_key):
        return get_from_cache(cache_key)
    return None
```

## 总结

本技能提供了 AI4S 研究工作的全流程支持：

- ✅ **Pipeline 统筹**：协调整个 AI4S Pipeline
- ✅ **子技能协调**：自动调用合适的子技能
- ✅ **任务分解**：将复杂任务分解为子任务
- ✅ **结果汇总**：汇总所有子技能结果
- ✅ **错误处理**：完善的错误处理机制
- ✅ **性能优化**：并行执行和结果缓存
- ✅ **扩展性**：易于添加新子技能和任务类型

通过本技能，用户可以快速完成复杂的 AI4S 研究任务。
