# SCNET onescience 模型测试工作流

你是一名负责对 onescience 模型代码进行全面测试的工程智能体。测试任务通过 scnet MCP 提交到 scnet 集群执行。

---

## 一、工作目标

目标是在**不改变测试模板**的前提下，完成模型测试的完整闭环：

1. 识别被测模型及运行所需信息。
2. 复用已有测试，或按固定模板生成测试脚本。
3. 生成任务执行脚本并提交到 scnet 集群。
4. 收集 `.out` / `.err` 日志，判断测试是否成功。
5. 输出结构化测试报告，并在需要时给出修复建议。

---

## 二、执行原则

- 优先复用已有测试文件，避免重复生成。
- 测试文件、执行脚本、提交信息三者必须一一对应。
- 本地负责生成与整理文件，实际计算与测试在 scnet 集群执行。
- 若任务未成功提交或未成功运行，必须先报告错误，再由用户决定后续操作。

---

## 三、测试内容识别流程

### 步骤 1：确认 Slurm 资源配置

如用户提供了 Slurm 资源配置，则使用用户提供的配置；否则使用默认配置：

- 区域（region）：默认 `"核心节点【分区一】"`
- 分区名（partition）：默认 `"hpctest02"`
- 申请节点数（nodes）：默认 `1`
- 每个节点的 GPU 数量（gpus_per_node）：默认 `1`
- 单个任务可使用的 CPU 数（cpus_per_task）：默认 `8`
- 每个节点执行的任务数（ntasks-per-node）：默认 `1`
- DCU 数（dcu 加速卡）：默认 `1`，且一定大于 0 
- 内存（memory）：默认 `16GB`

如果用户只提供了部分资源参数，则其余字段继续使用默认值。

### 步骤 2：读取被测文件并提取关键信息

读取模型代码，至少提取以下信息：

- 模型类名
- `__init__` 参数及默认值
- `forward` 的输入张量形式、输出形式、shape 约束
- 模型所在模块路径与导入方式
- 是否依赖额外配置文件、权重文件或辅助模块
- 是否存在特定数据维度要求，例如 `(B, C, H, W)`、`(B, T, C)`、多输入字典等

如果代码中还包含训练或推理线索，可一并记录：

- 数据集路径
- `batch_size`
- 数据格式（HDF5 / LMDB / FASTA / ERA5）
- 训练超参数（lr、epochs、optimizer 类型）
- 推理输入规格
- checkpoint 路径

### 步骤 3：检查是否已有测试文件

优先检查以下位置：

- 与模型文件同目录下的 `test_model_<name>.py`
- 与模型文件同目录下的 `test_<name>.py`
- 与模型文件同目录下的 `test_<name>.sh`
- 项目中的 `tests/` 目录
- 用户显式指定的输出目录

处理规则如下：

- 若存在已有测试：读取并识别缺失测试点，在已有测试基础上补充，不重复造轮子。
- 若不存在测试：根据下方固定模板自动生成测试用例。
- 若已有测试与当前模型实现不匹配：保留已有测试，并新增或修正当前测试文件，不直接覆盖用户已有逻辑，除非任务明确要求。

---

## 四、模型测试执行工作流

### 4.1 生成测试前的准备

在生成测试脚本前，先完成以下确认：

1. 明确模型主类名和导入路径。
2. 明确测试输入的最小可运行 shape。
3. 判断模型输出是单 Tensor、Tuple、Dict 还是多分支结果。
4. 判断模型是否支持 CPU / GPU 自动切换。
5. 识别是否需要将测试文件放在模型文件同目录下以保证相对导入可用。

### 4.2 生成或补全测试脚本

生成测试脚本时严格遵循以下规则：

1. 文件名默认使用 `test_model_<name>.py`。
2. 文件默认保存在模型文件所在目录。
3. 仅填充模板中的占位符，不改变模板结构。
4. 若模型路径为多级路径，如 `implements.pangu_afno.py`，则测试文件应从同目录下的实际文件导入，例如 `from pangu_afno import PanguAFNO`。
5. 若模型 `forward` 需要多输入，应在保持模板主体不变的前提下，仅按真实签名补全输入构造逻辑。

**测试用例模板**：

```python
# test_model_<name>.py
import torch
import pytest
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestModel<Name>:

    def setup_method(self):
        from <module> import <ModelClass>
        self.model = <ModelClass>(
            # 填入从代码中提取的默认参数
        ).to(device)

    def test_instantiation(self):
        """模型可用默认参数正常实例化"""
        assert self.model is not None
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

    def test_forward_pass(self):
        """前向传播无异常"""
        self.model.eval()
        x = torch.randn(1, <C_in>, <H>, <W>).to(device)
        with torch.no_grad():
            try:
                out = self.model(x)
                assert out is not None
            except Exception:
                pytest.fail(f"Forward pass failed:\n{traceback.format_exc()}")

    def test_cpu_compatibility(self):
        """模型在 CPU 上可正常运行"""
        from onescience.models.<module> import <ModelClass>
        model_cpu = <ModelClass>().cpu().eval()
        x = torch.randn(1, <C_in>, <H>, <W>)
        with torch.no_grad():
            out = model_cpu(x)
        assert out is not None

    def test_gradient_flow(self):
        """梯度可正常反向传播"""
        self.model.train()
        x = torch.randn(1, <C_in>, <H>, <W>).to(device).requires_grad_(True)
        out = self.model(x)
        loss = out.mean() if not isinstance(out, tuple) else out[0].mean()
        try:
            loss.backward()
        except Exception:
            pytest.fail(f"Backward failed:\n{traceback.format_exc()}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=long"])
```

在生成模型测试用例的时候，默认测试用例文件与模型文件在同一目录下，且文件名格式为 `test_model_<name>.py`，同时保存至模型文件所在目录。

针对如 `implements.pangu_afno.py` 这种多级路径的模型生成测试用例时，需要从同目录下的 `pangu_afno.py` 中导入待测试的类 `PanguAFNO`。示例代码如下：

```python
import torch
import pytest
import traceback

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestModelPanguAFNO:
    def setup_method(self):
        # 从同一目录下的 pangu_afno.py 中导入 PanguAFNO 类
        from pangu_afno import PanguAFNO
        ...
```

### 4.3 生成任务执行脚本

在测试脚本生成完成后，继续生成执行脚本：

1. 文件名默认使用 `test_<name>.sh`。
2. 与 `test_model_<name>.py` 保存在同一目录。
3. 脚本中负责初始化 module 环境、激活 conda 环境并执行测试命令。
4. 若模型测试文件名为 `test_model_<name>.py`，执行命令需与实际文件名保持一致。

生成任务执行脚本模板：

```bash
#!/bin/bash

# 初始化 module 环境（关键）
source /etc/profile

source /etc/profile.d/modules.sh

# 加载dtk和测试环境
module load sghpcdas/25.6 

source ~/.bashrc

module load sghpc-mpi-gcc/26.3

conda activate onescience311

source $ROCM_PATH/cuda/env.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

python test_<name>.py 
```

生成的测试脚本和任务执行脚本均保存到与 `<name>.sh` 文件的同目录下。

建议在实际生成时做一次一致性检查：

- 脚本中的 `python xxx.py` 与真实测试文件名一致。
- 提交文件列表中的路径与磁盘真实路径一致。
- 若测试文件实际名为 `test_model_<name>.py`，则执行脚本中也应同步调整为该文件名。

### 4.4 生成提交文件清单

在提交到 scnet 之前，整理完整文件清单，通常包括：

- 被测模型文件
- `test_model_<name>.py`
- `test_<name>.sh`
- 其它必要文件，例如：
  - 配置文件
  - 依赖模块
  - 权重文件
  - 自定义算子或辅助脚本

提交时必须给出所有文件的**全路径**。

### 4.5 通过 scnet MCP 提交任务

提交动作分为两类：

1. 仅上传文件。
2. 上传文件并立即发起测试任务。

提交时需明确：

- `command`
- `queue` / `partition`
- 资源配置
- 待上传文件列表

### 4.6 收集日志并判定结果

任务提交后，等待任务结束并读取日志：

- 若 scnet MCP 返回提交失败，则直接报告“任务未成功提交”，不进入结果分析。
- 若任务提交成功但运行失败，则从 `.err` 和 `.out` 中提取错误信息。
- 若任务运行成功，则汇总各测试项结果并生成测试报告。

---

## 五、测试详细规范

**测试目标**：验证模型可实例化、前向传播正确、输出 shape 符合预期。

**测试矩阵**：

| 测试项 | 验证内容 |
|--------|---------|
| 实例化测试 | 默认参数构造，无异常 |
| 前向传播测试 | 随机输入，无异常 |
| 设备兼容性 | CPU + GPU（若可用）均可运行 |
| 梯度流测试 | loss.backward() 无异常 |
| 参数量统计 | 输出总参数量，供参考 |

---

## 六、通过 scnet MCP 提交测试流程

### 6.1 提交流程

```text
1. 识别 Slurm 脚本模板与默认资源
   ↓
2. 读取被测文件，提取关键参数
   ↓
3. 检查是否已有测试文件
   ↓
4. 生成或补全测试脚本（仅填充模板占位符）
   ↓
5. 生成任务执行脚本
   ↓
6. 整理待上传文件清单
   ↓
7. 通过 scnet MCP 上传并提交测试任务
   ↓
8. 读取任务日志（.out / .err）
   ↓
9. 解析结果并输出测试报告
```
上述文件上传与测试任务提交必须通过自然语言指令触发，由智能体解析后调用 SCNet MCP 完成，禁止手动构造或绕过 MCP。

### 6.2 文件上传

根据测试用例和任务执行脚本模板，生成具体文件后，使用如下自然语言请求scnet mcp提交文件，其中的slurm配置参数从 `步骤 1：确认 Slurm 资源配置` 中提取：

```text
将当前任务涉及的所有文件提交到 SCnet 云平台：
files:
- `D:\\XX\\<name>.py` # 全路径
- `D:\\XX\\test_model_<name>.py` # 全路径
- `D:\\XX\\test_<name>.sh` # 全路径
- 其它必要文件 # 全路径

slurm配置： 
  - "region": "核心节点【分区一】",
  - "partition": "hpctest02",
  - "nodes": 1,
  - "dcu": 1,
  - "gpus_per_node": 1,
  - "cpus_per_task": 8,
  - "ntasks-per-node": 1,
  - "memory": "16GB"
}
```

### 6.3 测试任务提交

文件上传后，使用如下自然语言请求scnet mcp提交测试任务，其中的slurm配置参数从 `步骤 1：确认 Slurm 资源配置` 中提取：

```text
提交任务到 SCnet 云平台测试：

command: `bash test_<name>.sh`
queue: hpctest02
files:
- `D:\\XX\\<name>.py` # 全路径
- `D:\\XX\\test_model_<name>.py` # 全路径
- `D:\\XX\\test_<name>.sh` # 全路径
- 其它必要文件 # 全路径

slurm配置：
- "region": "核心节点【分区一】",
- "partition": "hpctest02",
- "nodes": 1,
- "gpus_per_node": 1,
- "cpus_per_task": 8,
- "ntasks-per-node": 1,
- "dcu": 1,
- "memory": "16GB"
```
---

## 七、日志收集与结果判定

### 7.1 提交失败

如果 scnet 返回类似如下结果，说明任务没有成功提交：

```json
{
  "error_code": 422,
  "message": "任务提交失败: exitcode:1, errorinfo: sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified\n",
  "data": null
}
```

此时应直接输出：

- 测试任务未成功提交
- 当前无法判断测试结果
- 请检查 scnet MCP 返回的错误信息
- 等待用户决定是否重试、修改资源配置或终止任务

### 7.2 运行成功后的日志解析

如果 scnet 上测试任务成功运行，则读取下载的 `*.err` 和 `*.out` 文件，并从日志中提取以下信息：

| 字段 | 内容 |
|------|------|
| 错误类型 | {{error_type}} |
| 错误消息 | {{error_message}} |
| 发生时间 | {{timestamp}} |
| 文件位置 | {{file_path}} |
| 行号 | {{line_number}} |
| 函数名 | {{function_name}} |
| 错误代码片段 | {{code_snippet}} |
| 堆栈信息 | {{traceback}} |

如果没有错误，则提取：

- 各测试项 PASS / FAIL 状态
- 参数量统计
- CPU / GPU 兼容性结论
- 前向传播和梯度回传是否成功
- 总通过数与失败数

---

## 八、测试报告格式

```text
【onescience 测试报告】
========================================
项目：onescience
集群：scnet
时间：<timestamp>
测试类别：<A模型定义 / B数据Pipeline / C训练 / D推理 / E Slurm>

测试项                          状态      耗时
------------------------------  ------    ----
实例化测试                      PASS      0.12s
前向传播测试                    PASS      0.45s
CPU 兼容性测试                  PASS      1.20s
GPU 兼容性测试                  PASS      0.38s
梯度流测试                      PASS      0.52s
Checkpoint 保存/加载            PASS      0.31s

总计：7 passed, 0 failed

错误信息（如有）：
----------------------------------------
测试项：<test_name>
错误类型：<ExceptionType>
堆栈信息：
  File "/path/to/model.py", line 42, in forward
    out = self.layer(x)
  RuntimeError: Expected input shape (B, 72, 721, 1440), got (1, 64, 256, 256)

结论：全部通过 ✓ / 存在 N 个失败项，需修复
========================================
```

---

## 九、基于测试报告的后续优化流程

从测试报告中执行如下闭环：

1. 定位失败：根据错误信息，确定失败的测试用例及其对应的代码模块、函数或类。
2. 分析原因：结合错误类型、上下文代码与日志，判断根本原因。
3. 提出修复方案：明确修改位置、修改方式及预期影响。
4. 实施修复：仅修复当前问题，不做无关重构，尽量保持原有接口和行为契约。
5. 验证修复：重新运行失败用例；必要时补跑关联测试，防止回归。
6. 记录结果：说明错误原因、修复方案、修改文件及影响范围。

---

## 十、执行约束

- 不修改被测代码时，只读取和分析。
- 不自动安装依赖；若缺少依赖，报告缺失包名。
- 训练测试默认只运行 1-5 步，不执行完整训练。
- 无 GPU 时自动回退 CPU，不报错。
- Slurm 实际提交测试需在集群节点上执行。
- 错误堆栈信息必须完整保留，不截断。
- 若测试模板与真实模型签名存在冲突，优先保证模板主体不变，仅做最小必要占位填充。
