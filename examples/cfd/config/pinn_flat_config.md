# PINN 扁平配置 (Physics-Informed Neural Networks)

适用于 PDENNEval/PINN。此模型配置结构与其他所有 CFD 示例完全不同，采用扁平 key-value 结构。

---

## 1. 核心特征

PINN 配置的独特之处：
- **无 `datapipe` 嵌套**：不使用 `datapipe.source.data_dir` 三层结构
- **无 `model` 段**：模型参数与全局参数混合在顶层
- **无 `optimizer`/`scheduler` 子对象**：仅指定 `learning_rate`
- 训练 epoch 数极高（25000），因为 PINN 通过优化物理方程残差训练
- 包含 PDE 特定的物理参数 (`aux_params`)

---

## 2. 完整结构

```yaml
# 模型与场景
model_name: "PINN"
scenario: "pde1D"                      # pde1D / CFD2D / ...

# PDE 选择
pde: "1dBgs"                           # 1dBgs (Burgers) / 1dAdv (Advection) / 1dDiff-react

# 数据路径（注意：不使用 ${ONESCIENCE_DATASETS_DIR} 环境变量）
root_path: "/path/to/data/1D/Burgers/Train/"
filename: "1D_Burgers_Sols_Nu0.001.hdf5"

# 训练参数
epochs: 25000                          # PINN 需要大量 epoch
learning_rate: 1.e-3                   # 学习率
model_update: 500                      # 模型更新/日志频率

# 模型参数（顶层，非嵌套）
input_ch: 2                            # 输入通道 (x, t)
output_ch: 1                           # 输出通道 (u)

# PDE 物理参数
aux_params: [0.001]                    # PDE 参数（如 Burgers 方程的粘性系数 nu）

# 边界条件
if_periodic_bc: True                   # 是否使用周期边界条件

# 验证参数
val_num: 1                             # 验证样本数
val_time: 2.0                          # 验证时间点
val_batch_idx: 9000                    # 验证批次索引
period: 5000                           # 验证周期

# 随机种子
seed: "0000"                           # 注意：字符串类型

# 预训练模型路径
model_path: "/path/to/pretrained/model.pt"

# 数据降采样（唯一的嵌套结构）
dataset:
  reduced_resolution: 4                # 空间降采样
  reduced_resolution_t: 5              # 时间降采样
```

---

## 3. 不同 PDE 的参数对照

| PDE | `pde` | `filename` | `aux_params` | `input_ch` | `output_ch` |
|-----|-------|-----------|-------------|-----------|------------|
| Burgers 1D | `1dBgs` | `1D_Burgers_Sols_Nu0.001.hdf5` | `[0.001]` (nu) | 2 | 1 |
| Advection 1D | `1dAdv` | `1D_Advection_Sols_beta0.1.hdf5` | `[0.1]` (beta) | 2 | 1 |
| Reaction-Diffusion 1D | `1dDiff-react` | `ReacDiff_Nu0.5_Rho1.0.hdf5` | `[0.5, 1.0]` (nu, rho) | 2 | 1 |
| CFD 2D | (CFD2d) | 各类 2D 流场数据 | 按 PDE 而定 | 3 | 1-4 |

---

## 4. 与标准三段式结构的映射关系

| 标准结构 | PINN 对应字段 |
|---------|-------------|
| `datapipe.source.data_dir` | `root_path` |
| `datapipe.source.file_name` | `filename` |
| `datapipe.data.reduced_resolution` | `dataset.reduced_resolution` |
| `model.name` | `model_name` |
| `model.in_channels` | `input_ch` |
| `model.out_channels` | `output_ch` |
| `training.epochs` | `epochs` |
| `training.optimizer.lr` | `learning_rate` |
| `training.seed` | `seed` |
| `training.output_dir` | 无（由代码内部决定） |

---

## 5. 注意事项

1. **路径格式**：PINN 配置中的路径为硬编码绝对路径，不使用 `${ONESCIENCE_DATASETS_DIR}` 环境变量。如需适配项目规范，应改为环境变量格式。
2. **seed 类型**：PINN 中 `seed` 为字符串 `"0000"`，其他模型为整数 `0`。
3. **无早停**：PINN 不使用 `patience` 早停机制，依赖固定 epoch 数训练。
4. **无学习率调度**：仅设置初始学习率，无 StepLR/Cosine 调度。
5. **验证方式特殊**：不按 epoch 验证，而是按 `period` 间隔和特定 `val_batch_idx` 验证。

---

## 6. 模板

```yaml
model_name: "PINN"
scenario: "pde1D"
pde: "1dBgs"

root_path: "${ONESCIENCE_DATASETS_DIR}/PDENNEval/data/1D/Burgers/Train/"
filename: "1D_Burgers_Sols_Nu0.001.hdf5"

epochs: 25000
learning_rate: 1.e-3
model_update: 500

input_ch: 2
output_ch: 1

aux_params: [0.001]
if_periodic_bc: True

val_num: 1
val_time: 2.0
val_batch_idx: 9000
period: 5000

seed: "0000"
model_path: null

dataset:
  reduced_resolution: 4
  reduced_resolution_t: 5
```
