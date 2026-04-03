# PDE Operator 模型配置 (FNO / UNO / PINO / DeepONet / UNet)

适用于 PDENNEval 下的 PDE 求解器系列模型。这些模型共享高度统一的配置结构。

---

## 1. 通用结构

所有 PDE Operator 模型使用 `{model}_config` 包装键：

```yaml
{model}_config:              # fno_config / uno_config / pino_config / deeponet_config / unet_config
  datapipe:
    source: ...
    data: ...
    dataloader: ...
  model: ...
  training: ...
```

---

## 2. datapipe 段 — PDE 数据特有字段

```yaml
datapipe:
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/PDENNEval/data/"
    file_name: "1D_Advection_Sols_beta0.1.hdf5"    # HDF5 单文件数据集
  verbose: True
  data:
    name: "Advection"                  # PDE 名称标识
    single_file: True                  # 数据是否存储在单个 HDF5 文件中
    reduced_resolution: 4              # 空间降采样倍率
    reduced_resolution_t: 5            # 时间降采样倍率
    reduced_batch: 1                   # batch 维度降采样
    initial_step: 10                   # 自回归输入的初始时间步数
    test_ratio: 0.1                    # 测试集比例
    seed: 0

    # PINO 额外字段：
    reduced_resolution_pde: 4          # PDE 约束计算用的空间降采样
    reduced_resolution_pde_t: 5        # PDE 约束计算用的时间降采样
    if_grid_norm: False                # 是否对网格坐标归一化

    # MPNN 额外字段：
    pde_name: "1D_Advection"           # PDE 类型标识
    time_window: 10                    # 图时间窗口大小
    neighbors: 3                       # 图的邻居数
    temporal_domain: [0, 2]            # 时间域范围
    resolution_t: 201                  # 原始时间分辨率
    spatial_domain: [[0, 1]]           # 空间域范围
    resolution: [1024]                 # 原始空间分辨率
    variables:                         # PDE 参数
      beta: 0.1

  dataloader:
    batch_size: 64                     # Operator 模型通常 50-64
    num_workers: 4
    pin_memory: False                  # PDENNEval 中通常为 False
```

---

## 3. model 段 — 各模型参数

### 3.1 FNO (Fourier Neural Operator)

```yaml
model:
  name: "FNO"
  num_channels: 1                      # 输入物理量通道数
  modes: 12                            # 保留的傅里叶模态数
  width: 20                            # 隐藏层宽度（提升维度后的通道数）
```

### 3.2 UNO (U-shaped Neural Operator)

```yaml
model:
  name: "UNO"
  num_channels: 1                      # 输入物理量通道数
  width: 20                            # 隐藏层宽度
```

### 3.3 PINO (Physics-Informed Neural Operator)

```yaml
model:
  name: "PINO"
  width: 32                            # 隐藏层宽度
  modes1: [12, 12, 12, 12]            # 各层 X 方向傅里叶模态数
  modes2: [12, 12, 12, 12]            # 各层 Y/T 方向傅里叶模态数
  fc_dim: 128                          # 全连接层维度
  act: "gelu"                          # 激活函数
  in_channels: 1                       # 输入通道数
  out_channels: 1                      # 输出通道数
```

### 3.4 DeepONet

```yaml
model:
  name: "DeepONet"
  base_model: "MLP"                    # 基础网络类型
  input_size: 256                      # 输入空间离散点数
  act: "tanh"                          # 激活函数
  in_channels: 1                       # 输入物理量通道数
  out_channels: 1                      # 输出物理量通道数
  query_dim: 2                         # 查询坐标维度 (1D: 2, 2D: 3)
```

### 3.5 UNet (PDENNEval 版)

```yaml
model:
  name: "UNet"
  in_channels: 1                       # 输入通道数
  out_channels: 1                      # 输出通道数
  init_features: 32                    # 初始特征通道数
```

---

## 4. training 段 — Operator 模型共性

```yaml
training:
  pde_name: "1D_Advection"             # PDE 标识（FNO/UNO/UNet/PINO 使用）
  scenario: "1D_Advection"             # 场景标识（DeepONet 使用）
  save_name: "1d_Adv_256_41"           # 保存文件名前缀

  # 训练类型（核心参数）
  training_type: "autoregressive"      # autoregressive（自回归） / single（单步）
  t_train: 201                         # 自回归训练的时间步总数（FNO/UNO）
  unroll_step: 20                      # 自回归展开步数
  pushforward: True                    # 是否使用 pushforward trick（UNet）

  # 模式控制
  if_training: True                    # 是否执行训练
  continue_training: False             # 是否恢复训练
  model_path: null                     # 预训练模型路径
  output_dir: "./checkpoint/"

  epochs: 500
  save_period: 20
  seed: 0

  # 优化器（所有 Operator 模型统一）
  optimizer:
    name: "Adam"
    lr: 1.0e-3
    weight_decay: 1.0e-4
  scheduler:
    name: "StepLR"
    step_size: 100
    gamma: 0.5

  # PINO 特有的物理约束损失权重
  ic_loss: 2.0                         # 初始条件损失权重
  f_loss: 1.0                          # PDE 残差损失权重
  xy_loss: 10.0                        # 数据拟合损失权重

  # 可选：日志与可视化
  tensorboard: True
  log_dir: "./logs/tensorboard/"
```

---

## 5. 1D vs 2D 配置差异速查

| 参数 | 1D | 2D |
|------|----|----|
| `in_channels` / `out_channels` | 1 | 1-4 |
| `query_dim` (DeepONet) | 2 (x, t) | 3 (x, y, t) |
| `input_size` (DeepONet) | 256 | 128 |
| `reduced_resolution` | 4 | 1-4 |
| `initial_step` | 1 或 10 | 1 或 10 |
| 典型 `batch_size` | 50-64 | 16-64 |

---

## 6. 完整模板 (以 FNO 为例)

```yaml
fno_config:
  datapipe:
    source:
      data_dir: "${ONESCIENCE_DATASETS_DIR}/PDENNEval/data/"
      file_name: "1D_Advection_Sols_beta0.1.hdf5"
    verbose: True
    data:
      single_file: True
      reduced_resolution: 4
      reduced_resolution_t: 5
      reduced_batch: 1
      initial_step: 10
      pde_name: "1D_Advection"
    dataloader:
      batch_size: 64
      num_workers: 4
      pin_memory: False

  model:
    name: "FNO"
    num_channels: 1
    modes: 12
    width: 20

  training:
    output_dir: "./checkpoint/"
    if_training: True
    continue_training: False
    model_path: null
    save_period: 20
    seed: 0
    training_type: "autoregressive"
    t_train: 201
    unroll_step: 20
    epochs: 500
    optimizer:
      name: "Adam"
      lr: 1.0e-3
      weight_decay: 1.0e-4
    scheduler:
      name: "StepLR"
      step_size: 100
      gamma: 0.5
```
