# CNN 模型配置 (UNet / UNetEx / ResNet / 多模型 Benchmark)

适用于基于卷积神经网络的 CFD 模型，包括 DeepCFD、CFDBench。

---

## 1. 核心特征

CNN 模型的配置特点：
- 使用 `root` 包装键
- 处理规则网格数据，`batch_size` 通常为 32-128
- 模型参数以通道数 (`in_channels`, `out_channels`) 为核心
- CFDBench 支持在单个配置文件中定义多个模型的参数

---

## 2. datapipe 段

### 2.1 DeepCFD 风格 (简单 pickle 数据)

```yaml
datapipe:
  verbose: False
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/DeepCFD"
    data_x_name: "dataX.pkl"           # 输入数据文件名
    data_y_name: "dataY.pkl"           # 输出数据文件名
  data:
    split_ratio: 0.7                   # 训练集比例
    seed: 0
  dataloader:
    batch_size: 32
    num_workers: 4
```

### 2.2 CFDBench 风格 (多场景数据集)

```yaml
datapipe:
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/CFDBench/"
    # 数据子集选择:
    # tube_prop_bc_geo (管道流), cavity_geo (顶盖驱动流),
    # cylinder_bc (圆柱绕流), dam_prop (溃坝问题)
    data_name: "tube_prop_bc_geo"
  verbose: True
  data:
    # 任务类型
    task_type: "auto"                  # auto (自回归/瞬态) 或 static (稳态)

    # 数据预处理
    norm_props: true                   # 归一化物理属性
    norm_bc: true                      # 归一化边界条件

    # 自回归模式参数
    delta_time: 0.1                    # 时间步长
    stable_state_diff: 0.001           # 稳态判断阈值

    # 通用参数
    split_ratios: [0.8, 0.1, 0.1]
    seed: 0
    num_rows: 64                       # 网格分辨率 (高度)
    num_cols: 64                       # 网格分辨率 (宽度)
  dataloader:
    batch_size: 128
    eval_batch_size: 16                # 验证/测试批次大小（独立设置）
    num_workers: 4
```

---

## 3. model 段

### 3.1 DeepCFD — 单模型

```yaml
model:
  name: "UNetEx"                       # UNet 或 UNetEx
  in_channels: 3                       # 输入通道 (SDF1, 区域, SDF2)
  out_channels: 3                      # 输出通道 (Ux, Uy, p)
  base_channels: 8                     # 初始特征通道数
  num_stages: 3                        # 下采样/上采样层数
  kernel_size: 5                       # 卷积核大小
  bilinear: true                       # 是否双线性插值上采样
  normtype: "none"                     # 归一化: bn / in / none
```

### 3.2 CFDBench — 多模型集中配置

CFDBench 在单个配置文件中定义所有模型的参数，通过 `name` 选择：

```yaml
model:
  # 模型选择: auto_deeponet, auto_ffn, auto_edeeponet, fno, resnet, unet, deeponet, ffn
  name: "deeponet"

  # 通用参数
  in_chan: 2                           # 输入通道数
  out_chan: 2                          # 输出通道数

  # --- FFN 参数 ---
  ffn_depth: 8
  ffn_width: 100

  # --- Auto-FFN 参数 ---
  autoffn_depth: 8
  autoffn_width: 200

  # --- DeepONet 参数 ---
  deeponet_width: 100
  branch_depth: 8
  trunk_depth: 8
  act_fn: "relu"
  act_scale_invariant: true
  act_on_output: false

  # --- Auto-EDeepONet 参数 ---
  autoedeeponet_width: 100
  autoedeeponet_depth: 8
  autoedeeponet_act_fn: "relu"

  # --- FNO 参数 ---
  fno_depth: 4
  fno_hidden_dim: 32
  fno_modes_x: 12
  fno_modes_y: 12

  # --- U-Net 参数 ---
  unet_dim: 12
  unet_insert_case_params_at: "input"

  # --- ResNet 参数 ---
  resnet_depth: 4
  resnet_hidden_chan: 16
  resnet_kernel_size: 7
  resnet_padding: 3
```

> **设计模式说明**: CFDBench 将所有模型参数集中在一个配置文件中，代码根据 `name` 字段动态选择对应的参数块。这种模式适合 benchmark 场景，方便快速切换模型对比。

---

## 4. training 段

### 4.1 DeepCFD 风格 (简洁)

```yaml
training:
  output_dir: "./result/deepcfd"
  num_epochs: 1000
  lr: 0.001
  weight_decay: 0.005
  patience: 300                        # 早停耐心值
  log_interval: 10                     # 日志间隔 (epoch)
  eval_interval: 10                    # 验证间隔 (epoch)
  save_interval: 50                    # 保存间隔 (epoch)
```

### 4.2 CFDBench 风格 (含模式切换)

```yaml
training:
  output_dir: "./result"
  mode: "train_test"                   # train / test / train_test
  num_epochs: 200
  lr: 1.0e-3
  lr_step_size: 20                     # StepLR 步长
  lr_gamma: 0.9                        # StepLR 衰减系数
  log_interval: 100                    # 日志间隔 (step)
  eval_interval: 20                    # 验证间隔 (epoch)
  loss_name: "nmse"                    # nmse / mse / mae / nmae
  gpuid: 0
```

---

## 5. DeepCFD vs CFDBench 差异速查

| 参数 | DeepCFD | CFDBench |
|------|---------|----------|
| 根键 | `root` | `root` |
| 模型数 | 1-2 (UNet/UNetEx) | 8+ 模型 |
| `batch_size` | 32 | 128 |
| `epochs` | 1000 | 200 |
| 早停 | `patience: 300` | 无 |
| 学习率调度 | 无（手动 weight_decay） | StepLR |
| 损失函数 | MSE (隐式) | nmse (显式) |
| 运行模式 | 无 | `mode: train_test` |

---

## 6. 完整模板 (以 DeepCFD 为例)

```yaml
root:
  datapipe:
    verbose: False
    source:
      data_dir: "${ONESCIENCE_DATASETS_DIR}/YourDataset"
      data_x_name: "input.pkl"
      data_y_name: "output.pkl"
    data:
      split_ratio: 0.7
      seed: 0
    dataloader:
      batch_size: 32
      num_workers: 4

  model:
    name: "UNetEx"
    in_channels: 3
    out_channels: 3
    base_channels: 8
    num_stages: 3
    kernel_size: 5
    bilinear: true
    normtype: "none"

  training:
    output_dir: "./result/"
    num_epochs: 1000
    lr: 0.001
    weight_decay: 0.005
    patience: 300
    log_interval: 10
    eval_interval: 10
    save_interval: 50
```
