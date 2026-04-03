# 通用配置结构 (Common Config Structure)

所有 CFD 示例配置文件的通用骨架。无论使用哪种模型架构，配置文件都应遵循此三段式结构。

---

## 1. 整体结构

CFD 配置文件统一采用 **datapipe + model + training** 三段式结构：

```yaml
# 可选的根包装键（二选一）：
# 方式 A: 带模型名称前缀的包装键（PDENNEval 风格）
fno_config:           # 或 deeponet_config, unet_config, ...
  datapipe: ...
  model: ...
  training: ...

# 方式 B: 通用根键（CFDBench/DeepCFD 风格）
root:
  datapipe: ...
  model: ...
  training: ...

# 方式 C: 无包装键，顶层直接展开（Transolver/Vortex_shedding/Eagle 风格）
datapipe: ...
model: ...
training: ...
```

**选择建议**：
- 如果一个项目下有多个模型各自独立的配置 → 方式 A（用 `{model}_config` 区分）
- 如果是单模型项目 → 方式 B 或 C 均可
- 方式 C 最简洁，推荐新项目使用

---

## 2. datapipe 段

`datapipe` 负责描述数据来源、数据语义和 DataLoader 参数，分为三个子段：

```yaml
datapipe:
  # --- 2.1 数据源 (source) ---
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/YourDataset/"   # 必填，数据集根目录
    # 以下按需选填：
    file_name: "data.hdf5"            # 单文件数据集的文件名
    stats_dir: "./stats/"             # 归一化统计量 (mean/std) 存放路径
    cache_dir: "./cache/"             # 缓存目录
    cluster_dir: "..."                # 聚类信息目录 (GraphViT)
    splits_dir: "./splits/"           # 数据划分文件目录
    preprocessed_save_dir: "..."      # 预处理结果保存目录

  # 是否打印详细的数据加载日志
  verbose: True

  # --- 2.2 数据描述 (data) ---
  data:
    name: "YourDataset"               # 数据集名称标识
    seed: 0                           # 随机种子，保证数据划分可复现

    # 数据划分（三种常见方式，选其一）：
    # 方式 1: 比例划分
    split_ratio: 0.8                  # train 占比，剩余为 test
    # 方式 2: 三段比例
    split_ratios: [0.8, 0.1, 0.1]    # [train, val, test]
    # 方式 3: 数量划分
    ntrain: 900
    ntest: 100
    # 方式 4: 比例 + 名称 (Transolver 风格)
    splits:
      task: "full"
      train_name: "full_train"
      val_split_ratio: 0.1
      test_name: "full_test"

    # 其他按需添加的数据参数（取决于具体数据集和模型）

  # --- 2.3 数据加载器 (dataloader) ---
  dataloader:
    batch_size: 32                    # 训练批次大小
    num_workers: 4                    # CPU 数据加载进程数
    pin_memory: True                  # 是否固定内存（加速 GPU 传输）
    # 可选：
    eval_batch_size: 16               # 验证/测试批次大小（CFDBench 风格）
```

### 关键约定

| 字段 | 约定 |
|------|------|
| `data_dir` | 始终使用 `${ONESCIENCE_DATASETS_DIR}/` 环境变量前缀 |
| `batch_size` | GNN 模型通常为 1，CNN/Operator 模型通常为 32-128 |
| `num_workers` | 通常 0-4，视数据加载复杂度而定 |
| `seed` | 必须提供，保证实验可复现 |

---

## 3. model 段

`model` 描述模型架构和超参数，结构因模型类型而异，但有统一的入口：

```yaml
model:
  name: "ModelName"                   # 必填，模型名称标识

  # --- 方式 A: 扁平参数（简单模型，如 FNO/UNet/DeepONet）---
  width: 20
  modes: 12
  in_channels: 1
  out_channels: 1

  # --- 方式 B: specific_params 字典（多模型切换，如 Transolver/Vortex_shedding）---
  specific_params:
    ModelA:
      param1: value1
      param2: value2
    ModelB:
      param1: value3
      param2: value4
```

**选择建议**：
- 配置文件只服务一个模型 → 方式 A
- 配置文件支持多个可切换模型 → 方式 B（通过 `name` 字段选择）

---

## 4. training 段

`training` 描述训练流程的所有参数：

```yaml
training:
  # === 必填参数 ===
  epochs: 500                         # 或 max_epoch / num_epochs
  output_dir: "./checkpoint/"         # 或 checkpoint_dir / result_dir

  # === 优化器（两种组织方式）===
  # 方式 A: 嵌套对象（PDENNEval 风格）
  optimizer:
    name: "Adam"
    lr: 1.0e-3
    weight_decay: 1.0e-4
  scheduler:
    name: "StepLR"
    step_size: 100
    gamma: 0.5

  # 方式 B: 扁平参数（Transolver/Vortex_shedding 风格）
  lr: 0.001
  lr_decay_rate: 0.9999991            # 或 lr_step_size + lr_gamma

  # === 常用可选参数 ===
  seed: 0                             # 随机种子
  gpuid: 0                            # 单卡训练 GPU ID
  save_period: 20                     # 或 save_interval，模型保存频率 (epoch)
  log_interval: 100                   # 日志打印频率 (step 或 epoch)
  eval_interval: 20                   # 验证频率 (epoch)
  patience: 50                        # 早停耐心值

  # === 损失函数 ===
  loss_criterion: "MSE"               # 或 loss_name，支持 MSE/MAE/NMSE 等

  # === 训练模式控制 ===
  if_training: True                   # 是否执行训练
  continue_training: False            # 是否从 checkpoint 恢复
  model_path: null                    # 预训练模型路径

  # === 高级选项 ===
  amp: False                          # 混合精度训练
  jit: False                          # TorchScript 编译
  training_type: "autoregressive"     # 训练类型：autoregressive / single / static
  mode: "train_test"                  # 运行模式：train / test / train_test
```

### 命名不一致速查表

不同示例对同一概念使用了不同的字段名，新建配置时请统一：

| 概念 | 已有变体 | 推荐用法 |
|------|---------|---------|
| 训练轮数 | `epochs`, `max_epoch`, `num_epochs` | `epochs` |
| 学习率 | `lr`, `learning_rate` | `lr` |
| 输出目录 | `output_dir`, `checkpoint_dir`, `result_dir` | `output_dir` |
| 保存频率 | `save_period`, `save_interval` | `save_period` |

---

## 5. inference 段（可选）

部分示例包含独立的推理配置段：

```yaml
inference:
  gpuid: 0                            # 推理 GPU ID
  result_dir: "./results/"            # 推理结果保存目录
  viz_vars: ["u", "v", "p"]          # 可视化的物理量
  frame_skip: 10                      # 跳帧间隔
  frame_interval: 1                   # 帧间隔
  save_vtk: True                      # 是否保存 VTK 格式
  visualize: True                     # 是否生成可视化
```

---

## 6. 完整模板

以下是一个最小化的完整配置模板，可直接复制修改：

```yaml
# conf/your_model.yaml

datapipe:
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/YourDataset/"
    stats_dir: "./stats/"
  verbose: True
  data:
    name: "YourDataset"
    seed: 0
    split_ratios: [0.8, 0.1, 0.1]
  dataloader:
    batch_size: 32
    num_workers: 4
    pin_memory: True

model:
  name: "YourModel"
  # ... 模型特定参数，参见对应分类 skill

training:
  epochs: 500
  output_dir: "./checkpoint/"
  seed: 0
  gpuid: 0
  optimizer:
    name: "Adam"
    lr: 1.0e-3
    weight_decay: 1.0e-4
  scheduler:
    name: "StepLR"
    step_size: 100
    gamma: 0.5
  save_period: 20
  log_interval: 100
  patience: 50
  loss_criterion: "MSE"
```
