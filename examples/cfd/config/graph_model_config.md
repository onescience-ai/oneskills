# Graph 模型配置 (MeshGraphNet / HeteroGNS / MPNN)

适用于基于图神经网络的 CFD 模型，包括 Vortex_shedding_mgn、BENO、PDENNEval/MPNN。

---

## 1. 核心特征

Graph 模型的配置有以下共同特点：
- `batch_size` 通常为 **1**（每个图的节点数不同，难以批处理）
- 需要描述 **节点/边特征维度** 和 **图构建参数**
- 通常需要 **噪声注入** (`noise_std`) 提高泛化性
- 训练 epoch 数较少（图数据每个 epoch 计算量大）

---

## 2. datapipe 段

### 2.1 时序图数据 (MeshGraphNet / Vortex_shedding 风格)

```yaml
datapipe:
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/your_dataset/"
    stats_dir: "${ONESCIENCE_DATASETS_DIR}/your_dataset/stats"   # 归一化统计量
  verbose: True
  data:
    # 轨迹划分（时序图数据特有）
    train_samples: 400                 # 训练轨迹数
    train_steps: 300                   # 每条轨迹使用的时间步数
    val_samples: 10
    val_steps: 300
    test_samples: 10
    test_steps: 300
    noise_std: 0.02                    # 训练噪声标准差
  dataloader:
    batch_size: 1                      # 图网络必须 batch_size=1
    num_workers: 4
```

### 2.2 静态图数据 (BENO 风格)

```yaml
datapipe:
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/BENO/data/Dirichlet/"
    cache_dir: "./cache_data/"
    file_prefix: "N32_4c"              # 数据文件名前缀
  verbose: True
  data:
    name: "BENO_Elliptic"
    ntrain: 900
    ntest: 100
    resolution: 32                     # 网格分辨率
    ns: 10                             # 图构建时每个节点的邻居数
    domain_bounds: [[0, 1], [0, 1]]    # 计算域范围
  dataloader:
    batch_size: 1
    num_workers: 0
    pin_memory: True
```

### 2.3 PDE 图数据 (MPNN 风格)

```yaml
datapipe:
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/PDENNEval/data/"
    file_name: "1D_Advection_Sols_beta1.0.hdf5"
  verbose: True
  data:
    pde_name: "1D_Advection"
    single_file: True
    reduced_resolution: 4
    reduced_resolution_t: 5
    reduced_batch: 1
    test_ratio: 0.1
    # MPNN 图构建参数
    time_window: 10                    # 时间窗口大小
    neighbors: 3                       # 每个节点的邻居数
    # PDE 域描述
    temporal_domain: [0, 2]
    resolution_t: 201
    spatial_domain: [[0, 1]]
    resolution: [1024]
    variables:
      beta: 0.1                        # PDE 参数
  dataloader:
    batch_size: 64                     # MPNN 可以用较大 batch
    num_workers: 4
    pin_memory: True
```

---

## 3. model 段

### 3.1 MeshGraphNet (Vortex_shedding_mgn)

使用 `specific_params` 字典组织：

```yaml
model:
  name: "MeshGraphNet"
  specific_params:
    MeshGraphNet:
      num_input_features: 6            # 节点输入特征 (速度u,v + 压力p + 节点类型等)
      num_edge_features: 3             # 边特征 (相对坐标差 + 距离)
      num_output_features: 3           # 输出特征 (下一时刻速度/加速度)
      do_concat_trick: False           # 是否拼接输入到 Message Passing 输出
      num_processor_checkpoint_segments: 0  # 梯度检查点分段数 (0=不使用)
      recompute_activation: False      # 是否重计算激活函数 (True → silu, False → relu)
```

### 3.2 HeteroGNS (BENO)

扁平参数：

```yaml
model:
  name: "HeteroGNS"
  nnode_in_features: 10                # 节点输入特征维度
  nnode_out_features: 1                # 节点输出特征维度
  nedge_in_features: 7                 # 边特征维度
  nmlp_layers: 2                       # MLP 层数
  act: "relu"                          # 激活函数
  boundary_dim: 128                    # 边界特征编码维度
  trans_layer: 3                       # 消息传递层数
  width: 64                            # 网络宽度
  ker_width: 256                       # 核宽度
```

### 3.3 MPNN

```yaml
model:
  name: "MPNN"
  hidden_features: 128                 # 隐藏层特征维度
  hidden_layer: 6                      # 隐藏层数
  num_outputs: 1                       # 输出维度
```

---

## 4. training 段

### 4.1 MeshGraphNet 风格 (扁平参数)

```yaml
training:
  max_epoch: 25                        # 图模型 epoch 少但每 epoch 计算量大
  lr: 0.0001
  lr_decay_rate: 0.9999991             # LambdaLR 指数衰减
  amp: False                           # 混合精度
  jit: False                           # TorchScript
  checkpoint_dir: "./checkpoints"
  loss_criterion: "MSE"
  patience: 50                         # 早停
  log_interval: 1000
  gpuid: 0
```

### 4.2 BENO 风格 (嵌套 optimizer/scheduler)

```yaml
training:
  output_dir: "./model/"
  save_dir_name: "Resolution_32_poisson"
  epochs: 1000
  save_period: 1
  seed: 2025
  optimizer:
    name: "Adam"
    lr: 1.0e-5
    weight_decay: 5.0e-4
  scheduler:
    name: "CosineAnnealingWarmRestarts"
    T_0: 16
    T_mult: 2
```

### 4.3 MPNN 风格 (自回归展开)

```yaml
training:
  if_training: True
  continue_training: False
  model_path: null
  output_dir: "./checkpoint/"
  save_period: 20
  seed: 0
  tensorboard: True
  log_dir: "./logs/tensorboard/"
  unrolling: 1                         # 是否启用自回归展开
  unroll_step: 20                      # 展开步数
  epochs: 500
  optimizer:
    name: "Adam"
    lr: 1.0e-4
    weight_decay: 1.0e-8
  scheduler:
    name: "StepLR"
    step_size: 100
    gamma: 0.5
```

---

## 5. inference 段 (可选，Vortex_shedding)

```yaml
inference:
  viz_vars: ["u", "v", "p"]           # 可视化物理量
  frame_skip: 10                       # 跳帧间隔
  frame_interval: 1
```

---

## 6. 完整模板 (以 MeshGraphNet 为例)

```yaml
datapipe:
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/your_dataset/"
    stats_dir: "${ONESCIENCE_DATASETS_DIR}/your_dataset/stats"
  verbose: True
  data:
    train_samples: 400
    train_steps: 300
    val_samples: 10
    val_steps: 300
    test_samples: 10
    test_steps: 300
    noise_std: 0.02
  dataloader:
    batch_size: 1
    num_workers: 4

model:
  name: "MeshGraphNet"
  specific_params:
    MeshGraphNet:
      num_input_features: 6
      num_edge_features: 3
      num_output_features: 3
      do_concat_trick: False
      num_processor_checkpoint_segments: 0
      recompute_activation: False

training:
  max_epoch: 25
  lr: 0.0001
  lr_decay_rate: 0.9999991
  amp: False
  jit: False
  checkpoint_dir: "./checkpoints"
  loss_criterion: "MSE"
  patience: 50
  log_interval: 1000
  gpuid: 0

inference:
  viz_vars: ["u", "v", "p"]
  frame_skip: 10
  frame_interval: 1
```
