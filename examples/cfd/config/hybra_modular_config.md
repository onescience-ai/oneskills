# Hydra 模块化配置 (Lagrangian_MGN)

适用于使用 Hydra 框架进行配置组合的复杂项目。当前仅 Lagrangian_MGN 使用此模式。

---

## 1. 核心特征

Hydra 模块化配置的核心特点：
- 使用 `defaults` 列表组合多个子配置文件
- 使用 `_target_` 字段实现动态类实例化
- 配置文件分布在多个子目录中（data/, model/, optimizer/, loss/, lr_scheduler/, experiment/）
- 支持 `${...}` 变量插值引用其他配置值
- 通过 experiment 文件一次性覆盖多个配置

---

## 2. 目录结构

```
conf/
├── config.yaml                    # 主配置（入口）
├── logging/
│   └── python/
│       └── default.yaml           # Python logging 配置
├── data/
│   └── lagrangian_dataset.yaml    # Dataset 类的实例化参数
├── model/
│   ├── mgn.yaml                   # MeshGraphNet 基础参数
│   ├── mgn_2d.yaml                # 2D 版本
│   └── mgn_3d.yaml                # 3D 版本
├── optimizer/
│   └── adam.yaml                  # Adam 优化器
├── lr_scheduler/
│   └── cosine.yaml                # Cosine 调度器
├── loss/
│   └── mseloss.yaml               # MSE 损失
└── experiment/
    ├── water.yaml                 # Water 实验
    └── sand.yaml                  # Sand 实验
```

---

## 3. 主配置 (config.yaml)

```yaml
# Hydra 默认配置组合
defaults:
  - /logging/python: default
  - override hydra/job_logging: disabled
  - _self_

hydra:
  run:
    dir: ${output}
  output_subdir: hydra

# === 全局参数 ===
dim: 2                                 # 空间维度 (2D/3D)
output: outputs                        # 输出根目录
resume_dir: ${output}                  # checkpoint 恢复目录

# === 数据集描述（抽象层）===
data:
  data_dir: ???                        # 必须由 experiment 或命令行指定
  name: "dataset"
  num_history: 5                       # 输入历史帧数
  num_node_types: 6                    # 节点类型数
  noise_std: 0.0003                    # 训练噪声
  train:
    split: "train"
    num_sequences: 1000
  valid:
    split: "valid"
    num_sequences: 30
  test:
    split: "test"
    num_sequences: 30

# === 训练配置 ===
train:
  batch_size: 20
  epochs: 100
  checkpoint_save_freq: 5
  dataloader:
    batch_size: ${..batch_size}        # 引用父级 batch_size
    shuffle: true
    num_workers: 2
    pin_memory: true
    drop_last: true

# === 测试配置 ===
test:
  batch_size: 1                        # 图网络推理必须 batch=1
  device: cuda
  dataloader:
    batch_size: ${..batch_size}
    shuffle: false
    num_workers: 1
    pin_memory: true
    drop_last: false

# === Datapipe（实际数据加载层）===
datapipe:
  source:
    data_dir: ${data.data_dir}         # 变量插值
  verbose: True
  data:
    name: ${data.name}
    num_history: ${data.num_history}
    num_node_types: ${data.num_node_types}
    noise_std: ${data.noise_std}
    train: ${data.train}
    valid: ${data.valid}
    test: ${data.test}
  dataloader:
    train: ${train.dataloader}
    valid: ${test.dataloader}
    test: ${test.dataloader}

# === 动态实例化占位（由 experiment 填充）===
model: ???                             # _target_: onescience.models.xxx
loss: ???                              # _target_: torch.nn.MSELoss
optimizer: ???                         # _target_: torch.optim.Adam
lr_scheduler: ???                      # _target_: torch.optim.lr_scheduler.xxx

# === 高级选项 ===
compile:
  enabled: false
  args:
    backend: inductor

amp:
  enabled: false

# === 日志 ===
loggers:
  wandb:
    _target_: loggers.WandBLogger
    project: meshgraphnet
    entity: onescience
    name: l-mgn
    mode: disabled                     # disabled / online / offline
    dir: ${output}
  tensorboard:
    _target_: loggers.TensorBoardLogger
    log_dir: ${output}/tensorboard

# === 推理 ===
inference:
  frame_skip: 1
  frame_interval: 1
```

---

## 4. 子配置文件

### 4.1 data/lagrangian_dataset.yaml

```yaml
_target_: onescience.datapipes.gnn.lagrangian_dataset.LagrangianDataset
_convert_: all

name: ${data.name}
data_dir: ${data.data_dir}
split: ???                             # 由 experiment 覆盖
num_sequences: ???                     # 由 experiment 覆盖
num_history: ${..num_history}
num_steps:                             # 可选
num_node_types: ${..num_node_types}
noise_std: 0.0003
radius:                                # 由数据集元数据填充
dt:                                    # 由数据集元数据填充
bounds:                                # 由数据集元数据填充
```

### 4.2 model/mgn.yaml

```yaml
_target_: onescience.models.meshgraphnet.MeshGraphNet
_convert_: all

input_dim_nodes: ???                   # 由 2D/3D 变体指定
input_dim_edges: ???
output_dim: ???
processor_size: 10                     # Message Passing 层数
aggregation: sum                       # 聚合方式: sum / mean
hidden_dim_processor: 128
hidden_dim_node_encoder: 256
hidden_dim_edge_encoder: 256
hidden_dim_node_decoder: 256
mlp_activation_fn: relu
do_concat_trick: false
num_processor_checkpoint_segments: 0
recompute_activation: false
```

### 4.3 optimizer/adam.yaml

```yaml
_target_: torch.optim.Adam
lr: 1e-4
weight_decay: 0.0
```

### 4.4 lr_scheduler/cosine.yaml

```yaml
_target_: torch.optim.lr_scheduler.CosineAnnealingLR
T_max: ${train.epochs}
eta_min: 0
```

---

## 5. experiment 文件

experiment 文件通过 `@package _global_` 覆盖主配置中的多个字段：

```yaml
# @package _global_

defaults:
  - /data@data.train: lagrangian_dataset    # Dataset 类 → data.train
  - /data@data.valid: lagrangian_dataset    # Dataset 类 → data.valid
  - /data@data.test: lagrangian_dataset     # Dataset 类 → data.test
  - /model: mgn_2d                          # 2D MeshGraphNet
  - /loss: mseloss                          # MSE 损失
  - /optimizer: adam                        # Adam 优化器
  - /lr_scheduler: cosine                   # Cosine 调度器

data:
  name: Water
  train:
    num_sequences: 1000
  valid:
    num_sequences: 30
    num_steps: 206
  test:
    num_sequences: 30
    num_steps: 206
```

---

## 6. 使用方式

```bash
# 基础运行
python train.py +experiment=water data.data_dir=/path/to/water_data

# 覆盖参数
python train.py +experiment=water train.batch_size=10 train.epochs=200

# 切换模型
python train.py +experiment=water model=mgn_3d

# 多运行 (sweep)
python train.py -m +experiment=water train.batch_size=10,20,50
```

---

## 7. 关键语法速查

| 语法 | 含义 | 示例 |
|------|------|------|
| `???` | 必填参数（未填则报错） | `data_dir: ???` |
| `${...}` | 变量插值 | `${data.data_dir}` |
| `${..field}` | 相对引用（父级字段） | `${..batch_size}` |
| `_target_` | Hydra 动态实例化的类路径 | `_target_: torch.optim.Adam` |
| `_convert_: all` | 将 OmegaConf 自动转为原生 Python 类型 | |
| `@package _global_` | experiment 文件覆盖全局配置 | |
| `defaults` | 配置组合列表 | `- /model: mgn_2d` |

---

## 8. 何时使用 Hydra 模块化

**适合场景**：
- 同一个训练管道需要支持多个数据集 × 多个模型 × 多个优化器的排列组合
- 需要通过命令行灵活覆盖任意参数
- 项目规模较大，单配置文件超过 200 行

**不适合场景**：
- 简单的单模型、单数据集项目 → 用扁平 YAML 即可
- 团队对 Hydra 不熟悉 → 增加上手成本
