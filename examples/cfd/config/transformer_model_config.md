# Transformer 模型配置 (Transolver / GraphViT)

适用于基于 Transformer/注意力机制的 CFD 模型，包括 Transolver-Airfoil-Design、Transolver-Car-Design、EagleMeshTransformer。

---

## 1. 核心特征

Transformer 模型的配置特点：
- 无根包装键，直接使用顶层 `datapipe` + `model` + `training`
- `model` 段使用 `specific_params` 支持多模型切换
- 通常处理非结构化网格/点云数据，`batch_size` 通常为 1-2
- 需要图构建参数（`r`, `max_neighbors`）用于局部注意力
- 有独立的 `inference` 段

---

## 2. datapipe 段

### 2.1 气动外形数据 (AirfRANS / Transolver-Airfoil 风格)

```yaml
datapipe:
  name: "AirfRANS"                     # 数据管道名称标识
  verbose: False
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/Transolver-Airfoil-Design/Dataset"
    stats_dir: "./dataset"             # 归一化统计量
  data:
    splits:
      task: "full"                     # 任务类型: full / scarce / reynolds / aoa
      train_name: "full_train"
      val_split_ratio: 0.1
      test_name: "full_test"
    sampling:
      sample_strategy: null            # 采样策略: null / uniform / mesh
      n_boot: 500000                   # 自举采样数量
      surf_ratio: 0.1                  # 表面点比例
    crop: null                         # 裁剪边界 [xmin, xmax, ymin, ymax]
    subsampling: 32000                 # 每样本最大点数
  dataloader:
    batch_size: 1
    num_workers: 4
```

### 2.2 汽车外形数据 (ShapeNetCar / Transolver-Car 风格)

```yaml
datapipe:
  name: "ShapeNetCar"
  task: "Transolver-Car-Design"
  verbose: True
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/Transolver-Car-Design/mlcfd_data/training_data"
    preprocessed_save_dir: "${ONESCIENCE_DATASETS_DIR}/Transolver-Car-Design/mlcfd_data/preprocessed_data"
    stats_dir: "${ONESCIENCE_DATASETS_DIR}/Transolver-Car-Design/mlcfd_data/stats"
    preprocessed: 1                    # 是否使用预处理数据 (1: 是, 0: 否)
  data:
    splits:
      fold_id: 0                       # 交叉验证折数
  dataloader:
    batch_size: 1
    num_workers: 4
    pin_memory: True
```

### 2.3 流场时序数据 (Eagle / GraphViT 风格)

```yaml
datapipe:
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/Eagle/Eagle_dataset"
    cluster_dir: "${ONESCIENCE_DATASETS_DIR}/Eagle/Eagle_dataset"   # 聚类信息
    splits_dir: "./splits/"
  verbose: True
  data:
    window_length_train: 6             # 训练时间窗口
    window_length_val: 25              # 验证时间窗口
    window_length_test: 25             # 测试时间窗口
    n_cluster: 40                      # 节点聚类数量（GraphViT 特有）
    normalized: True                   # 是否归一化
    type_as_onehot: True               # 节点类型是否 one-hot 编码
    with_cells: True                   # 是否包含单元信息
  dataloader:
    batch_size: 2
    num_workers: 4
```

---

## 3. model 段

### 3.1 Transolver (多模型切换)

```yaml
model:
  name: "Transolver"                   # 通过修改此字段切换模型
  specific_params:
    # Transolver / Transolver_plus 共享参数（YAML 锚点）
    Transolver: &transolver_params
      n_hidden: 256                    # 隐藏层维度
      n_layers: 8                      # Transformer 层数
      space_dim: 7                     # 空间特征维度 (位置 + 法向量 + 边界条件)
      fun_dim: 0                       # 函数特征维度
      n_head: 8                        # 多头注意力头数
      mlp_ratio: 2                     # MLP 扩展比例
      out_dim: 4                       # 输出维度 (速度u,v + 压力p + 湍流粘度nut)
      slice_num: 32                    # 物理切片数量
      unified_pos: 1                   # 是否统一位置编码 (1: 是, 0: 否)
      build_graph: True                # 是否构建图结构
      r: 0.05                          # 邻域半径 (Airfoil: 0.05, Car: 0.2)
      max_neighbors: 64               # 最大邻居数
    Transolver_plus: *transolver_params

    # 基线模型
    GraphSAGE:
      encoder: [7, 64, 64, 8]
      decoder: [8, 64, 64, 4]
      nb_hidden_layers: 3
      size_hidden_layers: 64
      bn_bool: True
      build_graph: True
      r: 0.05
      max_neighbors: 64

    PointNet:
      encoder: [7, 64, 64, 8]
      decoder: [8, 64, 64, 4]
      base_nb: 8
      build_graph: False
      r: null
      max_neighbors: null

    MLP:
      encoder: [7, 64, 64, 8]
      decoder: [8, 64, 64, 4]
      nb_hidden_layers: 3
      size_hidden_layers: 64
      bn_bool: True
      build_graph: False
      r: null
      max_neighbors: null

    GUNet:
      encoder: [7, 64, 64, 8]
      decoder: [8, 64, 64, 4]
      layer: "SAGE"
      pool: "random"
      nb_scale: 5
      pool_ratio: [0.5, 0.5, 0.5, 0.5]
      list_r: [0.05, 0.2, 0.5, 1, 10]
      size_hidden_layers: 8
      batchnorm: True
      res: False
      build_graph: True
      r: 0.05
      max_neighbors: 64
```

### 3.2 GraphViT (EagleMeshTransformer)

简洁的扁平参数：

```yaml
model:
  name: "GraphViT"
  w_size: 512                          # 隐空间特征维度
  state_size: 4                        # 状态向量维度 (速度+压力)
```

---

## 4. training 段

Transformer 模型的 training 使用扁平参数风格：

```yaml
training:
  max_epoch: 500                       # Airfoil: 500, Car: 200, Eagle: 1000
  lr: 0.001                            # 初始学习率（Eagle: 1e-4）
  gpuid: 0
  patience: 50                         # 早停耐心值（Eagle: 100）

  # 损失函数
  loss_criterion: "MSE_weighted"       # MSE / MAE / MSE_weighted
  loss_weight: 1.0                     # 表面损失权重 (Airfoil)
  loss_alpha: 0.1                      # 压力项权重 (Eagle)

  # 保存路径
  checkpoint_dir: "./checkpoints/transolver_airfrans"
  result_dir: "./results"

  # 测试
  n_test: 3                            # 测试时随机抽取的样本数 (Airfoil)
  val_iter: 1                          # 验证频率 (Car)
  max_anim_on_infer: 5                 # 推理动画数量 (Eagle)
```

---

## 5. inference 段 (Transolver-Car)

```yaml
inference:
  gpuid: 0
  result_dir: "./results/ShapeNetCar"
  save_vtk: True                       # 保存 VTK 格式结果
  visualize: True                      # 生成可视化图片
```

---

## 6. Airfoil vs Car vs Eagle 差异速查

| 参数 | Airfoil | Car | Eagle |
|------|---------|-----|-------|
| 模型 | Transolver | Transolver_plus | GraphViT |
| `batch_size` | 1 | 1 | 2 |
| `max_epoch` | 500 | 200 | 1000 |
| `lr` | 0.001 | 0.001 | 1e-4 |
| `r` (邻域半径) | 0.05 | 0.2 | N/A |
| `out_dim` | 4 (u,v,p,nut) | 4 (u,v,w,p) | 4 |
| 数据划分 | task/split 名称 | fold_id | splits_dir |
| `loss_criterion` | MSE_weighted | MSE | 自定义 (loss_alpha) |

---

## 7. 完整模板 (以 Transolver 为例)

```yaml
datapipe:
  name: "YourDataset"
  verbose: True
  source:
    data_dir: "${ONESCIENCE_DATASETS_DIR}/YourDataset/"
    stats_dir: "./stats/"
  data:
    splits:
      task: "full"
      train_name: "full_train"
      val_split_ratio: 0.1
      test_name: "full_test"
    subsampling: 32000
  dataloader:
    batch_size: 1
    num_workers: 4

model:
  name: "Transolver"
  specific_params:
    Transolver:
      n_hidden: 256
      n_layers: 8
      space_dim: 7
      fun_dim: 0
      n_head: 8
      mlp_ratio: 2
      out_dim: 4
      slice_num: 32
      unified_pos: 1
      build_graph: True
      r: 0.05
      max_neighbors: 64

training:
  max_epoch: 500
  lr: 0.001
  gpuid: 0
  patience: 50
  loss_criterion: "MSE"
  checkpoint_dir: "./checkpoints/"
  result_dir: "./results/"
```
