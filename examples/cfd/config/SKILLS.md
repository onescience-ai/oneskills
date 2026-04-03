# CFD 示例配置文件编写指南 (Skills)

本目录是 OneScience CFD 领域示例配置文件的编写规范，基于 12 个已有示例（BENO、CFDBench、DeepCFD、EagleMeshTransformer、Lagrangian_MGN、PDENNEval、Transolver-Airfoil-Design、Transolver-Car-Design、Vortex_shedding_mgn、PINNsformer、GP_for_TO、CFD_Benchmark）的配置文件总结而成。

---

## 分层结构

CFD 示例的配置文件遵循 **三层 Skills** 体系：

| 层级 | 文件 | 覆盖范围 |
|------|------|---------|
| **Level 1: 通用** | [common_config.md](common_config.md) | 所有示例共享的 `datapipe` + `model` + `training` 三段式结构 |
| **Level 2: 按模型分类** | 4 个分类文件 | 同类模型架构的配置模板 |
| **Level 3: 特例** | 2 个特例文件 | 结构独特的配置 |

### Level 2 — 模型分类 Skills

| 文件 | 适用模型 | 适用示例 |
|------|---------|---------|
| [operator_model_config.md](operator_model_config.md) | FNO, UNO, PINO, DeepONet | PDENNEval (6 个子模型) |
| [graph_model_config.md](graph_model_config.md) | MeshGraphNet, HeteroGNS, MPNN | BENO, Vortex_shedding_mgn, PDENNEval/MPNN |
| [transformer_model_config.md](transformer_model_config.md) | Transolver, GraphViT | Transolver-Airfoil-Design, Transolver-Car-Design, EagleMeshTransformer |
| [cnn_model_config.md](cnn_model_config.md) | UNet, UNetEx, ResNet | DeepCFD, CFDBench |

### Level 3 — 特例 Skills

| 文件 | 适用示例 | 特殊之处 |
|------|---------|---------|
| [hydra_modular_config.md](hydra_modular_config.md) | Lagrangian_MGN | Hydra 模块化配置，`defaults` + `_target_` 动态实例化 |
| [pinn_flat_config.md](pinn_flat_config.md) | PDENNEval/PINN | 扁平 key-value 结构，无 `datapipe` 嵌套 |

---

## 快速决策流程

新建 CFD 示例配置时，按以下流程选择模板：

```
你的模型是什么类型？
├── FNO / UNO / PINO / DeepONet → operator_model_config.md
├── MeshGraphNet / HeteroGNS / MPNN → graph_model_config.md
├── Transolver / GraphViT / 自注意力 → transformer_model_config.md
├── UNet / ResNet / CNN → cnn_model_config.md
├── PINN (物理约束，无需数据管道) → pinn_flat_config.md
├── 需要 Hydra 动态组合 → hydra_modular_config.md
└── 不确定 → 先读 common_config.md，选最接近的分类
```

---

## 无配置文件的示例

以下示例不使用 YAML 配置文件，参数硬编码在脚本或 Notebook 中：
- **PINNsformer** — 参数在 Python 脚本中
- **GP_for_TO** — 参数在 Python 脚本中
- **CFD_Benchmark** — 参数在 Python 脚本中
