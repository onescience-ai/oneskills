# 训练脚本编写指南 (Train.py Skills)

本文档是 OneScience CFD 领域训练入口脚本 (train.py) 的编写规范，基于 17 个已有训练脚本总结而成。

---

## 1. 整体结论

所有 CFD 训练脚本共享一个 **7 步骨架**，差异主要体现在模型初始化和训练循环内部。

```
main()
 ├── Step 1: 分布式初始化          ← 100% 统一
 ├── Step 2: 配置加载              ← 2 种模式 (YParams / Hydra)
 ├── Step 3: 数据加载              ← 95% 统一 (Datapipe 模式)
 ├── Step 4: 模型初始化 + DDP 包装 ← 差异较大（按模型分类）
 ├── Step 5: 优化器/调度器/损失函数 ← 3 种模式
 ├── Step 6: 训练循环              ← 骨架统一，内部逻辑分类
 └── Step 7: 清理                  ← 100% 统一
```

---

## 2. 分类 Skills

| 文件 | 适用模型 | 适用示例 |
|------|---------|---------|
| [train_common.md](train_common.md) | 所有模型 | 通用骨架 + 7 步流程 |
| [train_operator.md](train_operator.md) | FNO, UNO, PINO, DeepONet, UNet (PDENNEval) | PDENNEval 系列 |
| [train_graph.md](train_graph.md) | MeshGraphNet, HeteroGNS, MPNN | BENO, Vortex_shedding, Lagrangian_MGN |
| [train_transformer.md](train_transformer.md) | Transolver, GraphViT | Transolver-Airfoil/Car, Eagle |
| [train_cnn.md](train_cnn.md) | UNet/UNetEx, CFDBench 多模型 | DeepCFD, CFDBench |

---

## 3. 快速决策流程

```
你的训练脚本属于哪种类型？
├── PDE Operator (PDENNEval 风格) → train_operator.md
│   特征: argparse 接收 config 路径, get_model() 按维度选模型, 自回归/单步训练
├── Graph 模型 → train_graph.md
│   特征: DGL 图数据, graph.ndata/edata 接口, AMP 支持, load/save_checkpoint
├── Transformer → train_transformer.md
│   特征: specific_params 多模型切换, 表面/体积加权损失, 训后测试
├── CNN → train_cnn.md
│   特征: 简单 tensor batch, 自定义 loss_func, 早停
├── Hydra + Trainer 类 → train_graph.md (Lagrangian_MGN 部分)
│   特征: @hydra.main 装饰器, hydra.utils.instantiate, Trainer 类封装
└── 不确定 → 先读 train_common.md
```
