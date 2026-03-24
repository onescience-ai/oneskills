# OneScience 项目架构文档

## 项目概述

OneScience 是基于深度学习框架打造的科学计算工具包，专注于 AI for Science 领域，支持地球科学、计算流体、结构力学、生物信息、材料化学等多个科学计算领域。

**版本**: v0.2.1
**开发团队**: sugon-ai4s
**许可证**: Apache License 2.0

---

## 项目目录结构

```
onescience/
├── .claude/                    # Claude AI 配置
│   ├── checks/                # 检查规则
│   ├── skills/                # 技能定义
│   └── settings.json         # 配置文件
│
├── src/onescience/            # 核心源代码
│   ├── configs/              # 配置模块
│   ├── datapipes/            # 数据管道
│   ├── deploy/               # 部署工具
│   ├── distributed/          # 分布式训练
│   ├── flax_models/          # Flax 模型
│   ├── kernels/              # 自定义算子
│   ├── launch/               # 启动脚本
│   ├── memory/               # 内存管理
│   ├── metrics/              # 评估指标
│   ├── models/               # 模型库 ⭐
│   ├── modules/              # 基础模块 ⭐
│   ├── monitoring/           # 监控工具
│   ├── optimizers/           # 优化器
│   ├── registry/             # 注册机制
│   ├── sciui/                # 科学可视化
│   ├── utils/                # 工具函数
│   └── verification/         # 验证工具
│
├── examples/                  # 应用示例 ⭐
│   ├── biosciences/          # 生物信息 (10个模型)
│   ├── cfd/                  # 计算流体 (12个模型)
│   ├── earth/                # 地球科学 (12个模型)
│   ├── MaterialsChemistry/   # 材料化学 (2个模型)
│   ├── structural/           # 结构力学 (2个模型)
│   └── configs/              # 示例配置
│
├── tests/                     # 测试代码
├── benchmark/                 # 性能基准
├── doc/                       # 文档资源
├── docker/                    # Docker 配置
│
├── setup.py                   # 安装配置
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明
```

---

## 核心模块详解

### 1. models/ - 模型库

包含各领域的深度学习模型实现：

**地球科学模型**:
- `afno` - Adaptive Fourier Neural Operator (气象预报)
- `graphcast` - 图神经网络天气预报
- `fourcastnet` - FourCastNet 全球天气预报
- `fengwu` - 风乌天气预报模型
- `fuxi` - 伏羲天气预报模型
- `nowcastnet` - 短临降雨预报

**计算流体模型**:
- `meshgraphnet` - 网格图神经网络
- `deepcfd` - 深度学习 CFD 求解器
- `beno` - 边界元神经算子
- `cfdbench/cfd_benchmark` - CFD 基准测试模型集

**材料化学模型**:
- `UMA` - 通用原子尺度模拟
- `mace` - 原子间势函数拟合

**生物信息模型**:
- `evo2` - 基因序列分析

**通用组件**:
- `diffusion` - 扩散模型基础组件
- `GPs` - 高斯过程模型
- `graphvit` - 图视觉 Transformer
- `mesh_reduced` - 网格降维

### 2. modules/ - 基础模块

可复用的神经网络组件：

- `afno` - AFNO 层实现
- `attention` - 注意力机制
- `diffusion` - 扩散模块
- `embedding` - 嵌入层
- `encoder/decoder` - 编解码器
- `equivariant` - 等变神经网络
- `fourier/fft` - 傅里叶变换层
- `evolution` - 演化算法组件
- `layer` - 通用层定义
- `loss` - 损失函数
- `head` - 预测头
- `edge` - 边特征处理
- `linear/fc` - 全连接层
- `fuser` - 特征融合

### 3. datapipes/ - 数据管道

数据加载和预处理工具，支持多种科学数据格式。

### 4. distributed/ - 分布式训练

支持多 GPU/多节点训练的分布式框架。

### 5. optimizers/ - 优化器

包含各种优化算法实现。

### 6. metrics/ - 评估指标

科学计算领域的专用评估指标。

---

## 应用示例详解

### 生物信息学 (biosciences/)

包含 10 个生物信息学模型：

1. **AlphaFold3** - 蛋白质结构预测 (Pairformer + Diffusion)
2. **Protenix** - 蛋白质结构预测 (Transformer + Diffusion)
3. **RFdiffusion** - 蛋白质骨架设计 (Diffusion)
4. **ProteinMPNN** - 蛋白质序列设计 (MPNN)
5. **PT-DiT** - 蛋白质设计优化 (Diffusion + Transformer)
6. **Evo2** - 突变预测/基因分类 (StripedHyena2)
7. **MolSculptor** - 药物设计 (Latent Diffusion)
8. **OpenFold** - AlphaFold 开源实现
9. **SimpleFold** - 简化版蛋白质折叠
10. **AlphaFold** - 原始 AlphaFold 实现

### 地球科学 (earth/)

包含 12 个气象/海洋预报模型：

1. **CorrDiff** - 降尺度 (Unet + Diffusion)
2. **FourCastNet** - 中期天气预报 (AFNO)
3. **GraphCast** - 中期天气预报 (GNN)
4. **Pangu Weather** - 中期天气预报 (3D Transformer)
5. **NowCastNet** - 短临降雨 (GAN)
6. **FengWu** - 中期天气预报 (3D Transformer)
7. **Fuxi** - 中长期天气预报 (3D Transformer)
8. **Oceancast** - 海洋预报 (AFNO)
9. **Xihe** - 气象模型
10. **GraphCast JAX** - GraphCast JAX 版本
11. **ERA5 Dataset Prepare** - ERA5 数据集准备工具
12. **Pangu Weather Distributed** - Pangu 分布式版本

### 计算流体力学 (cfd/)

包含 12 个 CFD 模型：

1. **Transolver-Car-Design** - 汽车设计 (Transformer)
2. **Transolver-Airfoil-Design** - 翼型设计 (Transformer)
3. **Vortex Shedding MGN** - 圆柱绕流 (GNN)
4. **DeepCFD** - 2D 几何体绕流 (U-Net)
5. **PDENNEval** - PDE 求解模型集 (多种架构)
6. **PINNsformer** - 物理驱动 PDE 求解 (PINN)
7. **CFDBench** - 不可压流体 (多种模型)
8. **BENO** - 椭圆偏微分方程 (Transformer + GNN)
9. **Lagrangian MGN** - 拉格朗日网格 (GNN)
10. **CFD Benchmark** - 流体模型基准测试
11. **EagleMeshTransformer** - 湍流模拟 (Transformer)
12. **GP for TO** - 拓扑优化 (Gaussian Processes)

### 材料化学 (MaterialsChemistry/)

包含 2 个材料模拟模型：

1. **UMA** - 通用原子尺度模拟 (E(3)-等变 GNN)
   - 支持数据集: OC20, OMat24, OMol25, ODAC23, OMC25
2. **MACE** - 原子间势函数拟合 (E(3)-等变 GNN)
   - 支持数据集: MPTrj, SPICE, OMat24

### 结构力学 (structural/)

包含 2 个结构力学模型：

1. **DEM for Plasticity** - 弹塑性力学 (PINN)
2. **Plane Stress** - 2D 平面应力 (PINN)

---

## 技术特性

### 支持的深度学习框架
- PyTorch (主要)
- JAX/Flax (部分模型)

### 支持的硬件平台
- NVIDIA GPU
- 海光 DCU

### 分布式训练支持
- 多 GPU 训练
- 多节点训练
- DeepSpeed 集成
- Megatron-Core 支持

### 数据格式支持
- NetCDF4 (气象数据)
- HDF5
- PDB (蛋白质结构)
- XYZ (原子坐标)
- LMDB
- WebDataset

---

## 依赖管理

项目采用模块化依赖管理，根据应用领域安装对应依赖：

```bash
# 基础安装
pip install -e .

# 地球科学
pip install -e .[earth]

# 生物信息
pip install -e .[bio]

# 计算流体
pip install -e .[cfd]

# 材料化学
pip install -e .[chem]

# 量子计算
pip install -e .[quantum]

# 完整安装
pip install -e .[all]
```

### 核心依赖

**基础依赖**: numpy, torch, tqdm, hydra-core, wandb, mlflow, einops, omegaconf, mpi4py

**地球科学**: xarray, netcdf4, s3fs, pytz

**计算流体**: shapely, seaborn, deepxde, gpytorch, tensorflow

**生物信息**: biopython, rdkit, dm-haiku, jax, flax, transformers, datasets

**材料化学**: e3nn, ase, pymatgen, rdkit, matscipy, lmdb

---

## 快速开始

### 安装
```bash
cd onescience
pip install -e .
```

### 示例代码
```python
import torch
from onescience.models.unet import UNet

# 创建输入
inputs = torch.randn(1, 1, 96, 96, 96).cuda()

# 初始化模型
model = UNet(
    in_channels=1,
    out_channels=1,
    model_depth=5,
    feature_map_channels=[64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024],
    num_conv_blocks=2,
).cuda()

# 前向推理
output = model(inputs)
print(f"输出形状: {output.shape}")
```

---

## 项目特色

1. **多领域覆盖**: 支持物理、化学、生物、地球科学等多个领域
2. **模型丰富**: 集成 38+ 前沿 AI4S 模型
3. **工程化**: 完整的训练、推理、部署流程
4. **高性能**: 支持分布式训练和多硬件平台
5. **易扩展**: 模块化设计，易于添加新模型
6. **开放生态**: 开源协作，持续更新
