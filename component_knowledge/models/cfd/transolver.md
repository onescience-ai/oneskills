# Transolver 模型编写指南 (Skills)

本文档是 OneScience Transolver 系列模型的使用规范与修改指南。基于已有的 Transolver2D、Transolver2D_plus、Transolver3D、Transolver3D_plus 以及辅助模块 MLP、NN、GraphSAGE、PointNet、GUNet 总结而成。

---

## 1. 整体架构

### 1.1 模型总览

Transolver 是一系列基于 Transformer 的物理场求解器模型，专为非结构化网格上的偏微分方程求解而设计。核心思想是通过 **Physics-Attention（物理注意力）** 机制将输入点云切分为多个物理切片（slice），在切片内进行注意力计算，从而高效捕捉物理场的空间依赖。

| 模型 | 维度 | 注意力类型 | 适用场景 |
|------|------|-----------|---------|
| `Transolver2D` | 2D | `unstructured` | 2D CFD、弹性力学等标准 2D 物理场 |
| `Transolver2D_plus` | 2D | `unstructured_plus` | 2D 不规则网格，需要更强建模能力 |
| `Transolver3D` | 3D | `unstructured` | 3D CFD 等标准 3D 物理场 |
| `Transolver3D_plus` | 3D | `unstructured_plus` | 3D 不规则网格，需要更强建模能力 |

### 1.2 辅助模型

| 模型 | 说明 | 依赖 |
|------|------|------|
| `MLP` | 多层感知机，支持 BatchNorm 和 Dropout | `torch_geometric.nn.Linear` |
| `NN` | 编码器-MLP-解码器结构 | `MLP` |
| `GraphSAGE` | 基于 SAGEConv 的图神经网络 | `torch_geometric` |
| `PointNet` | 点云处理网络，含全局池化 | `torch_geometric`, `MLP` |
| `GUNet` | 图 U-Net，支持多尺度下采样/上采样 | `torch_geometric` |

### 1.3 文件结构

```
src/onescience/models/transolver/
├── __init__.py              # 导出 Transolver2D/3D 及 plus 变体
├── Transolver2D.py          # 2D 标准版
├── Transolver2D_plus.py     # 2D 增强版 (unstructured_plus)
├── Transolver3D.py          # 3D 标准版
├── Transolver3D_plus.py     # 3D 增强版 (unstructured_plus)
├── MLP.py                   # 通用 MLP (基于 torch_geometric)
├── NN.py                    # 编码器-MLP-解码器
├── GraphSAGE.py             # GraphSAGE 图网络
├── PointNet.py              # PointNet 点云网络
└── GUNet.py                 # 图 U-Net 多尺度网络
```

---

## 2. 核心模型详解

### 2.1 模型架构流程

所有 Transolver 变体遵循相同的三阶段流程：

```
输入 data (含 x, pos)
    │
    ▼
[可选] 统一位置编码 (unified_pos) ── get_grid() 生成参考网格距离特征
    │
    ▼
预处理 MLP (OneMlp: input_dim → n_hidden*2 → n_hidden)
    │
    ▼
加上 placeholder 可学习偏置
    │
    ▼
N 层 Transolver Block (OneTransformer, style="Transolver_block")
    │  ├── 物理注意力 (Physics-Attention)
    │  ├── 切片机制 (slice_num 控制切片数)
    │  └── 最后一层输出 out_dim
    ▼
输出 (N, out_dim)
```

### 2.2 构造参数说明

所有 Transolver 变体共享以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `space_dim` | int | 1 | 空间维度（与输入坐标维度一致，2D 问题设为 2，3D 问题设为 3） |
| `n_layers` | int | 5 | Transolver Block 的堆叠层数，层数越多建模能力越强但计算量越大 |
| `n_hidden` | int | 256 | 隐藏层特征维度，决定模型容量 |
| `dropout` | float | 0 | Dropout 概率，用于正则化 |
| `n_head` | int | 8 | 注意力头数，需整除 n_hidden |
| `act` | str | 'gelu' | 激活函数类型 |
| `mlp_ratio` | float | 1 | Transolver Block 中 FFN 的膨胀比率 |
| `fun_dim` | int | 1 | 输入物理场的特征维度（不含坐标） |
| `out_dim` | int | 1 | 输出特征维度 |
| `slice_num` | int | 32 | 物理注意力中的切片数量，越大分辨率越高 |
| `ref` | int | 8 | 统一位置编码的参考网格分辨率 |
| `unified_pos` | bool | False | 是否启用统一位置编码 |

### 2.3 标准版 vs Plus 版区别

唯一区别在于 `OneTransformer` 的 `geotype` 参数：

- **标准版** (`Transolver2D`, `Transolver3D`): `geotype='unstructured'`
  - 标准物理注意力切片机制
- **Plus 版** (`Transolver2D_plus`, `Transolver3D_plus`): `geotype='unstructured_plus'`
  - 使用 Gumbel Softmax 加权切片，增强对不规则网格的建模能力

### 2.4 2D vs 3D 版区别

| 特性 | 2D | 3D |
|------|----|----|
| `unified_pos` 输入维度 | `fun_dim + space_dim + ref^2` | `fun_dim + ref^3` |
| `get_grid()` 参考网格 | 2D 网格 (ref x ref) | 3D 网格 (ref x ref x ref) |
| `get_grid()` 坐标范围 | x:[-2,4], y:[-1.5,1.5] | x:[-1.5,1.5], y:[0,2], z:[-4,4] |
| 距离特征维度 | `ref * ref` | `ref * ref * ref` |

---

## 3. 依赖的 OneScience 模块

Transolver 核心依赖两个 OneScience 统一模块：

### 3.1 OneMlp

```python
from onescience.modules import OneMlp

# Transolver 中的预处理 MLP
self.preprocess = OneMlp(
    style="StandardMLP",        # MLP 风格
    input_dim=input_dim,        # 输入维度
    hidden_dims=[n_hidden * 2], # 隐藏层维度列表
    output_dim=n_hidden,        # 输出维度
    activation=act,             # 激活函数
    use_bias=True,              # 使用偏置
    use_skip_connection=False   # 无跳跃连接
)
```

### 3.2 OneTransformer

```python
from onescience.modules import OneTransformer

# Transolver Block
OneTransformer(
    style="Transolver_block",     # 固定风格
    num_heads=n_head,             # 注意力头数
    hidden_dim=n_hidden,          # 隐藏维度
    dropout=dropout,              # Dropout
    act=act,                      # 激活函数
    mlp_ratio=mlp_ratio,          # FFN 膨胀比
    out_dim=out_dim,              # 输出维度 (仅最后一层使用)
    slice_num=slice_num,          # 物理切片数
    last_layer=True/False,        # 是否为最后一层
    geotype='unstructured'        # 或 'unstructured_plus'
)
```

---

## 4. 输入/输出格式

### 4.1 输入格式

Transolver 接受一个 `data` 对象（通常为 `torch_geometric.data.Data` 或类似结构），需包含：

| 属性 | 形状 | 说明 |
|------|------|------|
| `data.x` | `(N, fun_dim + space_dim)` | 拼接了坐标和物理特征的输入张量 |
| `data.pos` | `(N, space_dim)` | 节点坐标（仅 `unified_pos=True` 时需要） |

其中 `N` 为节点数。注意：模型内部会在 batch 维度上增加一维 `[1, N, C]`。

### 4.2 输出格式

| 形状 | 说明 |
|------|------|
| `(N, out_dim)` | 每个节点的预测物理场值 |

---

## 5. 调用示例

### 5.1 基本使用

```python
from onescience.models.transolver import Transolver2D, Transolver3D

# === 2D 问题 ===
model_2d = Transolver2D(
    space_dim=2,       # 2D 坐标
    fun_dim=3,         # 3 个输入物理量 (如边界条件标记等)
    out_dim=3,         # 3 个输出物理量 (如 Ux, Uy, p)
    n_layers=5,
    n_hidden=256,
    n_head=8,
    slice_num=32,
)
output = model_2d(data)  # data.x: (N, 5), 输出: (N, 3)

# === 3D 问题 ===
model_3d = Transolver3D(
    space_dim=3,       # 3D 坐标
    fun_dim=4,         # 4 个输入物理量
    out_dim=4,         # 4 个输出物理量 (如 Ux, Uy, Uz, p)
    n_layers=8,
    n_hidden=512,
    n_head=8,
    slice_num=64,
)
output = model_3d(data)  # data.x: (N, 7), 输出: (N, 4)
```

### 5.2 使用增强版

```python
from onescience.models.transolver import Transolver2D_plus, Transolver3D_plus

# 增强版适用于高度不规则的网格
model = Transolver2D_plus(
    space_dim=2,
    fun_dim=3,
    out_dim=3,
    n_layers=5,
    n_hidden=256,
    slice_num=64,      # 可增大切片数以提高分辨率
)
```

### 5.3 启用统一位置编码

```python
model = Transolver2D(
    space_dim=2,
    fun_dim=3,
    out_dim=3,
    unified_pos=True,  # 启用统一位置编码
    ref=8,             # 参考网格分辨率 (会增加 ref^2 维特征)
)
# 注意：启用后 data 对象必须包含 data.pos 属性
```

---

## 6. 辅助模型调用示例

### 6.1 MLP

```python
from onescience.models.transolver.MLP import MLP

mlp = MLP(
    channel_list=[64, 128, 256, 3],  # 输入64 → 128 → 256 → 输出3
    dropout=0.1,
    batch_norm=True,
    relu_first=False,
)
out = mlp(x)  # x: (N, 64) → out: (N, 3)
```

### 6.2 NN (编码器-MLP-解码器)

```python
from onescience.models.transolver.NN import NN

hparams = {
    'nb_hidden_layers': 3,
    'size_hidden_layers': 256,
    'bn_bool': True,
    'encoder': [5, 64],      # encoder 最后一层输出 64
    'decoder': [64, 3],
}
encoder = nn.Linear(5, 64)
decoder = nn.Linear(64, 3)
model = NN(hparams, encoder, decoder)
out = model(data)  # data.x: (N, 5) → out: (N, 3)
```

### 6.3 GraphSAGE

```python
from onescience.models.transolver.GraphSAGE import GraphSAGE

hparams = {
    'nb_hidden_layers': 3,
    'size_hidden_layers': 128,
    'bn_bool': True,
    'encoder': [5, 64],
    'decoder': [64, 3],
}
encoder = nn.Linear(5, 64)
decoder = nn.Linear(64, 3)
model = GraphSAGE(hparams, encoder, decoder)
out = model(data)  # data 需包含 data.x 和 data.edge_index
```

### 6.4 PointNet

```python
from onescience.models.transolver.PointNet import PointNet

hparams = {
    'base_nb': 32,
    'encoder': [5, 64],
}
encoder = nn.Linear(5, 64)
decoder = nn.Linear(64, 3)
model = PointNet(hparams, encoder, decoder)
out = model(data)  # data 需包含 data.x, data.batch
```

### 6.5 GUNet

```python
from onescience.models.transolver.GUNet import GUNet

hparams = {
    'nb_scale': 3,
    'layer': 'SAGE',           # 或 'GAT'
    'pool': 'topk',            # 或 'random'
    'pool_ratio': [0.5, 0.5],
    'list_r': [0.1, 0.2],
    'size_hidden_layers': 64,
    'max_neighbors': 64,
    'encoder': [5, 64],
    'decoder': [64, 3],
    'batchnorm': True,
    'res': True,               # 残差连接
}
encoder = nn.Linear(5, 64)
decoder = nn.Linear(64, 3)
model = GUNet(hparams, encoder, decoder)
out = model(data)  # data 需包含 data.x, data.edge_index
```

---

## 7. 常见修改场景

### 7.1 调整模型规模

```python
# 小模型 (适合快速实验)
model = Transolver2D(n_layers=3, n_hidden=128, n_head=4, slice_num=16)

# 大模型 (适合复杂物理场)
model = Transolver3D(n_layers=12, n_hidden=512, n_head=16, slice_num=128)
```

### 7.2 修改输入/输出维度

根据具体物理问题调整 `space_dim`、`fun_dim`、`out_dim`：

```python
# 示例：2D 翼型绕流
# 输入: 坐标(x,y) + SDF + 法向量(nx,ny) → space_dim=2, fun_dim=3
# 输出: 速度(Ux,Uy) + 压力(p) → out_dim=3
model = Transolver2D(space_dim=2, fun_dim=3, out_dim=3)
# data.x 形状应为 (N, 5)，即 [x, y, sdf, nx, ny] 拼接
```

### 7.3 修改统一位置编码的坐标范围

如果数据的坐标范围与默认值不匹配，需修改 `get_grid()` 中的 `np.linspace` 范围：

```python
# Transolver2D.get_grid() 中默认范围
gridx = torch.tensor(np.linspace(-2, 4, self.ref), ...)    # x 方向
gridy = torch.tensor(np.linspace(-1.5, 1.5, self.ref), ...) # y 方向

# 修改为适合你的数据的范围
gridx = torch.tensor(np.linspace(x_min, x_max, self.ref), ...)
gridy = torch.tensor(np.linspace(y_min, y_max, self.ref), ...)
```

### 7.4 新增 Transolver 变体

如需创建新变体（如支持结构化网格的版本），按以下模板：

```python
import torch
import torch.nn as nn
import numpy as np
from timm.layers import trunc_normal_
from onescience.modules import OneMlp, OneTransformer

class TransolverXX(nn.Module):
    """新变体说明"""
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256,
                 dropout=0, n_head=8, act='gelu', mlp_ratio=1,
                 fun_dim=1, out_dim=1, slice_num=32,
                 ref=8, unified_pos=False):
        super().__init__()
        # 1. 计算 input_dim
        # 2. 创建 preprocess (OneMlp)
        # 3. 创建 blocks (OneTransformer 列表)
        # 4. 初始化权重
        # 5. 创建 placeholder

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data):
        # 1. 提取 data.x
        # 2. [可选] 统一位置编码
        # 3. 预处理 MLP + placeholder
        # 4. 逐层通过 blocks
        # 5. 返回结果
        pass
```

### 7.5 修改 geotype 以使用结构化网格

如果数据是规则网格（如图像形式的物理场），可将 `geotype` 改为 `'structured'`（需确认 `OneTransformer` 支持该类型）：

```python
self.blocks = nn.ModuleList([
    OneTransformer(
        style="Transolver_block",
        geotype='structured',   # 结构化网格注意力
        ...
    )
    for _ in range(n_layers)
])
```

---

## 8. 注册新模型

在 `__init__.py` 中添加导入：

```python
# src/onescience/models/transolver/__init__.py
from .Transolver3D import Transolver3D
from .Transolver3D_plus import Transolver3D_plus
from .Transolver2D import Transolver2D
from .Transolver2D_plus import Transolver2D_plus
from .TransolverXX import TransolverXX  # 新增变体
```

---

## 9. 关键设计约束与注意事项

1. **输入格式**: `data.x` 必须包含坐标和物理特征的拼接，模型内部会自动添加 batch 维度 `[1, N, C]`
2. **batch size**: 当前实现中 forward 方法对 `data.x` 做 `x[None, :, :]`，即隐式 batch_size=1。如需支持 batch 训练，需配合 `torch_geometric.loader.DataLoader` 使用
3. **权重初始化**: 使用 `trunc_normal_` 初始化 Linear 权重，LayerNorm/BatchNorm 初始化为标准值
4. **placeholder**: 可学习偏置参数 `(1/n_hidden) * rand(n_hidden)`，在预处理后加到特征上
5. **统一位置编码**: 启用后会显著增加输入维度（2D: +ref^2, 3D: +ref^3），需确保 `data.pos` 可用
6. **slice_num**: 物理切片数是影响性能和精度的关键超参数，建议根据问题复杂度调整（简单问题 16-32，复杂问题 64-128）
7. **n_head 必须整除 n_hidden**: 否则注意力计算会报错

---

## 10. Checklist

新建或修改 Transolver 模型时请确认：

- [ ] `space_dim` 与输入坐标维度一致
- [ ] `fun_dim` 与输入物理特征维度一致（不含坐标部分，除非 data.x 中已拼接坐标）
- [ ] `out_dim` 与目标物理量维度一致
- [ ] `n_head` 能整除 `n_hidden`
- [ ] 如启用 `unified_pos`，确保 `data.pos` 可用且坐标范围与 `get_grid()` 匹配
- [ ] 新变体已在 `__init__.py` 中注册
- [ ] 使用 `OneMlp` (style="StandardMLP") 作为预处理层
- [ ] 使用 `OneTransformer` (style="Transolver_block") 构建 blocks
- [ ] 权重初始化使用 `trunc_normal_` + constant 方案
- [ ] 最后一层 block 设置 `last_layer=True`
