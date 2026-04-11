# Materials 数据卡 (AI友好版)

## 数据集基本信息

**数据集名称**: Materials (材料化学数据)
**数据集版本**: v1
**数据集来源**: 多源材料数据库
**数据类型**: 原子级材料结构与性质
**时间范围**: N/A (非时间序列数据)
**空间分辨率**: 原子级分辨率 (Å)
**时间分辨率**: N/A
**数据格式**: LMDB, HDF5, XYZ, CIF, POSCAR
**更新频率**: 持续更新

## 数据存储结构

### OneScience标准存储结构

```
{ONESCIENCE_DATASETS_DIR}/Materials/
├── metadata.json          # 数据集元信息
├── data/                  # 主数据
│   ├── dataset1.lmdb      # LMDB格式数据
│   ├── dataset2.h5        # HDF5格式数据
│   ├── dataset3.xyz       # XYZ格式数据
│   └── ...
├── stats/                 # 统计信息
│   ├── global_means.npy   # 全局均值
│   └── global_stds.npy    # 全局标准差
└── features/              # 提取的特征
    ├── embeddings/        # 分子嵌入
    └── graphs/            # 分子图特征
```

### 文件命名规则

- **LMDB**: `[dataset_name].lmdb`
- **HDF5**: `[dataset_name].h5` 或 `[dataset_name].hdf5`
- **XYZ**: `[dataset_name].xyz` 或 `[dataset_name].extxyz`
- **CIF**: `[dataset_name].cif`
- **POSCAR**: `POSCAR_[dataset_name]`

### 数据结构

#### LMDB/HDF5格式

- **单个样本**:
  - 原子坐标: `[N_atoms, 3]`
  - 原子类型: `[N_atoms]`
  - 能量: `float`
  - 力: `[N_atoms, 3]`
  - 应力: `[3, 3]`

#### XYZ/CIF格式

- **单个样本**:
  - 原子坐标: `[N_atoms, 3]`
  - 原子类型: `[N_atoms]`
  - 晶格参数: `[3, 3]` (可选)

## 元数据信息

### metadata.json 结构

```json
{
    "dataset_type": "materials",
    "domain": "chemistry/physics",
    "total_samples": 1000000,
    "data_formats": ["lmdb", "hdf5", "xyz", "cif"],
    "atom_types": ["H", "C", "N", "O", ...],
    "property_stats": {
        "energy": {
            "min": -1000.0,
            "max": 1000.0,
            "mean": 0.0,
            "std": 100.0
        },
        "forces": {
            "min": -50.0,
            "max": 50.0,
            "mean": 0.0,
            "std": 10.0
        }
    },
    "r_max": 6.0,
    "z_table": [1, 6, 7, 8, ...]
}
```

## 支持的变量

### 原子属性

| 变量名 | 描述 | 单位 | 维度 |
|--------|------|------|------|
| atomic_numbers | 原子序数 | - | [N_atoms] |
| positions | 原子坐标 | Å | [N_atoms, 3] |
| forces | 原子受力 | eV/Å | [N_atoms, 3] |
| charges | 原子电荷 | e | [N_atoms] |
| magnetic_moments | 磁矩 | μB | [N_atoms] |

### 分子属性

| 变量名 | 描述 | 单位 |
|--------|------|------|
| energy | 总能量 | eV |
| forces | 原子力 | eV/Å |
| stress | 应力张量 | eV/Å³ |
| dipole_moment | 偶极矩 | e·Å |
| polarizability | 极化率 | Å³ |

### 晶体属性

| 变量名 | 描述 | 单位 |
|--------|------|------|
| lattice_vectors | 晶格矢量 | Å |
| periodic_boundary_conditions | 周期性边界条件 | - |
| cell_volume | 晶胞体积 | Å³ |

## 数据读取方法

### OneScience库读取 (推荐)

```python
from onescience.datapipes.materials import MaterialsDatapipe
from onescience.utils.YParams import YParams

# 读取配置文件
config_file_path = "conf/config.yaml"
cfg = YParams(config_file_path, "datapipe")

# 初始化Datapipe
datapipe = MaterialsDatapipe(
    params=cfg, 
    distributed=False,
    r_max=6.0,           # 截断半径
    z_table=[1, 6, 7, 8] # 原子序数表
)

# 获取DataLoader
train_dataloader, train_sampler = datapipe.train_dataloader()
val_dataloader, val_sampler = datapipe.val_dataloader()

# 使用DataLoader
for batch in train_dataloader:
    graph = batch["graph"]        # 图数据
    energy = batch["energy"]      # 能量
    forces = batch["forces"]      # 力
    # 训练代码...
```

### 自定义读取方法

```python
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MaterialsDataset(Dataset):
    """材料化学数据集"""
    
    def __init__(
        self,
        data_dir: str,
        data_format: str = "lmdb",
        r_max: float = 6.0,
        z_table: list = None,
        **kwargs
    ):
        self.data_dir = data_dir
        self.data_format = data_format
        self.r_max = r_max
        self.z_table = z_table or [1, 6, 7, 8]
        
        self._init_data_list()
        self._init_reader()
    
    def _init_data_list(self):
        """初始化数据列表"""
        if self.data_format == "lmdb":
            self.data_list = self._load_lmdb_data()
        elif self.data_format == "hdf5":
            self.data_list = self._load_hdf5_data()
        elif self.data_format == "xyz":
            self.data_list = self._load_xyz_data()
        else:
            raise ValueError(f"Unknown format: {self.data_format}")
    
    def _init_reader(self):
        """初始化数据读取器"""
        if self.data_format == "lmdb":
            from onescience.datapipes.materials.pyg_stack.storage.lmdb_dataset import LMDBDataset
            self.reader = LMDBDataset(
                file_path=self.data_dir,
                r_max=self.r_max,
                z_table=self.z_table
            )
        elif self.data_format == "hdf5":
            from onescience.datapipes.materials.pyg_stack.storage.hdf5_dataset import HDF5Dataset
            self.reader = HDF5Dataset(
                file_path=self.data_dir,
                r_max=self.r_max,
                z_table=self.z_table
            )
    
    def _load_lmdb_data(self):
        """加载LMDB数据"""
        import lmdb
        env = lmdb.open(self.data_dir, readonly=True, lock=False)
        with env.begin() as txn:
            num_samples = txn.stat()['entries']
        return list(range(num_samples))
    
    def _load_hdf5_data(self):
        """加载HDF5数据"""
        import h5py
        with h5py.File(self.data_dir, 'r') as f:
            num_samples = len(f.keys())
        return list(range(num_samples))
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """获取样本"""
        if self.data_format == "lmdb":
            return self._get_lmdb_sample(idx)
        elif self.data_format == "hdf5":
            return self._get_hdf5_sample(idx)
    
    def _get_lmdb_sample(self, idx):
        """获取LMDB样本"""
        from onescience.datapipes.materials.pyg_stack.core.atomic_data import AtomicData
        atoms = self.reader.AseDB.get_atoms(idx)
        
        # 构建图数据
        graph = AtomicData.from_atoms(
            atoms,
            r_max=self.r_max,
            z_table=self.z_table
        )
        
        # 提取属性
        data = {
            "graph": graph,
            "energy": atoms.info.get("energy", 0.0),
            "forces": atoms.arrays.get("forces", np.zeros_like(atoms.positions))
        }
        
        return data
    
    def _get_hdf5_sample(self, idx):
        """获取HDF5样本"""
        import h5py
        with h5py.File(self.data_dir, 'r') as f:
            batch_idx = idx // self.batch_size
            config_idx = idx % self.batch_size
            
            grp = f[f"config_batch_{batch_idx}"]
            subgrp = grp[f"config_{config_idx}"]
            
            # 读取数据
            positions = subgrp["positions"][:]
            atomic_numbers = subgrp["atomic_numbers"][:]
            energy = subgrp.attrs.get("energy", 0.0)
            forces = subgrp.get("forces", None)
            
            # 构建图数据
            from onescience.datapipes.materials.pyg_stack.core.atomic_data import AtomicData
            graph = AtomicData(
                positions=positions,
                atomic_numbers=atomic_numbers,
                r_max=self.r_max,
                z_table=self.z_table
            )
            
            data = {
                "graph": graph,
                "energy": energy,
                "forces": forces
            }
            
            return data
```

## 配置文件示例

```yaml
# conf/config.yaml
model:
  # 模型配置
  name: "MACE"
  
datapipe:
  dataset:
    type: "materials"
    
    # 数据源
    source:
      path: "$ONESCIENCE_DATASETS_DIR/Materials/data.lmdb"
    
    # 材料特定参数
    extra:
      r_max: 6.0                    # 截断半径
      z_table: [1, 6, 7, 8, 9, 11]  # 原子序数表
      head: "Default"               # 头名称
      heads: ["Default"]            # 头列表
    
  dataloader:
    batch_size: 32
    num_workers: 8
    shuffle: true
    pin_memory: true
```

## 数据处理流程

1. **数据获取**: 从材料数据库获取原始结构数据
2. **数据转换**: 转换为OneScience标准格式(LMDB/HDF5)
3. **数据读取**: 使用MaterialsDatapipe或自定义Dataset读取
4. **数据预处理**:
   - 图构建: 构建原子图表示
   - 特征提取: 提取原子/分子特征
   - 数据标准化: 使用stats目录下的统计数据
5. **数据加载**: 通过DataLoader批量加载数据
6. **模型训练/推理**: 将数据输入模型进行训练或推理
7. **结果分析**: 分析模型输出结果

## 模型接入接口

### 输入规范

#### 图数据

- **原子坐标**: `torch.Tensor [N_atoms, 3]`
- **原子类型**: `torch.Tensor [N_atoms]`
- **边索引**: `torch.Tensor [2, N_edges]`
- **边特征**: `torch.Tensor [N_edges, D]`

#### 批量数据

- **图数据**: `Batch`对象
- **能量**: `torch.Tensor [B]`
- **力**: `torch.Tensor [N_atoms_total, 3]`

### 输出规范

- **能量预测**: `torch.Tensor [B]`
- **力预测**: `torch.Tensor [N_atoms_total, 3]`
- **嵌入**: `torch.Tensor [N_atoms_total, D]`

### 批量处理支持

```python
# 示例: 批量数据加载
dataloader = MaterialsDatapipe(
    params=config,
    distributed=False,
    r_max=6.0,
    z_table=[1, 6, 7, 8]
).train_dataloader()

for batch in dataloader:
    graph = batch["graph"]
    energy = batch["energy"]
    forces = batch["forces"]
    # 训练代码...
```

## 数据分析功能

### 结构分析

```python
def analyze_structure(atoms):
    """分析原子结构"""
    return {
        'num_atoms': len(atoms),
        'bond_lengths': calculate_bond_lengths(atoms),
        'bond_angles': calculate_bond_angles(atoms),
        'dihedral_angles': calculate_dihedral_angles(atoms)
    }

def calculate_bond_lengths(atoms, cutoff=2.0):
    """计算键长"""
    from ase.geometry import get_distances
    distances, _ = get_distances(atoms)
    return distances[distances < cutoff]
```

### 能量分析

```python
def analyze_energy(energies):
    """分析能量分布"""
    return {
        'min': float(np.min(energies)),
        'max': float(np.max(energies)),
        'mean': float(np.mean(energies)),
        'std': float(np.std(energies)),
        'heat_capacity': calculate_heat_capacity(energies)
    }

def calculate_heat_capacity(energies, temperature=300):
    """计算热容"""
    k_B = 8.617e-5  # eV/K
    energy_sq_mean = np.mean(energies**2)
    energy_mean_sq = np.mean(energies)**2
    Cv = (energy_sq_mean - energy_mean_sq) / (k_B * temperature**2)
    return Cv
```

### 力分析

```python
def analyze_forces(forces):
    """分析原子力"""
    return {
        'max_force': float(np.max(np.abs(forces))),
        'mean_force': float(np.mean(np.abs(forces))),
        'force_std': float(np.std(forces))
    }
```

## 依赖库

- **核心库**:
  - numpy - 数值计算
  - torch - 深度学习框架
  - ase - 原子模拟环境
  - lmdb - 键值存储
  - h5py - HDF5支持

- **OneScience库**:
  - onescience.datapipes.materials.MaterialsDatapipe
  - onescience.datapipes.materials.pyg_stack
  - onescience.utils.YParams

## 注意事项

1. **数据量**: 材料数据量大，建议使用LMDB/HDF5格式
2. **计算资源**: 原子级模拟需要大量计算资源
3. **内存管理**: 处理大体系时注意内存使用
4. **截断半径**: 合理设置r_max参数影响计算效率
5. **原子类型**: 确保z_table包含所有原子类型

## 故障排查

### 常见问题

1. **文件路径错误**
   - 症状: 找不到数据文件
   - 解决方案: 检查数据路径配置是否正确

2. **原子类型缺失**
   - 症状: 原子类型不在z_table中
   - 解决方案: 更新z_table包含所有原子类型

3. **截断半径过小**
   - 症状: 图连接不完整
   - 解决方案: 增加r_max参数

4. **内存不足**
   - 症状: OOM错误
   - 解决方案: 减小batch_size或使用更小的r_max

## 参考资源

- [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/)
- [LMDB Documentation](https://lmdb.readthedocs.io/)
- [MACE Documentation](https://github.com/ACEsuit/mace)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

## 更新日志

- **v1.0** (2025-04): 初始版本，支持LMDB/HDF5/XYZ/CIF格式