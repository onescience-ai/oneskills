# Biology 数据卡 (AI友好版)

## 数据集基本信息

**数据集名称**: Biology (生物信息学数据)
**数据集版本**: v1
**数据集来源**: 多源生物数据
**数据类型**: 生物分子结构与序列数据
**时间范围**: N/A (非时间序列数据)
**空间分辨率**: 原子级分辨率 (Å)
**时间分辨率**: N/A
**数据格式**: PDB, CIF, FASTA, A3M, SDF, Mol2
**更新频率**: 持续更新

## 数据存储结构

### OneScience标准存储结构

```
{ONESCIENCE_DATASETS_DIR}/Biology/
├── metadata.json          # 数据集元信息
├── structures/            # 结构数据
│   ├── pdb/               # PDB格式结构
│   ├── cif/               # CIF格式结构
│   └── ...
├── sequences/             # 序列数据
│   ├── fasta/             # FASTA格式序列
│   └── ...
├── msa/                   # 多序列比对
│   ├── a3m/               # A3M格式MSA
│   └── ...
└── features/              # 提取的特征
    ├── embeddings/        # 序列嵌入
    └── graphs/            # 分子图特征
```

### 文件命名规则

- **结构文件**: `[PDB_ID]_[chain].{pdb,cif}`
  - 示例: `1A2B_A.pdb`, `7XYZ_B.cif`
  
- **序列文件**: `[PDB_ID]_[chain].fasta`
  - 示例: `1A2B_A.fasta`
  
- **MSA文件**: `[PDB_ID]_[chain].a3m`
  - 示例: `1A2B_A.a3m`

### 数据结构

#### 结构数据

- **PDB/CIF文件**:
  - 原子坐标: `[N_atoms, 3]`
  - 原子类型: `[N_atoms]`
  - 残基类型: `[N_residues]`
  - 二级结构: `[N_residues]`

#### 序列数据

- **FASTA文件**:
  - 序列: `str`
  - 长度: `N_residues`
  - 字母表: 20种氨基酸 + 特殊字符

#### MSA数据

- **A3M文件**:
  - 多序列: `[N_seqs, N_cols]`
  - 序列身份: `[N_seqs]`
  - 一致性: `float`

## 元数据信息

### metadata.json 结构

```json
{
    "dataset_type": "biology",
    "domain": "protein/genome/multimer",
    "total_samples": 100000,
    "data_formats": ["pdb", "cif", "fasta", "a3m"],
    "sequence_stats": {
        "min_length": 10,
        "max_length": 5000,
        "avg_length": 350
    },
    "structure_stats": {
        "total_chains": 50000,
        "total_residues": 17500000
    }
}
```

## 支持的数据类型

### 蛋白质数据

| 数据类型 | 格式 | 描述 | 用途 |
|----------|------|------|------|
| 蛋白质结构 | PDB/CIF | 原子级三维结构 | 结构预测、设计 |
| 蛋白质序列 | FASTA | 氨基酸序列 | 序列分析、嵌入 |
| 多序列比对 | A3M | 同源序列比对 | MSA特征提取 |

### 基因组数据

| 数据类型 | 格式 | 描述 | 用途 |
|----------|------|------|------|
| 基因组序列 | FASTA | DNA/RNA序列 | 基因组分析 |
| 基因注释 | GFF/GTF | 基因注释信息 | 功能分析 |

### 多聚体数据

| 数据类型 | 描述 | 用途 |
|----------|------|------|
| 蛋白质复合物 | 多链蛋白质结构 | 相互作用分析 |
| 蛋白质-配体复合物 | 蛋白质与小分子复合物 | 药物设计 |

## 数据读取方法

### OneScience库读取 (推荐)

```python
from onescience.datapipes.biology import (
    ProteinDataset,
    GenomeDataset,
    MultimerDataset,
    get_protein_dataloader,
)

# 方式1: 直接使用数据集类
from onescience.datapipes.core.config import DatasetConfig

config = {
    "source": {
        "path": "/data/proteins"
    },
    "data": {
        "extra": {
            "use_msa": True,
            "use_structure": True,
            "max_msa_seqs": 100
        }
    }
}

dataset = ProteinDataset(config)

# 方式2: 使用统一加载器
dataloader = get_protein_dataloader(
    data_path="/data/proteins",
    batch_size=4,
    use_msa=True,
    use_structure=True
)

# 使用DataLoader
for batch in dataloader:
    features = batch["features"]
    atom_array = batch["atom_array"]
    # 训练代码...
```

### 自定义读取方法

```python
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader

class BiologyDataset(Dataset):
    """生物信息学数据集"""
    
    def __init__(
        self,
        data_dir: str,
        data_type: str = "protein",
        use_msa: bool = False,
        use_structure: bool = False,
        max_length: int = 512,
        **kwargs
    ):
        self.data_dir = Path(data_dir)
        self.data_type = data_type
        self.use_msa = use_msa
        self.use_structure = use_structure
        self.max_length = max_length
        
        self._init_data_list()
        self._init_parsers()
    
    def _init_data_list(self):
        """初始化数据列表"""
        if self.data_type == "protein":
            self.data_list = self._load_protein_data()
        elif self.data_type == "genome":
            self.data_list = self._load_genome_data()
        else:
            raise ValueError(f"Unknown data type: {self.data_type}")
    
    def _init_parsers(self):
        """初始化解析器"""
        from onescience.datapipes.biology.common.sequence.fasta_parser import FASTAParser
        from onescience.datapipes.biology.common.json.json_parser import JSONParser
        
        self.fasta_parser = FASTAParser()
        self.json_parser = JSONParser()
    
    def _load_protein_data(self):
        """加载蛋白质数据"""
        data_list = []
        fasta_files = list(self.data_dir.glob("**/*.fasta"))
        
        for fasta_file in fasta_files:
            entry = {
                "fasta_path": str(fasta_file),
                "pdb_id": fasta_file.stem.split('_')[0],
                "chain": fasta_file.stem.split('_')[1] if '_' in fasta_file.stem else None
            }
            
            if self.use_msa:
                msa_file = fasta_file.with_suffix('.a3m')
                if msa_file.exists():
                    entry["msa_path"] = str(msa_file)
            
            if self.use_structure:
                pdb_file = fasta_file.with_suffix('.pdb')
                if pdb_file.exists():
                    entry["pdb_path"] = str(pdb_file)
            
            data_list.append(entry)
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """获取样本"""
        entry = self.data_list[idx]
        
        # 解析序列
        sequence = self.fasta_parser.parse(entry["fasta_path"])
        
        # 构建特征
        features = {
            "sequence": sequence,
            "sequence_length": len(sequence)
        }
        
        # 解析MSA (可选)
        if self.use_msa and "msa_path" in entry:
            from onescience.datapipes.biology.common.msa.msa_parser import MSAParser
            msa_parser = MSAParser()
            msa = msa_parser.parse(entry["msa_path"])
            features["msa"] = msa
        
        # 解析结构 (可选)
        if self.use_structure and "pdb_path" in entry:
            from onescience.datapipes.biology.common.structure.structure_parser import StructureParser
            structure_parser = StructureParser()
            structure = structure_parser.parse(entry["pdb_path"])
            features["structure"] = structure
        
        return features
```

## 配置文件示例

```yaml
# conf/config.yaml
model:
  # 模型配置
  name: "ProteinStructureModel"
  
datapipe:
  dataset:
    type: "biology"
    
    # 数据源
    source:
      path: "$ONESCIENCE_DATASETS_DIR/Biology/"
    
    # 数据类型
    data_type: "protein"  # protein/genome/multimer
    
    # 额外参数
    extra:
      use_msa: true
      use_structure: true
      max_msa_seqs: 100
      max_length: 512
      sequence_type: "protein"  # protein/dna/rna
    
  dataloader:
    batch_size: 4
    num_workers: 8
    shuffle: true
    pin_memory: true
```

## 数据处理流程

1. **数据获取**: 从PDB、AlphaFold DB等获取生物数据
2. **数据预处理**:
   - 序列提取: 从结构文件中提取序列
   - MSA生成: 使用HHblits等工具生成MSA
   - 结构解析: 解析PDB/CIF文件
   - 特征提取: 提取序列/结构特征
3. **特征工程**:
   - 序列编码: 氨基酸/核苷酸编码
   - MSA特征: MSA序列比对
   - 结构特征: 原子坐标、距离图等
4. **数据加载**: 通过DataLoader批量加载数据
5. **模型训练/推理**: 将数据输入模型进行训练或推理
6. **结果分析**: 分析模型输出结果

## 模型接入接口

### 输入规范

#### 序列输入

- **形状**: `[B, L]` 或 `[B, L, C]`
  - B: 批量大小
  - L: 序列长度
  - C: 编码通道数

#### MSA输入

- **形状**: `[B, N, L]` 或 `[B, N, L, C]`
  - N: 序列数量
  - L: 序列长度
  - C: 编码通道数

#### 结构输入

- **原子坐标**: `[B, N_atoms, 3]`
- **原子类型**: `[B, N_atoms]`
- **距离图**: `[B, N_atoms, N_atoms]`

### 输出规范

- **结构预测**: 原子坐标 `[B, N_atoms, 3]`
- **序列预测**: 序列概率 `[B, L, 20]`
- **嵌入**: 特征向量 `[B, L, D]`

### 批量处理支持

```python
# 示例: 批量数据加载
dataloader = get_protein_dataloader(
    data_path="/data/proteins",
    batch_size=4,
    use_msa=True,
    use_structure=True
)

for batch in dataloader:
    features = batch["features"]
    atom_array = batch["atom_array"]
    # 训练代码...
```

## 数据分析功能

### 序列分析

```python
def analyze_sequence(sequence):
    """分析序列特性"""
    return {
        'length': len(sequence),
        'composition': {aa: sequence.count(aa) for aa in set(sequence)},
        'hydrophobicity': calculate_hydrophobicity(sequence)
    }

def calculate_hydrophobicity(sequence):
    """计算疏水性"""
    hydrophobicity_scale = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
        'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
        'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
        'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
        'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    return sum(hydrophobicity_scale.get(aa, 0) for aa in sequence) / len(sequence)
```

### 结构分析

```python
def analyze_structure(atom_array):
    """分析结构特性"""
    return {
        'num_atoms': len(atom_array),
        'num_residues': len(np.unique(atom_array.residues)),
        'centroid': calculate_centroid(atom_array.positions),
        'radius_of_gyration': calculate_radius_of_gyration(atom_array)
    }

def calculate_centroid(positions):
    """计算质心"""
    return np.mean(positions, axis=0)

def calculate_radius_of_gyration(atom_array):
    """计算回转半径"""
    positions = atom_array.positions
    centroid = calculate_centroid(positions)
    distances = np.sqrt(np.sum((positions - centroid)**2, axis=1))
    return np.sqrt(np.mean(distances**2))
```

## 依赖库

- **核心库**:
  - numpy - 数值计算
  - torch - 深度学习框架
  - biotite - 生物信息学
  - rdkit - 化学信息学

- **OneScience库**:
  - onescience.datapipes.biology
  - onescience.datapipes.biology.common
  - onescience.datapipes.core

## 注意事项

1. **数据量**: 生物数据量大，建议使用高效的数据加载器
2. **计算资源**: 结构预测需要大量计算资源
3. **内存管理**: 处理大蛋白质时注意内存使用
4. **数据质量**: 检查PDB文件质量，过滤低分辨率结构
5. **MSA生成**: MSA生成耗时，建议预计算并缓存

## 故障排查

### 常见问题

1. **文件路径错误**
   - 症状: 找不到数据文件
   - 解决方案: 检查数据路径配置是否正确

2. **序列长度超限**
   - 症状: 序列长度超过模型限制
   - 解决方案: 增加max_length或截断序列

3. **MSA生成失败**
   - 症状: MSA文件不存在或格式错误
   - 解决方案: 使用HHblits等工具重新生成MSA

4. **结构解析失败**
   - 症状: PDB文件格式错误
   - 解决方案: 检查PDB文件格式，使用标准格式

## 参考资源

- [Protein Data Bank](https://www.rcsb.org/)
- [AlphaFold Database](https://alphafold.ebi.ac.uk/)
- [biotite Documentation](https://biotite-python.org/)
- [RDKit Documentation](https://www.rdkit.org/docs/)

## 更新日志

- **v1.0** (2025-04): 初始版本，支持蛋白质、基因组、多聚体数据