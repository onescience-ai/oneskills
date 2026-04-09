---
name: onescience-data-processing
description: OneScience 数据处理技能，基于数据卡驱动架构，支持多种数据集的读写、处理和分析，提供与OneScience生态的无缝集成能力。
triggers:
  - 数据处理
  - 数据读写
  - 数据分析
  - 数据清洗
  - 数据转换
  - data processing
  - data analysis
  - 数据集处理
  - 数据管道
---

# OneScience 数据处理技能 (数据卡驱动版)

## 角色定位

你是 OneScience 数据处理技能，采用**数据卡驱动架构**，负责：
1. 参考数据集数据卡（Data Card）获取数据集元信息
2. 基于数据卡生成标准化的数据处理代码
3. 提供与 OneScience Datapipe 的无缝集成
4. 支持数据集在不同AI模型上的灵活接入

## 核心架构

### 数据卡驱动设计

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   数据卡文件     │────▶│  人工参考数据卡   │────▶│  数据管道生成器  │
│  (ERA5.md等)    │     │   编写代码       │     │ (PipelineBuilder)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                           ┌──────────────────────────────┼──────────────┐
                           ▼                              ▼              ▼
                    ┌─────────────┐              ┌──────────────┐  ┌─────────────┐
                    │ OneScience  │              │   自定义      │  │  模型适配    │
                    │  Datapipe   │              │  Dataset     │  │   接口       │
                    └─────────────┘              └──────────────┘  └─────────────┘
```

### 支持的数据集类型

| 数据集 | 数据卡文件 | 数据格式 | 主要用途 |
|--------|-----------|---------|---------|
| ERA5 | ERA5.md | HDF5/NetCDF | 气象预报模型 |
| CMEMS | CMEMS.md | NetCDF | 海洋数据模型 |
| GFS | GFS.md | GRIB/NetCDF | 天气预报模型 |
| DeepCFD | DeepCFD.md | HDF5 | CFD仿真模型 |
| 自定义 | custom.md | 多种格式 | 通用AI模型 |

## 数据卡规范

### 数据卡结构

每个数据卡必须包含以下部分：

```markdown
# 数据集名称

## 数据集基本信息
- 名称、版本、来源、类型
- 时间/空间分辨率、数据格式

## 数据存储结构
- 目录结构
- 文件命名规则
- 数据维度说明

## 元数据信息
- metadata.json 结构
- 变量列表和描述
- 统计信息位置

## 支持的变量
- 变量名、描述、单位、层级

## 数据读取方法
- OneScience Datapipe 方式
- 自定义 Dataset 方式

## 配置文件示例
- YAML配置模板

## 数据处理流程
- 预处理步骤
- 标准化方法
- 数据增强

## 模型接入接口
- 输入输出规范
- 数据形状要求
- 批量处理支持
```

### 数据卡配置对象

参考数据卡后，生成统一的数据集配置：

```python
@dataclass
class DatasetConfig:
    name: str                    # 数据集名称
    version: str                 # 版本
    data_dir: str                # 数据根目录
    format: str                  # 数据格式 (hdf5, netcdf, grib等)
    
    # 数据维度
    spatial_resolution: float    # 空间分辨率
    temporal_resolution: str     # 时间分辨率
    shape: Dict[str, int]        # 数据形状 {C, H, W, T}
    
    # 变量信息
    variables: List[Variable]    # 变量列表
    pressure_levels: List[int]   # 气压层
    
    # 存储结构
    metadata_path: str           # 元数据文件路径
    stats_dir: str               # 统计文件目录
    static_dir: str              # 静态数据目录
    data_structure: str          # 数据组织结构
    
    # 预处理配置
    normalization: bool          # 是否标准化
    normalization_stats: str     # 统计文件路径
    
    # 模型接入
    input_spec: InputSpec        # 输入规范
    output_spec: OutputSpec      # 输出规范
```

## 代码生成策略

### 1. OneScience Datapipe 集成模式 (推荐)

适用于使用 OneScience 框架的模型：

```python
# 生成的代码结构
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams

def create_datapipe(config_path: str, mode: str = 'train'):
    """创建 OneScience Datapipe"""
    cfg_data = YParams(config_path, "datapipe")
    
    datapipe = ERA5Datapipe(
        params=cfg_data,
        distributed=False,
        input_steps=cfg_data.dataset.input_steps,
        output_steps=cfg_data.dataset.output_steps
    )
    
    if mode == 'train':
        return datapipe.train_dataloader()
    elif mode == 'val':
        return datapipe.val_dataloader()
    else:
        return datapipe.test_dataloader()
```

### 2. 自定义 Dataset 模式

适用于需要自定义处理的场景：

```python
# 生成的代码结构
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class ERA5CustomDataset(Dataset):
    """基于数据卡生成的自定义Dataset"""
    
    def __init__(self, data_card_config: DatasetConfig, **kwargs):
        self.config = data_card_config
        self._init_from_datacard()
        
    def _init_from_datacard(self):
        """根据数据卡配置初始化"""
        # 读取元数据
        # 初始化文件路径
        # 加载统计信息
        pass
        
    def __getitem__(self, idx):
        """根据数据卡规范读取数据"""
        pass
```

### 3. 通用数据读取模式

适用于快速数据探索和分析：

```python
# 生成的代码结构
class DataCardReader:
    """通用数据卡读取器"""
    
    def __init__(self, data_card_path: str):
        self.config = self._parse_datacard(data_card_path)
        
    def read_sample(self, **kwargs):
        """读取单个样本"""
        pass
        
    def read_batch(self, **kwargs):
        """读取批量数据"""
        pass
```

## 模型接入适配器

### 适配器设计

> **注意**: 模型适配器参考 OneScience 项目中的现有实现

```python
class ModelDataAdapter:
    """模型数据适配器 - 将数据集适配到不同模型"""
    
    def __init__(self, dataset_config: DatasetConfig, model_type: str):
        self.dataset_config = dataset_config
        self.model_type = model_type
        
    def get_input_shape(self) -> Tuple:
        """获取模型输入形状"""
        adapters = {
            'pangu': self._pangu_adapter,
            'fuxi': self._fuxi_adapter,
            'fourcastnet': self._fourcastnet_adapter,
            'custom': self._custom_adapter
        }
        return adapters.get(self.model_type, self._custom_adapter)()
        
    def _pangu_adapter(self):
        """Pangu-Weather 模型适配"""
        return {
            'input_shape': (self.dataset_config.shape['C'], 
                           self.dataset_config.shape['H'], 
                           self.dataset_config.shape['W']),
            'required_variables': self._get_pangu_variables()
        }
```

### 支持的模型类型

| 模型 | 适配器 | 特殊要求 |
|------|--------|---------|
| Pangu-Weather | PanguAdapter | 特定变量顺序 |
| FuXi | FuXiAdapter | 时间步处理 |
| FourCastNet | FourCastNetAdapter | 傅里叶变换预处理 |
| GraphCast | GraphCastAdapter | 图结构构建 |
| 自定义 | CustomAdapter | 灵活配置 |

## 使用流程

### 步骤1: 参考数据卡编写代码

```python
# 人工参考数据卡 assets/ERA5.md 中的信息
# 获取数据集配置信息:
# - 数据目录结构
# - 变量列表
# - 数据形状
# - 预处理方法

# 示例: 从数据卡获取的关键信息
dataset_info = {
    'name': 'ERA5',
    'data_dir': '$ONESCIENCE_DATASETS_DIR/ERA5/newh5/',
    'format': 'hdf5',
    'shape': {'C': 69, 'H': 721, 'W': 1440},
    'variables': [...],  # 变量列表
}
```

### 步骤2: 代码生成

基于数据卡信息生成数据处理代码：

```python
# 生成 OneScience Datapipe 代码
# 生成自定义 Dataset 代码
# 生成配置文件
# 生成模型适配代码
```

### 步骤3: 模型接入

参考 OneScience 项目中的模型适配器实现：

```python
# 使用 OneScience 提供的模型适配器
# 或基于数据卡信息自定义适配逻辑
```

## 输出文件结构

```
output/
├── datapipe.py              # OneScience Datapipe 代码
├── dataset.py               # 自定义 Dataset 代码
├── reader.py                # 通用数据读取代码
├── config.yaml              # 数据集配置文件
├── adapter.py               # 模型适配器代码
└── examples/
    ├── basic_usage.py       # 基础使用示例
    ├── model_integration.py # 模型集成示例
    └── data_analysis.py     # 数据分析示例
```

## 与其他技能的协作

```
┌─────────────────────────────────────────────────────────────┐
│                     OneScience 生态                         │
├─────────────────────────────────────────────────────────────┤
│  onescience-manager (任务编排)                               │
│         │                                                   │
│         ▼                                                   │
│  onescience-data-processing (本技能)                         │
│         │                                                   │
│    ┌────┴────┬─────────────┐                               │
│    ▼         ▼             ▼                               │
│  数据卡参考  代码生成      模型适配                          │
│    │         │             │                               │
│    └────┬────┴─────────────┘                               │
│         │                                                   │
│         ▼                                                   │
│  onescience-runtime (作业提交)                               │
└─────────────────────────────────────────────────────────────┘
```

## 最佳实践

### 1. 数据卡维护

- 保持数据卡与数据集版本同步
- 详细记录变量含义和单位
- 提供完整的数据处理示例

### 2. 代码复用

- 优先使用 OneScience Datapipe
- 自定义 Dataset 继承基础类
- 共享通用的预处理函数

### 3. 模型接入

- 使用适配器模式解耦数据和模型
- 定义清晰的输入输出规范
- 支持多种批量处理策略

### 4. 性能优化

- 使用分块读取处理大数据集
- 实现数据缓存机制
- 支持多线程/多进程数据加载

## 示例场景

### 场景1: ERA5数据用于Pangu模型

```python
# 用户输入: "使用ERA5数据训练Pangu模型"

# 1. 参考ERA5数据卡获取信息
# 从 assets/ERA5.md 获取:
# - 数据目录: $ONESCIENCE_DATASETS_DIR/ERA5/newh5/
# - 变量列表: 69个变量
# - 数据形状: [69, 721, 1440]
# - Pangu所需变量: 37个特定变量

# 2. 生成Pangu适配的数据管道代码
# - datapipe.py: 配置OneScience Datapipe
# - config.yaml: 数据集配置
# - adapter.py: 模型适配逻辑

# 3. 输出文件:
#    - pangu_era5_pipeline/datapipe.py
#    - pangu_era5_pipeline/config.yaml
#    - pangu_era5_pipeline/adapter.py
```

### 场景2: 自定义数据集接入

```python
# 用户输入: "为我的自定义数据集生成数据读取代码"

# 1. 用户提供数据卡或参考现有数据卡
# 参考 assets/DATACARD_TEMPLATE.md 格式

# 2. 基于数据卡信息生成代码
# 生成自定义Dataset类
# 生成数据读取函数
# 生成配置文件

# 3. 输出代码到指定目录
```

## 故障排查

### 常见问题

1. **数据卡信息不完整**
   - 检查数据卡是否包含所有必要字段
   - 参考 DATACARD_TEMPLATE.md 补充缺失信息

2. **代码生成错误**
   - 检查数据集配置是否正确
   - 验证输出目录权限

3. **模型接入失败**
   - 确认模型类型是否支持
   - 检查输入输出形状匹配
   - 参考 OneScience 项目中的适配器实现

4. **数据读取错误**
   - 验证数据文件路径
   - 检查数据格式是否正确

## 扩展开发

### 添加新的数据卡支持

1. 在 `assets/` 目录下创建新的数据卡文件
2. 遵循数据卡规范编写文档
3. 参考现有数据卡格式

### 添加新的模型适配器

1. 参考 OneScience 项目中的适配器实现
2. 在生成的代码中集成适配逻辑
3. 测试适配器与数据集的兼容性

## 参考资源

- [assets/ERA5.md](assets/ERA5.md) - ERA5数据卡示例
- [assets/DATACARD_TEMPLATE.md](assets/DATACARD_TEMPLATE.md) - 数据卡模板
- [assets/MODEL_ADAPTER_GUIDE.md](assets/MODEL_ADAPTER_GUIDE.md) - 模型适配器参考指南
- OneScience 项目中的模型适配器代码

## 总结

OneScience 数据处理技能采用数据卡驱动架构，通过参考数据集数据卡中的元信息，生成标准化的数据处理代码。模型适配器部分参考 OneScience 项目中的现有实现，实现数据集与AI模型的无缝集成。
