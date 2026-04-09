# 数据集名称 (AI友好版)

## 数据集基本信息

**数据集名称**: [数据集名称]
**数据集版本**: [版本号，如 v1.0]
**数据集来源**: [数据来源机构/项目]
**数据类型**: [气象/海洋/CFD/通用等]
**时间范围**: [起始时间] 至 [结束时间]
**空间分辨率**: [分辨率，如 0.25° × 0.25°]
**时间分辨率**: [时间间隔，如 6小时]
**数据格式**: [HDF5/NetCDF/GRIB/CSV等]
**更新频率**: [每日/每周/每月等]

## 数据存储结构

### OneScience标准存储结构

```
{ONESCIENCE_DATASETS_DIR}/[DATASET_NAME]/
├── metadata.json          # 数据集元信息
├── stats/                 # 统计信息
│   ├── global_means.npy   # 全局均值
│   └── global_stds.npy    # 全局标准差
├── static/                # 静态数据
│   └── [静态数据文件]      # 地形、掩码等
└── data/                  # 主数据
    ├── [year1]/           # 年份目录
    │   ├── [timestamp1].[ext]
    │   └── ...
    └── [year2]/
        └── ...
```

### 文件命名规则

- 数据文件: `[时间戳格式].[扩展名]`
  - 示例: `1979010100.h5`, `20200101.nc`

### 数据结构

- **单个数据文件**:
  - 数据域: `[数据域名称]`
  - 维度: `[维度说明，如 [C, H, W] 或 [T, C, H, W]]`
    - C: 变量数量
    - H: 高度方向像素数
    - W: 宽度方向像素数
    - T: 时间步数 (如适用)
  - 变量顺序: 严格按照 `metadata.json` 中的 `variables` 顺序

## 元数据信息

### metadata.json 结构

```json
{
    "years": ["年份列表"],
    "variables": ["变量名列表"],
    "total_files": 每年文件数,
    "shape": ["C", "H", "W"],
    "resolution": {
        "spatial": "空间分辨率",
        "temporal": "时间分辨率"
    }
}
```

## 支持的变量

### [变量类别1，如表面层变量]

| 变量名 | 描述 | 单位 | 备注 |
|--------|------|------|------|
| [var1] | [描述] | [单位] | [备注] |
| [var2] | [描述] | [单位] | [备注] |

### [变量类别2，如压力层变量]

| 变量名 | 描述 | 单位 | 层级 |
|--------|------|------|------|
| [var1] | [描述] | [单位] | [层级] |
| [var2] | [描述] | [单位] | [层级] |

## 数据读取方法

### OneScience库读取 (推荐)

```python
from onescience.datapipes.[domain] import [DatasetName]Datapipe
from onescience.utils.YParams import YParams

# 读取配置文件
config_file_path = "conf/config.yaml"
cfg = YParams(config_file_path, "datapipe")

# 初始化Datapipe
datapipe = [DatasetName]Datapipe(
    params=cfg, 
    distributed=False,
    input_steps=cfg.dataset.input_steps,
    output_steps=cfg.dataset.output_steps
)

# 获取DataLoader
train_dataloader, train_sampler = datapipe.train_dataloader()
val_dataloader, val_sampler = datapipe.val_dataloader()

# 使用DataLoader
for data in train_dataloader:
    input_data = data[0]   # 输入数据
    target_data = data[1]  # 目标数据
    # 训练代码...
```

### 自定义读取方法

```python
import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class [DatasetName]Dataset(Dataset):
    """自定义数据集"""
    
    def __init__(
        self, 
        data_dir: str,
        used_years: list,
        used_variables: list,
        input_steps: int = 1,
        output_steps: int = 1,
        normalize: bool = True
    ):
        self.data_dir = data_dir
        self.used_years = used_years
        self.used_variables = used_variables
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.normalize = normalize
        
        self._init_paths()
        self._init_normalization()
        self._init_files()
    
    def _init_paths(self):
        """初始化路径和元数据"""
        meta_path = os.path.join(self.data_dir, 'metadata.json')
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        self.variables = self.metadata["variables"]
        
        # 验证变量和年份
        missing_vars = [v for v in self.used_variables if v not in self.variables]
        if missing_vars:
            raise ValueError(f"Missing variables: {missing_vars}")
    
    def _init_normalization(self):
        """初始化标准化参数"""
        if self.normalize:
            self.channel_indices = [self.variables.index(v) for v in self.used_variables]
            means = np.load(os.path.join(self.data_dir, 'stats', "global_means.npy"))
            stds = np.load(os.path.join(self.data_dir, 'stats', "global_stds.npy"))
            self.means = means[:, self.channel_indices, :, :]
            self.stds = stds[:, self.channel_indices, :, :]
    
    def _init_files(self):
        """初始化文件列表"""
        self.files = {}
        for year in self.used_years:
            path = os.path.join(self.data_dir, 'data', str(year))
            files = sorted(glob.glob(os.path.join(path, "*.[ext]")))
            self.files[year] = files
        self.total_samples = sum(len(f) for f in self.files.values())
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 实现数据读取逻辑
        pass
```

## 配置文件示例

```yaml
# conf/config.yaml
model:
  # 模型配置
  name: "[模型名称]"
  
datapipe:
  dataset:
    type: "[数据格式]"
    data_dir: "$ONESCIENCE_DATASETS_DIR/[DATASET_NAME]/"
    stats_dir: "$ONESCIENCE_DATASETS_DIR/[DATASET_NAME]/stats/"
    static_dir: "$ONESCIENCE_DATASETS_DIR/[DATASET_NAME]/static/"
    
    # 数据集划分
    train_years: [训练年份列表]
    val_years: [验证年份列表]
    test_years: [测试年份列表]
    
    # 时间步设置
    input_steps: 1
    output_steps: 1
    
    # 变量列表
    channels: ["变量名列表"]
    
  dataloader:
    batch_size: 32
    num_workers: 8
    shuffle: true
    pin_memory: true
```

## 数据处理流程

1. **数据获取**: 从数据源获取原始数据
2. **数据转换**: 转换为OneScience标准格式
3. **数据读取**: 使用Datapipe或自定义Dataset读取
4. **数据预处理**:
   - 变量筛选
   - 时间窗切片
   - 数据标准化
   - [其他预处理步骤]
5. **数据加载**: 通过DataLoader批量加载
6. **模型训练/推理**: 输入模型进行训练或推理
7. **结果分析**: 分析模型输出

## 模型接入接口

### 输入规范

- **形状**: `[B, C, H, W]` 或 `[B, T, C, H, W]`
  - B: 批量大小
  - T: 时间步数 (如适用)
  - C: 通道/变量数
  - H: 高度
  - W: 宽度

- **数据类型**: `torch.float32`

- **数值范围**: [说明数值范围，如标准化后的范围]

### 输出规范

- **形状**: 与输入相同或根据模型任务定义
- **数据类型**: `torch.float32`

### 批量处理支持

```python
# 示例: 批量数据加载
dataset = [DatasetName]Dataset(...)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

## 数据分析功能

### 时间序列分析

```python
def analyze_timeseries(ds, variable, location):
    """分析特定位置的时间序列"""
    ts = ds[variable].sel(**location)
    return {
        'mean': float(ts.mean()),
        'std': float(ts.std()),
        'min': float(ts.min()),
        'max': float(ts.max())
    }
```

### 空间分析

```python
def analyze_spatial(ds, variable, time):
    """分析特定时间的空间分布"""
    spatial_data = ds[variable].sel(time=time)
    return {
        'mean': float(spatial_data.mean()),
        'std': float(spatial_data.std()),
    }
```

## 依赖库

- **核心库**:
  - numpy, torch, xarray, h5py
  - [其他必要的库]

- **OneScience库**:
  - onescience.datapipes.[domain]
  - onescience.utils.YParams

## 注意事项

1. **数据量**: [说明数据量大小和处理建议]
2. **计算资源**: [说明需要的计算资源]
3. **内存管理**: [内存使用建议和优化方法]
4. **版本兼容**: [数据版本和代码版本兼容性]

## 故障排查

### 常见问题

1. **文件路径错误**
   - 症状: [错误描述]
   - 解决方案: [解决方法]

2. **变量不存在**
   - 症状: [错误描述]
   - 解决方案: [解决方法]

3. **内存不足**
   - 症状: [错误描述]
   - 解决方案: [解决方法]

## 参考资源

- [官方文档链接]
- [数据下载链接]
- [相关论文/报告]

## 更新日志

- **[版本号]** ([日期]): [更新内容]
