# CMEMS 数据卡 (AI友好版)

## 数据集基本信息

**数据集名称**: CMEMS (Copernicus Marine Environment Monitoring Service)
**数据集版本**: v1
**数据集来源**: Copernicus Marine Environment Monitoring Service
**数据类型**: 海洋再分析数据
**时间范围**: 1993年至今
**空间分辨率**: 1/12° × 1/12° (约8km × 8km)
**时间分辨率**: 日数据/月数据
**数据格式**: NetCDF, HDF5 (OneScience处理后)
**更新频率**: 每日更新

## 数据存储结构

### OneScience标准存储结构

```
{ONESCIENCE_DATASETS_DIR}/CMEMS/h5/
├── metadata.json          # 数据集元信息
├── stats/                 # 统计信息
│   ├── global_means.npy   # 全局均值
│   └── global_stds.npy    # 全局标准差
├── static/                # 静态数据
│   ├── bathymetry.nc      # 海洋深度
│   └── mask.nc            # 海洋掩码
└── data/                  # 主数据
    ├── 1993/              # 年份目录
    │   ├── 19930101.h5    # 1993年1月1日数据
    │   ├── 19930102.h5    # 1993年1月2日数据
    │   └── ...
    ├── 1994/
    └── ...
```

### 文件命名规则

- HDF5文件命名: `yyyymmdd.h5`
  - yyyy: 4位年份
  - mm: 2位月份
  - dd: 2位日期

### 数据结构

- **单个HDF5文件**:
  - 数据域: `fields`
  - 维度: `[C, H, W]`
    - C: 变量数量
    - H: 高度方向像素数
    - W: 宽度方向像素数
  - 变量顺序: 严格按照 `metadata.json` 中的 `variables` 顺序

## 元数据信息

### metadata.json 结构

```json
{
    "years": ["1993", "1994", ..., "2025"],
    "variables": [
        "sea_surface_temperature",
        "sea_surface_height",
        ...
    ],
    "total_files": 365,
    "shape": ["C", "H", "W"],
    "resolution": {
        "spatial": "1/12° × 1/12°",
        "temporal": "日数据"
    }
}
```

- `years`: 可用年份列表
- `variables`: 包含的海洋变量
- `total_files`: 每年的文件数 (日数据约365个)

## 支持的变量

### 海面变量
| 变量名 | 描述 | 单位 |
|--------|------|------|
| sea_surface_temperature | 海面温度 | °C |
| sea_surface_height | 海面高度 | m |
| sea_surface_temperature_anomaly | 海面温度异常 | °C |
| sea_surface_height_anomaly | 海面高度异常 | m |
| surface_eastward_sea_water_velocity | 东向海流速度 | m/s |
| surface_northward_sea_water_velocity | 北向海流速度 | m/s |

### 水体变量
| 变量名 | 描述 | 单位 | 层级 |
|--------|------|------|------|
| ocean_temperature | 海水温度 | °C | 多层 |
| ocean_salinity | 海水盐度 | PSU | 多层 |
| ocean_density | 海水密度 | kg/m³ | 多层 |
| ocean_mixed_layer_thickness | 混合层厚度 | m | - |
| ocean_vertical_diffusivity | 垂向扩散系数 | m²/s | - |

### 其他变量
| 变量名 | 描述 | 单位 |
|--------|------|------|
| sea_ice_concentration | 海冰浓度 | % |
| sea_ice_thickness | 海冰厚度 | m |
| wind_speed | 风速 | m/s |
| air_temperature | 气温 | °C |

## 数据读取方法

### OneScience库读取 (推荐)

```python
from onescience.datapipes.climate import CMEMSDatapipe
from onescience.utils.YParams import YParams

# 读取配置文件
config_file_path = "conf/config.yaml"
cfg = YParams(config_file_path, "datapipe")

# 初始化Datapipe
datapipe = CMEMSDatapipe(
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
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CMEMSHDF5Dataset(Dataset):
    """CMEMS HDF5数据集"""
    
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
        self.years = list(map(int, self.metadata["years"]))
        self.variables = self.metadata["variables"]
        
        # 验证变量和年份
        missing_vars = [v for v in self.used_variables if v not in self.variables]
        if missing_vars:
            raise ValueError(f"Missing variables: {missing_vars}")
        
        missing_years = [y for y in self.used_years if y not in self.years]
        if missing_years:
            raise ValueError(f"Missing years: {missing_years}")
        
        self.channel_indices = [self.variables.index(v) for v in self.used_variables]
    
    def _init_normalization(self):
        """初始化标准化参数"""
        if self.normalize:
            means = np.load(os.path.join(self.data_dir, 'stats', "global_means.npy"))
            stds = np.load(os.path.join(self.data_dir, 'stats', "global_stds.npy"))
            self.means = means[:, self.channel_indices, :, :]
            self.stds = stds[:, self.channel_indices, :, :]
    
    def _init_files(self):
        """初始化文件列表"""
        self.files = {}
        for year in self.used_years:
            path = os.path.join(self.data_dir, 'data', str(year))
            files = sorted(glob.glob(os.path.join(path, "*.h5")))
            self.files[year] = files
        
        self.samples_per_year = len(files) - self.output_steps - (self.input_steps - 1)
        self.total_samples = len(self.used_years) * self.samples_per_year
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """获取单个样本"""
        year_idx = idx // self.samples_per_year
        step_idx = idx % self.samples_per_year
        year = self.used_years[year_idx]
        files = self.files[year]
        
        file_indices = range(step_idx, step_idx + self.input_steps + self.output_steps)
        
        data_list = []
        for i in file_indices:
            with h5py.File(files[i], "r") as f:
                data = f["fields"][:]  # [C, H, W]
                data = data[self.channel_indices]
                data_list.append(data)
        
        data = np.stack(data_list, axis=0)  # [T, C, H, W]
        invar = torch.as_tensor(data[:self.input_steps])
        outvar = torch.as_tensor(data[self.input_steps:])
        
        if self.normalize:
            invar = (invar - self.means) / self.stds
            outvar = (outvar - self.means) / self.stds
        
        return invar.squeeze(0), outvar.squeeze(0)
```

## 配置文件示例

```yaml
# conf/config.yaml
model:
  # 模型配置
  name: "OceanForecast"
  
datapipe:
  dataset:
    type: "hdf5"
    data_dir: "$ONESCIENCE_DATASETS_DIR/CMEMS/h5/"
    stats_dir: "$ONESCIENCE_DATASETS_DIR/CMEMS/h5/stats/"
    static_dir: "$ONESCIENCE_DATASETS_DIR/CMEMS/h5/static/"
    
    # 数据集划分
    train_ratio: 0.7    # 训练集比例
    val_ratio: 0.15     # 验证集比例
    test_ratio: 0.15    # 测试集比例
    
    # 时间步设置
    input_steps: 5      # 输入时间步数
    output_steps: 1     # 输出时间步数
    
    # 变量列表
    channels: [
        "sea_surface_temperature",
        "sea_surface_height",
        "surface_eastward_sea_water_velocity",
        "surface_northward_sea_water_velocity"
    ]
    
    # 数据参数
    img_size: [C, H, W]
    time_res: 24        # 时间分辨率 (小时)
    
  dataloader:
    batch_size: 32
    num_workers: 8
    shuffle: true
    pin_memory: true
```

## 数据处理流程

1. **数据获取**: 从CMEMS官网或Copernicus Marine Service下载原始NetCDF数据
2. **数据转换**: 将原始NetCDF数据转换为OneScience标准HDF5格式
3. **数据读取**: 使用CMEMSDatapipe或自定义Dataset读取数据
4. **数据预处理**:
   - 变量筛选: 根据模型需求选择相关变量
   - 时间窗切片: 切取连续时间窗样本
   - 数据标准化: 使用stats目录下的统计数据进行标准化
   - NaN处理: CMEMS数据可能包含NaN值，需要特殊处理
5. **数据加载**: 通过DataLoader批量加载数据
6. **模型训练/推理**: 将数据输入模型进行训练或推理
7. **结果分析**: 分析模型输出结果

## 模型接入接口

### 输入规范

- **形状**: `[B, C, H, W]` 或 `[B, T, C, H, W]`
  - B: 批量大小
  - T: 时间步数 (如适用)
  - C: 通道/变量数
  - H: 高度
  - W: 宽度

- **数据类型**: `torch.float32`

- **数值范围**: 标准化后的范围

### 输出规范

- **形状**: 与输入相同或根据模型任务定义
- **数据类型**: `torch.float32`

### 批量处理支持

```python
# 示例: 批量数据加载
dataset = CMEMSHDF5Dataset(...)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

for batch_idx, data in enumerate(dataloader):
    invar, outvar = data
    # 训练代码...
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
        'min': float(spatial_data.min()),
        'max': float(spatial_data.max())
    }
```

### 海洋指标计算

```python
def calculate_ocean_indices(ds):
    """计算海洋指标"""
    indices = {}
    
    # 计算月平均海面温度
    if 'sea_surface_temperature' in ds:
        monthly_sst = ds['sea_surface_temperature'].resample(time='1M').mean()
        indices['monthly_sst'] = monthly_sst
    
    # 计算海面高度异常
    if 'sea_surface_height' in ds:
        ssh_anom = ds['sea_surface_height'] - ds['sea_surface_height'].mean()
        indices['ssh_anomaly'] = ssh_anom
    
    # 计算海流速度
    if 'surface_eastward_sea_water_velocity' in ds:
        u = ds['surface_eastward_sea_water_velocity']
        v = ds['surface_northward_sea_water_velocity']
        current_speed = np.sqrt(u**2 + v**2)
        indices['current_speed'] = current_speed
    
    return indices
```

## 依赖库

- **核心库**:
  - numpy, torch, xarray, h5py
  - pandas - 时间序列处理

- **可视化库**:
  - matplotlib - 基本可视化
  - cartopy - 地理空间可视化
  - xarray - 数据处理

- **OneScience库**:
  - onescience.datapipes.climate.CMEMSDatapipe
  - onescience.utils.YParams - 配置文件处理

## 注意事项

1. **数据量**: CMEMS数据量大，建议使用分块读取和处理
2. **计算资源**: 大规模数据分析可能需要高性能计算资源
3. **数据版本**: 不同版本的CMEMS数据可能存在差异
4. **坐标系统**: CMEMS使用特定的网格系统，注意坐标转换
5. **NaN处理**: CMEMS数据可能包含NaN值，需要特殊处理
6. **时间处理**: CMEMS时间格式为UTC，注意时区转换
7. **数据标准化**: 使用提供的统计数据进行标准化
8. **内存管理**: 处理大规模数据时注意内存使用

## 故障排查

### 常见问题

1. **文件路径错误**
   - 症状: 找不到数据文件或统计文件
   - 解决方案: 检查`ONESCIENCE_DATASETS_DIR`环境变量是否正确设置

2. **变量不存在**
   - 症状: 报错"Missing required variables"
   - 解决方案: 检查`metadata.json`中的变量列表，确保使用正确的变量名

3. **内存不足**
   - 症状: OOM错误或进程被杀死
   - 解决方案: 减小batch_size，使用分块处理，或使用更强大的计算资源

4. **NaN值处理**
   - 症状: 训练过程中出现NaN
   - 解决方案: 检查数据中的NaN值，使用均值填充或忽略NaN值

5. **时间步错误**
   - 症状: 时间窗切片失败
   - 解决方案: 确保`input_steps`和`output_steps`设置合理，不超过文件数量

## 参考资源

- [Copernicus Marine Service](https://marine.copernicus.eu/)
- [CMEMS Documentation](https://resources.marine.copernicus.eu/)
- [xarray Documentation](https://xarray.pydata.org/en/stable/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

## 更新日志

- **v1.0** (2025-04): 初始版本，支持CMEMS全球海洋再分析数据