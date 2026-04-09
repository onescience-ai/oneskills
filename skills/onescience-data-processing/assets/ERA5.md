# ERA5 数据卡 (AI友好版)

## 数据集基本信息

**数据集名称**：ERA5
**数据集版本**：v1
**数据集来源**：European Centre for Medium-Range Weather Forecasts (ECMWF)
**数据类型**：气象再分析数据
**时间范围**：1940年至今
**空间分辨率**：0.25° × 0.25°
**数据格式**：NetCDF, GRIB (原始)；HDF5 (OneScience处理后)
**更新频率**：每日更新
**时间分辨率**：6小时 (每小时数据也可获取)

## 数据存储结构

### OneScience标准存储结构

```
{ONESCIENCE_DATASETS_DIR}/ERA5/newh5/
├── metadata.json        # 数据集元信息
├── stats/              # 统计信息
│   ├── global_means.npy  # 全局均值
│   └── global_stds.npy   # 全局标准差
├── static/             # 静态数据
│   ├── geopotential.nc   # 位势高度
│   ├── land_mask.npy     # 陆地掩码
│   ├── land_sea_mask.nc  # 海陆掩码
│   ├── soil_type.npy     # 土壤类型
│   └── topography.npy    # 地形数据
└── data/               # 主数据
    ├── 1979/            # 年份目录
    │   ├── 1979010100.h5  # 2020年1月1日00时数据
    │   ├── 1979010106.h5  # 2020年1月1日06时数据
    │   └── ...
    ├── 1980/
    └── ...
```

### 文件命名规则

- HDF5文件命名：`yyyymmddhh.h5`
  - yyyy: 4位年份
  - mm: 2位月份
  - dd: 2位日期
  - hh: 2位小时 (00, 06, 12, 18)

### 数据结构

- **单个HDF5文件**：
  - 数据域：`fields`
  - 维度：`[C, H, W]`
    - C: 变量数量
    - H: 高度方向像素数 (721)
    - W: 宽度方向像素数 (1440)
  - 变量顺序：严格按照 `metadata.json` 中的 `variables` 顺序

- **统计文件**：
  - 维度：`[1, C, 1, 1]`
  - 用于数据标准化

## 元数据信息

### metadata.json 结构

```json
{
    "years": ["1979", "1980", ..., "2025"],
    "variables": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        ...,
        "vertical_velocity_500"
    ],
    "total_files": 1460
}
```

- `years`：可用年份列表
- `variables`：包含的气象变量
- `total_files`：每年的文件数 (6小时分辨率，365×4=1460)

## 支持的变量

### 表面层变量
| 变量名 | 描述 | 单位 |
|--------|------|------|
| 10m_u_component_of_wind | 10米东向风 | m/s |
| 10m_v_component_of_wind | 10米北向风 | m/s |
| 2m_temperature | 2米温度 | K |
| 2m_dewpoint_temperature | 2米露点温度 | K |
| mean_sea_level_pressure | 平均海平面气压 | Pa |
| total_precipitation | 总降水量 | m |
| surface_solar_radiation_downwards | 表面向下短波辐射 | W/m² |

### 压力层变量
| 变量名 | 描述 | 单位 | 层级 |
|--------|------|------|------|
| geopotential_1000 | 位势高度 | m²/s² | 1000hPa |
| geopotential_925 | 位势高度 | m²/s² | 925hPa |
| geopotential_850 | 位势高度 | m²/s² | 850hPa |
| geopotential_700 | 位势高度 | m²/s² | 700hPa |
| geopotential_600 | 位势高度 | m²/s² | 600hPa |
| geopotential_500 | 位势高度 | m²/s² | 500hPa |
| geopotential_400 | 位势高度 | m²/s² | 400hPa |
| geopotential_300 | 位势高度 | m²/s² | 300hPa |
| geopotential_250 | 位势高度 | m²/s² | 250hPa |
| geopotential_200 | 位势高度 | m²/s² | 200hPa |
| geopotential_150 | 位势高度 | m²/s² | 150hPa |
| geopotential_100 | 位势高度 | m²/s² | 100hPa |
| geopotential_50 | 位势高度 | m²/s² | 50hPa |
| temperature_1000 | 温度 | K | 1000hPa |
| temperature_925 | 温度 | K | 925hPa |
| temperature_850 | 温度 | K | 850hPa |
| temperature_700 | 温度 | K | 700hPa |
| temperature_600 | 温度 | K | 600hPa |
| temperature_500 | 温度 | K | 500hPa |
| temperature_400 | 温度 | K | 400hPa |
| temperature_300 | 温度 | K | 300hPa |
| temperature_250 | 温度 | K | 250hPa |
| temperature_200 | 温度 | K | 200hPa |
| temperature_150 | 温度 | K | 150hPa |
| temperature_100 | 温度 | K | 100hPa |
| temperature_50 | 温度 | K | 50hPa |
| specific_humidity_1000 | 比湿 | kg/kg | 1000hPa |
| specific_humidity_925 | 比湿 | kg/kg | 925hPa |
| specific_humidity_850 | 比湿 | kg/kg | 850hPa |
| specific_humidity_700 | 比湿 | kg/kg | 700hPa |
| specific_humidity_600 | 比湿 | kg/kg | 600hPa |
| specific_humidity_500 | 比湿 | kg/kg | 500hPa |
| specific_humidity_400 | 比湿 | kg/kg | 400hPa |
| specific_humidity_300 | 比湿 | kg/kg | 300hPa |
| specific_humidity_250 | 比湿 | kg/kg | 250hPa |
| specific_humidity_200 | 比湿 | kg/kg | 200hPa |
| specific_humidity_150 | 比湿 | kg/kg | 150hPa |
| specific_humidity_100 | 比湿 | kg/kg | 100hPa |
| specific_humidity_50 | 比湿 | kg/kg | 50hPa |
| u_component_of_wind_1000 | 东向风 | m/s | 1000hPa |
| u_component_of_wind_925 | 东向风 | m/s | 925hPa |
| u_component_of_wind_850 | 东向风 | m/s | 850hPa |
| u_component_of_wind_700 | 东向风 | m/s | 700hPa |
| u_component_of_wind_600 | 东向风 | m/s | 600hPa |
| u_component_of_wind_500 | 东向风 | m/s | 500hPa |
| u_component_of_wind_400 | 东向风 | m/s | 400hPa |
| u_component_of_wind_300 | 东向风 | m/s | 300hPa |
| u_component_of_wind_250 | 东向风 | m/s | 250hPa |
| u_component_of_wind_200 | 东向风 | m/s | 200hPa |
| u_component_of_wind_150 | 东向风 | m/s | 150hPa |
| u_component_of_wind_100 | 东向风 | m/s | 100hPa |
| u_component_of_wind_50 | 东向风 | m/s | 50hPa |
| v_component_of_wind_1000 | 北向风 | m/s | 1000hPa |
| v_component_of_wind_925 | 北向风 | m/s | 925hPa |
| v_component_of_wind_850 | 北向风 | m/s | 850hPa |
| v_component_of_wind_700 | 北向风 | m/s | 700hPa |
| v_component_of_wind_600 | 北向风 | m/s | 600hPa |
| v_component_of_wind_500 | 北向风 | m/s | 500hPa |
| v_component_of_wind_400 | 北向风 | m/s | 400hPa |
| v_component_of_wind_300 | 北向风 | m/s | 300hPa |
| v_component_of_wind_250 | 北向风 | m/s | 250hPa |
| v_component_of_wind_200 | 北向风 | m/s | 200hPa |
| v_component_of_wind_150 | 北向风 | m/s | 150hPa |
| v_component_of_wind_100 | 北向风 | m/s | 100hPa |
| v_component_of_wind_50 | 北向风 | m/s | 50hPa |
| vertical_velocity_500 | 垂直速度 | Pa/s | 500hPa |

## 数据读取方法

### OneScience库读取 (推荐)

```python
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams

# 读取配置文件
config_file_path = "conf/config.yaml"
cfg = YParams(config_file_path, "model")
cfg_data = YParams(config_file_path, "datapipe")

# 初始化Datapipe
datapipe = ERA5Datapipe(
    params=cfg_data, 
    distributed=dist.is_initialized(), 
    input_steps=cfg_data.dataset.input_steps,
    output_steps=cfg_data.dataset.output_steps
)

# 获取DataLoader
train_dataloader, train_sampler = datapipe.train_dataloader()
val_dataloader, val_sampler = datapipe.val_dataloader()

# 使用DataLoader
for j, data in enumerate(train_dataloader):
    invar = data[0]  # 维度: [B, S, C, H, W] 或 [B, C, H, W]
    outvar = data[1]  # 维度: [B, S, C, H, W] 或 [B, C, H, W]
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

class ERA5HDF5Dataset(Dataset):
    def __init__(self, data_dir, used_years, used_variables, output_steps=1, input_steps=1, normalize=True):
        self.data_dir = data_dir
        self.used_years = used_years
        self.used_variables = used_variables
        self.output_steps = output_steps
        self.input_steps = input_steps
        self.normalize = normalize
        
        # 初始化路径和标准化数据
        self._init_paths()
        self._init_normalization()
        self._init_files()
    
    def _init_paths(self):
        # 读取元数据
        meta_path = os.path.join(self.data_dir, 'metadata.json')
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        self.years = list(map(int, self.metadata["years"]))
        self.variables = self.metadata["variables"]
        
        # 检查变量和年份是否存在
        missing_ch = [ch for ch in self.used_variables if ch not in self.variables]
        if missing_ch:
            raise ValueError(f"Missing required variables: {missing_ch}")
        missing_year = [year for year in self.used_years if year not in self.years]
        if missing_year:
            raise ValueError(f"Missing required years: {missing_year}")
    
    def _init_normalization(self):
        # 加载标准化统计数据
        self.channel_indices = [self.variables.index(v) for v in self.used_variables]
        means = np.load(os.path.join(self.data_dir, 'stats', "global_means.npy"))
        stds = np.load(os.path.join(self.data_dir, 'stats', "global_stds.npy"))
        self.means = means[:, self.channel_indices, :, :]
        self.stds = stds[:, self.channel_indices, :, :]
    
    def _init_files(self):
        # 初始化文件路径
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
        # 获取样本
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
  # 模型相关配置
  ...

datapipe:
  # 数据管道配置
  dataset:
    type: "hdf5"
    stats_dir: "$ONESCIENCE_DATASETS_DIR/ERA5/newh5/stats/"
    static_dir: "$ONESCIENCE_DATASETS_DIR/ERA5/newh5/static/"
    data_dir: "$ONESCIENCE_DATASETS_DIR/ERA5/newh5/"
    
    # 数据集划分
    train_ratio: 42    # 训练年份数
    val_ratio: 3       # 验证年份数
    test_ratio: 2      # 测试年份数
    
    # 时间步设置
    input_steps: 1     # 输入时间步数
    output_steps: 1    # 输出时间步数

    img_size: [2, 721, 1440]
    img_res: 6
    verbose: true
    cache: false
    
    # 气象变量
    channels:  [
        'mean_sea_level_pressure', 
        '10m_u_component_of_wind', 
        '10m_v_component_of_wind', 
        '2m_temperature', 
        'geopotential_1000', 
        'geopotential_925', 
        'geopotential_850', 
        'geopotential_700', 
        'geopotential_600', 
        'geopotential_500', 
        'geopotential_400', 
        'geopotential_300', 
        'geopotential_250', 
        'geopotential_200', 
        'geopotential_150', 
        'geopotential_100', 
        'geopotential_50',
        'specific_humidity_1000', 
        'specific_humidity_925', 
        'specific_humidity_850', 
        'specific_humidity_700', 
        'specific_humidity_600',
        'specific_humidity_500', 
        'specific_humidity_400', 
        'specific_humidity_300', 
        'specific_humidity_250', 
        'specific_humidity_200', 
        'specific_humidity_150', 
        'specific_humidity_100', 
        'specific_humidity_50', 
        'temperature_1000', 
        'temperature_925', 
        'temperature_850', 
        'temperature_700', 
        'temperature_600', 
        'temperature_500', 
        'temperature_400', 
        'temperature_300', 
        'temperature_250', 
        'temperature_200',
        'temperature_150', 
        'temperature_100', 
        'temperature_50', 
        'u_component_of_wind_1000', 
        'u_component_of_wind_925', 
        'u_component_of_wind_850', 
        'u_component_of_wind_700', 
        'u_component_of_wind_600', 
        'u_component_of_wind_500', 
        'u_component_of_wind_400', 
        'u_component_of_wind_300', 
        'u_component_of_wind_250', 
        'u_component_of_wind_200', 
        'u_component_of_wind_150', 
        'u_component_of_wind_100', 
        'u_component_of_wind_50', 
        'v_component_of_wind_1000', 
        'v_component_of_wind_925', 
        'v_component_of_wind_850', 
        'v_component_of_wind_700', 
        'v_component_of_wind_600', 
        'v_component_of_wind_500', 
        'v_component_of_wind_400', 
        'v_component_of_wind_300', 
        'v_component_of_wind_250', 
        'v_component_of_wind_200', 
        'v_component_of_wind_150', 
        'v_component_of_wind_100', 
        'v_component_of_wind_50'
    ]
```

## 数据处理流程

1. **数据获取**：从ECMWF官网或Copernicus Climate Data Store下载原始ERA5数据
2. **数据转换**：将原始NetCDF/GRIB数据转换为OneScience标准HDF5格式
3. **数据读取**：使用ERA5Datapipe或自定义Dataset读取数据
4. **数据预处理**：
   - 变量筛选：根据模型需求选择相关变量
   - 时间窗切片：切取连续时间窗样本
   - 数据标准化：使用stats目录下的统计数据进行标准化
   - 太阳天顶角计算：根据时间和经纬度计算太阳天顶角
5. **数据加载**：通过DataLoader批量加载数据
6. **模型训练/推理**：将数据输入模型进行训练或推理
7. **结果分析**：分析模型输出结果

## 数据分析功能

### 时间序列分析

```python
def time_series_analysis(ds, variable, lat, lon):
    """分析特定位置的时间序列"""
    # 提取特定位置的时间序列
    ts = ds[variable].sel(latitude=lat, longitude=lon, method='nearest')
    
    # 计算基本统计量
    stats = {
        'mean': float(ts.mean()),
        'std': float(ts.std()),
        'min': float(ts.min()),
        'max': float(ts.max())
    }
    
    return ts, stats
```

### 空间分析

```python
def spatial_analysis(ds, variable, time):
    """分析特定时间的空间分布"""
    # 提取特定时间的空间场
    spatial_data = ds[variable].sel(time=time, method='nearest')
    
    # 计算空间统计量
    stats = {
        'mean': float(spatial_data.mean()),
        'std': float(spatial_data.std()),
        'min': float(spatial_data.min()),
        'max': float(spatial_data.max())
    }
    
    return spatial_data, stats
```

### 气候指标计算

```python
def calculate_climatic_indices(ds):
    """计算气候指标"""
    indices = {}
    
    # 计算月平均温度
    if '2m_temperature' in ds:
        monthly_temp = ds['2m_temperature'].resample(time='1M').mean()
        indices['monthly_temperature'] = monthly_temp
    
    # 计算月总降水量
    if 'total_precipitation' in ds:
        monthly_precip = ds['total_precipitation'].resample(time='1M').sum()
        indices['monthly_precipitation'] = monthly_precip
    
    # 计算年平均温度
    if '2m_temperature' in ds:
        annual_temp = ds['2m_temperature'].resample(time='1Y').mean()
        indices['annual_temperature'] = annual_temp
    
    # 计算年总降水量
    if 'total_precipitation' in ds:
        annual_precip = ds['total_precipitation'].resample(time='1Y').sum()
        indices['annual_precipitation'] = annual_precip
    
    return indices
```

## 依赖库

- **核心库**：
  - xarray - 用于处理NetCDF数据
  - h5py - 用于处理HDF5数据
  - numpy - 数值计算
  - torch - 深度学习框架
  - pandas - 时间序列处理

- **可视化库**：
  - matplotlib - 基本可视化
  - cartopy - 地理空间可视化
  - seaborn - 统计可视化

- **OneScience库**：
  - onescience.datapipes.climate.ERA5Datapipe - 专用数据管道
  - onescience.utils.YParams - 配置文件处理

## 注意事项

1. **数据量**：ERA5数据量大，建议使用分块读取和处理
2. **计算资源**：大规模数据分析可能需要高性能计算资源
3. **数据版本**：不同版本的ERA5数据可能存在差异
4. **坐标系统**：ERA5使用经纬度坐标系统
5. **时间处理**：ERA5时间格式为UTC，注意时区转换
6. **变量命名**：不同数据源的变量命名可能不同，注意统一
7. **数据标准化**：使用提供的统计数据进行标准化，确保模型训练效果
8. **内存管理**：处理大规模数据时注意内存使用，避免OOM错误

## 适合的应用场景

- **全球天气预报模型**：使用ERA5数据训练深度学习天气预报模型
- **气候研究**：分析长期气候趋势和变化
- **极端天气事件分析**：研究台风、暴雨等极端天气事件
- **气候变化评估**：评估气候变化对不同地区的影响
- **可再生能源预测**：基于气象数据预测太阳能、风能等可再生能源产出

## 参考资源

- [ECMWF ERA5 Documentation](https://confluence.ecmwf.int/display/CKB/ERA5)
- [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- [xarray Documentation](https://xarray.pydata.org/en/stable/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [OneScience Documentation](https://onescience.readthedocs.io/)

## 故障排查

### 常见问题

1. **文件路径错误**
   - 症状：找不到数据文件或统计文件
   - 解决方案：检查`ONESCIENCE_DATASETS_DIR`环境变量是否正确设置

2. **变量不存在**
   - 症状：报错"Missing required variables"
   - 解决方案：检查`metadata.json`中的变量列表，确保使用正确的变量名

3. **内存不足**
   - 症状：OOM错误或进程被杀死
   - 解决方案：减小batch_size，使用分块处理，或使用更强大的计算资源

4. **标准化错误**
   - 症状：标准化后数据出现异常值
   - 解决方案：检查统计文件是否与数据匹配，确保使用正确的通道索引

5. **时间步错误**
   - 症状：时间窗切片失败
   - 解决方案：确保`input_steps`和`output_steps`设置合理，不超过文件数量

## 示例代码

### 完整的数据处理示例

```python
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams

# 配置文件路径
config_file = "conf/config.yaml"

# 读取配置
cfg = YParams(config_file, "model")
cfg_data = YParams(config_file, "datapipe")

# 初始化Datapipe
datapipe = ERA5Datapipe(
    params=cfg_data, 
    distributed=False, 
    input_steps=cfg_data.dataset.input_steps,
    output_steps=cfg_data.dataset.output_steps
)

# 获取DataLoader
train_loader, _ = datapipe.train_dataloader()

# 读取数据
for batch_idx, data in enumerate(train_loader):
    invar, outvar = data
    print(f"输入数据形状: {invar.shape}")
    print(f"输出数据形状: {outvar.shape}")
    break

# 数据可视化示例
# 注意：这里需要从原始数据文件中读取以进行可视化
# 实际应用中可以使用xarray读取NetCDF文件或h5py读取HDF5文件

print("✅ ERA5数据处理示例完成")
```

## 最佳实践

1. **数据预处理**：在训练前对数据进行充分的预处理，包括标准化、变量筛选等
2. **批量处理**：使用DataLoader进行批量处理，提高训练效率
3. **分布式训练**：对于大规模数据，使用分布式训练加速模型训练
4. **数据缓存**：合理使用数据缓存，减少I/O开销
5. **模型选择**：根据任务需求选择合适的模型架构
6. **超参数调优**：对模型超参数进行调优，提高模型性能
7. **结果验证**：使用独立的验证集验证模型性能
8. **模型部署**：将训练好的模型部署到生产环境，进行实时预测

## 总结

ERA5数据集是一个高质量的全球气象再分析数据集，通过OneScience的处理和组织，使其更适合用于深度学习模型的训练和推理。本数据卡提供了完整的数据集信息、存储结构、读取方法和使用示例，旨在帮助AI系统更有效地使用ERA5数据进行气象相关的研究和应用。