# SyntheticWeather 数据卡 (AI友好版)

## 数据集基本信息

**数据集名称**: SyntheticWeather (合成天气数据)
**数据集版本**: v1
**数据集来源**: OneScience合成数据生成器
**数据类型**: 合成气象数据
**时间范围**: 可配置
**空间分辨率**: 可配置 (默认 721 × 1440)
**时间分辨率**: 日数据
**数据格式**: NumPy数组 (内存生成)
**更新频率**: 实时生成

## 数据存储结构

### 内存生成结构

```
SyntheticWeatherDataset (内存生成)
├── 生成参数配置
│   ├── channels: [层次列表]
│   ├── num_samples_per_year: 每年样本数
│   ├── num_steps: 时间步数
│   └── grid_size: (纬度, 经度)
└── 生成数据
    ├── temperatures: [num_days, num_channels, lat, lon]
    └── 临时存储 (可选)
```

### 数据生成参数

```python
{
    "channels": [0, 1, 2, ...],           # 大气层次通道
    "num_samples_per_year": 365,          # 每年天数
    "num_steps": 1,                       # 时间步数
    "grid_size": (721, 1440),             # 网格大小
    "base_temp": 15.0,                    # 基础温度
    "amplitude": 10.0,                    # 温度变化幅度
    "noise_level": 2.0,                   # 噪声水平
    "device": "cuda"                      # 计算设备
}
```

### 数据结构

- **生成数据**:
  - 数据域: `temperatures`
  - 维度: `[num_days, num_channels, lat, lon]`
    - num_days: 天数
    - num_channels: 大气层次数量
    - lat: 纬度方向像素数 (721)
    - lon: 经度方向像素数 (1440)
  - 数据类型: `numpy.ndarray` → `torch.Tensor`

## 元数据信息

### 数据集元数据

```json
{
    "dataset_type": "synthetic",
    "source": "OneScience SyntheticWeatherDataLoader",
    "grid_size": [721, 1440],
    "channels": ["channel_0", "channel_1", ...],
    "temporal_resolution": "daily",
    "spatial_resolution": "0.25° × 0.25°",
    "time_range": "configurable",
    "data_format": "numpy/torch"
}
```

## 支持的变量

### 大气层次变量

| 变量名 | 描述 | 单位 | 层级 |
|--------|------|------|------|
| temperature_layer_0 | 层次0温度 | °C | 地面附近 |
| temperature_layer_1 | 层次1温度 | °C | 近地面 |
| temperature_layer_2 | 层次2温度 | °C | 中层大气 |
| ... | ... | ... | ... |
| temperature_layer_N | 层次N温度 | °C | 高层大气 |

### 变量特性

- **温度分布**: 基于纬度、高度和时间的综合效应
- **空间变化**: 纬度梯度 + 噪声
- **时间变化**: 年周期 + 日变化
- **层次效应**: 随高度递减

## 数据读取方法

### OneScience库读取 (推荐)

```python
from onescience.datapipes.climate import SyntheticWeatherDataLoader

# 初始化DataLoader
dataloader = SyntheticWeatherDataLoader(
    channels=[0, 1, 2, 3, 4],           # 选择的层次
    num_samples_per_year=365,           # 每年天数
    num_steps=2,                        # 时间步数
    device="cuda",                      # 设备
    grid_size=(721, 1440),              # 网格大小
    base_temp=15.0,                     # 基础温度
    amplitude=10.0,                     # 振幅
    noise_level=2.0,                    # 噪声
    batch_size=32,                      # 批量大小
    shuffle=True,                       # 打乱
    num_workers=4                       # 工作线程
)

# 使用DataLoader
for batch in dataloader:
    invar = batch[0]["invar"]      # 输入数据
    outvar = batch[0]["outvar"]    # 目标数据
    # 训练代码...
```

### 自定义读取方法

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticWeatherDataset(Dataset):
    """合成天气数据集"""
    
    def __init__(
        self,
        channels: list,
        num_samples_per_year: int,
        num_steps: int,
        device: str = "cuda",
        grid_size: tuple = (721, 1440),
        base_temp: float = 15,
        amplitude: float = 10,
        noise_level: float = 2,
        **kwargs
    ):
        self.num_days = num_samples_per_year
        self.num_steps = num_steps
        self.num_channels = len(channels)
        self.device = device
        self.grid_size = grid_size
        
        # 生成数据
        self.temperatures = self.generate_data(
            num_days=self.num_days,
            num_channels=self.num_channels,
            grid_size=self.grid_size,
            base_temp=base_temp,
            amplitude=amplitude,
            noise_level=noise_level
        )
        
        print(f"Generated {len(self.temperatures)} samples")
    
    def generate_data(
        self,
        num_days: int,
        num_channels: int,
        grid_size: tuple,
        base_temp: float,
        amplitude: float,
        noise_level: float
    ) -> np.ndarray:
        """生成合成温度数据"""
        days = np.arange(num_days)
        latitudes, longitudes = grid_size
        
        # 高度效应
        altitude_effect = np.arange(num_channels) * -0.5
        altitude_effect = altitude_effect[:, np.newaxis, np.newaxis]
        altitude_effect = np.tile(altitude_effect, (1, latitudes, longitudes))
        altitude_effect = altitude_effect[np.newaxis, :, :, :]
        altitude_effect = np.tile(altitude_effect, (num_days, 1, 1, 1))
        
        # 纬度效应
        lat_variation = np.linspace(-amplitude, amplitude, latitudes)
        lat_variation = lat_variation[:, np.newaxis]
        lat_variation = np.tile(lat_variation, (1, longitudes))
        lat_variation = lat_variation[np.newaxis, np.newaxis, :, :]
        lat_variation = np.tile(lat_variation, (num_days, num_channels, 1, 1))
        
        # 时间效应
        time_effect = np.sin(2 * np.pi * days / 365)
        time_effect = time_effect[:, np.newaxis, np.newaxis, np.newaxis]
        time_effect = np.tile(time_effect, (1, num_channels, latitudes, longitudes))
        
        # 噪声
        noise = np.random.normal(scale=noise_level, size=(num_days, num_channels, latitudes, longitudes))
        
        # 计算温度
        daily_temps = base_temp + altitude_effect + lat_variation + time_effect + noise
        
        return daily_temps
    
    def __len__(self):
        return self.num_days - self.num_steps
    
    def __getitem__(self, idx):
        """获取样本"""
        invar = torch.tensor(self.temperatures[idx], dtype=torch.float32).to(self.device)
        outvar = torch.tensor(
            self.temperatures[idx + 1: idx + self.num_steps + 1],
            dtype=torch.float32
        ).to(self.device)
        
        return [{"invar": invar, "outvar": outvar}]
```

## 配置文件示例

```yaml
# conf/config.yaml
model:
  # 模型配置
  name: "WeatherForecast"
  
datapipe:
  dataset:
    type: "synthetic"
    
    # 数据生成参数
    channels: [0, 1, 2, 3, 4]
    num_samples_per_year: 365
    num_steps: 1
    
    # 网格参数
    grid_size: [721, 1440]
    base_temp: 15.0
    amplitude: 10.0
    noise_level: 2.0
    
    # 设备
    device: "cuda"
    
  dataloader:
    batch_size: 32
    num_workers: 4
    shuffle: true
    pin_memory: true
```

## 数据处理流程

1. **数据生成**: 使用SyntheticWeatherDataLoader生成合成数据
2. **数据预处理**:
   - 变量筛选: 根据模型需求选择层次
   - 时间窗切片: 切取连续时间窗样本
   - 数据标准化: 可选的标准化处理
3. **数据加载**: 通过DataLoader批量加载数据
4. **模型训练/推理**: 将数据输入模型进行训练或推理
5. **结果分析**: 分析模型输出结果

## 模型接入接口

### 输入规范

- **形状**: `[B, C, H, W]` 或 `[B, T, C, H, W]`
  - B: 批量大小
  - T: 时间步数 (如适用)
  - C: 通道/层次数
  - H: 高度 (纬度)
  - W: 宽度 (经度)

- **数据类型**: `torch.float32`

- **数值范围**: 根据生成参数确定

### 输出规范

- **形状**: 与输入相同或根据模型任务定义
- **数据类型**: `torch.float32`

### 批量处理支持

```python
# 示例: 批量数据加载
dataloader = SyntheticWeatherDataLoader(
    channels=[0, 1, 2],
    num_samples_per_year=365,
    num_steps=1,
    batch_size=32,
    shuffle=True
)

for batch in dataloader:
    invar = batch[0]["invar"]
    outvar = batch[0]["outvar"]
    # 训练代码...
```

## 数据分析功能

### 时间序列分析

```python
def analyze_timeseries(dataset, channel, location):
    """分析特定位置的时间序列"""
    ts = dataset.temperatures[:, channel, location[0], location[1]]
    return {
        'mean': float(ts.mean()),
        'std': float(ts.std()),
        'min': float(ts.min()),
        'max': float(ts.max())
    }
```

### 空间分析

```python
def analyze_spatial(dataset, channel, day):
    """分析特定时间的空间分布"""
    spatial_data = dataset.temperatures[day, channel, :, :]
    return {
        'mean': float(spatial_data.mean()),
        'std': float(spatial_data.std()),
        'min': float(spatial_data.min()),
        'max': float(spatial_data.max())
    }
```

### 层次分析

```python
def analyze_layers(dataset, day, location):
    """分析特定位置的层次分布"""
    layer_profile = dataset.temperatures[day, :, location[0], location[1]]
    return {
        'profile': layer_profile,
        'lapse_rate': np.gradient(layer_profile)  # 温度递减率
    }
```

## 依赖库

- **核心库**:
  - numpy - 数值计算
  - torch - 深度学习框架

- **OneScience库**:
  - onescience.datapipes.climate.SyntheticWeatherDataLoader
  - onescience.datapipes.climate.SyntheticWeatherDataset

## 注意事项

1. **数据生成**: 合成数据是随机生成的，每次运行结果可能不同
2. **可重复性**: 设置随机种子以获得可重复的结果
3. **内存使用**: 大网格和长时间序列会占用大量内存
4. **设备选择**: 根据计算资源选择CPU或GPU
5. **数据规模**: 合成数据适合快速原型开发和测试

## 故障排查

### 常见问题

1. **内存不足**
   - 症状: OOM错误或进程被杀死
   - 解决方案: 减小grid_size、num_samples_per_year或batch_size

2. **生成时间过长**
   - 症状: 数据生成耗时过长
   - 解决方案: 减少num_samples_per_year或使用更小的grid_size

3. **数据形状不匹配**
   - 症状: 模型输入形状错误
   - 解决方案: 检查channels和grid_size参数是否正确

4. **设备不匹配**
   - 症状: CPU/GPU数据不匹配
   - 解决方案: 确保device参数与模型设备一致

## 参考资源

- [OneScience Documentation](https://onescience.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

## 更新日志

- **v1.0** (2025-04): 初始版本，支持合成天气数据生成