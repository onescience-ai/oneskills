# 中科天机数据集

## 数据集基本信息

**数据集名称**: 中科天机数据集
**数据集版本**: v1.0  
**数据集来源**: TianJi Weather Science and Technology Company  
**数据类型**: 气象  
**空间分辨率**: 0.025°~0.1°（视子集而定）  
**时间分辨率**: 1小时 / 6小时（视子集而定）  
**数据格式**: NetCDF (.nc)  
**更新频率**: 按预报批次更新  

### 子数据集说明

| 数据集  | 覆盖范围                                  | 分辨率             | 时间分辨率 | 预报时效         |
| --------| ------------------------------------------- | -------------------- | ------------ | ------------------ |
| TJ1-CN  | 中国区域（65°E~145°E，5°N~65°N） | 0.033°（约3km）   | 1小时      | 未来15天，逐小时 |
| TJ-CN   | 中国区域（65°E~145°E，5°N~65°N） | 0.025°（约2.5km） | 1小时      | 未来10天，逐小时 |
| TJ1-GB  | 全球（0°~360°E，-90°~90°N）  | 0.1°（约12km）    | 1小时     | 未来15天，逐小时 |
| TJ-GB| 全球（0°~360°E，-90°~90°N）  | 0.1°（约12km）    | 1小时/1~15天，6小时/16~46天     | 未来46天         |

---

## 数据存储结构

### OneScience标准存储结构

```
{ONESCIENCE_DATASETS_DIR}/TJWeather/
├── {dataset_type}/                 # 数据集类型（如：TJ1-CN, TJ-CN, TJ1-GB, TJ-GB）
│   ├── stats/                      # 统计信息
│   │   ├── global_means.nc         # 全局均值
│   │   └── global_stds.nc          # 全局标准差
│   │
│   ├── static/                     # 静态地理
│   │   ├── geopotential/           # 位势高度
│   │   ├── land_mask/              # 陆地掩码
│   │   ├── land_sea_mask/          # 海陆掩码
│   │   ├── soil_type/              # 土壤类型
│   │   └── topography/             # 地形数据
│   │
│   └── data/                       # 时序气象主数据（NetCDF 格式）
│       ├── {year}{timestamp}.nc    # 主数据，按时间组织的数据文件（如：2020010100.nc）
│       └── ...

```

### 文件命名规则

- 数据文件: `[年][时间戳].nc`
  - 示例: `20260409000297.nc`

### NC 文件内部结构

```
<xarray.Dataset>
Size: 2GB
Dimensions:  (time: 32, lat: xx, lon: xx)
Coordinates:
  * time     (time) datetime64[ns] 256B 2026-02-10T06:00:00 ... 2026-02-18
  * lat      (lat) float32 7kB -89.95 -89.85 -89.75 -89.65 ... 89.75 89.85 89.95
  * lon      (lon) float32 14kB 0.05 0.15 0.25 0.35 ... 359.6 359.8 359.9 360.0
Data variables:
    base_reflectivity  (time, lat, lon) float32
    bdsf_ave           (time, lat, lon) float32
    cape               (time, lat, lon) float32
    cldh               (time, lat, lon) float32
    cldl               (time, lat, lon) float32
Attributes:
    title:        TJ1-GB
    institution:  TianJi Weather Science and Technology Company
    source:       the Super Dynamics on Cube, Tianji Weather System
    references:   https://www.tjweather.com
    license:      CC BY-NC 4.0
```

- **单个数据文件**:
  - 维度: `[T, H, W]`（每变量），堆叠后为 `[C, T, H, W]`
    - C: 变量数量
    - T: 时间步数
    - H: 纬度方向像素数
    - W: 经度方向像素数
  - 变量顺序: 严格按照 `used_variables` 列表顺序

---

## 支持的变量

### 示例变量（以 TJ1-GB 为例）

| 变量名 | 描述 | 单位 | 备注 |
|--------|------|------|------|
| base_reflectivity | 基本反照率 | dBZ | 地面层 |
| bdsf_ave | 直接辐射 | W/m² | 地面层 |
| cape | 对流有效位能 | J/kg | 地面层 |
| cldh | 高云量 | % | 地面层 |
| cldl | 低云量 | % | 地面层 |
| cldm | 中云量 | % | 地面层 |
| cldt | 总云量 | % | 地面层 |
| dpt2m | 2米露点温度 | K | 地面层 |
| DSWRFsfc | 地表向下短波通量 | W/m² | 地面层 |
| gust | 阵风 | m/s | 地面层 |
| max_reflectivity | 最大反照率 | dBZ | 地面层 |
| PRATEsfc | 地表总降水率 | kg/(m²·s) | 地面层 |
| preg | 霰降水量（累积值） | kg/m² | 地面层 |
| prei | 降冰量（累积量） | kg/m² | 地面层 |
| prer | 降雨量（累积值） | kg/m² | 地面层 |
| pres | 降雪量（累积值） | kg/m² | 地面层 |
| psz | 订正后地表气压 | Pa | 地面层 |
| qnh | 修正海平面气压 | Pa | 地面层 |
| rh2m | 2米相对湿度 | % | 地面层 |
| ri_min | 理查德森数 | - | 地面层 |
| slp | 海平面气压 | Pa | 地面层 |
| SPFH2m | 2米比湿 | kg/kg | 地面层 |
| t2m_1km | 2米气温（分辨率为1千米） | K | 地面层 |
| t2mz | 2米气温 | K | 地面层 |
| TMPsfc | 地表温度 | K | 地面层 |
| u100m | 100米纬向风 | m/s | 地面层 |
| u110m | 110米纬向风 | m/s | 地面层 |
| u120m | 120米纬向风 | m/s | 地面层 |
| u130m | 130米纬向风 | m/s | 地面层 |
| u140m | 140米纬向风 | m/s | 地面层 |
| u150m | 150米纬向风 | m/s | 地面层 |
| u160m | 160米纬向风 | m/s | 地面层 |
| u170m | 170米纬向风 | m/s | 地面层 |
| u30m | 30米纬向风 | m/s | 地面层 |
| u50m | 50米纬向风 | m/s | 地面层 |
| u60m | 60米纬向风 | m/s | 地面层 |
| u70m | 70米纬向风 | m/s | 地面层 |
| u80m | 80米纬向风 | m/s | 地面层 |
| u90m | 90米纬向风 | m/s | 地面层 |
| UGRD10m | 10米纬向风 | m/s | 地面层 |
| v100m | 100米经向风 | m/s | 地面层 |
| v110m | 110米经向风 | m/s | 地面层 |
| v120m | 120米经向风 | m/s | 地面层 |
| v130m | 130米经向风 | m/s | 地面层 |
| v140m | 140米经向风 | m/s | 地面层 |
| v150m | 150米经向风 | m/s | 地面层 |
| v160m | 160米经向风 | m/s | 地面层 |
| v170m | 170米经向风 | m/s | 地面层 |
| v30m | 30米经向风 | m/s | 地面层 |
| v50m | 50米经向风 | m/s | 地面层 |
| v60m | 60米经向风 | m/s | 地面层 |
| v70m | 70米经向风 | m/s | 地面层 |
| v80m | 80米经向风 | m/s | 地面层 |
| v90m | 90米经向风 | m/s | 地面层 |
| VGRD10m | 10米经向风 | m/s | 地面层 |


---

## 路径与文件解析规则

从用户输入中智能识别并提取以下路径，未提供则默认 `None`，严禁硬编码假想路径：

- `data_path`（必填）：主数据目录或具体 `.nc` 文件路径，未提供则抛出 `ValueError`。
- `mean_path` / `std_path`（可选）：归一化统计文件路径，未提供则用 Welford 算法动态计算。
- 静态场文件（可选）：`geopotential`、`land_mask`、`land_sea_mask`、`soil_type`、`topography`，仅在涉及掩膜或地形分析时提取。

---

## 数据读取方法

### OneScience库读取（推荐）

```python
from onescience.datapipes.weather import TJDatapipe

datapipe = TJDatapipe(
    path="$ONESCIENCE_DATASETS_DIR/TJWeather/data/",
    used_variables=["dpt2m", "UGRD10m", "VGRD10m", "t2mz", "slp", "rh2m"],
    start_time="2026-01-01T00:00:00",
    end_time="2026-03-01T00:00:00",
    batch_size=32,
    input_steps=1,
    output_steps=1,
    normalize=True,
    means=means_array,
    stds=stds_array,
    distributed=False,
    num_workers=4,
)

train_loader, train_sampler = datapipe.get_dataloader(mode="train")

for invar, outvar in train_loader:
    # invar: [B, C, H, W], outvar: [B, C, H, W]
    pass
```

### 自定义读取方法

```python
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def _to_datetime64ns(x) -> np.ndarray:
    return np.asarray(x).astype("datetime64[ns]")


class NCForecastMetadata:
    """扫描路径下所有 .nc 文件，记录每个文件的 time/lat/lon/变量名。"""

    def __init__(self, path: str, time_name: str = "time", lat_name: str = "lat", lon_name: str = "lon"):
        self.path = os.path.abspath(path)
        self.time_name = time_name
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.files: Dict[str, Dict[str, Any]] = {}
        self.all_times: Optional[np.ndarray] = None
        self._build()

    def _build(self):
        nc_paths = sorted(os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(".nc"))
        if not nc_paths:
            raise FileNotFoundError(f"路径中无 .nc 文件: {self.path}")
        all_times_list = []
        for p in nc_paths:
            name = os.path.basename(p)
            ds = xr.open_dataset(p)
            for c in (self.time_name, self.lat_name, self.lon_name):
                if c not in ds:
                    ds.close()
                    raise KeyError(f"{name} 缺少坐标: {c}")
            t = np.sort(_to_datetime64ns(ds[self.time_name].values))
            self.files[name] = {
                "path": p,
                "time": t,
                "lat": np.asarray(ds[self.lat_name].values),
                "lon": np.asarray(ds[self.lon_name].values),
                "var_names": list(ds.data_vars.keys()),
            }
            all_times_list.extend(t.tolist())
            ds.close()
        self.all_times = np.sort(np.unique(np.array(all_times_list, dtype="datetime64[ns]")))

    def available_files(self) -> List[str]:
        return sorted(self.files.keys())

    def get_file_info(self, filename: str) -> Dict[str, Any]:
        if filename not in self.files:
            raise KeyError(f"未知文件: {filename}，可选: {self.available_files()}")
        return self.files[filename]


class TJDataset(Dataset):
    """按时间段在所有 .nc 文件中选样本窗口，返回 (invar, outvar)。"""

    def __init__(
        self,
        metadata: NCForecastMetadata,
        used_variables: List[str],
        start_time: str,
        end_time: str,
        input_steps: int = 1,
        output_steps: int = 1,
        normalize: bool = False,
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None,
    ):
        self.meta = metadata
        self.used_variables = used_variables
        self.start_time = np.datetime64(start_time).astype("datetime64[ns]")
        self.end_time = np.datetime64(end_time).astype("datetime64[ns]")
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.normalize = normalize
        self.means = means
        self.stds = stds
        self.total_steps = input_steps + output_steps
        self.samples: List[Tuple[str, int]] = []

        for fname, info in self.meta.files.items():
            missing = [v for v in self.used_variables if v not in info["var_names"]]
            if missing:
                raise ValueError(f"{fname} 缺少变量 {missing}，可用: {info['var_names']}")
            times = info["time"]
            T = len(times)
            if T < self.total_steps:
                continue
            for t0 in range(T - self.total_steps + 1):
                if times[t0] >= self.start_time and times[t0 + self.total_steps - 1] <= self.end_time:
                    self.samples.append((fname, t0))

        if not self.samples:
            raise ValueError("在给定时间段内，没有可用的样本窗口。")
        if self.normalize and (self.means is None or self.stds is None):
            raise ValueError("normalize=True 时必须提供 means 和 stds。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, t0 = self.samples[idx]
        info = self.meta.get_file_info(fname)
        ds = xr.open_dataset(info["path"])
        arrs = []
        for v in self.used_variables:
            a = ds[v].isel({self.meta.time_name: slice(t0, t0 + self.total_steps)}).values.astype(np.float32)
            if a.ndim != 3:
                ds.close()
                raise ValueError(f"{fname} 中变量 {v} 维度不是 (time,lat,lon)，shape={a.shape}")
            arrs.append(a)
        ds.close()
        data = np.stack(arrs, axis=1)  # [T, C, H, W]
        invar = torch.as_tensor(data[: self.input_steps])
        outvar = torch.as_tensor(data[self.input_steps:])
        if self.normalize:
            invar = (invar - self.means) / self.stds
            outvar = (outvar - self.means) / self.stds
        return invar.squeeze(0), outvar.squeeze(0)


class TJDatapipe:
    """中科天机数据端到端加载 Pipeline，接口与 ERA5Datapipe 保持一致。"""

    def __init__(
        self,
        path: str,
        used_variables: List[str],
        start_time: str,
        end_time: str,
        batch_size: int,
        input_steps: int = 1,
        output_steps: int = 1,
        normalize: bool = False,
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None,
        distributed: bool = False,
        num_workers: int = 1,
    ):
        self.path = path
        self.used_variables = used_variables
        self.start_time = start_time
        self.end_time = end_time
        self.batch_size = batch_size
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.normalize = normalize
        self.means = means
        self.stds = stds
        self.distributed = distributed
        self.num_workers = num_workers

    def get_dataloader(self, mode: str = "train"):
        meta = NCForecastMetadata(self.path)
        dataset = TJDataset(
            metadata=meta,
            used_variables=self.used_variables,
            start_time=self.start_time,
            end_time=self.end_time,
            input_steps=self.input_steps,
            output_steps=self.output_steps,
            normalize=self.normalize,
            means=self.means,
            stds=self.stds,
        )
        is_train = mode == "train"
        sampler = DistributedSampler(dataset, shuffle=is_train) if self.distributed else None
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=(is_train and not self.distributed),
            sampler=sampler,
            drop_last=self.distributed,
        )
        return loader, sampler
```

---

## 配置文件示例

```yaml
# conf/config.yaml
model:
  name: "TJForecast"

datapipe:
  dataset:
    type: "netcdf"
    data_dir: "$ONESCIENCE_DATASETS_DIR/TJWeather/data/"
    stats_dir: "$ONESCIENCE_DATASETS_DIR/TJWeather/stats/"
    static_dir: "$ONESCIENCE_DATASETS_DIR/TJWeather/static/"

    train_years: [2020, 2021, 2022, 2023]
    val_years: [2024]
    test_years: [2025]

    input_steps: 1
    output_steps: 1

    channels: ["dpt2m", "UGRD10m", "VGRD10m", "t2mz", "slp", "rh2m", "TMPsfc", "SPFH2m"]

  dataloader:
    batch_size: 32
    num_workers: 8
    shuffle: true
    pin_memory: true
```

---

## 归一化策略

**优先级：**
1. 用户提供统计文件 → 直接加载并应用 `(x - mean) / std`。
2. 未提供 → 使用 Chan's 并行 Welford 算法动态计算。

```python
from pathlib import Path
import xarray as xr
import numpy as np


def merge_stats(count_a, mean_a, m2_a, count_b, mean_b, m2_b):
    count = count_a + count_b
    if count == 0:
        return 0, 0.0, 0.0
    delta = mean_b - mean_a
    mean = mean_a + delta * (count_b / count)
    m2 = m2_a + m2_b + delta ** 2 * (count_a * count_b / count)
    return count, mean, m2


def compute_stats_for_var(nc_files: list, var_name: str):
    g_count, g_mean, g_m2 = np.int64(0), np.float64(0.0), np.float64(0.0)
    for fpath in nc_files:
        ds = xr.open_dataset(fpath, engine="netcdf4")
        if var_name not in ds:
            ds.close()
            continue
        data = ds[var_name].values.astype(np.float64)
        ds.close()
        count_b = np.int64((~np.isnan(data)).sum())
        if count_b == 0:
            continue
        mean_b = np.nanmean(data)
        m2_b = np.nanvar(data) * count_b
        g_count, g_mean, g_m2 = merge_stats(g_count, g_mean, g_m2, count_b, mean_b, m2_b)
    if g_count == 0:
        return 0.0, 1.0
    return float(g_mean), float(np.sqrt(g_m2 / g_count))


def compute_means_and_stds(data_path: str):
    nc_files = [str(f) for f in sorted(Path(data_path).glob("*.nc"))]
    if not nc_files:
        raise FileNotFoundError(f"No NC files found under {data_path}")
    with xr.open_dataset(nc_files[0], engine="netcdf4") as ds0:
        var_names = list(ds0.data_vars)
        global_attrs = ds0.attrs
    means_dict, stds_dict = {}, {}
    for var in var_names:
        means_dict[var], stds_dict[var] = compute_stats_for_var(nc_files, var)

    def build_dataset(values_dict, description):
        data_vars = {
            var: xr.DataArray(
                np.array([val], dtype=np.float32),
                dims=["n"],
                attrs={"description": description, "variable": var},
            )
            for var, val in values_dict.items()
        }
        ds = xr.Dataset(data_vars, attrs=global_attrs)
        ds.attrs["normalization_description"] = description
        return ds

    return (
        build_dataset(means_dict, "global mean over all valid grid points and time steps"),
        build_dataset(stds_dict, "global population std over all valid grid points and time steps"),
    )
```

---

## 掩膜处理规范

```python
import numpy as np

# mask 为数值型时转布尔型（> 0.5 为陆地），布尔型直接使用
if np.issubdtype(mask.dtype, np.number):
    mask_bool = mask > 0.5
else:
    mask_bool = mask.astype(bool)

if data.shape[-2:] != mask_bool.shape:
    raise ValueError(f"维度不匹配：数据 {data.shape[-2:]} vs 掩码 {mask_bool.shape}")

data_masked = np.where(mask_bool, data, np.nan)
```

---

## 数据处理流程

1. **数据获取**: 从中科天机平台获取 `.nc` 格式预报数据
2. **元数据扫描**: 使用 `NCForecastMetadata` 扫描目录，建立文件索引
3. **数据读取**: 使用 `xarray.open_dataset` 按需加载（`isel` 切片避免 OOM）
4. **数据预处理**:
   - 变量筛选（`used_variables`）
   - 时间窗切片（`input_steps` + `output_steps`）
   - 数据标准化（提供统计文件或动态 Welford 计算）
   - 掩膜处理（可选，需静态场文件）
5. **数据加载**: 通过 `TJDatapipe.get_dataloader()` 批量加载
6. **模型训练/推理**: 输入模型进行训练或推理

---

## 模型接入接口

### 输入规范

- **形状**: `[B, C, H, W]`（input_steps=1）或 `[B, T, C, H, W]`（input_steps>1）
  - B: 批量大小，T: 时间步数，C: 通道/变量数，H: 纬度，W: 经度
- **数据类型**: `torch.float32`
- **数值范围**: 归一化后为标准正态分布（若 normalize=True）

### 输出规范

- **形状**: 与输入相同
- **数据类型**: `torch.float32`

---

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

---

## 依赖库

- **核心库**: `numpy`, `torch`, `xarray`, `netcdf4`
- **OneScience库**: `onescience.datapipes.weather`, `onescience.utils.YParams`

---

## 代码结构要求

- 函数化：每个独立功能（读取、统计、变换）封装为函数
- 参数传递：禁止在函数内部硬编码路径或变量名
- 类封装：端到端流程必须复用 `NCForecastMetadata` + `TJDataset` + `TJDatapipe` 结构
- 导入规范：所有用到的库（含 `typing`、`torch` 等）必须在文件头部显式导入
- `NCForecastMetadata`、`compute_stats_for_var` 等核心组件必须在代码中完整实现，禁止外部导入

---

## 注意事项

1. **大文件处理**: 优先使用 `isel` 切片，避免 OOM；`xarray.Dataset` 转 `numpy` 后坐标元数据丢失，需提前缓存 `time`、`lat`、`lon`
2. **训练/验证/测试集划分**: 采用按需加载机制，各集合独立调用读取函数并传入对应时间区间，禁止先全量读取再切分
3. **时间格式**: 时间字段统一转换为 `datetime64[ns]`，格式为 `'yyyy-mm-ddThh:mm:ss'`
4. **路径规范**: `xarray.open_dataset` 输入必须是具体文件路径，不能是目录

---

## 故障排查

### 常见问题

1. **文件路径错误**
   - 症状: `FileNotFoundError: 路径中无 .nc 文件`
   - 解决方案: 确认 `data_path` 为包含 `.nc` 文件的目录，而非文件本身

2. **变量不存在**
   - 症状: `ValueError: xxx 缺少变量`
   - 解决方案: 检查 `used_variables` 是否与文件中实际变量名一致

3. **内存不足**
   - 症状: OOM 错误
   - 解决方案: 使用 `isel` 切片按需加载，减小 `batch_size` 或 `num_workers`

4. **坐标名不匹配**
   - 症状: `KeyError: xxx 缺少坐标`
   - 解决方案: 通过 `NCForecastMetadata` 的 `time_name`/`lat_name`/`lon_name` 参数自定义坐标名

---

## 参考资源

- [中科天机官方平台](https://www.tjweather.com)
- [xarray 文档](https://docs.xarray.dev)
- [PyTorch DataLoader 文档](https://pytorch.org/docs/stable/data.html)

---

## 更新日志

- **v1.0** (2026-04-14): 初始版本，支持 TJ1-CN / TJ-CN / TJ1-GB / TJ-GB 四类数据集
