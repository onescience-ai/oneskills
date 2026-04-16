# Datapipe: TJWeather

## 基本信息

- Datapipe 名：`TJWeather`
- 数据集名称: 中科天机数据集
- 数据格式: NetCDF (.nc)
- 空间分辨率: 0.025°~0.1°（视子集而定）
- 时间分辨率: 1小时 / 6小时（视子集而定）
- 数据类型：`climate`
- 主要任务：`forecast`
- 数据组织方式：`per_timestamp_nc`

## Datapipe 职责

将按年份目录存储的逐时间步 NetCDF 文件，组织成多步输入 / 多步输出的气象预报训练样本。

- 数据目录结构：`<data_dir>/data/<year><timestamp>.nc`，每个 `.nc` 文件对应一个时间步
- 负责变量筛选、归一化（z-score）、空间裁剪（patch 对齐）、cos 太阳天顶角生成
- 同时负责 `Dataset`（`TJDataset`）与 `DataLoader`（`TJDatapipe`）

## 数据根目录

onescience 平台数据根目录通过环境变量获取，天机数据根目录及完整路径构造如下：

```python
import os
# 天机数据根目录
data_root = os.path.join(os.environ.get("ONESCIENCE_DATASETS_DIR"), "TJWeather")
# dataset_type 须从用户输入解析，取值：TJ1-CN | TJ-CN | TJ1-GB | TJ-GB
data_dir = os.path.join(os.environ.get("ONESCIENCE_DATASETS_DIR"), "TJWeather", dataset_type)
```

## 子数据集说明

| 数据集  | 覆盖范围                            | 分辨率            | 时间分辨率  | 预报时效         |
|---------|-------------------------------------|-------------------|-------------|------------------|
| TJ1-CN  | 中国区域（65°E~145°E，5°N~65°N）   | 0.033°（约3km）   | 1小时       | 未来15天，逐小时 |
| TJ-CN   | 中国区域（65°E~145°E，5°N~65°N）   | 0.025°（约2.5km） | 1小时       | 未来10天，逐小时 |
| TJ1-GB  | 全球（0°~360°E，-90°~90°N）        | 0.1°（约12km）    | 1小时       | 未来15天，逐小时 |
| TJ-GB   | 全球（0°~360°E，-90°~90°N）        | 0.1°（约12km）    | 1小时/6小时 | 未来46天         |

## 标准目录结构

```
${ONESCIENCE_DATASETS_DIR}/TJWeather/
└── {dataset_type}/
    ├── stats/
    │   ├── global_means.nc
    │   └── global_stds.nc
    ├── static/
    │   ├── geopotential/
    │   ├── land_mask/
    │   ├── land_sea_mask/
    │   ├── soil_type/
    │   └── topography/
    └── data/
        ├── {year}{timestamp}.nc    # 示例：20260409000297.nc
        └── ...
```

## NC 文件内部结构

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


## 输入配置

| 参数 | 说明 |
|------|------|
| `params.dataset.data_dir` | 数据根目录，含 `data/<year><timestamp>.nc` 子路径 |
| `params.dataset.stats_dir` | 统计量目录，含 `global_means.npy` / `global_stds.npy` |
| `params.dataset.channels` | 需要使用的变量名列表 |
| `params.dataset.train_years` / `val_years` / `test_years` | 各阶段年份列表（必须显式指定） |
| `params.dataset.time_res` | 时间分辨率（小时） |
| `params.dataset.img_size` | 空间尺寸，用于生成 latlon 网格 |
| `input_steps` | 输入时间步数，默认 1 |
| `output_steps` | 输出时间步数，默认 1 |
| `normalize` | 是否 z-score 归一化，默认 `True` |
| `params.dataloader.batch_size` | 训练/验证 batch size |
| `params.dataloader.num_workers` | DataLoader 工作进程数 |

## 数据存储约定

- 主数据：`<data_dir>/data/<year><timestamp>.nc`
- 统计量：`<stats_dir>/global_means.npy`、`<stats_dir>/global_stds.npy`
- 每个 `.nc` 文件统计量形状为 `(1, num_all_channels, H, W)`，按 `channel_indices` 切片后使用

## 支持的变量（以 TJ1-GB 为例）

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

## 样本构造方式

- 输入：`(input_steps, C, H, W)` — 连续 `input_steps` 个时间步，归一化后裁剪至 patch 对齐尺寸
- 输出：`(output_steps, C, H, W)` — 紧随其后的 `output_steps` 个时间步
- 附加返回：`cos_zenith (output_steps, H, W)`、`step_idx (int)`、`time_index (List[str])`
- 时间步切片：以全局 `idx` 通过 `bisect` 定位年份和年内偏移
- 空间裁剪：裁剪至 `patch_size` 整数倍，默认 `[1,1]` 即不裁剪
- 附加返回项：`cos_zenith (output_steps, H, W)`、`step_idx (int)`、`time_index (List[str])`

## DataLoader 约定

- 训练：`batch_size` 来自 `params.dataloader`，分布式用 `DistributedSampler(shuffle=True)`
- 验证：同训练，但 `shuffle=False`
- 测试：`batch_size=1`，`shuffle=False`，不用 `DistributedSampler`

## 风险点

- `train_years`/`val_years`/`test_years` 必须显式配置，不支持 ratio 方式
- 各年份文件数可能不同（闰年），`samples_per_year` 可能为 `None`
- 统计量切片依赖 `channel_indices` 与 `metadata.json` 顺序一致，变量顺序变更会导致归一化错误
- `__getitem__` 逐文件打开 `.nc`，高 `num_workers` 下 I/O 压力大

## 常见问题

| 症状 | 解决方案 |
|------|----------|
| `FileNotFoundError: 路径中无 .nc 文件` | 确认 `data_path` 为含 `.nc` 文件的目录 |
| `ValueError: xxx 缺少变量` | 检查 `used_variables` 与文件实际变量名一致 |
| OOM 错误 | 用 `isel` 按需加载，减小 `batch_size` 或 `num_workers` |
| `KeyError: xxx 缺少坐标` | 通过 `time_name`/`lat_name`/`lon_name` 自定义坐标名 |

## 源码锚点

- `./onescience/src/onescience/datapipes/climate/tjweather.py`
- `./onescience/src/onescience/datapipes/climate/utils/zenith_angle.py`
- `./onescience/src/onescience/datapipes/climate/utils/invariant.py`
- `./onescience/src/onescience/datapipes/core.py`

## 输出要求

- 输出符合用户需求的代码，包括 Dataset 类和 DataLoader 类
- 禁止生成 `get_dataloader` 或 `get_*_dataloader` 函数

## 参考资源

- [中科天机官方平台](https://www.tjweather.com)
- [xarray 文档](https://docs.xarray.dev)
- [PyTorch DataLoader 文档](https://pytorch.org/docs/stable/data.html)
