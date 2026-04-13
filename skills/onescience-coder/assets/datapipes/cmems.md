# Datapipe: CMEMSDatapipe

## 基本信息

- Datapipe 名：`CMEMSDatapipe`
- 数据类型：`climate`
- 主要任务：`global ocean forecast`
- 数据组织方式：`per_timestamp_h5_under_year_dir`

## Datapipe 职责

`CMEMSDatapipe` 负责把按年份目录、按时刻存储的 CMEMS 海洋场数据切成连续时间窗样本，并组织成适合模型训练或推理的 `DataLoader`。

补充说明：

- 原始数据以“按年份分目录、目录下按时刻存 HDF5 文件”的方式组织
- datapipe 负责年份划分、变量筛选、时间窗切片、NaN 处理、归一化、太阳天顶角构造和 `DataLoader` 组装
- train / val / test 划分在 datapipe 内部通过配置完成

## 输入配置

- `data_dir`
  - CMEMS 数据根目录
- `channels`
  - 当前任务需要读取的变量名列表
- `train_ratio`
  - 训练阶段年份划分配置
- `val_ratio`
  - 验证阶段年份划分配置
- `test_ratio`
  - 测试阶段年份划分配置
- `time_res`
  - 时间分辨率，单位小时
- `input_steps`
  - 输入时间步数
- `output_steps`
  - 预测时间步数
- `img_size`
  - 目标空间尺寸，用于裁剪对齐

## 数据存储约定

- 主数据路径：`<data_dir>/h5/<year>/*.h5`
- 统计量路径：`<stats_dir>/global_means.npy` 与 `global_stds.npy`
- 元数据来源：`<data_dir>/metadata.json`

额外约定：

- `metadata.json` 中必须给出 `years` 与 `variables`
- 每个时刻文件中的 `fields` 形状为 `[Channels, Height, Width]`

## 样本构造方式

- 输入样本：`[input_steps, Channels, Height, Width]`
- 输出样本：`[output_steps, Channels, Height, Width]`
- 附加返回项：`cos_zenith`, `step_idx`, `time_index`

具体做法：

- 先从 `metadata.json` 读取可用年份和变量列表
- 再根据 `train_ratio`、`val_ratio`、`test_ratio` 计算当前模式下选用的年份
- 对每个年份目录下的时刻文件做排序，并按连续文件窗口切片
- 根据 `channels` 建立变量索引，只保留当前任务需要的通道
- 先把 NaN 替换为对应通道均值，再使用统计量做标准化
- 同时基于经纬度网格与时间戳构造太阳天顶角

## DataLoader 约定

- 训练阶段：若启用分布式训练，使用 `DistributedSampler(shuffle=True)`
- 验证阶段：不打乱，分布式时使用 `DistributedSampler(shuffle=False)`
- 测试阶段：不打乱，`batch_size` 与 `pin_memory` 由外部 dataloader 配置控制

## 适合优先使用的场景

- 使用 CMEMS 海洋数据训练全球海洋预测模型
- 原始数据已经按年份目录和时刻文件整理完成
- 需要在 datapipe 内部直接完成 train / val / test 年份划分

## 风险点

- 年份划分当前仍依赖 `train_ratio`、`val_ratio`、`test_ratio`，而不是显式年份列表
- 代码默认假设每个选中年份包含相同数量的可用样本，若不同年份文件数不同，索引逻辑需要额外确认
- 数据目录强依赖 `h5/<year>/*.h5`、`metadata.json` 与 `stats/` 结构
- 时间索引直接从文件名尾部切片得到，文件命名规则变化时需要同步修改

## 源码锚点

- `./onescience/src/onescience/datapipes/climate/cmems.py`
- `./onescience/src/onescience/datapipes/climate/utils/invariant.py`
- `./onescience/src/onescience/datapipes/climate/utils/zenith_angle.py`
