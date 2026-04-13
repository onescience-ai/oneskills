# Datapipe: ERA5Datapipe

## 基本信息

- Datapipe 名：`ERA5Datapipe`
- 数据类型：`climate`
- 主要任务：`global weather forecast`
- 数据组织方式：`per_year_h5`

## Datapipe 职责

`ERA5Datapipe` 负责从 OneScience 数据卡中读取已配置的 ERA5 气象数据，并切成连续时间窗样本，组织为适合天气预测模型训练或推理的 `DataLoader`。

重要说明：

- **数据已在数据卡中配置**：使用时直接引用数据卡中的数据路径和元数据，无需手动指定数据位置
- **数据卡自动注入**：数据路径、统计量路径、变量列表等信息自动从数据卡中读取
- **数据已就绪**：数据卡中定义的数据已在 `{ONESCIENCE_DATASETS_DIR}` 下存储，无需额外生成测试数据
- datapipe 负责变量筛选、时间窗切片、归一化、太阳天顶角构造和 `DataLoader` 组装
- train / val / test 的年份划分不在 datapipe 内部完成，而是由外部通过 `used_years` 显式传入

## 输入配置

- `dataset_dir`
  - ERA5 数据根目录（**从数据卡中自动读取，无需手动指定**）
- `used_years`
  - 当前 datapipe 使用的年份列表
- `used_variables`
  - 当前任务需要读取的变量名列表（**可从数据卡元数据中获取**）
- `input_steps`
  - 输入时间步数
- `output_steps`
  - 预测时间步数
- `normalize`
  - 是否使用统计量做标准化
- `batch_size`
  - `DataLoader` 批大小
- `num_workers`
  - `DataLoader` 工作进程数

## 数据存储约定

- 主数据路径：`{ONESCIENCE_DATASETS_DIR}/ERA5/newh5/data_merged/<year>.h5`
- 统计量路径：`{ONESCIENCE_DATASETS_DIR}/ERA5/newh5/stats/global_means.npy` 与 `global_stds.npy`
- 元数据来源：`fields` 数据集的 `variables` 与 `time_step` 属性

额外约定：

- 单个年份文件中的 `fields` 形状为 `[TimeSteps, Channels, Height, Width]`
- 所选变量必须都存在于 `fields.attrs["variables"]`

## 样本构造方式

- 输入样本：`[input_steps, Channels, Height, Width]`
- 输出样本：`[output_steps, Channels, Height, Width]`
- 附加返回项：`cos_zenith`, `step_idx`, `time_index`

具体做法：

- 先按 `used_years` 建立年份到文件路径的映射
- 再从每个年份文件中按连续时间步切出长度为 `input_steps + output_steps` 的窗口
- 通过 `used_variables` 建立通道索引，仅保留当前任务需要的变量
- 若 `normalize=True`，使用 `global_means.npy` 与 `global_stds.npy` 对输入和输出做标准化
- 根据时间戳和经纬度网格计算太阳天顶角

## DataLoader 约定

- 训练阶段：若启用分布式训练，使用 `DistributedSampler(shuffle=True)`；否则由 `DataLoader` 本身随机打乱
- 验证阶段：不打乱，分布式时使用 `DistributedSampler(shuffle=False)`
- 测试阶段：不打乱，接口仍通过 `get_dataloader(mode)` 统一获取

## 适合优先使用的场景

- 使用 ERA5 年尺度 HDF5 数据训练全球天气预报模型
- 需要显式控制训练、验证、测试所使用的年份集合
- 需要把数据直接组织成 `[TimeSteps, Channels, Height, Width]` 的连续时间窗

## 风险点

- 当前 train / val / test 划分不在 datapipe 内部处理，调用方必须为不同阶段显式传入对应 `used_years`
- 数据目录强依赖 `data_merged/<year>.h5` 与 `stats/` 结构，路径不一致时会直接失败
- 变量名依赖 HDF5 属性中的 `variables`，若数据构建时未正确写入属性，会导致变量校验失败
- `time_index` 与 `cos_zenith` 的时间步基于年份起始时间和 `time_step` 计算，若文件内部时间轴与该假设不一致，需要回到源码核对

## 源码锚点

- `./onescience/src/onescience/datapipes/climate/era5.py`
- `./onescience/src/onescience/datapipes/climate/utils/invariant.py`
- `./onescience/src/onescience/datapipes/climate/utils/zenith_angle.py`
