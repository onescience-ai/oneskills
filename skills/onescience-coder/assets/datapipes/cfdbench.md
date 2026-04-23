# Datapipe: CFDBenchDatapipe

## 基本信息

- Datapipe 名称：`CFDBenchDatapipe`
- 数据类型：`cfd`
- 主要任务：`steady_or_autoregressive benchmark regression`
- 数据组织方式：`case_directories_with_json_and_npy`

## Datapipe 职责

`CFDBenchDatapipe` 负责把 `tube/cavity/cylinder/dam` 四类基准数据从 case 目录读入，完成按 case 的 train/val/test 划分，并根据 `task_type` 组织成静态监督样本或自回归样本。

补充说明：

- 原始数据是 case 目录，不是单个大文件。
- datapipe 内部包含问题类型特定的边界填充、障碍物 mask 构建和物理参数归一化逻辑。
- datapipe 统一输出普通 `dict` batch，而不是图对象。

## 输入配置

- `source.data_dir`
  - CFDBench 根目录。
- `source.data_name`
  - 形如 `tube_prop`、`cavity_bc`、`cylinder_geo`、`dam_prop_bc` 的数据子集名，datapipe 会从前缀推断问题类型。
- `data.task_type`
  - `auto` 表示自回归一步预测；其它值走静态模式。
- `data.split_ratios`
  - train/val/test 的 case 划分比例。
- `data.seed`
  - case 目录打乱时的随机种子。
- `data.norm_props`
  - 是否对物性参数做归一化。
- `data.norm_bc`
  - 是否对边界条件做归一化。
- `data.delta_time`
  - `task_type=auto` 时使用的预测时间间隔。
- `dataloader.batch_size`
  - 训练集 batch 大小。
- `dataloader.eval_batch_size`
  - 验证集 batch 大小。
- `dataloader.num_workers`
  - DataLoader 工作进程数。

## 数据存储约定

- 主数据路径：`<data_dir>/<problem>/<subset>/case*/`
- case 元数据：`case.json`
- case 场数据：`u.npy`、`v.npy`
- 元数据来源：`case.json` 中的边界条件、几何尺寸和物性参数

额外约定：

- `problem` 只能是 `tube`、`cavity`、`cylinder`、`dam` 四类之一。
- `subset` 通过 `source.data_name` 中是否包含 `prop`、`bc`、`geo` 来选择。
- `cylinder` 和 `dam` 会额外根据 `case.json` 构造障碍物或坝体区域的 mask。

## 样本构造方式

- 静态模式输入样本：`{"case_params", "t"}`
- 静态模式输出样本：`{"label"}`
- 自回归模式输入样本：`{"inputs", "case_params"}`
- 自回归模式输出样本：`{"label", "mask"}`

具体说明：

- 原始特征统一以 `[u, v, mask]` 三通道组织。
- 静态模式按 frame 展开样本，`label` 是单个时间帧的三通道张量，`t` 是对应帧索引。
- 自回归模式先按 `delta_time` 取输入帧与目标帧，再在 `collate_fn` 里把最后一个通道拆为 `mask`，因此 batch 后的 `inputs/label` 实际上只保留速度通道。
- `case_params` 会在 batch 阶段整理为定长张量，但会忽略 `rotated/dx/dy` 这类键。

## DataLoader 约定

- 训练阶段：`train_dataloader()` 返回 `(DataLoader, sampler)`，使用与 `task_type` 对应的 `collate_fn`
- 验证阶段：`val_dataloader()` 返回 `(DataLoader, sampler)`
- 测试阶段：`test_dataloader()` 只返回 `DataLoader`，固定 `batch_size=1`

## 适合优先使用的场景

- 想复用 `CFD_Benchmark` 目录下的 FNO / U-FNO / UNO / U-Net 等示例。
- 新数据仍然是规则网格上的二维稳态或准时序 CFD 基准。
- 需要把 case 级物理参数和场张量一起送入模型。

## 风险点

- `task_type=auto` 与静态模式的 batch 协议完全不同，训练脚本不能直接混用。
- 当前实现只读取 `u.npy` 和 `v.npy`，没有额外压力通道。
- train/val/test 划分是运行时基于随机种子从 case 目录切出来的，不是数据集自带固定 split。
- `delta_time` 会被离散成 `time_step_size=int(delta_time / data_dt)`，如果不整除会产生隐式近似。

## 源码锚点

- `./onescience/src/onescience/datapipes/cfd/cfdbench.py`
- `./onescience/examples/cfd/CFD_Benchmark/train.py`
- `./onescience/examples/cfd/CFD_Benchmark/conf/`
