# OneScience 数据处理技能

## 概述

OneScience 数据处理技能采用**数据卡驱动架构**，提供标准化的数据集处理流程，支持多种数据集格式，并实现与 OneScience 生态的无缝集成。

## 核心特性

- 📋 **数据卡驱动**: 通过数据卡定义数据集元信息，自动生成处理代码
- 🔧 **代码复用**: 深度集成 OneScience Datapipe，复用现有数据管道
- 🎯 **模型适配**: 提供模型适配器，支持数据集快速接入不同AI模型
- 📝 **灵活扩展**: 支持自定义数据集和模型适配器

## 文件结构

```
onescience-data-processing/
├── SKILL.md                      # 技能主文档
├── README.md                     # 本文件
└── assets/
    ├── ERA5.md                   # ERA5数据集数据卡 (示例)
    ├── DATACARD_TEMPLATE.md      # 数据卡模板
    └── MODEL_ADAPTER_GUIDE.md    # 模型适配器指南
```

## 快速开始

### 1. 使用现有数据卡

```python
# 解析ERA5数据卡
from onescience.data_processing import DataCardParser, CodeGenerator

parser = DataCardParser()
config = parser.parse('assets/ERA5.md')

# 生成数据处理代码
generator = CodeGenerator(config)
generator.generate_all(output_dir='./era5_pipeline/')
```

### 2. 接入AI模型

```python
from onescience.data_processing import ModelAdapterFactory

# 创建Pangu模型适配器
adapter = ModelAdapterFactory.create_adapter(
    model_type='pangu',
    dataset_config=config,
    forecast_hours=24
)

# 获取数据加载器
train_loader = adapter.get_dataloader(mode='train', batch_size=32)
```

### 3. 创建自定义数据卡

参考 `assets/DATACARD_TEMPLATE.md` 创建新的数据卡文件。

## 工作流程

```
┌─────────────────┐
│   数据卡文件     │
│  (ERA5.md等)    │
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  数据卡解析器     │
│  (DataCardParser)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐
│  代码生成器       │────▶│ OneScience代码   │
│ (CodeGenerator)  │     │ (Datapipe等)     │
└────────┬─────────┘     └──────────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌──────────────────┐              ┌──────────────────┐
│  自定义Dataset   │              │  模型适配器       │
│    代码          │              │ (ModelAdapter)   │
└──────────────────┘              └────────┬─────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │   AI模型接入      │
                                  │ (Pangu/FuXi等)   │
                                  └──────────────────┘
```

## 支持的数据集

| 数据集 | 数据卡 | 格式 | 用途 |
|--------|--------|------|------|
| ERA5 | ✅ | HDF5/NetCDF | 气象预报 |
| CMEMS | 📝 | NetCDF | 海洋数据 |
| GFS | 📝 | GRIB/NetCDF | 天气预报 |
| DeepCFD | 📝 | HDF5 | CFD仿真 |

✅ 已完成 | 📝 待添加

## 支持的模型

| 模型 | 适配器 | 状态 |
|------|--------|------|
| Pangu-Weather | PanguAdapter | ✅ |
| FuXi | FuXiAdapter | ✅ |
| FourCastNet | FourCastNetAdapter | ✅ |
| GraphCast | GraphCastAdapter | 📝 |
| 自定义 | CustomAdapter | ✅ |

## 数据卡规范

### 必需字段

```markdown
# 数据集名称

## 数据集基本信息
- 名称、版本、来源
- 时间/空间分辨率
- 数据格式

## 数据存储结构
- 目录结构
- 文件命名规则

## 支持的变量
- 变量名、描述、单位

## 数据读取方法
- OneScience Datapipe 方式
- 自定义 Dataset 方式
```

### 完整示例

参考 `assets/ERA5.md` 获取完整的数据卡示例。

## 模型适配器使用

### 基础用法

```python
from onescience.data_processing import ModelAdapterFactory

# 列出支持的模型
models = ModelAdapterFactory.list_supported_models()
print(models)  # ['pangu', 'fuxi', 'fourcastnet', ...]

# 创建适配器
adapter = ModelAdapterFactory.create_adapter('pangu', config)

# 获取输入规范
spec = adapter.get_input_spec()
print(f"形状: {spec.shape}")
print(f"变量: {spec.variables}")
```

### 自定义适配器

```python
# 定义预处理函数
def custom_preprocess(data):
    return (data - data.mean()) / data.std()

# 创建自定义适配器
adapter = ModelAdapterFactory.create_adapter(
    model_type='custom',
    dataset_config=config,
    variables=['temperature_500', 'geopotential_500'],
    custom_preprocess=custom_preprocess
)
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

## 扩展开发

### 添加新的数据卡

1. 复制 `assets/DATACARD_TEMPLATE.md`
2. 填写数据集特定信息
3. 保存到 `assets/` 目录

### 添加新的模型适配器

1. 继承 `BaseModelAdapter`
2. 实现必需的方法
3. 注册到 `ModelAdapterFactory`

```python
class MyAdapter(BaseModelAdapter):
    def get_input_spec(self):
        # 实现
        pass

ModelAdapterFactory.register_adapter('mymodel', MyAdapter)
```

## 故障排查

### 常见问题

1. **数据卡解析失败**
   - 检查格式是否符合规范
   - 验证必要字段是否完整

2. **代码生成错误**
   - 检查数据集配置
   - 验证输出目录权限

3. **模型接入失败**
   - 确认模型类型支持
   - 检查输入输出形状

## 参考文档

- [SKILL.md](SKILL.md) - 技能详细文档
- [assets/ERA5.md](assets/ERA5.md) - ERA5数据卡示例
- [assets/DATACARD_TEMPLATE.md](assets/DATACARD_TEMPLATE.md) - 数据卡模板
- [assets/MODEL_ADAPTER_GUIDE.md](assets/MODEL_ADAPTER_GUIDE.md) - 模型适配器指南

## 更新日志

### v1.0 (2025-04-09)
- ✅ 数据卡驱动架构设计
- ✅ ERA5数据卡示例
- ✅ 模型适配器框架
- ✅ 数据卡模板
- ✅ 适配器开发指南

## 贡献

欢迎提交新的数据卡和模型适配器！

1. Fork 本仓库
2. 添加新的数据卡到 `assets/`
3. 提交 Pull Request
