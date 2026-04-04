# FourCastNet Global Region Hybrid - 代码生成完成

## 已生成的文件

### 1. 核心组件

#### [components/global_region_fusion.py](components/global_region_fusion.py)
- **GlobalRegionFusionBlock**: 全球特征与区域特征融合模块
  - 支持三种融合方式：`concat_mlp`、`concat_conv`、`gate`
  - 输入：区域特征 `(B, H, W, dim_region)` + 全球特征 `(B, H, W, dim_global)`
  - 输出：融合后的区域特征 `(B, H, W, dim_out)`

- **GlobalFeatureAligner**: 全球特征对齐模块
  - 支持两种对齐方式：`crop`（裁切）、`interpolate`（插值）
  - 根据区域经纬度范围对齐全球特征到区域 patch grid

#### [models/fourcastnet_global_region_hybrid.py](models/fourcastnet_global_region_hybrid.py)
- **FourCastNetGlobalRegionHybrid**: 全球+区域混合模型
  - 全球分支：预训练 FourCastNet，提取最后一层特征
  - 区域分支：训练新模型，在中间层融合全球特征
  - 支持冻结/解冻全球分支参数
  - 支持加载全球预训练权重

- **FourCastNetBase**: 基座 FourCastNet 模型
  - 用于快速训练获取 checkpoint
  - 复用 OneScience 的 `OneEmbedding` 和 `OneFuser` 组件

### 2. 工具函数

#### [utils/synthetic_data.py](utils/synthetic_data.py)
- **generate_synthetic_weather_data()**: 生成虚拟气象数据
  - 全球数据：0.25度分辨率 (720x1440)
  - 区域数据：0.01度分辨率 (1000x1000, 10度范围)
  - 支持自定义经纬度范围

- **SyntheticWeatherDataset**: 虚拟数据集类
- **create_dataloaders()**: 创建训练和验证数据加载器

#### [utils/checkpoint_utils.py](utils/checkpoint_utils.py)
- **save_checkpoint()**: 保存模型 checkpoint
- **load_checkpoint()**: 加载模型 checkpoint
- **save_checkpoint_early()**: 快速保存（训练中断时使用）
- **load_global_pretrained()**: 加载全球模型预训练权重
- **freeze_parameters() / unfreeze_parameters()**: 冻结/解冻参数
- **get_parameter_count() / print_parameter_stats()**: 参数统计

### 3. 训练和测试

#### [train_base.py](train_base.py)
- 基座模型训练脚本
- 支持 TensorBoard 日志记录
- 支持训练中断保存 checkpoint
- 命令行参数配置

#### [tests/](tests/)
拆分后的测试文件，按测试层级组织：

- **[test_data.py](tests/test_data.py)** - 数据层测试
  - `test_synthetic_data()`: 测试虚拟数据生成
  - `test_dataset_and_dataloader()`: 测试数据集和数据加载器

- **[test_components.py](tests/test_components.py)** - 组件层测试
  - `test_fusion_block()`: 测试融合模块
  - `test_fusion_block_modes()`: 测试融合模块的不同模式
  - `test_feature_aligner()`: 测试特征对齐模块
  - `test_feature_aligner_modes()`: 测试特征对齐模块的不同模式

- **[test_models.py](tests/test_models.py)** - 模型层测试
  - `test_base_model()`: 测试基座模型
  - `test_hybrid_model()`: 测试混合模型
  - `test_hybrid_model_fusion_modes()`: 测试混合模型的不同融合模式
  - `test_hybrid_model_align_modes()`: 测试混合模型的不同对齐模式
  - `test_hybrid_model_without_global_input()`: 测试混合模型在没有全球输入时的行为

- **[test_utils.py](tests/test_utils.py)** - 工具层测试
  - `test_checkpoint_save_load()`: 测试 checkpoint 保存和加载
  - `test_checkpoint_early_save()`: 测试快速保存 checkpoint
  - `test_freeze_unfreeze()`: 测试参数冻结和解冻
  - `test_model_freeze_methods()`: 测试模型的冻结和解冻方法
  - `test_parameter_count()`: 测试参数计数功能

- **[test_integration.py](tests/test_integration.py)** - 集成测试
  - `test_end_to_end()`: 端到端测试
  - `test_training_loop()`: 测试完整训练循环
  - `test_model_with_checkpoint()`: 测试模型的 checkpoint 保存和恢复
  - `test_model_with_frozen_global_branch()`: 测试冻结全球分支的模型

#### [run_all_tests.py](run_all_tests.py)
- 运行所有测试的脚本
- 按顺序执行各个测试文件
- 输出详细的测试结果汇总

## 复用的 OneScience 组件

| 组件 | 注册名 | 用途 |
|------|--------|------|
| `OneEmbedding` | `style="FourCastNetEmbedding"` | 2D patch embedding |
| `OneFuser` | `style="FourCastNetFuser"` | 主干特征融合（AFNO + MLP） |

## 主要 Shape 变化

### 全球分支
```
Input:        (Batch, 19, 720, 1440)
              ↓
Embedding:    (Batch, 16200, 768)
              ↓ reshape
Patch Grid:   (Batch, 90, 180, 768)
              ↓
Trunk x 12:   (Batch, 90, 180, 768)
              ↓ [提取最后一层特征]
Global Feat:  (Batch, 90, 180, 768)
```

### 区域分支
```
Input:              (Batch, 19, 1000, 1000)
                    ↓
Embedding:          (Batch, 15625, 768)
                    ↓ reshape
Patch Grid:         (Batch, 125, 125, 768)
                    ↓
Pre-fusion Blocks:  (Batch, 125, 125, 768)
                    ↓ [Fusion Block]
Fused Feature:      (Batch, 125, 125, 768)
                    ↓
Post-fusion Blocks: (Batch, 125, 125, 768)
                    ↓
Head:               (Batch, 125, 125, 19 * 8 * 8)
                    ↓ rearrange
Output:             (Batch, 19, 1000, 1000)
```

## 新增桥接模块说明

### 1. GlobalRegionFusionBlock
- **职责**：将全球特征与区域特征融合
- **融合方式**：
  - `concat_mlp`: 拼接后通过 MLP 融合
  - `concat_conv`: 拼接后通过 Conv 融合（推荐）
  - `gate`: 门控融合
- **位置**：插入在区域 trunk 的第 `fusion_layer_idx` 层（默认第 4 层）

### 2. GlobalFeatureAligner
- **职责**：将全球特征对齐到区域 patch grid
- **对齐方式**：
  - `crop`: 根据经纬度范围裁切全球特征
  - `interpolate`: 插值到区域分辨率
- **输入**：全球特征 `(B, 90, 180, 768)` + 区域经纬度范围
- **输出**：对齐后的全球特征 `(B, H_region, W_region, 768)`

## 使用方法

### 1. 训练基座模型
```bash
cd /Users/zhao/Desktop/OneScience/dev-earth-function+commit/oneskills/codex/case/fourcastnet_global_region_hybrid/V2
python train_base.py --num_epochs 10 --batch_size 4
```

### 2. 运行测试

#### 运行所有测试
```bash
python run_all_tests.py
```

#### 运行单个测试文件
```bash
# 数据层测试
python tests/test_data.py

# 组件层测试
python tests/test_components.py

# 模型层测试
python tests/test_models.py

# 工具层测试
python tests/test_utils.py

# 集成测试
python tests/test_integration.py
```

### 3. 使用混合模型
```python
import torch
from models.fourcastnet_global_region_hybrid import FourCastNetGlobalRegionHybrid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FourCastNetGlobalRegionHybrid(
    global_img_size=(720, 1440),
    global_patch_size=(8, 8),
    global_embed_dim=768,
    global_depth=12,
    global_checkpoint_path="./checkpoints/base_model/checkpoint_epoch_10.pth",

    region_img_size=(1000, 1000),
    region_patch_size=(8, 8),
    region_lat_range=(-5, 5),
    region_lon_range=(-5, 5),
    region_embed_dim=768,
    region_depth=6,

    in_chans=19,
    out_chans=19,
    fusion_layer_idx=3,
    fusion_mode="concat_conv",
    align_mode="crop",
).to(device)

model.load_global_pretrained()
model.freeze_global_branch()

x_region = torch.randn(2, 19, 1000, 1000).to(device)
x_global = torch.randn(2, 19, 720, 1440).to(device)

output = model(x_region, x_global)
print(output.shape)  # (2, 19, 1000, 1000)
```

## 注意事项

1. **环境要求**：需要在 OneScience 环境中运行，确保已安装 PyTorch、einops、timm 等依赖
2. **GPU 支持**：代码优先使用 GPU，如无 GPU 会自动回退到 CPU
3. **Checkpoint 路径**：训练基座模型后，将 checkpoint 路径传递给混合模型的 `global_checkpoint_path` 参数
4. **区域参数**：根据实际需求调整 `region_img_size`、`region_lat_range`、`region_lon_range` 等参数

## 下一步建议

1. **运行测试验证代码**：
   ```bash
   # 运行所有测试
   python run_all_tests.py

   # 或运行单个测试文件
   python tests/test_data.py
   ```

2. **训练基座模型**：
   ```bash
   python train_base.py --num_epochs 10 --batch_size 4
   ```

3. **使用基座模型 checkpoint 训练混合模型**

4. **根据实际数据调整虚拟数据生成参数**
