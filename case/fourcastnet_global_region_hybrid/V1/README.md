# Global + Region Hybrid FourCastNet

## 项目简介

本项目实现了一个全球+区域混合的天气预报系统，使用FourCastNet官方开源权重中最后一层的全球特征，同时结合局部区域的气象数据及相关经纬度信息进行区域映射，训练一个能够融合全球特征的FourCastNet模型进行高分辨率预测。

## 实现方案

### 核心组件

1. **GlobalRegionHybridAFNONet**：全球+区域混合模型，融合全球特征和局部区域数据
2. **GlobalFeatureExtractor**：全球特征提取器，从预训练的FourCastNet模型中提取最后一层特征
3. **RegionFeatureFusion**：区域特征融合模块，用于融合全球特征和局部区域特征
4. **GlobalRegionERA5Datapipe**：全球+区域混合数据加载器，用于加载局部区域数据、全球数据和区域信息

### 模型架构

1. **局部特征提取**：使用前半部分的AFNO blocks提取局部区域特征
2. **全球特征提取**：使用预训练的FourCastNet模型提取全球特征
3. **特征融合**：使用区域特征融合模块融合全球特征和局部区域特征
4. **特征处理**：使用后半部分的AFNO blocks处理融合后的特征
5. **输出预测**：通过线性层输出最终的预测结果

## 目录结构

```
fourcastnet_global_region_hybrid/
├── conf/                   # 配置文件
│   └── config.yaml         # 模型和数据配置
├── global_region_hybrid_afnonet.py  # 全球+区域混合模型
├── global_region_datapipe.py        # 全球+区域混合数据加载器
├── train.py                # 训练脚本
├── inference.py            # 推理脚本
└── README.md               # 项目说明
```

## 安装和依赖

本项目依赖以下库：

- PyTorch
- numpy
- einops
- apex (用于FusedAdam优化器)

## 使用方法

### 1. 准备数据

- 局部区域数据：存储在 `./data/region/` 目录下
- 全球数据：存储在 `./data/global/` 目录下
- 静态数据：存储在 `./data/static/` 目录下

### 2. 配置模型

修改 `conf/config.yaml` 文件，设置以下参数：

- `pretrained_model_path`：预训练的FourCastNet模型路径
- `data_dir`：局部区域数据路径
- `global_data_dir`：全球数据路径
- 其他模型和数据加载器配置

### 3. 训练模型

```bash
python train.py
```

### 4. 推理

```bash
python inference.py
```

## 实现细节

### 全球特征提取

使用预训练的FourCastNet模型提取全球特征，具体是提取模型的 `forward_features` 输出，即最后一层的特征。

### 区域特征融合

通过区域掩码选择与局部区域对应的全球特征，然后通过线性投影将全球特征映射到与局部特征相同的维度，最后通过AFNO block融合特征。

### 区域信息处理

区域信息包含经纬度范围 (lat_min, lat_max, lon_min, lon_max)，用于生成区域掩码，选择与局部区域对应的全球特征。

## 注意事项

1. 预训练模型路径需要正确设置，确保能够加载预训练的FourCastNet模型
2. 数据格式需要与原始FourCastNet保持一致，以便正确加载和处理
3. 区域信息需要正确提取，确保能够生成准确的区域掩码

## 未来工作

1. 优化区域特征融合策略，提高模型性能
2. 支持更多的区域和全球数据格式
3. 实现模型的分布式训练，加速训练过程
4. 添加更多的评估指标，全面评估模型性能
