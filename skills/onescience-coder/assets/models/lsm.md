# Model Card: LSM

## 基本信息

- 模型名：`LSM`
- 任务类型：`CFD / latent spectral multiscale operator`
- 当前状态：`stable`
- 主实现文件：`./onescience/src/onescience/models/cfd_benchmark/LSM.py`

## 模型定位

LSM 把 U-Net 的多尺度路径和 latent spectral block 结合起来，适合在规则网格或 GeoFNO 投影后的潜在网格上做多尺度全局建模。

补充说明：

- 结构上最接近“U-Net + spectral transformer hybrid”。
- 支持 structured 与 unstructured，两条路径都先统一到多尺度潜在网格主干。
- 若想比较纯 U-Net、纯 FNO 和混合型多尺度谱模型，LSM 是关键候选。

## 输入定义

- 输入 shape：`x -> (Batch, NumPoints, space_dim)`，`fx -> (Batch, NumPoints, fun_dim)`
- 输入变量组织：
  - `x`
    - 坐标或统一位置编码
  - `fx`
    - 点级场输入

## 输出定义

- 输出 shape：`(Batch, NumPoints, out_dim)`
- 输出变量组织：
  - 点级目标变量

## 主干结构

- `OneMlp`
  - 预处理
- 可选 `GeoSpectralConv2d`
  - 非结构化投影
- 4 级 U-Net 编码器/解码器
- 每个尺度插入 `OneTransformer(style="NeuralSpectralBlock*d")`
- `fc1 + fc2`

## 主要依赖组件

- `OneMlp`
- `OneFourier`
- `OneTransformer(style="NeuralSpectralBlock*d")`
- `unet_layer` 家族

## 主要 Shape 变化

- 预处理后：`(Batch, NumPoints, n_hidden)`
- reshape 后：`(Batch, n_hidden, *grid_shape)`
- 编码下采样时空间尺寸减小、通道数增大
- 解码恢复后再展平成点级输出

## 默认关键参数

- `num_token=4`
- `num_basis=12`
- `s1=96, s2=96`
- `task="steady"` 时默认走 `bn` 风格归一化

## 常见修改点

- 调整 `shapelist` 与 padding
- 依据任务修改 `num_token / num_basis`
- 对比实验时可直接与 `U_Net`、`F_FNO`、`MWT` 共用训练流

## 风险点

- 主干比纯 U-Net 和纯 FNO 都更重，显存和训练时间通常更高。
- 非结构化几何仍然依赖 GeoFNO 投影质量。
- 多尺度和谱块同时存在，改通道数时要同时检查 U-Net 和 spectral block 两边。

## 推荐检索顺序

1. 先看本模型卡
2. 再看 `U_Net (CFD_Benchmark)`
3. 再看 `OneTransformer` 和 `OneFourier`

## 组件契约入口

- `./contracts/onemlp.md`
- `./contracts/onefourier.md`
- `./contracts/onetransformer.md`

## 源码锚点

- `./onescience/src/onescience/models/cfd_benchmark/LSM.py`
- `./onescience/examples/cfd/CFD_Benchmark/exp/exp_steady.py`
