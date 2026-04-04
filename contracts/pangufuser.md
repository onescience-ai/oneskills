# Contract: PanguFuser

## 基本信息

- 组件名：`PanguFuser`
- 所属模块族：`fuser`
- 统一入口：`OneFuser`
- 注册名：`style="PanguFuser"`

## 组件职责

在三维 `(PressureLevels, Height, Width)` 网格上堆叠多层 `EarthTransformer3DBlock`，对 patch token 进行特征融合。

补充说明：

- 该组件处理的是 token 序列，不是原始图像张量
- 它位于 Pangu 主干中的编码和解码阶段
- 它是 Pangu 主模型最核心的三维特征交互模块之一

## 支持输入

- 2D 输入：`not_applicable`
- 3D 输入：`(Batch, PressureLevels * Height * Width, dim)`

内部统一做法：

- 不处理 2D/3D 双形态统一问题
- 输入默认已经被组织成三维网格对应的 token 序列
- 每一层内部调用 `OneTransformer(style="EarthTransformer3DBlock")`
- 多层 block 顺序堆叠完成融合

## 构造参数

- `dim`
  - 输入与输出 token 的特征维度
- `input_resolution`
  - 三维网格尺寸 `(PressureLevels, Height, Width)`
- `depth`
  - 3D Transformer block 堆叠层数
- `num_heads`
  - 多头注意力头数
- `window_size`
  - 三维窗口大小 `(Pl_window, H_window, W_window)`
- `drop_path`
  - 单个值或按层提供的序列
- `mlp_ratio`
  - MLP 隐层放大倍数
- `qkv_bias`, `qk_scale`, `drop`, `attn_drop`, `norm_layer`
  - 标准 Transformer 配置项

## 输出约定

- 2D 输出：`not_applicable`
- 3D 输出：`(Batch, PressureLevels * Height * Width, dim)`
 
明确约束：

- 输出 token 数与输入 token 数保持一致
- `input_resolution` 必须与输入 token 数匹配

## 典型调用位置

- Pangu 主模型 `layer1`
- Pangu 主模型 `layer2`
- Pangu 主模型 `layer3`
- Pangu 主模型 `layer4`

## 典型参数

- 主干第一层融合
  - `dim=192`
  - `input_resolution=(8, 181, 360)`
  - `depth=2`
  - `num_heads=6`
  - `window_size=(2, 6, 12)`
- 下采样后中间层融合
  - `dim=384`
  - `input_resolution=(8, 91, 180)`
  - `depth=6`
  - `num_heads=12`
  - `window_size=(2, 6, 12)`

## 风险点

- 该组件不是统一 2D/3D 组件，调用层不要直接给 2D token
- 输入 token 数必须与 `PressureLevels * Height * Width` 一致
- `dim` 必须和上下游 embedding / sample 模块的输出维度对齐
- `num_heads` 需要能和 `dim`、内部 attention 配置合理匹配

## 下层依赖契约入口

- `./oneskills/codex/contracts/onefuser.md`
- `./oneskills/codex/contracts/onetransformer.md`
- `./oneskills/codex/contracts/earthtransformer3dblock.md`
- `./oneskills/codex/contracts/oneattention.md`
- `./oneskills/codex/contracts/earthattention3d.md`

## 源码锚点

- `./onescience/src/onescience/modules/fuser/pangufuser.py`
- `./onescience/src/onescience/modules/fuser/onefuser.py`
- `./onescience/src/onescience/models/pangu/pangu.py`
