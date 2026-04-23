# Model Index

## 使用建议

- 用户明确提到模型名时，先读这里，再跳到对应模型卡。
- 若用户没提模型名，但目标很像已有案例，也先在这里找最接近的基线。
- 先分清楚模型吃的是哪种协议：`(x, fx)`、`PyG Data`、`DGLGraph`，还是 `(node_features, edge_features, graph)`。
- `CFD_Benchmark` 目录里存在若干与主库其它目录“同名但不同实现”的模型，尤其是 `Transolver` 和 `MeshGraphNet`，不要混用。

## 已登记模型

| 模型 | 任务类型 | 输入协议摘要 | 主干类型 | 状态 | 模型卡 |
| --- | --- | --- | --- | --- | --- |
| `Transolver` | CFD / pointwise field regression | `PyG Data.x + Data.pos` | pointwise physics transformer | `stable` | [transolver.md](./transolver.md) |
| `Transolver (CFD_Benchmark)` | CFD / benchmark token regression | `(x, fx, T)` | token transformer with slicing | `stable` | [transolver_benchmark.md](./transolver_benchmark.md) |
| `MeshGraphNet` | CFD / graph rollout & regression | `DGLGraph + node/edge features` | encode-process-decode graph net | `stable` | [meshgraphnet.md](./meshgraphnet.md) |
| `MeshGraphNet (CFD_Benchmark)` | CFD / explicit graph regression | `(node_features, edge_features, graph)` | encode-process-decode graph net | `stable` | [meshgraphnet_benchmark.md](./meshgraphnet_benchmark.md) |
| `FNO` | CFD / operator learning | `(x, fx)` | Fourier operator trunk | `stable` | [fno.md](./fno.md) |
| `F_FNO` | CFD / operator learning | `(x, fx)` | factorized Fourier trunk | `stable` | [f_fno.md](./f_fno.md) |
| `GFNO` | CFD / 2D operator learning | `(x, fx)` | group-equivariant Fourier trunk | `stable` | [gfno.md](./gfno.md) |
| `U_FNO` | CFD / operator learning | `(x, fx)` | Fourier trunk with parallel U-branch | `stable` | [u_fno.md](./u_fno.md) |
| `U_NO` | CFD / operator learning | `(x, fx)` | U-shaped neural operator | `stable` | [u_no.md](./u_no.md) |
| `U_Net (CFD_Benchmark)` | CFD / operator learning | `(x, fx)` | U-shape encoder/decoder | `stable` | [u_net_operator.md](./u_net_operator.md) |
| `MWT` | CFD / operator learning | `(x, fx)` | multiwavelet transform trunk | `stable` | [mwt.md](./mwt.md) |
| `LSM` | CFD / hybrid operator learning | `(x, fx)` | U-Net + latent spectral blocks | `stable` | [lsm.md](./lsm.md) |
| `DeepONet` | CFD / operator learning | `(x, fx)` | branch-trunk MLP operator | `stable` | [deeponet.md](./deeponet.md) |
| `ONO` | CFD / operator learning | `(x, fx)` | orthogonal neural operator blocks | `stable` | [ono.md](./ono.md) |
| `GNOT` | CFD / operator learning | `(x, fx)` | MoE-style transformer operator | `stable` | [gnot.md](./gnot.md) |
| `Transformer` | CFD / token baseline | `(x, fx)` | standard self-attention trunk | `stable` | [transformer.md](./transformer.md) |
| `Galerkin_Transformer` | CFD / token baseline | `(x, fx)` | linear-attention transformer | `stable` | [galerkin_transformer.md](./galerkin_transformer.md) |
| `Factformer` | CFD / token baseline | `(x, fx)` | factorized-attention transformer | `stable` | [factformer.md](./factformer.md) |
| `Swin_Transformer` | CFD / structured 2D baseline | `(x, fx)` | windowed transformer trunk | `stable` | [swin_transformer.md](./swin_transformer.md) |
| `GraphSAGE` | CFD / graph baseline | `(x, fx, edge_index)` | PyG message passing | `stable` | [graphsage.md](./graphsage.md) |
| `Graph_UNet` | CFD / graph baseline | `(x, fx, edge_index)` | hierarchical graph U-Net | `stable` | [graph_unet.md](./graph_unet.md) |
| `PointNet` | CFD / point baseline | `(x, fx)` | pointwise MLP + global pooling | `stable` | [pointnet.md](./pointnet.md) |
| `RegDGCNN` | CFD / point baseline | `(x, fx)` | dynamic kNN EdgeConv trunk | `stable` | [regdgcnn.md](./regdgcnn.md) |
| `Pangu` | weather / global forecasting | surface 2D + upper-air 3D | 3D token trunk | `stable` | [pangu.md](./pangu.md) |
| `FourCastNet` | weather / global forecasting | 2D fields | AFNO trunk | `stable` | [fourcastnet.md](./fourcastnet.md) |
| `Fuxi` | weather / spatiotemporal forecasting | multi-step 2D/3D blocks | U-shape Swin trunk | `stable` | [fuxi.md](./fuxi.md) |
| `FengWu` | weather / medium-range forecasting | multi-branch 2D inputs | encoder-decoder + 3D fuser | `stable` | [fengwu.md](./fengwu.md) |

## CFD_Benchmark 里当前已覆盖的模型

- 已单独成卡：
  - `DeepONet`
  - `Factformer`
  - `FNO`
  - `F_FNO`
  - `Galerkin_Transformer`
  - `GFNO`
  - `GNOT`
  - `GraphSAGE`
  - `Graph_UNet`
  - `LSM`
  - `MeshGraphNet (CFD_Benchmark)`
  - `MWT`
  - `ONO`
  - `PointNet`
  - `RegDGCNN`
  - `Swin_Transformer`
  - `Transformer`
  - `Transolver (CFD_Benchmark)`
  - `U_FNO`
  - `U_Net (CFD_Benchmark)`
  - `U_NO`

## 选型提示

- 规则网格算子学习：
  - 先看 `FNO`、`F_FNO`、`U_FNO`、`U_NO`、`U_Net (CFD_Benchmark)`、`MWT`、`LSM`
- 结构化网格上的纯注意力主干：
  - 先看 `Transformer`、`Galerkin_Transformer`、`Factformer`、`Swin_Transformer`
- branch-trunk / 双分支算子路线：
  - 先看 `DeepONet`、`ONO`、`GNOT`
- 点云或显式图基线：
  - 先看 `PointNet`、`RegDGCNN`、`GraphSAGE`、`Graph_UNet`
- 显式图 encode-process-decode：
  - 如果是通用 DGL MGN 案例，看 `MeshGraphNet`
  - 如果是 `CFD_Benchmark` 风格实验工厂，看 `MeshGraphNet (CFD_Benchmark)`
- Transolver 路线：
  - 如果训练脚本吃 `PyG Data`，看 `Transolver`
  - 如果训练脚本吃 `(x, fx)`，看 `Transolver (CFD_Benchmark)`

## 维护建议

- 新增模型卡时，优先写清输入协议、主干类型、最容易混淆的实现差异。
- 对同名不同实现，优先在索引里直接拆成两行，而不是只在单张卡片里顺带提一句。
