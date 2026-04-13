# Model Index

## 目标

本文档用于登记当前已经补充到 `oneskills` 的模型级知识。

它与 `contracts/component_index.md` 的分工不同：

- `model_index.md`
  - 解决“当前任务更接近哪个已有模型”
  - 提供模型级输入输出、主干结构、组件链路和常见修改点
- `component_index.md`
  - 解决“某个模块该从哪个入口初始化、`style` 写什么、shape 怎么对齐”

## 建议使用方式

推荐让智能体按下面顺序检索：

1. 若用户明确提到模型名，先读本文档
2. 若用户没提模型名，但任务明显接近某个已知模型，也可先读本文档
3. 再按需读取对应模型卡
4. 然后读取相关组件契约
5. 只有当模型卡和契约都不够时，再回到源码

## 当前已登记模型

| 模型 | 任务类型 | 输入形态摘要 | 主干类型 | 主要依赖组件 | 状态 | 模型卡 |
|---|---|---|---|---|---|---|
| Transolver | CFD / 点级物理场回归 | `PyG Data` 点集输入，`Data.x + Data.pos` | 2D/3D point trunk with physics slicing | `OneTransformer(style="Transolver_block")`, `StandardMLP` | `stable` | `./models/transolver.md` |
| MeshGraphNet | CFD / 图网格物理场回归 | `DGLGraph + node_features + edge_features` | graph encode-process-decode trunk | `OneMlp`, `OneEdge`, `OneNode` | `stable` | `./models/meshgraphnet.md` |
| FNO | CFD / 规则网格神经算子基线 | `(pos, fx)` 接口 | spectral operator trunk | `OneMlp`, `OneFourier` | `stable` | `./models/fno.md` |
| U_FNO | CFD / 谱算子 + 多尺度分支基线 | `(pos, fx)` 接口 | spectral trunk with parallel U-shape branch | `OneMlp`, `OneFourier`, `U_Net` | `stable` | `./models/u_fno.md` |
| U_NO | CFD / U-shaped neural operator | `(pos, fx)` 接口 | U-shape operator trunk | `OneMlp`, `OneFourier` | `stable` | `./models/u_no.md` |
| U_Net (CFD_Benchmark) | CFD / benchmark 风格 U-shape 基线 | `(pos, fx)` 接口 | U-shape encoder/decoder trunk | `OneMlp`, `OneFourier`, `OneEncoder`, `OneDecoder`, `OneHead` | `stable` | `./models/u_net_operator.md` |
| Pangu | 全球天气预报 / 多层大气与地表联合建模 | surface 2D + upper-air 3D | 3D token trunk | `PanguEmbedding`, `PanguFuser`, `PanguDownSample`, `PanguUpSample`, `PanguPatchRecovery` | `stable` | `./models/pangu.md` |
| FourCastNet | 全球天气预报 / 单时刻 2D 场建模 | 2D 场 | 2D patch-grid AFNO trunk | `FourCastNetEmbedding`, `FourCastNetFuser`, `FourCastNetAFNO2D`, `FourCastNetFC` | `stable` | `./models/fourcastnet.md` |
| Fuxi | 多时间步二维气象场预报 | 多时间步 3D 时空块输入 | 2D U-shape Swin trunk | `FuxiEmbedding`, `FuxiTransformer`, `FuxiDownSample`, `FuxiUpSample`, `FuxiFC` | `stable` | `./models/fuxi.md` |
| FengWu | 多变量中期天气预报 | 多分支 2D 场输入 | 多分支 2D encoder/decoder + 3D fuser | `FengWuEncoder`, `FengWuFuser`, `FengWuDecoder` | `stable` | `./models/fengwu.md` |

## 适合优先看模型卡的问题

- 当前任务更像哪个已有模型
- 某个模型的输入输出变量如何组织
- 某个模型的 trunk 是二维还是三维
- 做模块替换时，应先改哪些调用点
- 做跨模型组件替换时，哪些 shape 或语义最容易冲突
- 当用户说“U-Net”时，当前指的是 `CFD_Benchmark/U_Net` 还是 `DeepCFD/UNet(Ex)`
- 当用户说“MGN”时，当前通常指的是 `MeshGraphNet` 路线，但要继续确认具体是 DGL 版现有案例还是 benchmark 版实现

## 新增模型卡的维护建议

新增模型时，建议至少完成以下内容：

1. 复制 `./models/TEMPLATE.md`
2. 填写模型级输入输出、主干结构、主要依赖组件和风险点
3. 在本文档中登记
4. 检查 `README.md`、`task/SKILL.md`、`DEVELOPER_MANUAL.md` 是否需要同步更新
