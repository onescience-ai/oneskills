# Developer Manual

本手册面向维护 `./oneskills/` 的开发人员。

目标不是解释如何使用某个单独案例，而是说明：

1. 当前的文件结构是什么
2. 每一层文件分别承担什么职责
3. 当 `./onescience/` 中新增一个 module 或 model 后，如何补齐当前的 skills / models / contracts / case 体系
4. 当 `./onescience/` 中新增 datapipe 后，如何补齐当前的数据侧知识

---

## 1. 当前文件结构

当前推荐将当前目录理解为三层：

```text
./oneskills/
  README.md
  DEVELOPER_MANUAL.md
  models/
    TEMPLATE.md
    model_index.md
    <model>.md
  contracts/
    TEMPLATE.md
    naming_convention.md
    component_index.md
    <component>.md
  datapipes/
    TEMPLATE.md
    datapipe_index.md
    <datapipe>.md
  task/
    SKILL.md
  case/
    README.md
    first_round_guide.md
    second_round_guide.md
    ../query.md
    ../confirm.md
    <task_name>/
      query.md
      confirm.md
```

### 目录职责

- `models/`
  - 存放模型知识卡
  - 面向智能体，提供整模型的输入输出、主干结构、组件链路和常见修改点
  - 适合处理“这个任务像哪个模型”“替换某个模型中的哪个模块”这类问题

- `contracts/`
  - 存放组件契约卡片
  - 面向智能体，提供稳定、短小、结构化的组件信息
  - 不以大段源码转写为目标

- `task/`
  - 存放交互和执行流程
  - 规定用户第一轮 query、确认后第二轮生成代码的标准行为

- `datapipes/`
  - 存放数据处理与 `DataLoader` 知识卡
  - 面向智能体，提供数据目录结构、年份划分、变量组织、样本构造和 dataloader 行为
  - 适合处理“这个数据怎么读”“样本怎么切”“训练时 dataloader 怎么组织”这类问题

- `case/`
  - 存放测试案例和案例模板
  - 上层是通用说明与模板
  - 子目录是具体任务案例

---

## 2. 当前设计原则

### 原则 1：模型层与组件层分开，不复述源码

核心不是把 `./onescience/` 重新讲一遍，而是给智能体两个足够稳定的检索入口：

- 模型层
  - 解决整模型的输入输出组织、主干结构和调用链路
- 组件层
  - 解决单个模块的注册入口、参数和 shape 约定

组件卡片主要记录：

- 组件职责
- 注册入口
- 注册名
- 输入输出 shape
- 关键参数
- 典型调用位置
- 风险点
- 源码锚点

模型卡主要记录：

- 模型定位
- 输入输出定义
- 主干结构
- 主要依赖组件
- 主要 shape 变化
- 常见修改点
- 风险点
- 源码锚点

文档和代码不一致时，以 `./onescience/` 源码为准。

### 原则 1.5：数据侧知识单独成层，不混入组件契约

datapipe 文档与组件契约解决的问题不同：

- `contracts/`
  - 解决模块怎么接
- `datapipes/`
  - 解决数据怎么读、怎么切、怎么组织成样本和 dataloader

因此数据侧知识单独放在 `datapipes/`，不与组件契约混写。

### 原则 2：用户提示词尽量短

对用户侧，推荐只保留两轮交互：

1. 第一轮：自然语言需求 -> 结构化规格说明
2. 第二轮：确认规格说明 -> 生成完整代码

### 原则 3：案例优先复用通用模板

新增案例时，不要直接在根目录堆新的 md 文件，而是：

1. 通用模板继续放在 `case/`
2. 具体任务放到独立子目录

---

## 3. 当 OneScience 新增一个 module 或 model 时，要改哪些文件

这里的 “新增 module / model” 指：

- 新增一个当前可被复用的正式组件
- 新增一个当前希望被智能体识别和复用的正式模型
- 或已有组件发生明显接口变化
- 或已有模型的调用层、结构或变量组织发生明显变化
- 或某个组件开始被推荐给智能体优先使用

下面按优先级给出建议。

### 3.1 新增组件时的必改项

如果一个新组件需要让智能体可稳定使用，建议按下面顺序处理。

#### A. 先整理 `OneScience` 侧源码本体

在写文档前，先确认下面几件事：

1. 组件实现文件已经稳定存在
2. 组件注释已经说明清楚：
   - 输入输出 shape
   - 关键参数
   - 组件职责
3. 代码中的局部变量命名尽量与当前约定一致
4. 如果该组件通过某个统一入口调用，对应注册入口已经接好

常见统一入口例如：

- `OneEmbedding`
- `OneSample`
- `OneRecovery`
- `OneFuser`
- `OneTransformer`
- `OneAttention`

如果 `OneScience` 侧还在频繁变，不建议先写契约卡。

#### B. 在 `contracts/` 下新增组件契约卡片

做法：

1. 复制 `./oneskills/contracts/TEMPLATE.md`
2. 生成新的组件文档，例如：
   - `./oneskills/contracts/mynewmodule.md`
3. 按模板填写以下信息：
   - 基本信息
   - 组件职责
   - 支持输入
   - 构造参数
   - 输出约定
   - 典型调用位置
   - 典型参数
   - 风险点
   - 源码锚点

注意：

- 源码锚点统一使用 `./onescience/...` 相对路径
- 不写本机绝对路径
- 契约卡片内容要让智能体“先用起来”，不是写成源码赏析

现有可参考的契约卡：

- `./oneskills/contracts/panguembedding.md`
- `./oneskills/contracts/pangufuser.md`
- `./oneskills/contracts/earthtransformer3dblock.md`
- `./oneskills/contracts/earthattention3d.md`

#### C. 检查命名是否需要更新

如果新组件引入了新的高频语义名，或者已有文档中会出现命名混乱，需要更新：

- `./oneskills/contracts/naming_convention.md`

通常只有在以下情况才需要改：

- 引入了一个新的核心维度语义
- 引入了一个新的高频参数名
- 需要禁止某种容易混淆的旧写法

#### D. 在组件索引中登记

当前使用：

- `./oneskills/contracts/component_index.md`

它实际承担的是：

- 组件索引页
- 当前推荐组件登记页

如果新增组件希望被智能体优先检索，建议在这里登记。

建议补充的字段：

- 组件名
- 模块族
- 统一入口 / 调用入口
- 注册名
- 输入形态摘要
- 当前状态
- 契约卡片路径

#### E. 如果该组件是“下层关键模块”，也要补到底层

有些组件虽然不是用户第一眼看到的业务模块，但它们是高层组件继续往下追时一定会命中的关键节点，也要补进 `contracts/`。

典型情况：

- `fuser` 内部依赖某个 `transformer block`
- `transformer block` 内部依赖某个 `attention`
- 某个组件通过 `One*` wrapper 统一调度

这类情况建议一起补：

1. 对应 `One*` wrapper 契约
2. 对应底层 block 契约
3. 对应底层 attention / afno / fc 契约

当前现成例子：

- `PanguFuser -> EarthTransformer3DBlock -> EarthAttention3D`
- `FengWuEncoder -> EarthTransformer2DBlock -> EarthAttention2D`

### 3.2 新增模型时的必改项

如果一个已有模型需要让智能体稳定理解并引用，建议按下面顺序处理。

#### F. 先整理 `OneScience` 侧模型调用层

在补模型卡前，先确认模型源码本身已经清楚表达下面这些内容：

1. 输入是什么
2. 输出是什么
3. 主干结构是什么
4. 依赖哪些组件
5. 调用层关键参数是否已经正确透传
6. 局部变量命名是否尽量统一

如果模型的调用层本身还不稳定，模型卡会很快过期。

#### G. 先检查依赖组件是否都已有契约卡

模型卡不应悬空依赖不存在的组件卡。

因此在写模型卡前，先检查：

1. 主干依赖组件是否已有契约
2. 如果模型会继续往下追到底层 block / attention，这些卡是否也已存在
3. 若缺失，先补组件，再补模型

#### H. 在 `models/` 下新增模型卡

做法：

1. 复制 `./oneskills/models/TEMPLATE.md`
2. 生成新的模型文档，例如：
   - `./oneskills/models/mymodel.md`
3. 按模板填写以下信息：
   - 基本信息
   - 模型定位
   - 输入定义
   - 输出定义
   - 主干结构
   - 主要依赖组件
   - 主要 shape 变化
   - 默认关键参数
   - 常见修改点
   - 风险点
   - 组件契约入口
   - 源码锚点

注意：

- 模型卡是调用层视角，不是源码赏析
- 重点是帮助智能体先理解整模型，再下钻到组件

现有可参考模型卡：

- `./oneskills/models/pangu.md`
- `./oneskills/models/fourcastnet.md`
- `./oneskills/models/fuxi.md`
- `./oneskills/models/fengwu.md`

#### I. 在模型索引中登记

当前使用：

- `./oneskills/models/model_index.md`

建议补充的字段：

- 模型名
- 任务类型
- 输入形态摘要
- 主干类型
- 主要依赖组件
- 当前状态
- 模型卡路径

### 3.3 可能要改的文件

这些文件不是每次都要动，但出现下面情况时建议同步更新。

#### J. 更新 `README.md`

文件：

- `./oneskills/README.md`

适用场景：

- 新组件是高频组件
- 你希望开发者或用户能更容易发现它
- 组件范围发生明显变化

#### K. 更新 `task/SKILL.md`

文件：

- `./oneskills/task/SKILL.md`

适用场景：

- 新组件会改变智能体第一轮需求翻译或第二轮代码生成的默认策略
- 新组件需要在流程层面被强制优先复用
- 新组件改变了默认保存路径、交互格式或确认规则

如果只是新增一个普通组件，通常不需要改这里。

#### L. 更新或新增 `case/`

适用场景：

- 新组件需要通过案例显式测试
- 你希望验证智能体是否能通过 `oneskills` 检索到它

做法：

1. 在 `./oneskills/case/` 下新增一个任务子目录
2. 放入：
   - `query.md`
   - `confirm.md`
3. 在 `case/README.md` 中登记

建议复用通用流程文件：

- `./oneskills/case/first_round_guide.md`
- `./oneskills/case/second_round_guide.md`
- `./oneskills/query.md`
- `./oneskills/confirm.md`

现有案例可参考：

- `./oneskills/case/pangu_afno_replacement/query.md`
- `./oneskills/case/pangu_afno_replacement/confirm.md`

---

## 4. 新增 module / model 的推荐维护流程

推荐按下面顺序做，而不是一次性乱改所有文档。

### 第一步：先确认 OneScience 侧是否已经稳定

在写文档之前，先确认：

1. 组件实现文件已经存在
2. 对应注册入口已经接好
3. 注释基本可读
4. 调用方式和参数接口已相对稳定

如果 OneScience 侧还在频繁变，先不要急着写契约卡。

### 第二步：判断这是“模型层知识”还是“组件层知识”

简单判断方法：

- 如果你要表达的是“这个模块怎么初始化、shape 怎么接、style 写什么”
  - 优先写组件契约卡
- 如果你要表达的是“这个整模型的输入输出怎么组织、主干怎么走、应该先改哪里”
  - 优先写模型卡
- 若两者都会被高频用到，就模型卡和组件卡都写

### 第三步：优先补最直接对应的那一层卡片

这是最重要的一步。

如果只做一件事，就先做与当前新增内容最直接对应的那一类卡片。

简单原则：

1. 新增 `module`
   - 先补 `contracts/`
2. 新增 `model`
   - 先保证依赖组件卡完整，再补 `models/`

### 第四步：判断是否需要进入模型索引或组件索引

如果该组件：

- 高频
- 推荐优先使用
- 可能被案例引用

就把它登记到 `component_index.md`。

如果该模型：

- 已经是用户会直接点名的模型
- 经常会作为模块替换的宿主模型
- 你希望智能体在第一轮就能识别它

就把它登记到 `model_index.md`。

### 第五步：判断是否需要更新流程或案例

只在下面情况才动：

- 它影响默认交互流程
- 你要用它做回归测试
- 你希望让用户能通过案例稳定触发它

---

## 5. 三种常见场景的处理方式

### 场景 A：新增一个普通组件

例如新增一个新的 `fuser`、`embedding` 或 `recovery`。

最小改动：

1. 整理 `OneScience` 侧源码与注册入口
2. 新建契约卡
3. 视情况登记到组件索引

一个最小案例：

1. 在 `./onescience/` 中完成 `mynewfuser.py`
2. 确认它已注册到 `OneFuser`
3. 复制 `./oneskills/contracts/TEMPLATE.md`
4. 新建 `./oneskills/contracts/mynewfuser.md`
5. 在 `component_index.md` 中登记
6. 若某个模型已经使用它，再更新对应模型卡

### 场景 B：新增一个模型卡

例如已有 `pangu.py`、`fourcastnet.py` 希望让智能体稳定识别。

建议改动：

1. 整理模型调用层注释与参数透传
2. 先检查依赖组件契约是否齐全
3. 新建模型卡
4. 登记到 `model_index.md`
5. 检查 `README.md` 与 `task/SKILL.md` 是否需要同步入口顺序

一个最小案例：

1. 复制 `./oneskills/models/TEMPLATE.md`
2. 新建 `./oneskills/models/mymodel.md`
3. 在 `model_index.md` 中登记
4. 若需要回归测试，再新增 `case/<task_name>/query.md` 与 `confirm.md`

### 场景 C：旧组件接口发生变化

例如参数名改了、shape 语义改了、推荐入口变了。

建议改动：

1. 更新对应契约卡
2. 检查 `naming_convention.md`
3. 检查 `case/` 中是否有旧案例引用该组件

### 场景 D：历史分裂实现被整合为一个现行组件

这里不再强调“统一组件”的概念本身，而是把整合后的版本视为当前正式组件。

建议做法：

1. 为当前正式组件写一张新的契约卡
2. 在风险点中注明旧分裂实现不再优先推荐
3. 更新案例和流程中的默认引用

---

## 6. 新增 module 时的最小检查清单

新增一个 OneScience module 后，至少检查以下问题：

- [ ] `./onescience/` 中组件实现已稳定
- [ ] 对应注册入口已存在
- [ ] 源码注释、shape 和关键参数已经清楚
- [ ] 已新增契约卡
- [ ] 契约卡中的源码锚点使用 `./onescience/...`
- [ ] 组件命名与 `naming_convention.md` 不冲突
- [ ] 如有必要，已登记到 `component_index.md`
- [ ] 若它是下层关键模块，已补对应 wrapper / block / attention 入口
- [ ] 如需测试，已新增或更新案例

---

## 7. 新增 model 时的最小检查清单

新增一个 OneScience model 后，至少检查以下问题：

- [ ] `./onescience/` 中模型调用层已稳定
- [ ] 输入输出、主干结构和主要依赖组件已写清楚
- [ ] 依赖组件契约已经齐全
- [ ] 已新增模型卡
- [ ] 已登记到 `model_index.md`
- [ ] 如有必要，已同步更新 `README.md`
- [ ] 如有必要，已同步更新 `task/SKILL.md`
- [ ] 如需测试，已新增或更新案例

---

## 8. 当前建议

从当前状态看，后续维护时优先遵循下面顺序：

1. 先补 `contracts/`
2. 再看是否需要补 `case/`
3. 最后才考虑是否修改 `task/`

原因是：

- `contracts/` 是智能体真正检索组件的核心层
- `case/` 是测试入口
- `task/` 是全局流程，改动成本最高，应该最谨慎
