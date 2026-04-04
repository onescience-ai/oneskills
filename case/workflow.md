# Case Workflow

本目录中的案例建议遵循 `./oneskills/codex/task/SKILL.md` 中定义的主流程。

也就是说，默认不是让智能体直接根据一句自然语言开始写完整模型，而是先做需求翻译，再做代码生成。

## 阶段 1：需求翻译

输入：

- `./oneskills/codex/case/first_round_guide.md`
- 用户自己的自然语言需求
- 或某个具体案例目录下的 `query.md`

参考案例：

- `./oneskills/codex/case/weather_forecast/case.md`
- `./oneskills/codex/case/pangu_afno_replacement/query.md`

说明：

- `weather_forecast` 仍保留旧命名 `case.md`
- 新案例建议统一使用 `query.md`

目标：

- 将用户的自然语言需求翻译成一份结构化、可确认的任务说明
- 输出内容应包含固定确认区块，方便用户低成本确认

这一阶段建议智能体完成：

1. 理解用户的任务目标
2. 提取数据形态、预测目标、轻量化要求、组件复用要求
3. 若任务涉及主干建模，先判断是否存在可直接复用的 `fuser` 组件
4. 基于 `./oneskills/codex/contracts/` 和 `./onescience/` 生成一份规格草案
5. 将草案交给用户确认或修改

这一阶段的价值：

- 用户不需要一开始就给出非常详细的 AI 模型描述
- 智能体可以先把模糊需求转成明确约束
- 用户可以在生成代码前纠正变量数、输入输出、结构复杂度等内容

## 阶段 2：代码生成

输入：

- `./oneskills/codex/case/second_round_guide.md`
- 用户已确认的结构化规格书
- 推荐使用：
  - 智能体在第一阶段刚输出并已被确认的“详细执行信息”
  - `./oneskills/codex/case/confirm_template.md`
  - 某个具体案例目录下的 `confirm.md`

参考案例：

- `./oneskills/codex/case/weather_forecast/case_confirm.md`
- `./oneskills/codex/case/pangu_afno_replacement/confirm.md`

说明：

- `weather_forecast` 仍保留旧命名 `case_confirm.md`
- 新案例建议统一使用 `confirm.md`

目标：

- 基于确认后的规格书生成完整模型代码

这一阶段建议智能体完成：

1. 读取 `./oneskills/codex/task/SKILL.md`
2. 优先读取 `./oneskills/codex/contracts/component_index.md`
3. 再读取 `./oneskills/codex/contracts/naming_convention.md`
4. 按需读取相关 contract 文档
5. 若任务存在 trunk 或中间层特征提取，先检查是否应复用 `fuser` 组件
6. 优先复用已有推荐组件
7. 在契约不足时再回到 `./onescience/` 源码
8. 最终输出代码和 shape 说明
9. 若未额外指定保存路径，则默认将生成文件保存到对应的 `case` 目录下

## 推荐交互方式

推荐让智能体按以下顺序和用户交互：

1. 用户先给出自己的自然语言需求，或直接使用某个具体案例目录中的 `query.md`
2. 智能体先输出一份结构化规格草案，并在末尾附带固定确认区块
3. 用户确认或修改草案
4. 用户确认后，可直接回复一句确认，或使用 `confirm_template.md` / 具体案例目录中的 `confirm.md`
5. 智能体再根据确认后的草案生成最终代码

## 为什么这样更稳

- 更贴近真实用户的表达方式
- 更容易测试 `oneskills` 是否真的参与了“需求理解”和“代码生成”两个阶段
- 可以避免用户一句话需求过于模糊，导致直接产出的代码偏离预期
- 对天气预测这类任务，可以在第一轮就先锁定 trunk 应复用的组件，减少后续错误地退回到底层 block 拼装
