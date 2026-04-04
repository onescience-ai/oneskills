# Codex Cases

本目录用于放置面向 `oneskills/codex` 的测试案例。

这些案例既可以：

1. 用需求描述去触发智能体主动检索
2. 也可以直接点名某个已有模型，测试智能体是否会先读取模型卡

智能体可优先检索：

- `./oneskills/codex/models/*.md`
- `./oneskills/codex/contracts/*.md`
- `./oneskills/codex/task/*.md`
- `./onescience/` 中的实现代码

从而验证：

1. 智能体能否先读 models / contracts 再写代码
2. 智能体能否在不被直接提示模型名的情况下，找到合适的推荐组件
3. 智能体能否在被直接提示模型名时，先命中对应模型卡
4. 智能体能否保持命名、shape 和调用方式一致

当前案例：

- `first_round_guide.md`
- `second_round_guide.md`
- `confirm_template.md`
- `weather_forecast/case.md`
- `weather_forecast/case_confirm.md`
  - 旧命名示例，保留作历史参考
- `pangu_afno_replacement/query.md`
- `pangu_afno_replacement/confirm.md`
- `workflow.md`
