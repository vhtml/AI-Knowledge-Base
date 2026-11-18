# ai_pm_lab · AI 产品经理八周实验

**重点在两件事上：怎么判断一个需求该不该做，以及怎么证明做出来的东西是对的。**

配套原理笔记在 [`AI/AI-Product/`](../../../../AI/AI-Product/)，一周两个主题、一篇笔记、一个能跑的实验，
最后产出一份**能通过校验的完整 AI 产品方案包**。
所有实验**离线运行**：不联网、不消耗 token、不调用模型。跑挂了就是没学会。

---

## 快速开始

**必须在 `Demos/examples` 目录下运行**（包在 `examples` 下，Python 靠当前目录找包）：

```bash
cd Demos/examples

# 跑某一周的实验
uv run --project .. python -m ai_pm_lab week3

# 跑全部八周
uv run --project .. python -m ai_pm_lab all

# 列出八周主题
uv run --project .. python -m ai_pm_lab list

# 自检（不依赖 pytest）
uv run --project .. python -m ai_pm_lab selftest
```

实验只用标准库，裸解释器也能跑：

```bash
cd Demos/examples && python3 -m ai_pm_lab all
```

`all` 逐周跑，某周断言失败**不中断**整轮，最后汇总 `N/8 通过`；有失败退出码为 1。
单周命令（`week3`）在断言失败时同样返回非 0。

---

## 八周路线

| 周 | 主题 | 命令 | 产出 |
| --- | --- | --- | --- |
| 1 | 能力雷达与技术边界速查卡 | `week1` | 八维自评雷达 + 短板清单与补齐动作 + 11 类任务的边界速查卡 |
| 2 | 场景筛选与需求优先级 | `week2` | 12 个场景的 AI Fit 打分与档位判定 + 价值/成本排序 + 不确定性预算 |
| 3 | 竞品地图与定价锚点 | `week3` | 6 家竞品的二维象限定位 + 从用户省时价值推出来的定价区间 |
| 4 | 信任式回答与检索策略评估 | `week4` | 回答模板的四要素体检 + 召回/引用/忠实度三指标 |
| 5 | 自动化等级与工具权限 | `week5` | 六级自动化判定与每级兜底项 + 工具权限清单体检（含越权检出） |
| 6 | 评测门禁与单位经济 | `week6` | 20 条用例的评分与通过率 + 不许变差的门禁 + 单位经济公式链 |
| 7 | 风险登记册与指标树 | `week7` | 风险登记册体检 + 北极星/护栏/诊断三层校验 + 漏斗转化率 |
| 8 | 毕业作品校验 | `week8` | 方案包五份产物的逐项 checklist 与总评（不通过则退出码非 0） |

---

## 目录结构

```
ai_pm_lab/
├── __init__.py          包入口
├── __main__.py          命令行入口（week1..week8 / all / list / selftest）
├── labkit.py            公共工具：素材加载、数据模型、表格渲染（按显示宽度对齐）
├── selftest.py          自检（9 项）
├── weeks/
│   ├── __init__.py      WEEKS 登记表：加一周只改这一处
│   ├── week1.py … week8.py   每周一个模块，断言写在代码里
├── fixtures/
│   ├── capabilities.json     八维能力的档位判据、样例自评与补齐动作
│   ├── scenarios.json        12 个候选场景（含 AI Fit 打分所需属性）
│   ├── competitors.json      6 条竞品记录（纯定性，不含市场份额）
│   ├── eval_cases.json       20 条评测用例（关键错误 + 软错误）
│   ├── retrieval_cases.json  10 条检索样本（召回 / 引用 / 忠实度）
│   ├── cost_model.json       成本与延迟参数（单价一律用变量名表达）
│   └── metrics.json          北极星 / 护栏 / 诊断 + 漏斗样本
└── capstone/
    └── ai-product-spec/      毕业作品：一份完整的 AI 产品方案包
        ├── PRD.md             产品需求文档（十节齐全）
        ├── eval_set.json      评测集
        ├── metrics.md         指标字典与指标树
        ├── unit_economics.md  单位经济测算（公式 + 变量定义）
        └── risk_register.md   风险登记册（必填字段齐全）
```

---

## 几个可以自己动手改的地方

**改 `fixtures/scenarios.json` 里一个场景的判据，看 `week2` 的档位怎么变。**
把 `has_determinism` 从 `true` 改成 `false`，它的分数会掉 30 分——通常直接从 build 掉到 pilot。
这是整套教程里反馈最直接的地方：判据不是形容词，是可以加减的分。

**改 `fixtures/cost_model.json` 里的 `cache_hit_ratio` 或 `sessions_per_user`，看 `week6` 的回本周期怎么动。**
成本模型里最容易被忽略的往往不是 token 单价，是输入长度和会话频次。

**拿 `capstone/ai-product-spec/` 当模板，改成你自己的产品方案。**
五份产物缺一不可：PRD 十节、评测集、指标字典、单位经济、风险登记册。
改完之后跑 `week8`——它会告诉你缺哪一节，比自己回忆可靠。

**给 `weeks/week1.py` 的速查卡加一行你踩过的坑。**
`TASK_TYPES` 是一个普通的字典，加一行就跑起来了，不用改别的地方。

---

## 自检覆盖什么

| 检查项 | 验证内容 |
| --- | --- |
| 素材可解析且条数正确 | 7 份素材的条数与结构（8 维 / 12 场景 / 6 竞品 / 20 用例 / 10 检索样本） |
| 素材字段与枚举合法 | 场景字段与布尔类型、竞品坐标轴取值、评测用例结构 |
| 八周模块齐全 | 八个模块都在、都导出 `run()`、键与 `week1..week8` 一致 |
| 公共工具边界行为 | 显示宽度、截断、表格渲染、Markdown 表格解析、缺素材报错 |
| 各周关键函数 | 打分区间、象限名、自动化等级、门禁、单位经济、指标树的边界输入 |
| 毕业作品五份产物 | 五份文件齐全且非空、评测集可解析 |
| PRD 十节齐全 | 十个必需小节都在 |
| 残缺方案包被拦下 | 临时复制一份残缺方案包，校验器必须判不通过且 CLI 退出码非 0 |
| CLI list 可用 | `list` 子命令能跑通并列出八周 |

自检不验证"教程内容讲对了"——那由各周模块里的断言负责。
自检验证的是**骨架没坏**：素材是完整的、模块都能跑、校验器确实会拦住不合格的方案包。

---

## 和仓库其它部分的关系

| 位置 | 关系 |
| --- | --- |
| `AI/AI-Product/` | 本教程的原理笔记（对应第 01–16 章） |
| `Demos/examples/skill_lab/` | 同一套写法的另一个样本：笔记 + 离线实验 |
| `Demos/examples/langgraph_lab/` | 同上，十七章版本 |
| `Demos/examples/rag-demo/backend/rag.py` | `GOLD` 黄金集 + `eval` 子命令，第 4、6 周评测集的写法参考 |
