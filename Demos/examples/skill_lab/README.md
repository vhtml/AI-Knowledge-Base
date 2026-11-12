# skill_lab · Skill 制作七天实验

**重点在两件事上：技能的加载机制（省了多少上下文）和触发设计（能不能被用对）。**

配套原理笔记在 [`AI/Skill/`](../../../../AI/Skill/)，每天一个主题、一篇笔记、一个能跑的实验。
所有实验**离线运行**：不联网、不消耗 token、不调用模型。跑挂了就是没学会。

---

## 快速开始

**必须在 `Demos/examples` 目录下运行**（包在 `examples` 下，Python 靠当前目录找包）：

```bash
cd Demos/examples

# 跑某一天的实验
uv run --project .. python -m skill_lab day3

# 跑全部七天
uv run --project .. python -m skill_lab all

# 列出每天的主题
python -m skill_lab list

# 自检（不依赖 pytest）
python -m skill_lab.selftest
```

实验只用标准库，裸解释器也能跑：

```bash
cd Demos/examples && python3 -m skill_lab all
```

---

## 七天路线

| 天 | 主题 | 命令 | 产出 |
| --- | --- | --- | --- |
| 1 | 技能的本质与边界 | `day1` | 本机技能的常驻成本 / 正文成本实测账本 |
| 2 | 触发设计 | `day2` | 官方 description 范式统计 + 三版 description |
| 3 | 正文与自由度分级 | `day3` | 正文体检器 + demo-assistant 重写对照 |
| 4 | 资源拆分与渐进式披露 | `day4` | 技能目录体检 + 脚手架生成与清理 |
| 5 | 脚本化 | `day5` | 校验脚本实跑 + 故意搞坏再修好 |
| 6 | 评测与迭代 | `day6` | 触发评测集 + 三轮迭代得分 |
| 7 | 打包分发与生态 | `day7` | 校验 → 打包 → 安装模拟 |

---

## 目录结构

```
skill_lab/
├── __init__.py          包入口
├── __main__.py          命令行入口（day1..day7 / all / list / selftest）
├── labkit.py            公共工具：token 估算、frontmatter 解析、技能盘点
├── selftest.py          自检（7 项）
├── lessons/             七天实验，每天一个模块，断言写在代码里
│   ├── day1_context.py
│   ├── day2_description.py
│   ├── day3_freedom.py
│   ├── day4_resources.py
│   ├── day5_scripts.py
│   ├── day6_eval.py
│   └── day7_packaging.py
├── scripts/
│   └── validate_skill.py   技能目录校验脚本（第 5、7 天复用）
├── fixtures/
│   └── trigger_cases.json  触发评测集：10 正 + 10 负（含 4 近邻负例）
├── capstone/
│   └── kb-note-writer/     毕业作品：一个可发布的完整技能包
│       ├── SKILL.md
│       ├── references/      仓库约定、笔记模板
│       ├── scripts/         validate_note.py
│       └── assets/          笔记模板文件
└── dist/                 打包产物（.gitignore，不提交）
```

---

## 几个可以自己动手改的地方

**改 `lessons/day2_description.py` 里的三版 description，看 `day6` 的得分怎么变。**
这是整个实验里反馈最直接的地方——加一个触发词、删一个边界词，得分立刻不一样。

**往 `fixtures/trigger_cases.json` 里加自己的用例。**
评测集是给你自己的技能用的，官方素材只是示范。加完记得同步 `selftest.py` 里的条数断言。

**拿 `capstone/kb-note-writer` 当模板，改成你自己的技能。**
四类资源都齐了：`SKILL.md` 管流程、`references/` 查细节、`scripts/` 做机械动作、`assets/` 出模板。

---

## 自检覆盖什么

| 检查项 | 验证内容 |
| --- | --- |
| token 估算 | CJK / 英文的估算口径 |
| frontmatter 解析 | 正常 / 缺分隔 / 未闭合三种情况 |
| 技能盘点 | 数字自洽（常驻 + 正文 = 总量） |
| lesson 可导入 | 七个模块都在，都导出 `run()` |
| 评测集素材 | 条数、kind 合法性、query 非空 |
| 毕业作品结构 | 四类资源齐全、正文不超 500 行、frontmatter 合法 |
| 校验脚本 | 对毕业作品返回退出码 0 |

---

## 和其它示例的关系

| 位置 | 关系 |
| --- | --- |
| `Demos/examples/skills/demo-assistant/` | 仓库原有的最小 skill 样例，第 3 天的改写对象 |
| `Demos/examples/study_planner/` | 本仓库的 skill 化实战场景，第 7 天毕业作品的灵感来源 |
| `Demos/examples/rag-demo/backend/rag.py` | `GOLD` 黄金集 + `eval` 子命令，第 6 天评测集的写法参考 |
| `Demos/examples/chat_cli/` | token 估算口径（`/context` 命令）来源一致 |
