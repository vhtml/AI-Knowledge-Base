# vibe_film_lab · Vibe 知识动画教程实验

**重点在两件事上：一条知识短片的内容设计（形象、事实、分镜）和一条代码渲染管线（纯函数、逐帧、导出）。**

配套原理笔记在 [`AI/Knowledge-Film/`](../../../../AI/Knowledge-Film/)，一章一个主题、一篇笔记、一个能跑的实验。
所有实验**离线运行**：不联网、不消耗 token、不调用模型。跑挂了就是没学会。

渲染相关的实验需要本机装了 **Chrome/Chromium/Edge** 和 **ffmpeg**；
没装的话那几章会打印"跳过"并继续，不会中断整轮。

---

## 快速开始

**必须在 `Demos/examples` 目录下运行**（包在 `examples` 下，Python 靠当前目录找包）：

```bash
cd Demos/examples

# 跑某一章
uv run --project .. python -m vibe_film_lab ch07

# 跑全部 12 章
uv run --project .. python -m vibe_film_lab all

# 列出各章主题
uv run --project .. python -m vibe_film_lab list

# 自检（不依赖 pytest）
uv run --project .. python -m vibe_film_lab.selftest
```

除渲染那几章外，实验只用标准库，裸解释器也能跑：

```bash
cd Demos/examples && python3 -m vibe_film_lab all
```

---

## 十二章路线

| 章 | 主题 | 命令 | 产出 |
| --- | --- | --- | --- |
| 1 | 片型解剖：先把形态量出来 | `ch01` | 四条参考片的规格统计 + 硬指标归纳 |
| 2 | 形象优先：先找形象，再写文案 | `ch02` | 可画性判定器 + 四句坏文案的打回意见 |
| 3 | 事实表：每个数字都要有出处 | `ch03` | 双向核对（字幕 ↔ 事实表） |
| 4 | 文案即字幕：写稿就是排时间轴 | `ch04` | 句长/停留/语速/染色的体检表 |
| 5 | 分镜表：唯一的施工图 | `ch05` | 分镜体检器 + 故意做坏的对照 |
| 6 | 两套外观：夜曲与寓言 | `ch06` | 选题 → 外观的判定表 + 颜色语义锁 |
| 7 | 渲染引擎：画面必须是时间的纯函数 | `ch07` | 不确定性来源静态检查 + 确定性实跑 |
| 8 | 时间轴与卡点 | `ch08` | 字幕时间轴 + 拍点偏差表 + 片名卡位置 |
| 9 | 声音：没有人声，节奏靠声音撑 | `ch09` | 事件音表 + 真合成一条音轨 |
| 10 | 工程管线：先渲片段，再渲全片 | `ch10` | 渲染速度实测 + 并行加速比 |
| 11 | 终检：八道门槛 | `ch11` | 门槛检查器 + 人工确认项清单 |
| 12 | 工具对照与选型 | `ch12` | 六条路线的打分选型表 |

---

## 目录结构

```
vibe_film_lab/
├── __init__.py          包入口
├── __main__.py          命令行入口（ch01..ch12 / all / list / selftest）
├── labkit.py            公共口径层：字幕时间轴、分镜表、事实表、门槛检查
├── selftest.py          自检（10 项）
├── lessons/             十二章实验，每章一个模块，断言写在代码里
├── engine/
│   ├── template.html    渲染页模板：把 URL 上的 t 交给 seek(t)
│   ├── render.py        逐帧渲染 + ffmpeg 编码 + sheet/stills/motion/det
│   └── audio.py         事件音合成 + BGM 混音
├── fixtures/            教学素材
│   ├── reference.json   四条参考片的测量（教学用样例数据）
│   ├── facts.json       事实表
│   ├── script.json      字幕稿 + 四句坏文案
│   ├── storyboard.json  分镜表
│   └── worlds.json      两套外观的规格
└── capstone/
    └── fourier_C/       毕业作品：一条 30 秒的完整短片
        ├── index.html
        ├── film.js      全部画面（字幕、七个场景、片名卡、渲染引擎）
        ├── storyboard.md 分镜表 + 事实表 + 时间轴
        └── README.md    怎么跑
```

---

## 几个可以自己动手改的地方

**改 `fixtures/script.json` 里的一句字幕，再跑 `ch04`。**
停留时长、语速、染色统计会立刻变——这章的价值全在反馈快。

**往 `fixtures/storyboard.json` 加一镜，再跑 `ch05`。**
体检器会指出新镜是不是缺三问、是不是和前一镜撞了画法。

**改 `capstone/fourier_C/film.js` 里的 `SUBS`，再跑 `ch07` 和 `ch11`。**
两章都会报"字幕和文案稿不一致"——这就是"分镜表是唯一真相"的强制手段。

**把 `capstone/fourier_C/film.js` 里的 `seeded()` 换成 `Math.random()`，再跑 `ch07`。**
静态检查会当场抓住你；就算躲过静态检查，`--det` 也会因为两帧不一致而失败。

---

## 自检覆盖什么

| 检查项 | 验证内容 |
| --- | --- |
| lesson 可导入 | 十二个模块都在，都导出 `run()` |
| 素材完整 | 五个 fixture 文件齐全，分镜表字段完整 |
| 染色标记解析 | 标记只染前一段；写错的颜色词不被吞掉 |
| 时间轴口径 | 单调、不重叠、停留时长被钳制 |
| 分镜体检器 | 坏分镜表抓得住 |
| 事实双向核对 | 正反两个方向都能报出问题 |
| 引擎文件 | 四个文件齐全；film.js 里没有不确定性来源 |
| film.js 与稿件一致 | 字幕逐字一致、场景铺满全片 |
| 端到端渲染 | 渲帧 → 编码 → ffprobe 校验分辨率/帧率 |
| 渲染脚本入口 | `--det` 命令行跑得通 |

---

## 和其它示例的关系

| 位置 | 关系 |
| --- | --- |
| `AI/Knowledge-Film/` | 本教程的原理笔记 |
| `Demos/examples/chat_cli/` | 上下文压缩的做法可以借鉴到这里——长片也是要控成本的 |
| `Demos/examples/rag-demo/` | 检索评估的思路用在"事实表双向核对"上 |
| `Demos/examples/infra_lab/` | 逐帧渲染和连续批处理是同一个问题的两面 |
