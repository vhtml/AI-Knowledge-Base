# 附录-B · 模板库

直接复制就能用的模板。每条都标了它对应教程里的哪一条规矩。

---

## B1 · 分镜表模板

```markdown
| 镜 | 字幕 | 画法 | 屏上是什么 | 它在干什么 | 小字 | 观众明白了什么 | 声音 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  |  |
```

七列的顺序不要改。第三列（画法）只能是六种之一或"不画"；
第七列是给自己看的，填不出来说明这镜不该存在。

---

## B2 · 事实表模板

```markdown
| 事实 | 上屏的值 | 类型 | 出处 | 上屏？ |
| --- | --- | --- | --- | --- |
| 提出年份 | 1822 | 史实 | Fourier, *Théorie analytique de la chaleur*, 1822 | 是 |
| 最低音示例 | 220 | 示意 | 标准钢琴 A4 为 440 Hz，取其下方八度 | 是 |
```

出处要精确到别人能自己查。示意值必须写清楚为什么是示意。

---

## B3 · 字幕稿模板

```json
{
  "cues": [
    { "t": 0.6,  "text": "这是一段乱糟糟的声音", "hold": 3.0 },
    { "t": 3.9,  "text": "听到的是噪音，看见的却是一团乱线", "hold": 3.2 },
    { "t": 7.8,  "text": "把它{金色}拆开，里面全是简单的波", "hold": 3.2 }
  ]
}
```

- 每句 ≤ 24 字，停留 3–5 秒
- 染色标记染**它前面那一段**：`把它{金色}拆开` → "把它拆开"是金色
- 一句最多染一处
- 最后一条是片名卡，停留可以更长（不短于 3 秒）

---

## B4 · 渲染页模板

```html
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>短片 · 渲染页</title>
<style>
  html, body { margin: 0; padding: 0; background: #080c18; overflow: hidden; }
  canvas { display: block; width: 100vw; height: auto; }
</style>
</head>
<body>
<canvas id="stage"></canvas>
<script src="film.js"></script>
<script>
  (function () {
    const params = new URLSearchParams(location.search);
    const t = parseFloat(params.get('t') || '0');
    const canvas = document.getElementById('stage');
    canvas.width = CONFIG.W;
    canvas.height = CONFIG.H;
    seek(canvas.getContext('2d'), t);
  })();
</script>
</body>
</html>
```

渲染页只做一件事：把 URL 上的 `t` 交给 `seek(t)`。

---

## B5 · film.js 骨架

```javascript
const CONFIG = { W: 1920, H: 1080, FPS: 30, DUR: 30, WORLD: 'C', SUB_BOTTOM: 0.88 };
const WORLD  = { bg: '#080c18', bgLift: '#0f1730', ink: '#e9e4d6',
                  dim: '#5b6480', gold: '#e6c07b', rose: '#e07a8c', cyan: '#7fb4c9' };
const SUBS   = [ /* 一句一行 */ ];
const SCENES = [ { t0: 0.0, t1: 7.8, name: '乱线', draw: drawNoise } ];

function seeded(i) {                    // 确定性伪随机：星点靠它
  const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
}

function render(ctx, t) {               // 纯函数：只有 t
  ctx.clearRect(0, 0, CONFIG.W, CONFIG.H);
  drawBackground(ctx, t);
  for (const s of SCENES) {
    if (t >= s.t0 && t < s.t1) { s.draw(ctx, t); break; }
  }
  drawVignette(ctx);
  drawSubtitle(ctx, t);
}
function seek(ctx, t) { render(ctx, t); }
```

铁律：`render` 里不许有 `Math.random` / `Date.now`；场景只画自己那段；
字幕和常驻元素在最外层。

---

## B6 · 字幕染色模板

```javascript
function parseSegments(text) {
  const COLORS = { '金色': WORLD.gold, '玫红': WORLD.rose, '青色': WORLD.cyan };
  const re = /\{([^}]*)\}|\[([^\]]*)\]/g;
  const parts = []; let last = 0; let m;
  while ((m = re.exec(text)) !== null) {
    const token = m[1] || m[2] || '';
    if (token in COLORS) {
      parts.push({ text: text.slice(last, m.index), color: COLORS[token] });
      last = m.index + m[0].length;
    } else {
      parts.push({ text: m[0], color: null });
    }
  }
  parts.push({ text: text.slice(last), color: null });

  const out = [];
  for (const part of parts) {
    if (!part.color) { out.push(part); continue; }
    const clauses = part.text.split(/(?<=[，。、；：！？!?;])/);
    clauses.forEach((c, i) => out.push({
      text: c, color: i === clauses.length - 1 ? part.color : null,
    }));
  }
  return out.filter((s) => s.text.length);
}
```

---

## B7 · 渲染命令速查

```bash
# 联络表：几个时刻各截一帧拼一张
python -m vibe_film_lab.engine.render <工程> --sheet 3 8 15 22

# 均匀 16 帧的联络表
python -m vibe_film_lab.engine.render <工程> --sheet 16

# 原尺寸静帧
python -m vibe_film_lab.engine.render <工程> --stills 5.9 18.1

# 转场检查图
python -m vibe_film_lab.engine.render <工程> --motion 8.2 10.6

# 确定性检查（必须先过）
python -m vibe_film_lab.engine.render <工程> --det

# 渲片段
python -m vibe_film_lab.engine.render <工程> --clip 0 30

# 渲全片
python -m vibe_film_lab.engine.render <工程> --jobs 4

# 浏览器预览
python -m vibe_film_lab.engine.render <工程> --serve
```

---

## B8 · 事件音合成模板

```bash
ffmpeg -y \
  -f lavfi -i "sine=frequency=520:duration=0.14" \
  -f lavfi -i "sine=frequency=760:duration=0.12" \
  -filter_complex "[0:a]adelay=7800|7800,volume=0.35[a0];\
                   [1:a]adelay=14700|14700,volume=0.35[a1];\
                   [a0][a1]amix=inputs=2:duration=longest:normalize=0[out]" \
  -map "[out]" -ac 1 -ar 44100 events.wav
```

混 BGM：

```bash
ffmpeg -y -i film.mp4 -i bgm.mp3 \
  -filter_complex "[1:a]volume=-8dB[a1];[0:a][a1]amix=inputs=2:duration=first:normalize=0[out]" \
  -map 0:v -map "[out]" -c:v copy -c:a aac -b:a 192k -movflags +faststart final.mp4
```

---

## B9 · 终检清单（可直接贴进项目 README）

```markdown
## 终检

- [ ] 讲解：字幕↔分镜一致；抽五镜过三问；没有只是配图的画面
- [ ] 画与不画：提问/命名/结论没硬配图；没有连着几句画面不变
- [ ] 结构：片名卡在片长 5%–30%；结尾回到开头；最后一屏是片名 + 文献
- [ ] 字幕：每句一行；停留 3–5 秒；一句最多染一处
- [ ] 画面：字幕区没有别的东西；一镜只有一条小字；颜色含义一致
- [ ] 节奏：没有硬切；每 20–30 秒一段无字幕纯画面
- [ ] 事实：屏上每个数字和事实表一致；示意的地方写明了
- [ ] 技术：--det 通过；1920×1080 @30fps；没有水印和署名

人工确认项：
- [ ] 静音遮字幕看了一遍
- [ ] 只读字幕读了一遍
- [ ] 手机上完整看过一遍

未验事项：
-
```

---

## B10 · 选题候选表模板

```markdown
| 候选 | 标题 | 硬核模型 | 反直觉结论 | 核心画面 | 外观 | 评注 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  | C |  |
```

六个候选起步。**核心画面**那一列填不出来的，直接淘汰——
填不出来说明还没找到形象。
