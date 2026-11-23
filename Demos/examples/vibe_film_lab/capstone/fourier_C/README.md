# 毕业作品 · 《把复杂拆成简单》

一条 27 秒的 Vibe 知识短片，题目是傅里叶变换的直觉。
它是这套教程的**完整样例**：从分镜表到成片，中间每一步都能在这条片子里找到对应。

| 文件 | 是什么 |
| --- | --- |
| `index.html` | 渲染页：只做一件事，把 URL 上的 `t` 交给 `seek(t)` |
| `film.js` | 全部画面：字幕、七个场景、片名卡、渲染引擎 |
| `storyboard.md` | 分镜表 + 事实表 + 时间轴（唯一真相） |

机器可读的文案稿、事实表、分镜表在 [`../../fixtures/`](../../fixtures/)，
第 11 章的终检会拿它们和 `film.js` 逐字核对。

## 跑起来

```bash
cd Demos/examples

# 渲几张静帧拼成联络表（秒级，用来一眼看风格）
uv run --project .. python -m vibe_film_lab.engine.render \
    vibe_film_lab/capstone/fourier_C --sheet 2 8 12 15 19 22 25

# 看转场：在一段时间里连取 12 帧拼成一张图
uv run --project .. python -m vibe_film_lab.engine.render \
    vibe_film_lab/capstone/fourier_C --motion 7.4 8.2

# 确定性检查：同一个 t 渲两遍，必须逐字节一致
uv run --project .. python -m vibe_film_lab.engine.render \
    vibe_film_lab/capstone/fourier_C --det

# 渲前 10 秒（先渲片段，别一上来就渲全片）
uv run --project .. python -m vibe_film_lab.engine.render \
    vibe_film_lab/capstone/fourier_C --clip 0 10 --jobs 4

# 渲全片
uv run --project .. python -m vibe_film_lab.engine.render \
    vibe_film_lab/capstone/fourier_C --jobs 4
```

产物在 `vibe_film_lab/capstone/fourier_C/out/`：`film.mp4`（成片）、`sheet.jpg`（联络表）、
`motion.jpg`（转场检查图）、`frames/`（帧序列，体积大，不要提交）。

## 浏览器里预览

```bash
cd Demos/examples
uv run --project .. python -m vibe_film_lab.engine.render \
    vibe_film_lab/capstone/fourier_C --serve
# 打开 http://127.0.0.1:8737/
```

## 这条片子演示了哪些规矩

| 规矩 | 在这条片子里长什么样 |
| --- | --- |
| 字幕就是文案 | 8 句字幕就是全部内容，没有旁白 |
| 先找形象再写文案 | "拆开""叠回"都能在屏上发生，不是配图 |
| 每个事实带出处 | 1822 出自傅里叶 1822 年的著作；220/550/1210 标了"示意" |
| 该停的停 | 第 6 镜提问不画、第 8 镜片名不画 |
| 画面是纯函数 | `render(t)` 只依赖 t，随机星点用带种子的 `seeded()` |
| 常驻元素在最外层 | 字幕、颗粒、暗角都画在场景之上 |
| 结尾回到开头 | 片尾的乱线就是第 1 镜那条线 |
