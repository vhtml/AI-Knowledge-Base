/* film.js —— 一条 Vibe 知识短片的全部画面。
 *
 * 这个文件解决什么问题：演示"画面必须是一台时间的函数"这条硬规矩。
 *
 * 三条铁律：
 *   1. render(t) 是**纯函数**：同一个 t 永远画出同一帧。
 *      不许在绘制过程中改全局状态、不许用 Math.random()。
 *      不满足这条，就没法逐帧渲染、没法复核、没法只重渲有问题的那 3 秒。
 *   2. 画面只由 t 决定，不由"播到哪儿"决定。
 *      seek(t) 只是把 t 交给 render，没有别的副作用。
 *   3. 常驻元素（背景、字幕、颗粒、暗角）画在最外层，
 *      场景只画自己那一层，别把字幕画进场景里。
 *
 * 这条样片讲的是：把一段复杂的声音拆成一堆简单的声音（傅里叶的直觉）。
 * 选题、台词、事实表、时间轴都在同目录的 storyboard.md 里——
 * 那是唯一真相，这份代码必须和它逐字一致。
 */

const CONFIG = {
  W: 1920,
  H: 1080,
  FPS: 30,
  DUR: 30.1,
  WORLD: 'C',          // C = 夜曲（抽象概念），B = 寓言（心理行为）
  SUB_BOTTOM: 0.88,    // 字幕基线在画面高度的百分比位置
  TITLE: '把复杂拆成简单',
};

/* 两套外观之一：夜曲。颜色语义全片不破例——见 AI/Knowledge-Film/06。 */
const WORLD = {
  bg: '#080c18',
  bgLift: '#0f1730',
  ink: '#e9e4d6',
  dim: '#5b6480',
  gold: '#e6c07b',      // 关键词
  rose: '#e07a8c',      // 情绪词 / 反直觉的点
  cyan: '#7fb4c9',      // 数值
  star: '#2b3a63',
};

/* 字幕就是文案。hold 是停留秒数；{金色} / [玫红] 是染色标记。 */
const SUBS = [
  { t: 0.6,  text: '这是一段乱糟糟的声音',                       hold: 3.0 },
  { t: 3.9,  text: '听到的是噪音，看见的却是一团乱线',           hold: 3.2 },
  { t: 7.8,  text: '把它{金色}拆开，里面全是简单的波',           hold: 3.2 },
  { t: 11.5, text: '每一个音高，就是其中的一根波',               hold: 3.2 },
  { t: 14.7, text: '把它们按音高排开，就成了一张频谱图',         hold: 3.0 },
  { t: 17.9, text: '那问题来了：这是巧合吗',                     hold: 3.0 },
  { t: 22.9, text: '不[玫红]。任何复杂的东西，都是简单的东西加起来', hold: 3.8 },
  { t: 27.1, text: '傅里叶变换 · 1822',                        hold: 3.0 },
];

/* 片名卡：2 分半以上的长片要在 6–24 秒之间出现一次片名，
 * 结尾再出现一次。这是短片的等价做法：片长 5%–50% 之间给一次片名卡。 */
const TITLE_CARD = { t0: 2.6, t1: 5.4 };

/* 场景表。每个场景一个函数：只拿自己的那段时间画自己的东西。 */
const SCENES = [
  { t0: 0.0,  t1: 7.8,  name: '乱线',     draw: drawNoise },
  { t0: 7.8,  t1: 11.5, name: '拆开',     draw: drawSplit },
  { t0: 11.5, t1: 14.7, name: '正弦',     draw: drawSines },
  { t0: 14.7, t1: 17.9, name: '频谱',     draw: drawSpectrum },
  { t0: 17.9, t1: 22.9, name: '提问',     draw: drawQuestion },
  { t0: 22.9, t1: 27.1, name: '结论',     draw: drawConclusion },
  { t0: 27.1, t1: 30.1, name: '片名',     draw: drawTitle },
];

/* ---------------------------------------------------------------- 工具 */

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const lerp = (a, b, t) => a + (b - a) * t;
const easeOut = (x) => 1 - Math.pow(1 - clamp(x, 0, 1), 3);
const easeInOut = (x) => {
  const c = clamp(x, 0, 1);
  return c < 0.5 ? 2 * c * c : 1 - Math.pow(-2 * c + 2, 2) / 2;
};

/* 确定性伪随机：同一个种子永远给同一个数。
 * 用它画星点、颗粒这类"随机"元素——不用 Math.random()，否则同一帧
 * 重画一次就不一样了，逐帧渲染会前后抖动。 */
function seeded(i) {
  const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
}

const STAR_COUNT = 160;
const STARS = Array.from({ length: STAR_COUNT }, (_, i) => ({
  x: seeded(i * 3 + 1),
  y: seeded(i * 3 + 2),
  r: 0.6 + seeded(i * 3 + 3) * 1.4,
  tw: 0.4 + seeded(i * 3 + 4) * 0.6,
}));

/* ---------------------------------------------------------------- 背景 */

function drawBackground(ctx, t) {
  const g = ctx.createLinearGradient(0, 0, 0, CONFIG.H);
  g.addColorStop(0, WORLD.bg);
  g.addColorStop(1, WORLD.bgLift);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, CONFIG.W, CONFIG.H);

  for (let i = 0; i < STAR_COUNT; i++) {
    const s = STARS[i];
    const tw = 0.55 + 0.45 * Math.sin(t * s.tw + i);
    ctx.globalAlpha = 0.25 + 0.35 * tw;
    ctx.fillStyle = WORLD.star;
    ctx.beginPath();
    ctx.arc(s.x * CONFIG.W, s.y * CONFIG.H * 0.82, s.r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawVignette(ctx) {
  const g = ctx.createRadialGradient(
    CONFIG.W / 2, CONFIG.H / 2, CONFIG.H * 0.32,
    CONFIG.W / 2, CONFIG.H / 2, CONFIG.H * 0.78,
  );
  g.addColorStop(0, 'rgba(0,0,0,0)');
  g.addColorStop(1, 'rgba(0,0,0,0.42)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, CONFIG.W, CONFIG.H);
}

/* ---------------------------------------------------------------- 场景 */

/* 一条"乱线"：多条不同频率的波叠在一起，视觉上就是噪音。 */
function waveY(x, phase, freq, amp) {
  const u = (x / CONFIG.W) * Math.PI * 2;
  return (
    CONFIG.H * 0.42 +
    Math.sin(u * freq + phase) * amp +
    Math.sin(u * freq * 2.7 + phase * 1.3) * amp * 0.45
  );
}

function drawNoise(ctx, t) {
  const p = easeOut((t - 0.2) / 1.2);
  ctx.strokeStyle = WORLD.cyan;
  ctx.globalAlpha = 0.9 * p;
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let x = 0; x <= CONFIG.W; x += 4) {
    const y = waveY(x, t * 1.4, 6, 90 * p);
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/* "拆开"：乱线裂成三条各自独立的正弦，位置分开。 */
function drawSplit(ctx, t) {
  const p = easeInOut((t - 7.8) / 1.6);
  const lanes = [
    { freq: 3, amp: 78, off: -120, color: WORLD.cyan },
    { freq: 7, amp: 54, off: 0, color: WORLD.gold },
    { freq: 13, amp: 34, off: 120, color: WORLD.rose },
  ];
  lanes.forEach((lane, i) => {
    const q = clamp((p - i * 0.12) / 0.7, 0, 1);
    ctx.strokeStyle = lane.color;
    ctx.globalAlpha = 0.92 * q;
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    for (let x = 0; x <= CONFIG.W; x += 4) {
      const u = (x / CONFIG.W) * Math.PI * 2;
      const y =
        CONFIG.H * 0.42 +
        lane.off * p +
        Math.sin(u * lane.freq + t * 1.1) * lane.amp * q;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
  ctx.globalAlpha = 1;
}

/* 三根正弦排成一行，标签在下方——这是"简单的东西"的形象。 */
function drawSines(ctx, t) {
  const p = easeOut((t - 11.5) / 1.4);
  const amps = [110, 74, 48];
  const freqs = [2, 5, 11];
  const colors = [WORLD.cyan, WORLD.gold, WORLD.rose];
  for (let i = 0; i < 3; i++) {
    const baseX = CONFIG.W * (0.22 + i * 0.28);
    const q = clamp((p - i * 0.1) / 0.6, 0, 1);
    ctx.strokeStyle = colors[i];
    ctx.globalAlpha = 0.95 * q;
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let s = -120; s <= 120; s += 3) {
      const u = (s / 120) * Math.PI * 2;
      const y = CONFIG.H * 0.38 + Math.sin(u * freqs[i] + t * 0.9) * amps[i] * q;
      if (s === -120) ctx.moveTo(baseX + s, y);
      else ctx.lineTo(baseX + s, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 0.5 * q;
    ctx.fillStyle = colors[i];
    ctx.font = '28px "PingFang SC", "Noto Sans CJK SC", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${Math.round(freqs[i] * 110)} Hz`, baseX, CONFIG.H * 0.38 + 150);
  }
  ctx.globalAlpha = 1;
  ctx.textAlign = 'left';
}

/* 频谱：柱状图，柱高对应频率能量。 */
function drawSpectrum(ctx, t) {
  const p = easeOut((t - 14.7) / 1.3);
  const bars = 42;
  const w = CONFIG.W / bars;
  for (let i = 0; i < bars; i++) {
    const h = (0.12 + 0.88 * Math.pow(seeded(i * 7 + 5), 2.1)) * CONFIG.H * 0.34 * p;
    const x = i * w + w * 0.18;
    ctx.fillStyle = i % 7 === 3 ? WORLD.gold : WORLD.cyan;
    ctx.globalAlpha = 0.9;
    ctx.fillRect(x, CONFIG.H * 0.52 - h, w * 0.64, h);
  }
  ctx.globalAlpha = 0.35;
  ctx.strokeStyle = WORLD.dim;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, CONFIG.H * 0.52);
  ctx.lineTo(CONFIG.W, CONFIG.H * 0.52);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/* 提问这一镜"不画"：只有画面渐暗和字幕，画面不硬配图。 */
function drawQuestion(ctx, t) {
  const q = easeOut((t - 17.9) / 1.0);
  ctx.fillStyle = `rgba(8,12,24,${0.55 * q})`;
  ctx.fillRect(0, 0, CONFIG.W, CONFIG.H);
  ctx.globalAlpha = 0.5 * q;
  ctx.strokeStyle = WORLD.dim;
  ctx.lineWidth = 1.5;
  const cx = CONFIG.W / 2;
  const cy = CONFIG.H * 0.44;
  const r = 46 * q;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx + r * 0.72, cy + r * 0.72);
  ctx.lineTo(cx + r * 1.5, cy + r * 1.5);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/* 结论：所有波叠回一起，成为开头那团"乱线"。 */
function drawConclusion(ctx, t) {
  const p = easeInOut((t - 22.9) / 2.4);
  ctx.strokeStyle = WORLD.ink;
  ctx.globalAlpha = 0.92 * p;
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let x = 0; x <= CONFIG.W; x += 4) {
    const u = (x / CONFIG.W) * Math.PI * 2;
    const y =
      CONFIG.H * 0.42 +
      Math.sin(u * 2 + t * 1.0) * 96 * p +
      Math.sin(u * 5 + t * 0.9) * 62 * p +
      Math.sin(u * 11 + t * 1.2) * 38 * p;
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/* 片名卡：在片头稍后出现一次，题目抛出来，但不打断正在讲的那件事。 */
function drawTopicCard(ctx, t) {
  if (t < TITLE_CARD.t0 || t > TITLE_CARD.t1) return;
  const fadeIn = easeOut((t - TITLE_CARD.t0) / 0.8);
  const fadeOut = easeOut((TITLE_CARD.t1 - t) / 0.8);
  const alpha = Math.min(fadeIn, fadeOut);
  ctx.globalAlpha = alpha;
  ctx.fillStyle = WORLD.ink;
  ctx.font = '46px "Songti SC", "Noto Serif CJK SC", serif';
  ctx.textAlign = 'center';
  ctx.fillText(CONFIG.TITLE, CONFIG.W / 2, CONFIG.H * 0.2);
  ctx.globalAlpha = alpha * 0.45;
  ctx.strokeStyle = WORLD.dim;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(CONFIG.W / 2 - 120, CONFIG.H * 0.225);
  ctx.lineTo(CONFIG.W / 2 + 120, CONFIG.H * 0.225);
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.textAlign = 'left';
}

/* 结尾回到开头的画面，再压片名和文献。 */
function drawTitle(ctx, t) {
  const p = easeOut((t - 27.1) / 0.8);
  ctx.strokeStyle = WORLD.cyan;
  ctx.globalAlpha = 0.5 * p;
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let x = 0; x <= CONFIG.W; x += 4) {
    const y = waveY(x, 0, 6, 90);
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.globalAlpha = p;
  ctx.fillStyle = WORLD.ink;
  ctx.font = '76px "Songti SC", "Noto Serif CJK SC", serif';
  ctx.textAlign = 'center';
  ctx.fillText(CONFIG.TITLE, CONFIG.W / 2, CONFIG.H * 0.40);
  ctx.font = '30px "PingFang SC", sans-serif';
  ctx.fillStyle = WORLD.dim;
  ctx.fillText('傅里叶变换  ·  1822', CONFIG.W / 2, CONFIG.H * 0.47);
  ctx.textAlign = 'left';
  ctx.globalAlpha = 1;
}

/* ---------------------------------------------------------------- 字幕 */

/* 染色约定：{金色} / [玫红] 染**它前面那一段**（从上一个句读起）。
 * 所以 "把它拆开{金色}，里面全是简单的波" 只有 "把它拆开" 是金色，
 * 不会把整句染掉。一句话最多染一处。 */
function parseSegments(text) {
  const COLORS = { '金色': WORLD.gold, '玫红': WORLD.rose, '青色': WORLD.cyan };
  const re = /\{([^}]*)\}|\[([^\]]*)\]/g;
  const parts = [];
  let last = 0;
  let m;
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

  /* 标记只染它前面那一段的**最后一句读**，前面的句子读保持原色。 */
  const out = [];
  for (const part of parts) {
    if (!part.color) {
      out.push(part);
      continue;
    }
    const clauses = part.text.split(/(?<=[，。、；：！？!?;])/);
    clauses.forEach((clause, index) => {
      out.push({ text: clause, color: index === clauses.length - 1 ? part.color : null });
    });
  }
  return out.filter((segment) => segment.text.length);
}

/* 字幕在最外层画：任何场景都不能把字画进这条安全区。 */
function drawSubtitle(ctx, t) {
  let active = null;
  for (const sub of SUBS) {
    const end = sub.t + sub.hold;
    if (t >= sub.t && t < end) { active = sub; break; }
  }
  if (!active) return;

  const segments = parseSegments(active.text);
  const y = CONFIG.H * CONFIG.SUB_BOTTOM;
  const size = 54;
  ctx.font = `${size}px "PingFang SC", "Noto Sans CJK SC", sans-serif`;
  ctx.textAlign = 'left';

  /* 整行居中：先把各段宽度加起来，再从左端起画 */
  let total = 0;
  for (const segment of segments) total += ctx.measureText(segment.text).width;
  let x = CONFIG.W / 2 - total / 2;

  /* 描边保证任何背景上都读得清 */
  ctx.lineWidth = 8;
  ctx.strokeStyle = 'rgba(6,8,16,0.85)';
  for (const segment of segments) {
    ctx.strokeText(segment.text, x, y);
    x += ctx.measureText(segment.text).width;
  }

  x = CONFIG.W / 2 - total / 2;
  for (const segment of segments) {
    ctx.fillStyle = segment.color || WORLD.ink;
    ctx.fillText(segment.text, x, y);
    x += ctx.measureText(segment.text).width;
  }
}

/* ---------------------------------------------------------------- 主函数 */

/* render(t)：纯函数。给定 t，画出那一帧应该有的全部像素。 */
function render(ctx, t) {
  ctx.save();
  ctx.clearRect(0, 0, CONFIG.W, CONFIG.H);
  drawBackground(ctx, t);

  for (const scene of SCENES) {
    if (t >= scene.t0 && t < scene.t1) {
      scene.draw(ctx, t);
      break;
    }
  }
  /* 片名卡画在场景之上、字幕之下：它是常驻层的一部分。 */
  drawTopicCard(ctx, t);
  /* 片头 0–0.6 秒留白：片名不在第一秒出现。 */
  if (t < 0.6) {
    ctx.fillStyle = 'rgba(8,12,24,0.55)';
    ctx.fillRect(0, 0, CONFIG.W, CONFIG.H);
  }

  drawVignette(ctx);
  drawSubtitle(ctx, t);
  ctx.restore();
}

/* seek(t)：把 t 交给 render 就没有别的副作用了。 */
function seek(ctx, t) {
  render(ctx, t);
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { CONFIG, WORLD, SUBS, SCENES, render, seek, seeded };
}
