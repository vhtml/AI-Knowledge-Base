<!-- frontend/src/App.vue — Agent 聊天工作台 -->
<script setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'

/** 后端地址：可用 .env 里的 VITE_API_BASE 覆盖，默认直连本地 FastAPI */
const API_BASE = (import.meta.env.VITE_API_BASE ?? 'http://localhost:8000').replace(/\/+$/, '')
/** 与后端 ChatRequest 的 max_length 保持一致 */
const MAX_LENGTH = 4000

const STARTERS = [
  '帮我计算 128 × 3.5 的结果',
  '现在是什么时间？',
  '介绍一下这个 RAG Demo 能做什么',
]

let seq = 0
const uid = () => `m${++seq}-${Date.now().toString(36)}`
const clock = () =>
  new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })

const messages = ref([])
const input = ref('')
const loading = ref(false)
const elapsed = ref(0)
const failedTask = ref('')

const listRef = ref(null)
const composerRef = ref(null)

let timer = null

const canSend = computed(() => input.value.trim().length > 0 && !loading.value)
const roleName = (role) => (role === 'user' ? '我' : role === 'error' ? '系统' : 'Agent')

/* ---------- 滚动与输入框 ---------- */

function scrollToBottom(smooth = true) {
  nextTick(() => {
    const el = listRef.value
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' })
  })
}

function resize() {
  const el = composerRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 180)}px`
}

/* ---------- 计时器 ---------- */

function startTimer() {
  stopTimer()
  elapsed.value = 0
  timer = setInterval(() => {
    elapsed.value += 1
  }, 1000)
}

function stopTimer() {
  if (timer) clearInterval(timer)
  timer = null
}

/* ---------- 请求 ---------- */

async function request(text) {
  let res
  try {
    res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    })
  } catch {
    throw new Error(`无法连接后端服务（${API_BASE}），请确认 FastAPI 已启动`)
  }

  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json()
      detail = body?.detail ?? ''
    } catch {
      /* 响应体不是 JSON，忽略 */
    }
    throw new Error(detail ? `请求失败（${res.status}）：${detail}` : `请求失败（${res.status}）`)
  }

  const data = await res.json()
  if (typeof data?.answer !== 'string') {
    throw new Error('响应格式异常：缺少 answer 字段')
  }
  return data.answer
}

/* ---------- 发送 ---------- */

async function send(source) {
  const text = (typeof source === 'string' ? source : input.value).trim()
  if (!text || loading.value) return
  if (text.length > MAX_LENGTH) {
    messages.value.push({
      id: uid(),
      role: 'error',
      content: `输入超出长度限制（最多 ${MAX_LENGTH} 字）`,
      time: clock(),
    })
    scrollToBottom()
    return
  }

  // 用户回显与快捷发起的草稿消息
  if (typeof source !== 'string') {
    input.value = ''
    resize()
  }
  messages.value.push({ id: uid(), role: 'user', content: text, time: clock() })
  scrollToBottom()

  loading.value = true
  startTimer()

  try {
    const answer = await request(text)
    messages.value.push({ id: uid(), role: 'assistant', content: answer, time: clock() })
    failedTask.value = ''
  } catch (err) {
    failedTask.value = text
    messages.value.push({
      id: uid(),
      role: 'error',
      content: err?.message ?? '未知错误',
      time: clock(),
    })
  } finally {
    loading.value = false
    stopTimer()
    scrollToBottom()
    composerRef.value?.focus()
  }
}

function retry() {
  if (!failedTask.value) return
  const text = failedTask.value
  failedTask.value = ''
  send(text)
}

function reset() {
  if (loading.value) return
  messages.value = []
  failedTask.value = ''
  input.value = ''
  resize()
  composerRef.value?.focus()
}

function onKeydown(event) {
  if (event.key !== 'Enter' || event.shiftKey) return
  // 中文输入法组合态下不触发发送
  if (event.isComposing || event.keyCode === 229) return
  event.preventDefault()
  send()
}

async function copyText(text, id) {
  try {
    await navigator.clipboard.writeText(text)
    const target = messages.value.find((m) => m.id === id)
    if (target) {
      target.copied = true
      setTimeout(() => {
        const item = messages.value.find((m) => m.id === id)
        if (item) item.copied = false
      }, 1600)
    }
  } catch {
    /* 剪贴板不可用时静默忽略 */
  }
}

watch(messages, () => scrollToBottom(), { deep: true })
watch(input, () => resize())

onBeforeUnmount(stopTimer)
</script>

<template>
  <div class="workbench">
    <!-- 顶部栏 -->
    <header class="wb-header">
      <div class="wb-brand">
        <span class="wb-logo" aria-hidden="true">AI</span>
        <div class="wb-titles">
          <h1>Agent 聊天工作台</h1>
          <p>RAG Demo · <code>POST /chat</code></p>
        </div>
      </div>
      <div class="wb-actions">
        <span class="wb-status" :class="{ 'is-busy': loading }">
          <i class="wb-status-dot"></i>
          {{ loading ? `思考中 ${elapsed}s` : '就绪' }}
        </span>
        <button class="btn-ghost" type="button" :disabled="loading || !messages.length" @click="reset">
          清空对话
        </button>
      </div>
    </header>

    <!-- 消息区 -->
    <main ref="listRef" class="wb-list" role="log" aria-live="polite">
      <div v-if="!messages.length" class="wb-empty">
        <div class="wb-empty-icon" aria-hidden="true">💬</div>
        <h2>开始一段新的对话</h2>
        <p>Agent 支持精确计算与时间查询，可以先试试：</p>
        <div class="wb-starters">
          <button
            v-for="item in STARTERS"
            :key="item"
            type="button"
            class="starter"
            @click="send(item)"
          >
            {{ item }}
          </button>
        </div>
      </div>

      <article v-for="msg in messages" :key="msg.id" class="msg" :class="`msg--${msg.role}`">
        <div class="msg-avatar" :class="`avatar--${msg.role}`" aria-hidden="true">
          {{ roleName(msg.role) === '我' ? '我' : roleName(msg.role) === 'Agent' ? 'AI' : '!' }}
        </div>
        <div class="msg-main">
          <div class="msg-meta">
            <span class="msg-name">{{ roleName(msg.role) }}</span>
            <span class="msg-time">{{ msg.time }}</span>
          </div>
          <div class="bubble" :class="`bubble--${msg.role}`">
            <p class="bubble-text">{{ msg.content }}</p>
            <div v-if="msg.role === 'error'" class="bubble-actions">
              <button v-if="failedTask" type="button" class="link-btn" @click="retry">重试</button>
            </div>
            <div v-else-if="msg.role === 'assistant'" class="bubble-actions">
              <button type="button" class="link-btn" @click="copyText(msg.content, msg.id)">
                {{ msg.copied ? '已复制' : '复制' }}
              </button>
            </div>
          </div>
        </div>
      </article>

      <!-- 加载态 -->
      <article v-if="loading" class="msg msg--assistant">
        <div class="msg-avatar avatar--assistant" aria-hidden="true">AI</div>
        <div class="msg-main">
          <div class="msg-meta">
            <span class="msg-name">Agent</span>
            <span class="msg-time">正在处理</span>
          </div>
          <div class="bubble bubble--typing">
            <span class="typing-dots" aria-hidden="true">
              <i></i><i></i><i></i>
            </span>
            <span class="typing-text">思考中 {{ elapsed }}s</span>
          </div>
        </div>
      </article>
    </main>

    <!-- 输入区 -->
    <footer class="wb-composer">
      <form class="composer" @submit.prevent="send()">
        <textarea
          ref="composerRef"
          v-model="input"
          class="composer-input"
          rows="1"
          :maxlength="MAX_LENGTH"
          :disabled="loading"
          :placeholder="loading ? 'Agent 正在响应，请稍候…' : '输入你的任务，Enter 发送，Shift + Enter 换行'"
          aria-label="消息输入框"
          @keydown="onKeydown"
          @input="resize"
        ></textarea>
        <button class="btn-send" type="submit" :disabled="!canSend">
          <span v-if="loading" class="spinner" aria-hidden="true"></span>
          <span v-else>发送</span>
        </button>
      </form>
      <div class="composer-foot">
        <span class="hint">Enter 发送 · Shift + Enter 换行</span>
        <span class="counter" :class="{ 'is-limit': input.length > MAX_LENGTH * 0.9 }">
          {{ input.length }}/{{ MAX_LENGTH }}
        </span>
      </div>
    </footer>
  </div>
</template>

<style scoped>
.workbench {
  display: flex;
  flex-direction: column;
  height: 100%;
  max-width: 960px;
  margin: 0 auto;
  padding: 0 20px;
}

/* ---------- 顶部栏 ---------- */

.wb-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 0 14px;
  border-bottom: 1px solid var(--border);
}

.wb-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.wb-logo {
  display: grid;
  place-items: center;
  width: 38px;
  height: 38px;
  flex: none;
  border-radius: 11px;
  background: linear-gradient(135deg, var(--primary), #7b5cf0);
  color: #fff;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.4px;
  box-shadow: var(--shadow-sm);
}

.wb-titles {
  min-width: 0;
}

.wb-titles h1 {
  margin: 0;
  font-size: 16px;
  font-weight: 650;
  letter-spacing: -0.01em;
}

.wb-titles p {
  margin: 1px 0 0;
  font-size: 12.5px;
  color: var(--text-mute);
}

.wb-titles code {
  font-family: var(--mono);
  font-size: 11.5px;
  padding: 1px 5px;
  border-radius: 5px;
  background: var(--surface-soft);
  border: 1px solid var(--border);
}

.wb-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: none;
}

.wb-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 11px;
  border-radius: 999px;
  font-size: 12.5px;
  color: var(--text-soft);
  background: var(--surface);
  border: 1px solid var(--border);
  white-space: nowrap;
}

.wb-status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--success);
}

.wb-status.is-busy .wb-status-dot {
  background: var(--primary);
  animation: pulse 1.2s ease-in-out infinite;
}

.btn-ghost {
  padding: 6px 12px;
  border-radius: 9px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text-soft);
  font-size: 13px;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}

.btn-ghost:hover:not(:disabled) {
  color: var(--text);
  border-color: var(--border-strong);
  background: var(--surface-soft);
}

.btn-ghost:disabled {
  opacity: 0.5;
}

/* ---------- 消息区 ---------- */

.wb-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 24px 0 8px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  scroll-behavior: smooth;
}

/* 空状态 */
.wb-empty {
  margin: auto;
  text-align: center;
  padding: 32px 16px;
  max-width: 460px;
}

.wb-empty-icon {
  font-size: 40px;
  line-height: 1;
  margin-bottom: 12px;
}

.wb-empty h2 {
  margin: 0 0 6px;
  font-size: 17px;
  font-weight: 650;
}

.wb-empty p {
  margin: 0 0 18px;
  font-size: 13.5px;
  color: var(--text-soft);
}

.wb-starters {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.starter {
  padding: 10px 14px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
  font-size: 13.5px;
  text-align: left;
  box-shadow: var(--shadow-sm);
  transition: border-color 0.15s, transform 0.15s, box-shadow 0.15s;
}

.starter:hover {
  border-color: var(--primary);
  transform: translateY(-1px);
  box-shadow: var(--shadow);
}

/* 消息 */
.msg {
  display: flex;
  gap: 11px;
  align-items: flex-start;
  animation: rise 0.22s ease-out;
}

.msg--user {
  flex-direction: row-reverse;
}

.msg-avatar {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  flex: none;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 650;
  margin-top: 22px;
}

.avatar--user {
  background: var(--primary-soft);
  color: var(--primary-strong);
}

.avatar--assistant {
  background: linear-gradient(135deg, var(--primary), #7b5cf0);
  color: #fff;
}

.avatar--error {
  background: var(--danger-soft);
  color: var(--danger);
}

.msg-main {
  max-width: min(78%, 680px);
  min-width: 0;
}

.msg--user .msg-main {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

.msg-meta {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 5px;
  padding: 0 3px;
}

.msg-name {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--text-soft);
}

.msg-time {
  font-size: 11.5px;
  color: var(--text-mute);
  font-variant-numeric: tabular-nums;
}

.bubble {
  position: relative;
  padding: 11px 14px;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border);
  background: var(--surface);
}

.bubble--user {
  background: var(--primary);
  border-color: var(--primary);
  color: #fff;
  border-bottom-right-radius: 6px;
}

.bubble--assistant {
  border-bottom-left-radius: 6px;
}

.bubble--error {
  background: var(--danger-soft);
  border-color: var(--danger-border);
  color: var(--danger);
  border-bottom-left-radius: 6px;
}

.bubble-text {
  margin: 0;
  font-size: 14.5px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.bubble-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 6px;
}

.bubble--user .bubble-actions {
  display: none;
}

.link-btn {
  padding: 0;
  border: none;
  background: none;
  color: var(--text-mute);
  font-size: 12px;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.link-btn:hover {
  color: var(--primary);
}

.bubble--error .link-btn {
  color: var(--danger);
}

/* 加载态 */
.bubble--typing {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.typing-dots {
  display: inline-flex;
  gap: 4px;
}

.typing-dots i {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--primary);
  animation: bounce 1.3s ease-in-out infinite;
}

.typing-dots i:nth-child(2) {
  animation-delay: 0.16s;
}

.typing-dots i:nth-child(3) {
  animation-delay: 0.32s;
}

.typing-text {
  font-size: 13px;
  color: var(--text-mute);
  font-variant-numeric: tabular-nums;
}

/* ---------- 输入区 ---------- */

.wb-composer {
  position: sticky;
  bottom: 0;
  padding: 12px 0 18px;
  background: linear-gradient(to bottom, transparent, var(--bg) 22%);
}

.composer {
  display: flex;
  align-items: flex-end;
  gap: 10px;
  padding: 10px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--border);
  background: var(--surface);
  box-shadow: var(--shadow);
  transition: border-color 0.15s, box-shadow 0.15s;
}

.composer:focus-within {
  border-color: var(--primary);
  box-shadow: var(--shadow), 0 0 0 3px var(--primary-soft);
}

.composer-input {
  flex: 1;
  min-height: 26px;
  max-height: 180px;
  padding: 4px 2px;
  border: none;
  outline: none;
  resize: none;
  background: transparent;
  color: var(--text);
  line-height: 1.6;
  overflow-y: auto;
}

.composer-input::placeholder {
  color: var(--text-mute);
}

.composer-input:disabled {
  opacity: 0.6;
}

.btn-send {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 76px;
  height: 38px;
  padding: 0 18px;
  border: none;
  border-radius: 11px;
  background: var(--primary);
  color: #fff;
  font-size: 14px;
  font-weight: 600;
  transition: background 0.15s, opacity 0.15s, transform 0.1s;
}

.btn-send:hover:not(:disabled) {
  background: var(--primary-strong);
}

.btn-send:active:not(:disabled) {
  transform: scale(0.97);
}

.btn-send:disabled {
  opacity: 0.45;
}

.spinner {
  width: 15px;
  height: 15px;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.35);
  border-top-color: #fff;
  animation: spin 0.7s linear infinite;
}

.composer-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 8px;
  padding: 0 4px;
  font-size: 11.5px;
  color: var(--text-mute);
}

.counter {
  font-variant-numeric: tabular-nums;
}

.counter.is-limit {
  color: var(--danger);
}

/* ---------- 动画 ---------- */

@keyframes bounce {
  0%,
  60%,
  100% {
    transform: translateY(0);
    opacity: 0.45;
  }
  30% {
    transform: translateY(-5px);
    opacity: 1;
  }
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.3;
  }
}

@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

@media (max-width: 640px) {
  .workbench {
    padding: 0 12px;
  }

  .wb-header {
    flex-wrap: wrap;
    gap: 10px;
  }

  .msg-main {
    max-width: 86%;
  }

  .wb-titles p {
    display: none;
  }
}
</style>
