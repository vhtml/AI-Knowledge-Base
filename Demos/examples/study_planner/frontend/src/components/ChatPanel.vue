<!-- 对话面板：流式接收 Agent 的留痕与回答 -->
<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { usePlannerStore } from '../stores/planner'
import { copyText } from '../clipboard'

const store = usePlannerStore()

const STARTERS = [
  '我想学 Python 数据分析',
  '我每天只有 30 分钟',
  '生成计划',
  '复盘',
]

const input = ref('')
const listRef = ref(null)
const composerRef = ref(null)

const canSend = computed(() => input.value.trim().length > 0 && !store.streaming)

const copiedMessageId = ref('')
const copyErrorId = ref('')

async function copyMessage(message, event) {
  // currentTarget 只在同步事件处理期间可用，异步拒绝后仍需定位原文。
  const paragraph = event.currentTarget?.closest('.bubble')?.querySelector('.bubble-text')
  copiedMessageId.value = ''
  copyErrorId.value = ''
  try {
    await copyText(message.content)
    copiedMessageId.value = message.id
  } catch {
    copyErrorId.value = message.id
    if (paragraph) {
      const range = document.createRange()
      range.selectNodeContents(paragraph)
      const selection = window.getSelection()
      selection?.removeAllRanges()
      selection?.addRange(range)
    }
  }
}

/* ---------- 滚动 ---------- */

function scrollToBottom(smooth = true) {
  nextTick(() => {
    const el = listRef.value
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' })
  })
}

/* ---------- 发送 ---------- */

async function send(source) {
  const text = (typeof source === 'string' ? source : input.value).trim()
  if (!text || store.streaming) return

  if (typeof source !== 'string') {
    input.value = ''
  }

  await store.send(text)
  scrollToBottom()
  composerRef.value?.focus()
}

function onKeydown(event) {
  if (event.key !== 'Enter' || event.shiftKey) return
  // 中文输入法组合态下不触发发送
  if (event.isComposing || event.keyCode === 229) return
  event.preventDefault()
  send()
}

watch(
  () => store.messages.length + store.liveSteps.length,
  () => scrollToBottom(),
)
</script>

<template>
  <section class="chat">
    <button v-if="store.historyHasMore" type="button" @click="store.loadHistory(true)">加载更早对话</button>
    <!-- 消息流 -->
    <div ref="listRef" class="chat-list" role="log" aria-live="polite">
      <div v-if="!store.messages.length" class="chat-empty">
        <div class="chat-empty-icon" aria-hidden="true">🎯</div>
        <h2>开始规划你的学习</h2>
        <p>说出目标，我会记录下来并排出计划；每轮都会展示动了哪块记忆。</p>
        <div class="chat-starters">
          <button
            v-for="item in STARTERS"
            :key="item"
            type="button"
            class="starter"
            :disabled="store.streaming"
            @click="send(item)"
          >
            {{ item }}
          </button>
        </div>
      </div>

      <article
        v-for="msg in store.messages"
        :key="msg.id"
        class="msg"
        :class="`msg--${msg.role}`"
      >
        <div class="msg-avatar" :class="`avatar--${msg.role}`" aria-hidden="true">
          {{ msg.role === 'user' ? '我' : 'AI' }}
        </div>
        <div class="msg-main">
          <div class="msg-meta">
            <span class="msg-name">{{ msg.role === 'user' ? '我' : '规划师' }}</span>
            <span class="msg-time">{{ msg.time }}</span>
          </div>

          <!-- 留痕轨迹：这轮动了哪块记忆 -->
          <ul v-if="msg.steps?.length" class="trace">
            <li v-for="(step, index) in msg.steps" :key="index" class="trace-item">
              <span class="trace-phase">{{ step.phase }}</span>
              <span class="trace-title">{{ step.title }}</span>
              <span v-if="step.detail" class="trace-detail">{{ step.detail }}</span>
            </li>
          </ul>

          <div class="bubble" :class="`bubble--${msg.role}`">
            <p class="bubble-text">{{ msg.content }}</p>
            <div v-if="msg.role === 'assistant'" class="bubble-actions">
              <button
                type="button"
                class="link-btn"
                @click="copyMessage(msg, $event)"
              >
                {{ copiedMessageId === msg.id ? '已复制' : '复制' }}
              </button>
              <span v-if="copyErrorId === msg.id" role="status">浏览器限制自动复制，请选择回复文本并按 Ctrl+C（Mac：⌘C）</span>
            </div>
          </div>
        </div>
      </article>

      <!-- 流式中的这一轮 -->
      <article v-if="store.streaming" class="msg msg--assistant">
        <div class="msg-avatar avatar--assistant" aria-hidden="true">AI</div>
        <div class="msg-main">
          <div class="msg-meta">
            <span class="msg-name">规划师</span>
            <span class="msg-time">{{ store.liveStatus || '处理中' }}</span>
          </div>

          <ul v-if="store.liveSteps.length" class="trace trace--live">
            <li v-for="(step, index) in store.liveSteps" :key="index" class="trace-item">
              <span class="trace-phase">{{ step.phase }}</span>
              <span class="trace-title">{{ step.title }}</span>
              <span v-if="step.detail" class="trace-detail">{{ step.detail }}</span>
            </li>
          </ul>

          <div v-if="store.liveText" class="bubble bubble--assistant bubble--live">
            <p class="bubble-text">{{ store.liveText }}</p>
          </div>
          <div v-else class="bubble bubble--typing">
            <span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span>
            <span class="typing-text">{{ store.liveStatus || '正在处理…' }}</span>
          </div>
        </div>
      </article>
    </div>

    <!-- 错误条 -->
    <div v-if="store.error" class="error-bar">
      <span>{{ store.error }}</span>
      <button v-if="store.failedMessage" type="button" class="link-btn" @click="store.retry()">
        重试
      </button>
    </div>

    <!-- 输入区 -->
    <footer class="chat-composer">
      <form class="composer" @submit.prevent="send()">
        <textarea
          ref="composerRef"
          v-model="input"
          class="composer-input"
          rows="1"
          :disabled="store.streaming"
          placeholder="说出你的目标或今天想做什么，Enter 发送"
          aria-label="消息输入框"
          @keydown="onKeydown"
        ></textarea>
        <button v-if="store.streaming" type="button" class="btn-stop" @click="store.stop()">
          停止
        </button>
        <button v-else type="submit" class="btn-send" :disabled="!canSend">发送</button>
      </form>
    </footer>
  </section>
</template>

<style scoped>
.chat {
  display: flex;
  flex-direction: column;
  min-height: 0;
  height: 100%;
  padding-inline: clamp(14px, 3%, 32px);
}

.chat-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 20px 0 8px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

/* 空状态 */
.chat-empty {
  margin: auto;
  text-align: center;
  padding: 28px 16px;
  max-width: 440px;
}

.chat-empty-icon {
  font-size: 38px;
  line-height: 1;
  margin-bottom: 12px;
}

.chat-empty h2 {
  margin: 0 0 6px;
  font-size: 17px;
  font-weight: 650;
}

.chat-empty p {
  margin: 0 0 18px;
  font-size: 13.5px;
  color: var(--text-soft);
}

.chat-starters {
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

.starter:hover:not(:disabled) {
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
  margin-top: 20px;
}

.avatar--user {
  background: var(--primary-soft);
  color: var(--primary-strong);
}

.avatar--assistant {
  background: linear-gradient(135deg, var(--primary), #7b5cf0);
  color: #fff;
}

.msg-main {
  max-width: min(80%, 700px);
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

/* 留痕轨迹 */
.trace {
  list-style: none;
  margin: 0 0 8px;
  padding: 8px 10px;
  border-radius: var(--radius);
  border: 1px dashed var(--border-strong);
  background: var(--surface-soft);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.trace--live {
  border-style: solid;
  border-color: var(--primary-soft);
}

.trace-item {
  display: flex;
  align-items: baseline;
  gap: 7px;
  font-size: 12px;
  line-height: 1.6;
}

.trace-phase {
  flex: none;
  padding: 1px 6px;
  border-radius: 5px;
  background: var(--primary-soft);
  color: var(--primary-strong);
  font-size: 11px;
  font-weight: 600;
}

.trace-title {
  color: var(--text);
  font-weight: 550;
}

.trace-detail {
  color: var(--text-mute);
  overflow-wrap: anywhere;
}

/* 气泡 */
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

.msg--user .bubble-actions {
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

/* 打字态 */
.bubble--typing {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

/* 模型正在逐字吐出来的正文：末尾加一个闪烁的光标 */
.bubble--live {
  position: relative;
}

.bubble--live .bubble-text::after {
  content: '';
  display: inline-block;
  width: 2px;
  height: 1em;
  margin-left: 2px;
  vertical-align: text-bottom;
  background: var(--primary);
  animation: caret-blink 1s steps(2, start) infinite;
}

@keyframes caret-blink {
  to {
    visibility: hidden;
  }
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

/* 错误条 */
.error-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin: 0 0 8px;
  padding: 9px 13px;
  border-radius: var(--radius);
  border: 1px solid var(--danger-border);
  background: var(--danger-soft);
  color: var(--danger);
  font-size: 13px;
}

/* 输入区 */
.chat-composer {
  padding: 10px 0 16px;
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
  min-width: 0;
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

.btn-send,
.btn-stop {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 72px;
  height: 38px;
  padding: 0 18px;
  border: none;
  border-radius: 11px;
  font-size: 14px;
  font-weight: 600;
  transition: background 0.15s, opacity 0.15s, transform 0.1s;
}

.btn-send {
  background: var(--primary);
  color: #fff;
}

.btn-send:hover:not(:disabled) {
  background: var(--primary-strong);
}

.btn-send:disabled {
  opacity: 0.45;
}

.btn-stop {
  background: var(--surface-soft);
  border: 1px solid var(--border-strong);
  color: var(--text-soft);
}

.btn-stop:hover {
  color: var(--danger);
  border-color: var(--danger-border);
}

.btn-send:active:not(:disabled),
.btn-stop:active {
  transform: scale(0.97);
}

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
</style>
