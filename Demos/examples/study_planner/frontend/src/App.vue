<!-- 工作台外壳：左侧对话，右侧状态面板 -->
<script setup>
import { nextTick, onMounted, ref, watch } from 'vue'
import { apiJSON } from './api'
import ChatPanel from './components/ChatPanel.vue'
import SidePanel from './components/SidePanel.vue'
import { usePlannerStore } from './stores/planner'

const store = usePlannerStore()

const historyOpenKey = 'study-planner-history-open'
const historyOpen = ref(localStorage.getItem(historyOpenKey) === 'true')
const historyToggle = ref(null)
const historyClose = ref(null)
let listRequest = 0
const sessions = ref([])
const loading = ref(false)
const more = ref(false)
const historyError = ref('')
const sidePanel = ref(null)
const deletionDialog = ref(null)
const deletion = ref(null)
const deleting = ref(false)
const deletionError = ref('')

async function previewDelete(session) {
  historyError.value = ''
  try {
    const preview = await apiJSON(`/api/conversations/${encodeURIComponent(session.id)}/deletion-preview`)
    deletion.value = { session, preview }
    deletionError.value = ''
    await nextTick()
    if (!deletionDialog.value.open) deletionDialog.value.showModal()
  } catch (err) { historyError.value = err.message }
}

async function confirmDelete() {
  if (deleting.value) return
  deleting.value = true
  const { session, preview } = deletion.value
  try {
    await store.deleteConversation(session.id, preview)
    deletionDialog.value.close()
    await listSessions()
    historyClose.value?.focus()
  } catch (err) { deletionError.value = err.message }
  finally { deleting.value = false }
}

async function manageMemory(id) {
  deletionDialog.value.close()
  if (window.matchMedia('(max-width: 900px)').matches) closeHistory()
  await sidePanel.value.showMemory(id)
}

async function listSessions(append = false) {
  const request = ++listRequest
  loading.value = true
  historyError.value = ''
  try {
    const offset = append ? sessions.value.length : 0
    const rows = await apiJSON(`/api/conversations?limit=30&offset=${offset}`)
    if (request !== listRequest) return
    sessions.value = append ? [...sessions.value, ...rows] : rows
    more.value = rows.length === 30
  } catch (err) {
    if (request === listRequest) historyError.value = err.message
  } finally { if (request === listRequest) loading.value = false }
}

async function toggleHistory() {
  if (historyOpen.value) return closeHistory()
  historyOpen.value = true
  sessions.value = []
  more.value = false
  listSessions()
  await nextTick()
  historyClose.value?.focus()
}

function closeHistory() {
  historyOpen.value = false
  historyToggle.value?.focus()
}

watch(historyOpen, (open) => {
  try {
    localStorage.setItem(historyOpenKey, String(open))
  } catch {
    // 存储空间不足时仍允许在当前页面切换侧栏。
  }
})

async function selectSession(id) {
  try {
    await store.switchConversation(id)
    if (window.matchMedia('(max-width: 900px)').matches) closeHistory()
  } catch (err) { historyError.value = err.message }
}

async function openSource(conversation, message) {
  try {
    await store.switchConversation(conversation, message)
    if (historyOpen.value && window.matchMedia('(max-width: 900px)').matches) closeHistory()
    await nextTick()
    const target = document.getElementById(`message-${message}`)
    target?.focus({ preventScroll: true })
    target?.scrollIntoView({ block: 'center' })
  } catch (err) { store.error = err.message }
}

watch(() => [store.conversationId, store.sourceMessageId], ([conversation, message]) => {
  const url = new URL(window.location.href)
  if (message) {
    url.searchParams.set('conversation', conversation)
    url.searchParams.set('source', message)
  } else {
    url.searchParams.delete('conversation')
    url.searchParams.delete('source')
  }
  window.history.replaceState(null, '', url)
})

watch(() => store.streaming, (streaming) => {
  if (!streaming && historyOpen.value) listSessions()
})

onMounted(async () => {
  if (historyOpen.value) listSessions()
  const params = new URLSearchParams(window.location.search)
  await store.loadState()
  if (params.get('conversation') && params.get('source')) await openSource(params.get('conversation'), params.get('source'))
})
</script>

<template>
  <div class="workbench" @keydown.esc="historyOpen && !deletionDialog?.open && closeHistory()">
    <header class="wb-header">
      <div class="wb-brand">
        <span class="wb-logo" aria-hidden="true">学</span>
        <button ref="historyToggle" type="button" class="btn-ghost history-toggle" aria-controls="session-history"
                :aria-expanded="historyOpen" @click="toggleHistory">
          <svg aria-hidden="true" viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="2" y="3" width="16" height="14" rx="3" /><path d="M8 3v14" />
          </svg>
          历史会话
        </button>
        <div class="wb-titles">
          <h1>学习规划师工作台</h1>
          <p>会话记忆 · 长期学习档案</p>
        </div>
      </div>

      <div class="wb-actions">
        <span class="wb-status" :class="{ 'is-busy': store.streaming }">
          <i class="wb-status-dot"></i>
          {{ store.streaming ? '处理中' : '就绪' }}
        </span>
        <button
          type="button"
          class="btn-ghost"
          title="整理当前业务日，保留会话原文"
          @click="store.rollover()"
        >
          整理记忆
        </button>
        <button
          type="button"
          class="btn-ghost"
          :disabled="!store.messages.length"
          @click="store.clearChat()"
        >
          新对话
        </button>
      </div>
    </header>

    <main class="wb-body" :class="{ 'history-open': historyOpen }">
      <aside v-show="historyOpen" id="session-history" class="history-sidebar" aria-labelledby="history-title">
        <header>
          <h2 id="history-title">历史会话</h2>
          <button ref="historyClose" type="button" class="btn-ghost" aria-label="收起历史会话" @click="closeHistory">收起</button>
        </header>
        <p v-if="historyError" role="alert">{{ historyError }}</p>
        <p v-if="loading" role="status">正在读取会话…</p>
        <p v-else-if="!sessions.length && !historyError">还没有聊天记录。</p>
        <div v-for="session in sessions" :key="session.id" class="session-line">
          <button class="session-row" type="button"
                :aria-current="session.id === store.conversationId ? 'true' : undefined"
                :disabled="store.streaming || store.switching" @click="selectSession(session.id)">
            <strong>{{ session.title || '历史会话' }}</strong>
            <small>{{ new Date(session.last_activity).toLocaleString('zh-CN') }} {{ session.id === store.conversationId ? ' · 当前会话' : '' }}</small>
          </button>
          <details class="session-options">
            <summary :aria-label="`会话操作：${session.title || '历史会话'}`">···</summary>
            <button type="button" class="delete-action" :disabled="store.streaming || store.switching"
                    @click="$event.currentTarget.parentElement.open = false; previewDelete(session)">删除会话</button>
          </details>
        </div>
        <button v-if="more" type="button" class="btn-ghost" :disabled="loading" @click="listSessions(true)">加载更多会话</button>
      </aside>
      <ChatPanel :key="store.conversationId" />
      <SidePanel ref="sidePanel" @open-source="openSource" />
    </main>
    <dialog ref="deletionDialog" class="delete-dialog" aria-labelledby="delete-title"
            @cancel="deleting && $event.preventDefault()" @close="!$event.target.open && (deletion = null)">
      <template v-if="deletion">
        <h2 id="delete-title">删除「{{ deletion.session.title || '历史会话' }}」？</h2>
        <p>{{ deletion.preview.count }} 条聊天记录及其摘要、来源摘录将永久删除。学习档案、目标和计划会保留。</p>
        <template v-if="deletion.preview.memories.length">
          <h3>将保留的相关记忆</h3>
          <ul><li v-for="item in deletion.preview.memories" :key="item.id">
            {{ item.content }} <button type="button" class="btn-ghost" :disabled="deleting" @click="manageMemory(item.id)">管理记忆</button>
          </li></ul>
        </template>
        <p v-if="deletionError" role="alert">{{ deletionError }}</p>
        <footer>
          <button type="button" autofocus class="btn-ghost" :disabled="deleting" @click="deletionDialog.close()">取消</button>
          <button type="button" class="delete-action" :disabled="deleting || !!deletionError" @click="confirmDelete">确认删除</button>
        </footer>
      </template>
    </dialog>
  </div>
</template>

<style scoped>
.history-toggle { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.history-toggle[aria-expanded="true"] { background: var(--primary-soft); color: var(--primary-strong); }
.history-sidebar {
  min-height: 0;
  overflow-y: auto;
  padding: 16px 12px;
  border-right: 1px solid var(--border);
  background: var(--surface);
}
.history-sidebar header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.history-sidebar h2 { margin: 0; font-size: 15px; }
.session-row[aria-current="true"] { background: var(--primary-soft); color: var(--primary-strong); }
.session-row {
  display: grid; gap: 4px; width: 100%; padding: 10px;
  text-align: left; overflow-wrap: anywhere; border: 0; border-radius: var(--radius-sm);
  background: transparent; color: var(--text);
}
.session-row:hover, .session-row:focus-visible { background: var(--primary-soft); }
.session-row small { color: var(--text-soft); }
.session-line { display: flex; align-items: center; gap: 3px; }
.session-options { flex: none; position: relative; }
.session-options summary { list-style: none; cursor: pointer; padding: 4px 7px; color: var(--text-soft); }
.session-options summary::-webkit-details-marker { display: none; }
.session-options .delete-action { position: absolute; z-index: 1; right: 0; white-space: nowrap; box-shadow: var(--shadow); }
.delete-action { padding: 6px 12px; border: 1px solid var(--danger-border); border-radius: 8px; color: var(--danger); background: var(--danger-soft); }
.delete-dialog { width: min(420px, 90vw); padding: 20px; border: 1px solid var(--border); border-radius: var(--radius-lg); color: var(--text); background: var(--surface); }
.delete-dialog::backdrop { background: rgb(15 23 42 / 40%); }
.delete-dialog h2 { margin: 0; font-size: 17px; overflow-wrap: anywhere; }
.delete-dialog h3, .delete-dialog p, .delete-dialog li { font-size: 14px; }
.delete-dialog ul { max-height: 35vh; overflow-y: auto; padding-inline-start: 20px; }
.delete-dialog footer { display: flex; justify-content: flex-end; gap: 8px; }

.workbench {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.wb-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 20px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  flex: none;
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
  width: 36px;
  height: 36px;
  flex: none;
  border-radius: 11px;
  background: linear-gradient(135deg, var(--primary), #7b5cf0);
  color: #fff;
  font-size: 15px;
  font-weight: 700;
  box-shadow: var(--shadow-sm);
}

.wb-titles {
  min-width: 0;
}

.wb-titles h1 {
  margin: 0;
  font-size: 15.5px;
  font-weight: 650;
  letter-spacing: -0.01em;
}

.wb-titles p {
  margin: 1px 0 0;
  font-size: 12px;
  color: var(--text-mute);
}

.wb-titles code {
  font-family: var(--mono);
  font-size: 11px;
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
  background: var(--surface-soft);
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
  background: var(--surface-soft);
  color: var(--text-soft);
  font-size: 13px;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}

.btn-ghost:hover:not(:disabled) {
  color: var(--text);
  border-color: var(--border-strong);
  background: var(--surface);
}

.btn-ghost:disabled {
  opacity: 0.5;
}

.wb-body {
  position: relative;
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
}

.wb-body.history-open { grid-template-columns: clamp(200px, 19vw, 280px) minmax(0, 1fr) 340px; }

@media (max-width: 900px) {
  .wb-header { flex-wrap: wrap; gap: 8px; }
  .history-sidebar {
    position: absolute;
    inset: 0 auto 0 0;
    z-index: 2;
    width: min(280px, 85%);
    box-shadow: var(--shadow-lg);
  }
  .wb-body, .wb-body.history-open {
    grid-template-columns: minmax(0, 1fr);
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
</style>
