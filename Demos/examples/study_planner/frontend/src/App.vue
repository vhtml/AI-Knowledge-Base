<!-- 工作台外壳：左侧对话，右侧状态面板 -->
<script setup>
import { onMounted } from 'vue'
import ChatPanel from './components/ChatPanel.vue'
import SidePanel from './components/SidePanel.vue'
import { usePlannerStore } from './stores/planner'

const store = usePlannerStore()

onMounted(() => {
  store.loadState()
})
</script>

<template>
  <div class="workbench">
    <header class="wb-header">
      <div class="wb-brand">
        <span class="wb-logo" aria-hidden="true">学</span>
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

    <main class="wb-body">
      <ChatPanel />
      <SidePanel />
    </main>
  </div>
</template>

<style scoped>
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
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
}

@media (max-width: 900px) {
  .wb-body {
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
