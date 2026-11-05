<!-- 侧栏面板：今日待办、目标、两级记忆 -->
<script setup>
import { computed, nextTick, ref } from 'vue'
import { usePlannerStore } from '../stores/planner'

const store = usePlannerStore()
const emit = defineEmits(['open-source'])

function followSource(event, source) {
  if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return
  event.preventDefault()
  emit('open-source', source.conversation_id, source.source_id)
}

const tab = ref('today')
const showSuperseded = ref(false)
const editing = ref(null)
const deleting = ref(null)

async function showMemory(id) {
  tab.value = 'memory'
  showSuperseded.value = true
  while (!store.memory.long.some(item => item.id === id) && store.memoryHasMore) {
    const count = store.memory.long.length
    await store.loadMoreMemories()
    if (count >= store.memory.long.length) break
  }
  await nextTick()
  const card = document.getElementById(`memory-${id}`)
  if (card) {
    card.focus({ preventScroll: true })
    card.scrollIntoView({ block: 'center' })
  } else store.error ||= '该条记忆已更新，请刷新档案'
}

defineExpose({ showMemory })

function editMemory(item) {
  editing.value = { item, content: item.content, value: item.budget_value ?? '', scope: item.budget_scope ?? 'default' }
}

async function saveMemory() {
  const form = editing.value
  const changes = { content: form.content }
  if (form.value !== '') { changes.value = Number(form.value); changes.scope = form.scope }
  if (await store.updateMemory(form.item, changes)) editing.value = null
}

async function previewDeletion(item) {
  try { deleting.value = { item, preview: await store.previewDeletion(item) } }
  catch (err) { store.error = err.message }
}

async function confirmDeletion() {
  if (await store.deleteMemory(deleting.value.item)) deleting.value = null
}

const TABS = [
  { key: 'today', label: '今日' },
  { key: 'goals', label: '目标' },
  { key: 'memory', label: '记忆' },
]

/** 记忆面板要显示的长期条目 */
const longTermShown = computed(() =>
  showSuperseded.value ? store.memory.long : store.longTermActive,
)

/** 短期工作区里的摘要条目排前面——它是折叠出来的，值得先看到 */
const shortTermShown = computed(() => {
  const summary = store.shortTermSummary
  const rest = store.memory.short.filter((item) => item.category !== 'summary')
  return [...summary, ...rest]
})

function goalTitle(goalId) {
  return store.goals.find((goal) => goal.id === goalId)?.title ?? '未知目标'
}
</script>

<template>
  <aside class="side">
    <!-- 标签页 -->
    <nav class="tabs" role="tablist">
      <button
        v-for="item in TABS"
        :key="item.key"
        type="button"
        role="tab"
        class="tab"
        :class="{ 'is-active': tab === item.key }"
        :aria-selected="tab === item.key"
        @click="tab = item.key"
      >
        {{ item.label }}
        <span v-if="item.key === 'today' && store.todayPending.length" class="tab-badge">
          {{ store.todayPending.length }}
        </span>
        <span v-if="item.key === 'goals' && store.ready" class="tab-badge">{{ store.goals.length }}</span>
      </button>
    </nav>

    <!-- 今日 -->
    <div v-if="tab === 'today'" class="panel">
      <div class="panel-head">
        <h2>今日待办</h2>
        <span v-if="store.hasPlan" class="panel-sub">
          全部计划 {{ store.progress.done }}/{{ store.progress.total }} 已完成
        </span>
      </div>

      <div v-if="!store.today.length" class="panel-empty">
        <p v-if="!store.hasPlan">还没有计划。</p>
        <p v-else>今天没有排任务。</p>
        <span v-if="!store.hasPlan" class="panel-hint">在左侧说「生成计划」试试</span>
        <span v-else class="panel-hint">其他日期的任务计入全部计划进度</span>
      </div>

      <ul v-else class="task-list">
        <li v-for="item in store.today" :key="item.id" class="task">
          <button
            type="button"
            class="task-check"
            :class="{ 'is-done': item.done }"
            :aria-pressed="item.done"
            :aria-label="`标记「${item.title}」为${item.done ? '未完成' : '已完成'}`"
            @click="store.toggleTask(item.id)"
          >
            <span aria-hidden="true">{{ item.done ? '✓' : '' }}</span>
          </button>
          <div class="task-body">
            <p class="task-title" :class="{ 'is-done': item.done }">{{ item.title }}</p>
            <p class="task-meta">
              {{ item.minutes }} 分钟 · {{ goalTitle(item.goal_id) }}
            </p>
          </div>
        </li>
      </ul>
    </div>

    <!-- 目标 -->
    <div v-else-if="tab === 'goals'" class="panel">
      <div class="panel-head">
        <h2>目标</h2>
        <span class="panel-sub">{{ store.activeGoals.length }} 个进行中</span>
      </div>

      <div v-if="!store.goals.length" class="panel-empty">
        <p>还没有记录目标。</p>
        <span class="panel-hint">在左侧说「我想学 XXX」</span>
      </div>

      <ul v-else class="goal-list">
        <li v-for="goal in store.goals" :key="goal.id" class="goal">
          <div class="goal-head">
            <span class="goal-title">{{ goal.title }}</span>
            <span v-if="goal.status !== 'active'" class="goal-status">{{ goal.status }}</span>
          </div>
          <p v-if="goal.deadline" class="goal-meta">截止 {{ goal.deadline }}</p>
          <div class="goal-bar">
            <span
              class="goal-bar-fill"
              :style="{
                width: `${Math.round(
                  (store.items.filter((i) => i.goal_id === goal.id && i.done).length /
                    Math.max(1, store.items.filter((i) => i.goal_id === goal.id).length)) *
                    100,
                )}%`,
              }"
            ></span>
          </div>
        </li>
      </ul>
    </div>

    <!-- 记忆 -->
    <div v-else class="panel">
      <div class="panel-head memory-heading">
        <div>
          <h2>记忆</h2>
          <span class="panel-sub">短期 {{ store.memoryStats.short ?? 0 }} · 长期 {{ store.memoryStats.long ?? 0 }}</span>
        </div>
        <button type="button" class="mem-btn" @click="store.exportArchive()">
          <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4" /></svg>
          导出档案
        </button>
      </div>

      <form v-if="editing" class="memory-editor" @submit.prevent="saveMemory">
        <h3>纠正记忆</h3>
        <label>内容<textarea v-model="editing.content" required maxlength="1000"></textarea></label>
        <label>每日预算（仅修改时间时填写，分钟）<input v-model="editing.value" type="number" min="15" max="480" /></label>
        <label>适用范围<select v-model="editing.scope"><option v-if="editing.scope.startsWith('date:')" :value="editing.scope">{{ editing.scope.slice(5) }} 当天</option><option value="default">每天默认</option><option value="weekday">工作日</option><option value="weekend">周末</option></select></label>
        <div class="memory-actions">
          <button class="mem-btn" type="button" @click="editing = null">取消</button>
          <button class="mem-btn mem-btn-primary" type="submit">保存纠正</button>
        </div>
      </form>
      <div v-if="deleting" class="memory-editor" role="alert">
        <h3>确认清除影响</h3>
        <p>{{ deleting.preview.effect }}</p>
        <p>关联来源：{{ deleting.preview.source_ids.length }} 条</p>
        <div class="memory-actions">
          <button class="mem-btn" type="button" @click="deleting = null">取消</button>
          <button class="mem-btn mem-btn-danger" type="button" @click="confirmDeletion">确认删除</button>
        </div>
      </div>
      <!-- 长期档案 -->
      <section class="mem-section">
        <h3 class="mem-title">
          长期档案
          <span class="mem-count">{{ longTermShown.length }}</span>
        </h3>
        <p class="mem-note">跨会话保留，靠召回打分被使用</p>

        <div v-if="!longTermShown.length" class="panel-empty panel-empty--tight">
          <p>档案是空的。</p>
        </div>

        <ul v-else class="mem-list">
          <li
            v-for="item in longTermShown"
            :key="item.id"
            :id="`memory-${item.id}`"
            tabindex="-1"
            class="mem"
            :class="{ 'is-superseded': item.superseded }"
          >
            <div class="mem-head">
              <span class="mem-cat">{{ item.category }}</span>
              <span class="mem-mentions">{{ { active: '已生效', pending: '待确认', superseded: '已取代', expired: '已过期' }[item.status] || item.status }}</span>
            </div>
            <p class="mem-content">{{ item.content }}</p>
            <details v-if="item.sources?.length" class="memory-source">
              <summary>查看来源</summary>
              <p v-for="source in item.sources" :key="source.source_id" class="mem-note">
                <template v-if="source.deleted">来源消息已删除</template>
                <template v-else>
                  {{ source.excerpt }}
                  <a v-if="source.conversation_id" class="source-link"
                     :href="`?conversation=${encodeURIComponent(source.conversation_id)}&source=${encodeURIComponent(source.source_id)}`"
                     @click="followSource($event, source)">打开原始对话</a>
                </template>
              </p>
            </details>
            <div class="memory-actions">
              <button v-if="item.status === 'pending'" class="mem-btn mem-btn-primary" type="button" @click="store.updateMemory(item)">确认</button>
              <button v-if="['active','pending'].includes(item.status)" class="mem-btn mem-btn-edit" type="button" @click="editMemory(item)">
                <svg aria-hidden="true" viewBox="0 0 24 24"><path d="m15 5 4 4M4 20l4-1L20 7a2.8 2.8 0 0 0-4-4L4 15z" /></svg>
                纠正
              </button>
              <button class="mem-btn mem-btn-delete" type="button" @click="previewDeletion(item)">
                <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 10v7m4-7v7" /></svg>
                删除
              </button>
            </div>
          </li>
        </ul>

        <button v-if="store.memoryHasMore" class="mem-btn mem-btn-more" type="button" @click="store.loadMoreMemories()">加载更多记忆</button>

        <label v-if="store.longTermSuperseded.length" class="mem-toggle">
          <input v-model="showSuperseded" type="checkbox" />
          显示被取代的旧说法（{{ store.longTermSuperseded.length }}）
        </label>
      </section>

      <!-- 短期工作区 -->
      <section class="mem-section">
        <h3 class="mem-title">
          短期工作区
          <span class="mem-count">{{ shortTermShown.length }}</span>
        </h3>
        <p class="mem-note">会话原文持久保留，发送给模型的上下文按预算整理</p>

        <div v-if="!shortTermShown.length" class="panel-empty panel-empty--tight">
          <p>工作区是空的。</p>
        </div>

        <ul v-else class="mem-list">
          <li
            v-for="item in shortTermShown"
            :key="item.id"
            class="mem"
            :class="{ 'is-summary': item.category === 'summary' }"
          >
            <div class="mem-head">
              <span class="mem-cat">{{ item.category }}</span>
              <span class="mem-mentions">{{ item.role === 'user' ? '用户' : '助手' }}</span>
            </div>
            <p class="mem-content">{{ item.content }}</p>
          </li>
        </ul>
      </section>
    </div>
  </aside>
</template>

<style scoped>
.memory-editor { padding: 12px; border: 1px solid var(--border); border-radius: 10px; margin-bottom: 16px; }
.memory-editor label { display: grid; gap: 5px; margin-bottom: 10px; font-size: 12px; }
.memory-editor textarea, .memory-editor input, .memory-editor select { width: 100%; padding: 6px; border: 1px solid var(--border); border-radius: 5px; }
.panel-head.memory-heading {
  align-items: center;
  flex-wrap: wrap;
}

.memory-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
  margin-top: 8px;
}

.mem-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  min-height: 30px;
  padding: 4px 10px;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-soft);
  font: 550 12px/1.4 var(--font);
  white-space: nowrap;
}

.mem-btn svg {
  width: 14px;
  height: 14px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.mem-btn:hover, .mem-btn-edit {
  background: var(--primary-soft);
  color: var(--primary-strong);
}

.mem-btn-edit, .mem-btn-delete {
  border-color: transparent;
}

.mem-btn-delete {
  background: transparent;
}

.mem-btn-edit:hover {
  border-color: currentColor;
}

.mem-btn-danger, .mem-btn-danger:hover, .mem-btn-delete:hover {
  color: var(--danger);
  background: var(--danger-soft);
  border-color: var(--danger-border);
}

.mem-btn-primary {
  background: var(--primary-strong);
  border-color: transparent;
  color: var(--surface);
}

.mem-btn-primary:hover, .mem-btn-danger:hover {
  border-color: currentColor;
}

.mem-btn-more {
  width: 100%;
  margin-top: 8px;
}

.memory-source summary {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 28px;
  color: var(--text-soft);
  font-size: 12px;
  cursor: pointer;
  list-style: none;
}

.memory-source summary::-webkit-details-marker { display: none; }

.memory-source summary::before {
  content: '';
  width: 6px;
  height: 6px;
  border: solid currentColor;
  border-width: 1.5px 1.5px 0 0;
  transform: rotate(45deg);
}

.memory-source[open] summary::before { transform: rotate(135deg); }
.memory-source summary:hover { color: var(--primary-strong); }
.source-link { display: block; color: var(--primary-strong); text-underline-offset: 2px; }
.memory-source .mem-note {
  margin: 0 0 6px 3px;
  padding-left: 9px;
  border-left: 2px solid var(--border-strong);
  color: var(--text-soft);
  overflow-wrap: anywhere;
}

.side {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  background: var(--surface);
  border-left: 1px solid var(--border);
}

/* 标签页 */
.tabs {
  display: flex;
  gap: 2px;
  padding: 12px 14px 0;
  border-bottom: 1px solid var(--border);
  flex: none;
}

.tab {
  position: relative;
  padding: 8px 12px;
  border: none;
  background: none;
  color: var(--text-mute);
  font-size: 13.5px;
  font-weight: 550;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0;
  transition: color 0.15s, background 0.15s;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.tab:hover {
  color: var(--text-soft);
  background: var(--surface-soft);
}

.tab.is-active {
  color: var(--primary-strong);
  background: var(--primary-soft);
}

.tab-badge {
  min-width: 17px;
  height: 17px;
  padding: 0 5px;
  border-radius: 999px;
  background: var(--primary);
  color: #fff;
  font-size: 11px;
  font-weight: 600;
  display: inline-grid;
  place-items: center;
}

/* 面板 */
.panel {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 16px 14px 20px;
}

.panel-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
}

.panel-head h2 {
  margin: 0;
  font-size: 14px;
  font-weight: 650;
}

.panel-sub {
  font-size: 12px;
  color: var(--text-mute);
  font-variant-numeric: tabular-nums;
}

.panel-empty {
  padding: 26px 10px;
  text-align: center;
}

.panel-empty--tight {
  padding: 14px 10px;
}

.panel-empty p {
  margin: 0 0 4px;
  font-size: 13px;
  color: var(--text-soft);
}

.panel-hint {
  font-size: 12px;
  color: var(--text-mute);
}

/* 待办 */
.task-list,
.goal-list,
.mem-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.task {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  padding: 9px 10px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--surface-soft);
  transition: border-color 0.15s;
}

.task:hover {
  border-color: var(--border-strong);
}

.task-check {
  flex: none;
  width: 18px;
  height: 18px;
  margin-top: 2px;
  padding: 0;
  border-radius: 6px;
  border: 1.5px solid var(--border-strong);
  background: var(--surface);
  color: #fff;
  font-size: 11px;
  line-height: 1;
  display: grid;
  place-items: center;
  transition: background 0.15s, border-color 0.15s;
}

.task-check:hover {
  border-color: var(--primary);
}

.task-check.is-done {
  background: var(--primary);
  border-color: var(--primary);
}

.task-body {
  min-width: 0;
}

.task-title {
  margin: 0;
  font-size: 13.5px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.task-title.is-done {
  color: var(--text-mute);
  text-decoration: line-through;
}

.task-meta {
  margin: 2px 0 0;
  font-size: 11.5px;
  color: var(--text-mute);
}

/* 目标 */
.goal {
  padding: 10px 11px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--surface-soft);
}

.goal-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.goal-title {
  font-size: 13.5px;
  font-weight: 600;
  overflow-wrap: anywhere;
}

.goal-status {
  flex: none;
  padding: 1px 6px;
  border-radius: 5px;
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text-mute);
  font-size: 11px;
}

.goal-meta {
  margin: 3px 0 7px;
  font-size: 11.5px;
  color: var(--text-mute);
}

.goal-bar {
  height: 4px;
  border-radius: 999px;
  background: var(--border);
  overflow: hidden;
}

.goal-bar-fill {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--primary), #7b5cf0);
  transition: width 0.3s ease;
}

/* 记忆 */
.mem-section + .mem-section {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
}

.mem-title {
  margin: 0;
  font-size: 13px;
  font-weight: 650;
  display: flex;
  align-items: center;
  gap: 7px;
}

.mem-count {
  padding: 1px 6px;
  border-radius: 999px;
  background: var(--surface-soft);
  border: 1px solid var(--border);
  color: var(--text-mute);
  font-size: 11px;
  font-weight: 500;
}

.mem-note {
  margin: 3px 0 10px;
  font-size: 11.5px;
  color: var(--text-mute);
  line-height: 1.5;
}

.mem {
  padding: 8px 10px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--surface-soft);
}

.mem.is-summary {
  border-style: dashed;
  background: transparent;
}

.mem.is-superseded {
  opacity: 0.55;
}

.mem.is-superseded .mem-content {
  text-decoration: line-through;
}

.mem-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 3px;
}

.mem-cat {
  padding: 1px 6px;
  border-radius: 5px;
  background: var(--primary-soft);
  color: var(--primary-strong);
  font-size: 10.5px;
  font-weight: 600;
}

.mem-mentions {
  font-size: 11px;
  color: var(--text-mute);
  font-variant-numeric: tabular-nums;
}

.mem-content {
  margin: 0;
  font-size: 12.5px;
  line-height: 1.55;
  color: var(--text-soft);
  overflow-wrap: anywhere;
}

.mem-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 9px;
  font-size: 11.5px;
  color: var(--text-mute);
  cursor: pointer;
}
</style>
