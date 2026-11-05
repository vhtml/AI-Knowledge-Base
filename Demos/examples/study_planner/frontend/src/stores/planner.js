/** 服务端为唯一真相；用持久请求标识恢复断线操作。 */
import { defineStore } from 'pinia'
import { computed, ref, onScopeDispose } from 'vue'
import { apiJSON, createGoal as apiCreateGoal, fetchState, rollover as apiRollover, resetAll as apiResetAll, streamChat, toggleTask as apiToggleTask } from '../api'

const pendingKey = 'study-planner-pending'
const conversationKey = 'study-planner-conversation'
const clock = (value) => new Date(value ?? Date.now()).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })

export const usePlannerStore = defineStore('planner', () => {
  const goals = ref([]), items = ref([]), reviews = ref([])
  const memory = ref({ stats: {}, short: [], long: [] })
  const messages = ref([]), streaming = ref(false), liveSteps = ref([]), liveStatus = ref(''), liveText = ref('')
  const error = ref(''), failedMessage = ref(''), ready = ref(false)
  const conversationId = ref(localStorage.getItem(conversationKey) || 'default')
  const switching = ref(false)
  const sourceMessageId = ref('')
  const privacyRevision = ref(-1), timezone = ref('Asia/Shanghai'), now = ref(Date.now())
  const timer = setInterval(() => { now.value = Date.now() }, 60000)
  onScopeDispose(() => clearInterval(timer))
  let historyGeneration = 0
  const revision = ref(-1), memoryHasMore = ref(false), historyHasMore = ref(false)
  let controller = null, activeTurn = '', generation = 0
  let pending = null
  try { pending = JSON.parse(localStorage.getItem(pendingKey)) } catch { localStorage.removeItem(pendingKey) }

  const today = computed(() => items.value.filter((i) => i.day === new Date(now.value).toLocaleDateString('sv-SE', { timeZone: timezone.value })))
  const todayPending = computed(() => today.value.filter((i) => !i.done))
  const activeGoals = computed(() => goals.value.filter((i) => i.status === 'active'))
  const progress = computed(() => ({ done: items.value.filter((i) => i.done).length, total: items.value.length }))
  const longTermActive = computed(() => memory.value.long.filter((i) => !['superseded', 'deleted', 'expired'].includes(i.status)))
  const longTermSuperseded = computed(() => memory.value.long.filter((i) => i.status === 'superseded'))
  const shortTermSummary = computed(() => memory.value.short.filter((i) => i.category === 'summary'))
  const memoryStats = computed(() => memory.value.stats ?? {})
  const hasPlan = computed(() => items.value.length > 0)

  function applySnapshot(snapshot) {
    if (!snapshot || (snapshot.state_revision ?? 0) < revision.value || (snapshot.privacy_revision ?? 0) < privacyRevision.value) return
    privacyRevision.value = snapshot.privacy_revision ?? 0
    timezone.value = snapshot.timezone ?? timezone.value
    revision.value = snapshot.state_revision ?? 0
    goals.value = snapshot.goals ?? []; items.value = snapshot.items ?? []; reviews.value = snapshot.reviews ?? []
    memory.value = snapshot.memory ?? { stats: {}, short: [], long: [] }
    memoryHasMore.value = memory.value.has_more ?? false
    ready.value = true
  }

  function savePending(value) {
    pending = value
    if (value) localStorage.setItem(pendingKey, JSON.stringify(value))
    else localStorage.removeItem(pendingKey)
  }

  async function loadHistory(older = false) {
    const ticket = ++historyGeneration, conversation = conversationId.value
    const before = older && messages.value.length ? `&before_seq=${messages.value[0].seq}` : ''
    const data = await apiJSON(`/api/conversations/${encodeURIComponent(conversationId.value)}/messages?limit=50${before}`)
    if (ticket !== historyGeneration || conversation !== conversationId.value || data.privacy_revision < privacyRevision.value) return
    privacyRevision.value = data.privacy_revision
    const rows = data.messages.map((m) => ({ ...m, time: clock(m.created_at) }))
    messages.value = older ? [...rows, ...messages.value] : rows
    if (!older) sourceMessageId.value = ''
    historyHasMore.value = data.has_more
  }

  async function loadState() {
    try {
      const id = conversationId.value
      const snapshot = await fetchState(id)
      if (id !== conversationId.value) return
      applySnapshot(snapshot)
      await loadHistory()
      error.value = ''
      if (pending && !streaming.value) await recoverPending()
    } catch (err) { error.value = err.message ?? '加载失败' }
  }

  async function recoverPending() {
    if (!pending) return
    let result
    try {
      result = activeTurn
      ? await apiJSON(`/api/turns/${encodeURIComponent(activeTurn)}`)
      : await apiJSON(`/api/requests/${encodeURIComponent(pending.request_id)}`)
    } catch (err) {
      if (err.status === 404) {
        failedMessage.value = pending.message
        error.value = '请求尚未被接收，点击重试将复用原请求标识'
        return
      }
      throw err
    }
    if (['completed', 'cancelled', 'failed'].includes(result.status)) {
      if (result.status === 'failed') { failedMessage.value = pending.message; error.value = result.error_code || '处理失败' }
      savePending(null)
      applySnapshot(await fetchState(conversationId.value))
      await loadHistory()
    } else {
      activeTurn = result.turn_id
      error.value = '操作仍在处理中，正在查询结果'
      setTimeout(() => { if (pending && !streaming.value) recoverPending().catch((err) => { error.value = err.message }) }, 1000)
    }
  }

  async function send(text) {
    const content = text.trim()
    if (!content || streaming.value || switching.value) return
    if (sourceMessageId.value) {
      try { await switchConversation(conversationId.value) }
      catch (err) { error.value = err.message; return }
    }
    if (pending && pending.message !== content) {
      try { await recoverPending() } catch (err) { error.value = err.message; return }
      if (pending) return
    }
    error.value = ''; failedMessage.value = ''
    if (!pending) savePending({ message: content, request_id: crypto.randomUUID(), conversation_id: conversationId.value })
    if (!messages.value.some((m) => m.id === pending.request_id)) messages.value.push({ id: pending.request_id, role: 'user', content, time: clock() })
    const mine = ++generation
    activeTurn = ''; controller = new AbortController(); streaming.value = true
    liveSteps.value = []; liveText.value = ''; liveStatus.value = ''
    try {
      await streamChat(content, (frame) => {
        if (mine !== generation) return
        if (frame.event === 'operation') activeTurn = frame.data.turn_id
        else if (frame.event === 'status') liveStatus.value = frame.data.text ?? ''
        else if (frame.event === 'step') liveSteps.value.push(frame.data)
        else if (frame.event === 'delta') liveText.value += frame.data.text ?? ''
        else if (frame.event === 'state') applySnapshot(frame.data)
        else if (frame.event === 'done') {
          if (['completed', 'failed', 'cancelled'].includes(frame.data.status)) savePending(null)
          if (frame.data.status === 'failed') error.value = frame.data.error_code
        } else if (frame.event === 'error') error.value = frame.data.message ?? '处理失败'
      }, controller.signal, pending)
      if (pending) await recoverPending()
    } catch (err) {
      if (err.name !== 'AbortError') { error.value = err.message ?? '处理失败'; failedMessage.value = content }
    } finally {
      if (mine === generation) {
        streaming.value = false; liveText.value = ''; liveStatus.value = ''; controller = null
        try { await loadHistory() } catch { /* 待网络恢复后重新加载 */ }
      }
    }
  }

  async function stop() {
    try {
      if (!activeTurn && pending) activeTurn = (await apiJSON(`/api/requests/${encodeURIComponent(pending.request_id)}`)).turn_id
      if (activeTurn) {
        const result = await apiJSON(`/api/turns/${encodeURIComponent(activeTurn)}/cancel`, 'POST')
        error.value = result.status === 'cancelled' ? '已在提交前取消' : '变更已保存，停止接收回复不会撤销操作'
      }
    } catch (err) { error.value = err.message }
    controller?.abort(); generation += 1; streaming.value = false; liveText.value = ''
    if (pending) { try { await recoverPending() } catch { /* 保留请求标识供恢复 */ } }
  }

  async function change(action) {
    try { applySnapshot(await action()); error.value = ''; return true }
    catch (err) { error.value = err.message; return false }
  }
  async function toggleTask(id) {
    const target = items.value.find((i) => i.id === id)
    if (target) return change(() => apiToggleTask(id, !target.done, conversationId.value))
  }
  const createGoal = (payload) => change(() => apiCreateGoal(payload, conversationId.value))
  const deleteGoal = (id) => change(() => apiJSON(`/api/goals/${encodeURIComponent(id)}`, 'DELETE', {
    conversation_id: conversationId.value, request_id: crypto.randomUUID(),
  }))
  const rollover = () => change(() => apiRollover(conversationId.value))
  async function resetAll() { if (await change(() => apiResetAll(conversationId.value))) { savePending(null); await loadHistory() } }
  function retry() { if (failedMessage.value) return send(failedMessage.value) }
  async function switchConversation(id, source = '') {
    if (streaming.value || pending || switching.value) throw new Error('请先完成当前操作，再切换会话')
    switching.value = true
    try {
      const [snapshot, history] = await Promise.all([
        fetchState(id),
        apiJSON(`/api/conversations/${encodeURIComponent(id)}/messages?limit=50${source ? '&source=' + encodeURIComponent(source) : ''}`),
      ])
      if (history.privacy_revision !== snapshot.privacy_revision ||
          snapshot.privacy_revision < privacyRevision.value || snapshot.state_revision < revision.value) {
        throw new Error('档案刚刚更新，请重新打开会话')
      }
      ++historyGeneration
      conversationId.value = id
      localStorage.setItem(conversationKey, id)
      applySnapshot(snapshot)
      sourceMessageId.value = source
      privacyRevision.value = history.privacy_revision
      messages.value = history.messages.map((message) => ({ ...message, time: clock(message.created_at) }))
      historyHasMore.value = history.has_more
      activeTurn = ''; failedMessage.value = ''; error.value = ''
    } finally { switching.value = false }
  }

  function clearChat() {
    if (streaming.value || pending || switching.value) { error.value = '请先停止或确认当前操作结果'; return }
    conversationId.value = crypto.randomUUID(); localStorage.setItem(conversationKey, conversationId.value)
    messages.value = []; historyHasMore.value = false; error.value = ''
    sourceMessageId.value = ''
  }
  async function deleteConversation(id, preview) {
    if (pending || streaming.value || switching.value) throw new Error('请先结束当前操作，再删除会话')
    switching.value = true
    try {
      await apiJSON(`/api/conversations/${encodeURIComponent(id)}`, 'DELETE', {
        last_seq: preview.last_seq, revision: preview.revision,
      })
      ++historyGeneration
      const wasCurrent = conversationId.value === id
      if (wasCurrent) {
        conversationId.value = crypto.randomUUID()
        localStorage.setItem(conversationKey, conversationId.value)
        messages.value = []; historyHasMore.value = false; sourceMessageId.value = ''
        activeTurn = ''; failedMessage.value = ''
      }
      // 即使刷新暂时断网，也不能继续展示刚刚清除的原文摘录。
      memory.value = { stats: {}, short: [], long: [] }
      if (wasCurrent) await loadState()
      else {
        try { applySnapshot(await fetchState(conversationId.value)) }
        catch (err) { error.value = err.message }
      }
    } finally { switching.value = false }
  }
  async function loadMoreMemories() {
    try {
      const data = await apiJSON(`/api/memories?offset=${memory.value.long.length}`)
      if (data.state_revision !== revision.value) { await loadState(); return }
      memory.value.long.push(...data.items); memoryHasMore.value = data.has_more
    } catch (err) { error.value = err.message }
  }
  async function updateMemory(item, changes = {}) {
    const ok = await change(() => apiJSON(`/api/memories/${encodeURIComponent(item.id)}`, 'PATCH', { conversation_id: conversationId.value, expected_revision: item.revision, request_id: crypto.randomUUID(), ...changes }))
    if (ok) await loadHistory()
    return ok
  }
  async function previewDeletion(item) { return apiJSON(`/api/memories/${encodeURIComponent(item.id)}/deletion-preview`) }
  async function deleteMemory(item) {
    const ok = await change(() => apiJSON(`/api/memories/${encodeURIComponent(item.id)}`, 'DELETE', { conversation_id: conversationId.value, expected_revision: item.revision, request_id: crypto.randomUUID() }))
    if (ok) await loadHistory()
    return ok
  }
  async function exportArchive() {
    try {
      const data = await apiJSON('/api/export')
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }))
      const link = document.createElement('a')
      link.href = url; link.download = 'study-planner-export.json'; link.click()
      URL.revokeObjectURL(url)
    } catch (err) { error.value = err.message }
  }
  return { conversationId, switchConversation, switching, sourceMessageId, goals, items, reviews, memory, messages, streaming, liveSteps, liveStatus, liveText, error, failedMessage, ready,
    today, todayPending, activeGoals, progress, longTermActive, longTermSuperseded, shortTermSummary, memoryStats, hasPlan,
    memoryHasMore, historyHasMore, loadState, send, stop, toggleTask, createGoal, deleteGoal, rollover, resetAll, retry, clearChat,
    exportArchive, loadMoreMemories, loadHistory, updateMemory, previewDeletion, deleteMemory, deleteConversation, recoverPending }
})
