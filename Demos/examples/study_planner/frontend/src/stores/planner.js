/**
 * 工作台的状态中枢（Pinia）。
 *
 * 这个 store 要回答的核心问题是：**哪些状态归服务端，哪些归前端**。
 *
 * 归服务端（后端是唯一真相）：
 *   goals / items / reviews / memory
 *   这些数据在后端有持久化，前端只做镜像。任何改动都要走接口，
 *   拿到新快照后整体替换——不做局部合并，避免两边不一致。
 *
 * 归前端（只存在于这个页面）：
 *   messages / streaming / error
 *   对话记录和流式过程不需要落盘，刷新页面重来即可。
 *
 * 三处容易写错的地方，这里专门处理了：
 *
 * 1. **乐观更新要能回滚**
 *    勾选待办时先改本地再发请求，但失败时必须把状态改回去——
 *    否则界面显示"已完成"、实际没完成，用户再点一次就点不动了。
 *
 * 2. **流式请求可能乱序返回**
 *    快速连发两条消息时，先发的请求可能后到。用请求序号
 *    丢弃过期响应，避免旧回复覆盖新回复。
 *
 * 3. **状态整体替换而不是合并**
 *    后端推来的 state 是全量快照，直接替换。
 *    局部合并（只更新 items 保留 goals）在后端同时改了多个集合时
 *    会留下脏数据。
 */

import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import {
  createGoal as apiCreateGoal,
  fetchState,
  resetAll as apiResetAll,
  rollover as apiRollover,
  streamChat,
  toggleTask as apiToggleTask,
} from '../api'

let seq = 0
const uid = () => `m${++seq}-${Date.now().toString(36)}`

export const usePlannerStore = defineStore('planner', () => {
  // ----------------------------------------------------------
  // 服务端镜像
  // ----------------------------------------------------------

  const goals = ref([])
  const items = ref([])
  const reviews = ref([])
  const memory = ref({ stats: {}, short: [], long: [] })

  // ----------------------------------------------------------
  // 前端本地状态
  // ----------------------------------------------------------

  const messages = ref([])
  /** 流式状态：idle / streaming */
  const streaming = ref(false)
  /** 当前这一轮收到的留痕步骤 */
  const liveSteps = ref([])
  /** 当前这一轮的阶段提示 */
  const liveStatus = ref('')
  /** 当前这一轮模型回复的正文（逐段累积，流式显示用） */
  const liveText = ref('')
  const error = ref('')
  /** 最近一次失败的消息，供"重试"按钮使用 */
  const failedMessage = ref('')
  /** 已加载过一次快照，用来区分"首次加载"和"已就绪" */
  const ready = ref(false)

  let controller = null
  let requestId = 0

  // ----------------------------------------------------------
  // 派生状态：全部用 computed，不在 state 里存冗余副本
  // ----------------------------------------------------------

  const today = computed(() => {
    const day = new Date().toLocaleDateString('sv-SE')
    return items.value.filter((item) => item.day === day)
  })

  const todayPending = computed(() => today.value.filter((item) => !item.done))

  const activeGoals = computed(() => goals.value.filter((goal) => goal.status === 'active'))

  /** 整体进度：已完成条数 / 总条数 */
  const progress = computed(() => {
    const total = items.value.length
    const done = items.value.filter((item) => item.done).length
    return { done, total, ratio: total ? done / total : 0 }
  })

  /** 长期记忆：有效条目与被取代的旧说法分开 */
  const longTermActive = computed(() => memory.value.long.filter((item) => !item.superseded))
  const longTermSuperseded = computed(() => memory.value.long.filter((item) => item.superseded))

  /** 短期工作区里的摘要条目——它是折叠出来的，不是普通笔记 */
  const shortTermSummary = computed(() =>
    memory.value.short.filter((item) => item.category === 'summary'),
  )

  const memoryStats = computed(() => memory.value.stats ?? {})

  /** 有没有已排的任务——决定"今日"面板显示什么 */
  const hasPlan = computed(() => items.value.length > 0)

  // ----------------------------------------------------------
  // 动作
  // ----------------------------------------------------------

  /**
   * 用后端快照整体替换本地镜像。
   * @param {object} snapshot 后端返回的状态
   */
  function applySnapshot(snapshot) {
    if (!snapshot) return
    goals.value = snapshot.goals ?? []
    items.value = snapshot.items ?? []
    reviews.value = snapshot.reviews ?? []
    memory.value = snapshot.memory ?? { stats: {}, short: [], long: [] }
    ready.value = true
  }

  /** 拉取全量状态。 */
  async function loadState() {
    try {
      applySnapshot(await fetchState())
      error.value = ''
    } catch (err) {
      error.value = err?.message ?? '加载状态失败'
    }
  }

  /**
   * 发一条消息，走 SSE。
   *
   * @param {string} text 用户输入
   */
  async function send(text) {
    const content = text.trim()
    if (!content || streaming.value) return

    error.value = ''
    failedMessage.value = ''
    messages.value.push({ id: uid(), role: 'user', content, time: clock() })

    const mine = ++requestId
    controller = new AbortController()
    streaming.value = true
    liveSteps.value = []
    liveStatus.value = ''
    liveText.value = ''

    try {
      await streamChat(
        content,
        (frame) => {
          // 过期响应直接丢弃：先发的请求可能后到，不能让它覆盖新回复
          if (mine !== requestId) return

          if (frame.event === 'status') {
            liveStatus.value = frame.data.text ?? ''
          } else if (frame.event === 'step') {
            liveSteps.value.push(frame.data)
          } else if (frame.event === 'delta') {
            // 模型回复的文本增量：先累积到 liveText 里流式显示，
            // 最终消息以 reply 事件为准（那是完整正文，权威版本）
            liveText.value += frame.data.text ?? ''
          } else if (frame.event === 'reply') {
            messages.value.push({
              id: uid(),
              role: 'assistant',
              content: frame.data.text ?? '',
              steps: liveSteps.value.slice(),
              time: clock(),
            })
          } else if (frame.event === 'state') {
            applySnapshot(frame.data)
          } else if (frame.event === 'error') {
            error.value = frame.data.message ?? '处理失败'
          }
        },
        controller.signal,
      )
    } catch (err) {
      if (err?.name === 'AbortError') {
        // 用户主动中断，不算错误
      } else {
        error.value = err?.message ?? '未知错误'
        failedMessage.value = content
      }
    } finally {
      if (mine === requestId) {
        streaming.value = false
        liveSteps.value = []
        liveStatus.value = ''
        liveText.value = ''
        controller = null
      }
    }
  }

  /** 中断当前流式请求。 */
  function stop() {
    controller?.abort()
    requestId += 1
    streaming.value = false
    liveSteps.value = []
    liveStatus.value = ''
    liveText.value = ''
  }

  /**
   * 切换待办完成状态——乐观更新 + 失败回滚。
   *
   * @param {string} taskId 任务 id
   */
  async function toggleTask(taskId) {
    const target = items.value.find((item) => item.id === taskId)
    if (!target) return

    const previous = target.done
    target.done = !previous // 乐观：先翻转，界面立刻响应

    try {
      applySnapshot(await apiToggleTask(taskId, target.done))
      error.value = ''
    } catch (err) {
      target.done = previous // 回滚：不回滚的话界面会和服务端不一致
      error.value = err?.message ?? '更新失败'
    }
  }

  /**
   * 创建目标并排计划。
   *
   * @param {{title: string, deadline?: string, daily_minutes?: number}} payload 目标参数
   */
  async function createGoal(payload) {
    try {
      applySnapshot(await apiCreateGoal(payload))
      error.value = ''
      return true
    } catch (err) {
      error.value = err?.message ?? '创建目标失败'
      return false
    }
  }

  /** 模拟跨天。 */
  async function rollover() {
    try {
      applySnapshot(await apiRollover())
      error.value = ''
    } catch (err) {
      error.value = err?.message ?? '跨天失败'
    }
  }

  /** 清空全部数据。 */
  async function resetAll() {
    try {
      applySnapshot(await apiResetAll())
      messages.value = []
      error.value = ''
    } catch (err) {
      error.value = err?.message ?? '重置失败'
    }
  }

  /** 重试上一条失败的消息。 */
  function retry() {
    const text = failedMessage.value
    if (!text) return
    failedMessage.value = ''
    send(text)
  }

  /** 清空本地对话记录（不动服务端数据）。 */
  function clearChat() {
    messages.value = []
    failedMessage.value = ''
    error.value = ''
  }

  return {
    // 服务端镜像
    goals,
    items,
    reviews,
    memory,
    // 前端本地
    messages,
    streaming,
    liveSteps,
    liveStatus,
    liveText,
    error,
    failedMessage,
    ready,
    // 派生
    today,
    todayPending,
    activeGoals,
    progress,
    longTermActive,
    longTermSuperseded,
    shortTermSummary,
    memoryStats,
    hasPlan,
    // 动作
    loadState,
    send,
    stop,
    toggleTask,
    createGoal,
    rollover,
    resetAll,
    retry,
    clearChat,
  }
})

/** 当前时间，秒级。 */
function clock() {
  return new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}
