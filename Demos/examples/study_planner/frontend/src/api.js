/**
 * 后端 API 客户端。
 *
 * 两类接口：
 * - `fetchState()` / `toggleTask()` 等是普通 REST，一次拿全量
 * - `streamChat()` 是 SSE，走 POST + ReadableStream
 *
 * 为什么 SSE 要手写解析而不用 EventSource：
 * EventSource 只支持 GET 且不能带 body，而聊天请求要 POST 一段文本。
 * 所以这里用 fetch 拿流，自己按 SSE 规范切帧。
 */

/**
 * 后端地址。
 *
 * 开发时走 Vite 代理（同源），所以用相对路径即可——
 * 不用操心后端在哪个端口，也不会和别的示例抢端口。
 * 生产构建没有代理，这时才需要显式指定 VITE_API_BASE。
 */
export const API_BASE = import.meta.env.DEV
  ? ''
  : (import.meta.env.VITE_API_BASE ?? 'http://localhost:8000').replace(/\/+$/, '')

/** 与后端 ChatRequest 的 max_length 保持一致 */
export const MAX_LENGTH = 4000

/** 后端实际地址，用于错误提示（开发时是代理目标，不是页面地址） */
const BACKEND_HINT = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

/**
 * 统一的 fetch 包装：把网络错误翻译成人能看懂的话。
 *
 * 浏览器的原生报错是 "Failed to fetch"，用户看了不知道该做什么。
 * 这里统一成"连不上后端，请确认 FastAPI 已启动"并带上实际地址。
 *
 * @param {string} url 请求地址
 * @param {object} [options] fetch 选项
 * @returns {Promise<Response>} 原始响应
 */
async function request(url, options) {
  let res
  try {
    res = await fetch(url, options)
  } catch (err) {
    if (err?.name === 'AbortError') throw err
    throw new Error(`无法连接后端服务（${BACKEND_HINT}），请确认 FastAPI 已启动`)
  }

  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.json())?.detail ?? ''
    } catch {
      /* 响应体不是 JSON，忽略 */
    }
    const error = new Error(detail ? `请求失败（${res.status}）：${detail}` : `请求失败（${res.status}）`)
    error.status = res.status
    throw error
  }

  return res
}

/**
 * 拉取全量状态快照。
 * @returns {Promise<object>} 后端的状态对象
 */
export async function fetchState(conversationId = "default") {
  const res = await request(`${API_BASE}/api/state?conversation_id=${encodeURIComponent(conversationId)}`)
  return res.json()
}

/**
 * 切换某条待办的完成状态。
 * @param {string} taskId 任务 id
 * @param {boolean} done 是否标记为已完成
 * @returns {Promise<object>} 更新后的状态快照
 */
export async function toggleTask(taskId, done, conversationId = 'default') {
  const res = await request(`${API_BASE}/api/tasks/${encodeURIComponent(taskId)}/toggle`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ done, conversation_id: conversationId, request_id: crypto.randomUUID() }),
  })
  return res.json()
}

/**
 * 创建一个目标并自动排计划。
 * @param {{title: string, deadline?: string, daily_minutes?: number}} payload 目标参数
 * @returns {Promise<object>} 更新后的状态快照
 */
export async function createGoal(payload, conversationId = 'default') {
  const res = await request(`${API_BASE}/api/goals`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversation_id: conversationId, request_id: crypto.randomUUID(), ...payload }),
  })
  return res.json()
}

/**
 * 模拟跨天：清空短期工作区，值得沉淀的进长期档案。
 * @returns {Promise<object>} 更新后的状态快照
 */
export async function rollover(conversationId = 'default') {
  const res = await request(`${API_BASE}/api/rollover`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ conversation_id: conversationId, request_id: crypto.randomUUID() }) })
  return res.json()
}

/**
 * 清空全部数据。
 * @returns {Promise<object>} 更新后的状态快照
 */
export async function resetAll(conversationId = 'default') {
  const res = await request(`${API_BASE}/api/reset`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ conversation_id: conversationId, request_id: crypto.randomUUID() }) })
  return res.json()
}

/**
 * 把一段 SSE 文本流切成一个个事件。
 *
 * SSE 帧的格式是：
 *   event: <名字>\n
 *   data: <JSON>\n
 *   \n            ← 空行表示一帧结束
 *
 * 网络分片会把一帧切成好几块，所以要按空行切帧、
 * 把不完整的尾巴留在缓冲区里等下一块。
 *
 * @param {string} buffer 已累积的文本
 * @returns {{frames: Array<{event: string, data: object}>, rest: string}}
 */
export function parseSSE(buffer) {
  const frames = []
  let rest = buffer

  // \r\n\r\n 和 \n\n 都是合法的帧分隔
  const parts = rest.split(/\r?\n\r?\n/)
  rest = parts.pop() ?? ''

  for (const part of parts) {
    if (!part.trim()) continue

    let event = 'message'
    const dataLines = []

    for (const line of part.split(/\r?\n/)) {
      if (line.startsWith('event:')) {
        event = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim())
      }
    }

    if (!dataLines.length) continue

    try {
      frames.push({ event, data: JSON.parse(dataLines.join('\n')) })
    } catch {
      /* 数据不是合法 JSON，跳过这一帧 */
    }
  }

  return { frames, rest }
}

/**
 * 发起一次流式对话。
 *
 * @param {string} message 用户输入
 * @param {(frame: {event: string, data: object}) => void} onFrame 每收到一帧的回调
 * @param {AbortSignal} [signal] 中止信号
 * @returns {Promise<void>} 流结束时 resolve
 */
export async function streamChat(message, onFrame, signal, metadata = {}) {
  let res
  try {
    res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, ...metadata }),
      signal,
    })
  } catch (err) {
    if (err?.name === 'AbortError') throw err
    throw new Error(`无法连接后端服务（${BACKEND_HINT}），请确认 FastAPI 已启动`)
  }

  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.json())?.detail ?? ''
    } catch {
      /* 响应体不是 JSON，忽略 */
    }
    const error = new Error(detail ? `请求失败（${res.status}）：${detail}` : `请求失败（${res.status}）`)
    error.status = res.status
    throw error
  }

  if (!res.body) throw new Error('响应缺少可读流')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()

    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const { frames, rest } = parseSSE(buffer)
    buffer = rest

    for (const frame of frames) onFrame(frame)
  }
}


export async function apiJSON(path, method = 'GET', body) {
  const res = await request(`${API_BASE}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  return res.json()
}
