/** 用真实 Vue/Pinia 加可控 HTTP 响应验证恢复与隐私竞态。 */
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import vm from 'node:vm'
import * as vue from 'vue'
import * as pinia from 'pinia'

const source = await readFile(new URL('./src/stores/planner.js', import.meta.url), 'utf8')
const snapshot = { goals: [], items: [], reviews: [], memory: { short: [], long: [], stats: {} }, state_revision: 0, privacy_revision: 0 }
async function createStore(api, initial = {}) {
  const storage = new Map(Object.entries(initial))
  const context = vm.createContext({ console, Date, JSON, setInterval: () => 1, clearInterval: () => {}, setTimeout: () => 1,
    AbortController, crypto: { randomUUID: () => 'new-key' }, localStorage: {
      getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: (key) => storage.delete(key),
    } })
  const defaults = { fetchState: async () => snapshot, createGoal: async () => snapshot, resetAll: async () => snapshot,
    toggleTask: async () => snapshot, streamChat: async () => {}, apiJSON: async () => ({ messages: [], privacy_revision: 0 }) }
  const module = new vm.SourceTextModule(source, { context })
  await module.link(async (name) => {
    const exports = name === 'vue' ? vue : name === 'pinia' ? pinia : { ...defaults, ...api }
    return new vm.SyntheticModule(Object.keys(exports), function () { for (const [key, value] of Object.entries(exports)) this.setExport(key, value) }, { context })
  })
  await module.evaluate()
  pinia.setActivePinia(pinia.createPinia())
  return module.namespace.usePlannerStore()
}

const pending = { message: '我想学 Python', request_id: 'original-key', conversation_id: 'default' }
let submitted
const recovery = await createStore({
  apiJSON: async (path) => {
    if (path.startsWith('/api/requests/')) throw Object.assign(new Error('not received'), { status: 404 })
    return { messages: [], privacy_revision: 0 }
  },
  streamChat: async (text, onFrame, signal, metadata) => {
    submitted = metadata.request_id
    onFrame({ event: 'done', data: { status: 'completed' } })
  },
}, { 'study-planner-pending': JSON.stringify(pending) })
await recovery.loadState()
assert.equal(recovery.failedMessage, pending.message)
await recovery.send('另一条消息')
assert.match(recovery.error, /重试/)
await recovery.retry()
assert.equal(submitted, 'original-key')
recovery.$dispose()

const responses = []
const history = await createStore({ apiJSON: () => new Promise((resolve) => responses.push(resolve)) })
const old = history.loadHistory()
const fresh = history.loadHistory()
responses[1]({ messages: [], privacy_revision: 1 })
await fresh
responses[0]({ messages: [{ id: 'secret', content: '已删除的秘密', role: 'user' }], privacy_revision: 0 })
await old
assert.equal(history.messages.length, 0)
history.$dispose()
console.log('store 自检通过：404 可同键重试；旧历史不能恢复已删除正文')

let continuation
const switching = await createStore({
  apiJSON: async (path) => {
    if (!path.includes('/session-b/messages')) throw new Error('读取失败')
    return { messages: [{ id: 'previous', content: '之前的消息', role: 'user' }], privacy_revision: 0 }
  },
  streamChat: async (_text, onFrame, _signal, metadata) => {
    continuation = metadata.conversation_id
    onFrame({ event: 'done', data: { status: 'completed' } })
  },
})
await switching.switchConversation('session-b')
assert.equal(switching.messages[0].content, '之前的消息')
await assert.rejects(switching.switchConversation('missing'), /读取失败/)
assert.equal(switching.conversationId, 'session-b')
await switching.send('继续聊')
assert.equal(continuation, 'session-b')
switching.$dispose()
console.log('会话切换通过：恢复旧消息、加载失败保留原会话、续聊使用选中的会话 ID')
