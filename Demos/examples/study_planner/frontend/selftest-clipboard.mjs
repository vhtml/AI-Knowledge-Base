/** 验证受限剪贴板回退与清理，防止失败时误报成功。 */
import assert from 'node:assert/strict'
import test from 'node:test'
import { copyText } from './src/clipboard.js'

function legacyBrowser(result = true) {
  const state = { copied: null, nodes: [], selectionRestored: false, focused: false, caret: null }
  const range = {}
  const active = { selectionStart: 2, selectionEnd: 4, selectionDirection: 'backward',
    focus() { state.focused = true }, setSelectionRange(...args) { state.caret = args } }
  const document = {
    activeElement: active,
    getSelection: () => ({ rangeCount: 1, getRangeAt: () => ({ cloneRange: () => range }),
      removeAllRanges() {}, addRange(value) { state.selectionRestored = value === range } }),
    body: { appendChild(node) { state.nodes.push(node) } },
    createElement: () => ({ style: {}, setAttribute() {}, focus() {}, select() {}, setSelectionRange() {},
      remove() { state.nodes.pop() } }),
    execCommand(command) {
      assert.equal(command, 'copy')
      if (result instanceof Error) throw result
      if (result) state.copied = state.nodes.at(-1).value
      return result
    },
  }
  return { document, state }
}

test('现代接口成功时直接复制原文', async () => {
  let copied
  await copyText('中文\n第二行', { navigator: { clipboard: { async writeText(value) { copied = value } } } })
  assert.equal(copied, '中文\n第二行')
})

test('未提供 Clipboard API 时回退，保留中文、换行与输入位置', async () => {
  const browser = legacyBrowser()
  await copyText('中文\n  缩进', browser)
  assert.equal(browser.state.copied, '中文\n  缩进')
  assert.equal(browser.state.nodes.length, 0)
  assert.equal(browser.state.focused, true)
  assert.equal(browser.state.selectionRestored, true)
  assert.deepEqual(browser.state.caret, [2, 4, 'backward'])
})

test('现代接口拒绝权限时仍尝试回退', async () => {
  const browser = legacyBrowser()
  browser.navigator = { clipboard: { async writeText() { throw new Error('NotAllowedError') } } }
  await copyText('回复内容', browser)
  assert.equal(browser.state.copied, '回复内容')
})

for (const result of [false, new Error('copy blocked')]) {
  test(`传统复制${result === false ? '返回失败' : '抛异常'}时不误报成功，并恢复界面`, async () => {
    const browser = legacyBrowser(result)
    await assert.rejects(copyText('内容', browser))
    assert.equal(browser.state.copied, null)
    assert.equal(browser.state.nodes.length, 0)
    assert.equal(browser.state.focused, true)
  })
}

test('所有复制能力均不可用时显式失败', async () => {
  await assert.rejects(copyText('内容', {}), /Clipboard unavailable/)
})
