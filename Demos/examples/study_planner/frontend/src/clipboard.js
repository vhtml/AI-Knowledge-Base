/** 先使用剪贴板接口，受限时回退到用户点击触发的传统复制。 */
export async function copyText(text, browser = globalThis) {
  try {
    if (typeof browser.navigator?.clipboard?.writeText === 'function') {
      await browser.navigator.clipboard.writeText(text)
      return
    }
  } catch {
    // 内嵌浏览器可能暴露接口却拒绝写入，仍应尝试传统复制。
  }

  const document = browser.document
  if (!document?.body || typeof document.execCommand !== 'function') {
    throw new Error('Clipboard unavailable')
  }
  const active = document.activeElement
  const selection = document.getSelection?.()
  const ranges = selection ? Array.from({ length: selection.rangeCount }, (_, i) => selection.getRangeAt(i).cloneRange()) : []
  const inputSelection = active && typeof active.selectionStart === 'number'
    ? [active.selectionStart, active.selectionEnd, active.selectionDirection]
    : null
  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.setAttribute('readonly', '')
  textarea.style.cssText = 'position:fixed;top:0;left:-9999px;font-size:16px;'
  try {
    document.body.appendChild(textarea)
    textarea.focus({ preventScroll: true })
    textarea.select()
    textarea.setSelectionRange(0, text.length)
    if (!document.execCommand('copy')) throw new Error('Copy rejected')
  } finally {
    textarea.remove()
    // 不让回退复制改变输入框焦点、光标或用户原来的选区。
    try {
      active?.focus({ preventScroll: true })
      if (inputSelection) active.setSelectionRange(...inputSelection)
      if (selection) {
        selection.removeAllRanges()
        ranges.forEach((range) => selection.addRange(range))
      }
    } catch { /* 原节点可能已被界面更新移除 */ }
  }
}
