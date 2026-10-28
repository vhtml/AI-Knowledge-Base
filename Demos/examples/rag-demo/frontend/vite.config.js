import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    // 显式绑定 IPv4，避免仅监听 ::1 时浏览器无法通过 localhost 访问
    host: '127.0.0.1',
    port: 5173,
    strictPort: false,
  },
})
