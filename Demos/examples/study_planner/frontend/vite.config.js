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
    // 开发期把 /api 代理到后端，前端和后端同源，
    // 就不必纠结端口号和 CORS，也不会和别的示例抢 8000 端口
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: process.env.VITE_API_BASE ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
