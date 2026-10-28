# Agent 聊天工作台（前端）

Vue 3 + Vite 实现的 RAG Demo 前端，对接后端 `POST /chat`。

## 启动

```bash
cd Demos/examples/rag-demo/frontend
npm install
npm run dev      # http://localhost:5173
```

后端需另行启动（在 `Demos/` 目录下）：

```bash
uv run uvicorn examples.rag-demo.backend.main:app --reload --port 8000
```

## 功能

- 消息气泡：用户消息靠右、Agent 消息靠左，错误消息独立配色
- 输入区：自动增高文本框、`Enter` 发送 / `Shift + Enter` 换行、字数统计
- 加载态：三点动画 + 思考计时，发送期间禁用输入
- 其他：空状态快捷指令、清空对话、复制回复、失败重试、自动滚动到底部

## 配置

复制 `.env.example` 为 `.env` 后可覆盖后端地址：

```
VITE_API_BASE=http://localhost:8000
```

后端 CORS 已放行 `http://localhost:5173`。

## 构建

```bash
npm run build     # 产物在 dist/
npm run preview   # 本地预览构建产物
```
