# study_planner 前端

学习规划师工作台。Vue 3 + Vite + Pinia。

## 跑起来

后端要先起来（另开一个终端）：

```bash
cd Demos
uv run uvicorn examples.study_planner.backend.main:app --reload --port 8000
```

然后：

```bash
cd Demos/examples/study_planner/frontend
npm install
npm run dev
```

打开 http://localhost:5173。

## 配置

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `VITE_API_BASE` | 后端地址 | 开发期走 Vite 代理（`/api` → `http://127.0.0.1:8000`） |

开发时前后端同源，不存在跨域问题，也不用关心端口号。
后端换了端口就带上环境变量：

```bash
VITE_API_BASE=http://127.0.0.1:9000 npm run dev
```

生产构建没有代理，这时必须显式指定 `VITE_API_BASE`。

## 结构

```
src/
├── api.js            API 客户端
│                      · REST 调用统一走 request()，把网络错误翻译成人话
│                      · parseSSE() 按 SSE 规范切帧（EventSource 不支持 POST）
├── stores/planner.js Pinia 状态中枢
│                      · 服务端镜像（goals/items/reviews/memory）整体替换
│                      · 前端本地状态（messages/streaming/error）
│                      · 乐观更新 + 失败回滚、请求序号防乱序
├── App.vue           工作台外壳
└── components/
    ├── ChatPanel.vue  对话区 + 留痕轨迹
    └── SidePanel.vue  今日 / 目标 / 记忆 三个标签页
```

## 状态管理要点

**分清服务端和前端。** 目标、待办、复盘、记忆归服务端，后端是唯一真相，
收到快照就整体替换，不做局部合并。对话记录和流式过程只存在于这个页面。

**乐观更新必须能回滚。** 勾选待办时先翻转本地状态让界面立刻响应，
请求失败要把状态改回去，否则界面和服务端会不一致。

**流式请求会乱序返回。** 快速连发消息时先发的请求可能后到，
用请求序号丢弃过期响应。

**派生状态用 computed。** 今日待办、进度、长期记忆的有效/被取代分组，
都不在 state 里存冗余副本。

## 构建

```bash
npm run build
```

产物在 `dist/`，不提交。
