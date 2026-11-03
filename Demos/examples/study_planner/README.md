# study_planner · 学习规划师

面向个人本机或单机使用的学习工作台：记录目标、按时间预算排期、勾选任务、复盘，以及有来源的会话与长期记忆。

当前 Web 与 CLI 共用 SQLite；操作有幂等标识和可查询结果。会话原文持久保存，发给模型的上下文有预算，长期记忆可以确认、纠正、删除。仅监听本机，不提供多用户认证和云同步。

## 运行

Python 3.13，依赖由 `Demos/` 的 uv 项目管理；前端使用 npm。

```bash
cd Demos
uv sync
uv run uvicorn examples.study_planner.backend.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

另开终端：

```bash
cd Demos/examples/study_planner/frontend
npm install
npm run dev
```

打开终端打印的本地地址。开发代理默认指向 `http://127.0.0.1:8000`；启动 Vite 时设置 `VITE_API_BASE` 可修改代理目标，构建时同名变量设置部署后的后端地址。

CLI 固定走规则模式，不调用模型：

```bash
cd Demos/examples
uv run --project .. python -m study_planner.backend --data-dir /tmp/my-study --task '我想学 Python'
uv run --project .. python -m study_planner.backend --data-dir /tmp/my-study
```

交互命令：`/memory` 查看记忆，`/recall 关键词` 检索，`/today` 查看待办，`/rollover` 整理当前业务日，`/exit` 退出。单句重试可传同一个 `--request-id`；`--conversation` 选择会话。

可选模型配置见 [`Demos/.env.example`](../../.env.example)，复制需要的设置到 `Demos/.env`。有 key 时模型负责理解与回复，调用失败退回规则或已提交的事实回复。默认输入预算 6000、输出 1500、安全余量 500，模型窗口配置至少覆盖它们；对未知兼容模型用保守 UTF-8 字节上界计数，可能提前拒绝长输入。没有 embedding 外发或向量服务。

## 记忆如何工作

- **领域事实**：目标、待办、复盘和类型化时间预算保存到同一个事务库。每日预算支持默认、工作日、周末和具体日期，日期优先级最高；排期直接读当前有效约束。
- **短期工作记忆**：保存原始会话，按完整轮次选择近期历史，较早消息生成带来源的提取式摘要。整理不删除原文；当前实现无需额外模型调用生成摘要。
- **长期记忆**：目标和明确时间预算可按规则生效。模型推测、复杂/含糊预算和普通学习偏好先进入待确认区，在记忆面板确认后生效；历史版本保留但不参加有效召回。
- **召回**：中文词片＋英文词匹配，先过滤状态/有效期，再排序；不把全量档案重新塞给模型。40 个固定离线样本的词法 Recall@5/MRR 为 0.6667，10 个负样本全部无匹配；语义改写仍有明显缺口，P2 默认未启用。
- **用户控制**：面板支持查看来源、确认、纠正和导出 JSON 阅览档案（完整恢复使用数据库快照）。删除前展示影响，删除会失效来源、关联回复/摘要和硬约束；不撤销已提交的目标与待办。删除预算后可能要求重新确认。

默认数据目录 `~/.study-planner/`，主要文件是 `planner.db`、`deletions.jsonl` 和 `backups/`。数据库是持久真相；`memory.py` 的列表实现仅用于兼容旧领域内核的单轮计算及旧自检，不再作为 Web/CLI 的持久状态。

## 已有 JSON 用户：先迁移

**先停止旧版 Web/CLI 的全部写入。不要直接启动新版本覆盖旧数据。** 新版检测到非空、未导入的旧 JSON 时，接口返回带迁移说明的 503，原文件不修改。若旧 JSON 与 SQLite 都为空，会自动完成空档案初始化并保留原文件快照和导入审计；若 SQLite 已有数据，空旧 JSON 不会覆盖它。

```bash
cd Demos/examples
uv run --project .. python -m study_planner.backend.migrate_data --source /path/to/old-data --target /path/to/new-data --dry-run
uv run --project .. python -m study_planner.backend.migrate_data --source /path/to/old-data --target /path/to/new-data
```

dry-run 检查损坏 JSON、冲突 ID、日期与悬空目标；正式导入保留原文件快照、旧 ID 和导入审计。同日复盘采用最后一份，重复批次不会重复导入；不同批次不能覆盖已有数据。旧长期文本默认待确认，发现旧预算时阻止排期，需先在记忆面板确认或明确提供新的默认预算。

Web 设置 `STUDY_PLANNER_DATA_DIR=/path/to/new-data`，CLI 使用 `--data-dir /path/to/new-data`。新库可通过 `STUDY_PLANNER_TIMEZONE` 选择 IANA 时区；已有库保留自己的时区。

## 断线、取消与重试

聊天和业务写入支持 `request_id`。相同标识、相同输入返回同一操作；相同标识换输入返回 409。前端保存未决请求标识，刷新后查询原操作，未接收的请求可复用原标识重试。

`GET /api/turns/{id}` 查询操作；`GET /api/requests/{request_id}` 可在尚未收到 turn ID 时恢复；`POST /api/turns/{id}/cancel` 只取消提交前操作。已经提交时停止输出不会撤销操作。回复失败保存事实回复，状态仍可查询。

工作租约为 900 秒；进程异常退出后，已提交操作在租约过期时以事实回复恢复，未提交操作标记失败。Web 每 5 秒维护一次，CLI 在启动/查询时检查；数据库事务不覆盖模型请求。

## 备份与恢复

成功操作后检查当日备份，Web 维护线程也会检查，保留最近 7 份每日快照。`/health` 报告 SQLite 版本、失败作业数和最近备份时间；后台错误只记录类型，不写原始对话或模型 prompt。

手动备份可以在线执行：

```bash
cd Demos/examples
uv run --project .. python -m study_planner.backend.backup --data-dir /path/to/data --backup /path/to/snapshot.db
```

恢复必须先停止使用该目录的 Web/CLI；运行锁会拒绝覆盖正在使用的库：

```bash
uv run --project .. python -m study_planner.backend.backup --data-dir /path/to/data --restore /path/to/snapshot.db --dry-run
uv run --project .. python -m study_planner.backend.backup --data-dir /path/to/data --restore /path/to/snapshot.db
```

备份 `.db` 与旁边的 `.db.json` 校验清单一起保管。恢复同时校验数据库、版本、外键，并在开放读取前重放数据目录中独立的 `deletions.jsonl`。该清单缺失或损坏时不允许恢复；它不能随旧备份覆盖。删除意图先 fsync，再提交数据库，崩溃后可补做。

删除是应用层来源失效，不是介质物理擦除，历史快照和迁移原文件可能仍含旧内容；需要物理清除时应同时处理这些副本。切换后回退到旧 JSON 版本不能保证保留新写入，应先备份并验证导出，不直接重启旧程序写旧目录。

## 验证

```bash
cd Demos/examples
uv run --project .. python -m study_planner.backend.selftest
uv run --project .. python -m study_planner.backend.selftest_api
uv run --project .. python -m unittest study_planner.backend.selftest_storage study_planner.backend.selftest_turns study_planner.backend.selftest_memory study_planner.backend.selftest_retrieval study_planner.backend.selftest_backup
uv run --project .. python -m chat_cli.selftest
uv run --project .. python -m study_planner.backend.eval_memory
uv run --project .. python -m study_planner.backend.benchmark_memory
```

前端运行 `npm run test:store`、`npm run test:clipboard` 和 `npm run build`。复制自检覆盖现代接口成功、接口缺失、权限拒绝、传统复制失败及焦点/选区恢复。store 自检使用真实 Vue/Pinia 与可控 HTTP 响应，验证 404 重试和删除后的历史响应乱序。

本次合成规模实测（Darwin arm64 / Python 3.13.2 / SQLite 3.47.1 / DELETE journal / 词片索引）：1 万条活跃经验、100001 条消息、5 并发、50 次样本，检索＋上下文构建冷启动 71.59ms、p50 74.17ms、p95 110.74ms；该数据集快照恢复 679.89ms。结果不包含模型与网络，不代表其他机器或真实语料上的 SLA。

SQLite 默认 DELETE journal，不依赖本机旧运行库的 WAL。若手动启用 WAL，实际 SQLite 运行库必须至少为 3.51.3；低版本启动会拒绝 WAL 数据库。当前文件锁基于 POSIX `flock`，支持此项目的 macOS/Linux 单机环境。

## 代码入口与方案

- `backend/turn_service.py`：操作接收、分析、提交、回复、取消与恢复。
- `backend/db.py` / `repositories.py` / `migrations/`：事务库、运行锁和查询。
- `backend/memory_policy.py` / `context.py` / `retrieval.py`：事实规则、上下文预算和召回。
- `backend/maintenance.py` / `deletion.py` / `backup.py`：整理作业、删除重放和恢复。
- `backend/main.py` / `cli.py`：Web 与 CLI 接入。
- `frontend/src/stores/planner.js`：会话恢复与服务端镜像。

[技术方案](../../../study_planner-记忆系统-技术方案.md)与[实施计划](../../../docs/superpowers/plans/2026-11-03-study-planner-memory.md)记录设计和阶段验收；`architecture.drawio` 及其导出图是原示例版本的架构记录，当前结构以方案和本 README 为准。P0/P1 已接入，P2 的远程 embedding 服务商及外发范围尚未选择，默认关闭。
