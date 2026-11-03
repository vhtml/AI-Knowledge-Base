# study_planner Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> 本文仅为实施计划，不自动执行上述工作流。执行方式和可用技能在用户批准实施后确定；不因计划文档启动产品改造。

**Goal:** 将 study_planner 的字符串记忆示例改造成适合个人单机长期使用的可靠记忆系统。

**Architecture:** SQLite 统一持久化领域状态、会话、操作与记忆，TurnService 管理幂等和短事务。ContextBuilder 为模型决策和回复构造有预算的输入；长期记忆以证据和有效性管理，语义索引是可关闭的派生能力。

**Tech Stack:** Python 3.13 / sqlite3 / dataclass / FastAPI / Vue 3 / Pinia / uv；语义检索阶段复用已有 NumPy。保持现有 OpenAI 兼容文本模型接入。

**Spec:** [记忆系统技术方案](../../../study_planner-记忆系统-技术方案.md)，代码基线 `6916eca`。状态：待评审，所有实施复选框均未执行。

## Global Constraints

- 用户已确认：个人长期使用，本地或单机部署；不做多租户或远程开放。
- 所有 Python 仅位于 `Demos/examples/study_planner/backend/`；依赖与 `.env` 统一位于 `Demos/`。
- SQLite 单库事务覆盖领域变更、确定性记忆和操作结果；任何模型或网络请求不得发生在数据库事务内。
- 默认 SQLite DELETE journal、FULL synchronous、foreign_keys ON、busy_timeout 3000ms；WAL 可选，启用时实际运行库版本至少 3.51.3。
- 输入/输出/余量建议值为 6000/1500/500 tokens，必须按实际模型窗口校验；明确输入和硬约束不静默截断。
- 没有已确认来源的模型推测不可直接变成排期硬约束；旧约束、删除证据和失效摘要不可进入上下文。
- P0/P1 不增加 Redis、独立向量数据库、外部队列或 Agent 框架；P2 远程 embedding 默认关闭。
- 一个任务完成后立即本地提交中文 commit；不 push，不把多个任务的修改攒到最后。

## Review Focus

1. 数据损坏或导入中断被当成空数据：T1/T2 必须阻断并报告，不能覆盖原文件。
2. 相同 request_id 不同输入、断开后重试、提交与取消竞争：T3 必须返回稳定结果，不重复执行。
3. “今天例外”“周末”“引用他人”“否定”被误当永久预算：T4 必须验证范围与证据。
4. 旧摘要作业在用户删除后发布、模型计数器未知：T5/T6 必须阻止旧内容复活和输入超限。
5. 旧备份恢复带回已删除内容、CLI 与 Web 同时写：T3/T7 必须验证删除清单重放和版本冲突。

## 文件与接口约定

以下路径除 `Demos/pyproject.toml`、锁文件外均相对 `Demos/examples/study_planner/`；所有新文件是建议路径，不表示已有实现。

- `backend/db.py`、`backend/migrations/*.sql`：连接配置、迁移和事务生命周期。
- `backend/repositories.py`：领域、会话、记忆与作业读写，不创建隐式跨网络事务。
- `backend/memory_types.py`：共享 dataclass，包括 `MemoryCandidate`、`OperationResult`、`ContextBundle`。
- `OperationResult` 最少包含 `turn_id / status / state_revision / facts / reply / error_code`。
- `ContextBundle` 最少包含 `messages / token_count / source_ids / degraded / clipping_reasons`。
- `backend/turn_service.py`：请求幂等、决策校验、版本比较、提交、查询与取消。
- `backend/memory_policy.py`：类型化事实、作用域、生效规则和证据有效性。
- `backend/context.py`、`backend/retrieval.py`：有预算的模型输入和可替换的检索实现。
- `backend/maintenance.py`：以数据库租约运行摘要及每日整理；P2 再增加 embedding 作业。
- `backend/deletion.py`：删除影响预览、intent 清单持久化、事务删除及启动重放；不将跨文件协议混入纯记忆判断规则。
- `backend/migrate_data.py`、`backend/backup.py`：离线导入及备份恢复。
- `backend/selftest_storage.py`、`selftest_turns.py`、`selftest_memory.py`、`selftest_retrieval.py`：可 `python -m` 运行的离线验证，不依赖真实模型。

测试使用标准库 `unittest`/临时目录/假时钟/可脚本化模型；故障注入用子进程真实中止和独立数据库连接，不只 mock 掉提交。实现可按模块体量细分文件，不能改变表、事务与契约的职责。

## T1：事务存储与数据约束（R1、R6 / P0）

**Files:** 新增 `backend/db.py`、`backend/migrations/001_initial.sql`、`backend/repositories.py`、`backend/memory_types.py`、`backend/selftest_storage.py`；修改 `backend/models.py`。

**Interfaces:** `open_database(path: Path) -> sqlite3.Connection`、显式事务上下文；Repository 提供领域状态读取和 `expected_revision` 检查。表和必需字段按方案第 4 节定义，公共 dataclass 在本任务落地。

- [ ] 编写失败测试：全新库建表、迁移重复执行、外键违反、同日复盘唯一性、当前事实 key/scope 唯一性、数据库损坏和不支持的 schema 版本；断言错误显式可见。
- [ ] 执行 `cd Demos/examples && uv run --project .. python -m study_planner.backend.selftest_storage`，确认尚未实现的能力失败。
- [ ] 实现初始化及 SQL 迁移；默认 DELETE journal，检查配置与运行库；同一事务写一条任务和一条记忆，第二次写故障时两者都不存在。
- [ ] 重跑测试，并在独立进程事务中写入后中止，验证未提交记录不存在、已提交记录重启仍在。锁超时映射为 STORE_BUSY。
- [ ] 本地提交：`study_planner: 新增统一事务存储与版本迁移`。

## T2：旧 JSON 的可验证导入（R1、R2、R6 / P0，依赖 T1）

**Files:** 新增 `backend/migrate_data.py`；扩展 `backend/selftest_storage.py`；修改 `backend/storage.py`、README。

**Interfaces:** `python -m study_planner.backend.migrate_data --source PATH --target PATH --dry-run` 返回结构化映射与异常报告；去掉 dry-run 才正式导入。成功批次幂等，旧文件只读，目标库已有其他数据时拒绝覆盖。

- [ ] 编写失败测试：空目录、正常四份 JSON、坏 JSON、悬空目标、同 ID 不同内容、同日多份复盘、缺失来源、已被取代记忆和中途进程中止。
- [ ] 运行 storage 自检，确认迁移断言失败。
- [ ] 实现 dry-run、输入摘要、临时目标库、单事务导入和对账报告。同日复盘取最后记录；无法识别来源的长期记忆 pending，预算待确认，不自动猜测激活。
- [ ] 重跑测试：重复相同批次不增记录；失败不更改源 JSON；中断后目标无法误判为已成功；修复后重试成功；有效旧 ID 保持不变。
- [ ] 本地提交：`study_planner: 增加旧数据校验和幂等导入`。

## T3：将 Web 与 CLI 收敛到可恢复操作（R1、R6 / P0，依赖 T1/T2）

**Files:** 新增 `backend/turn_service.py`、`backend/selftest_turns.py`；修改 `backend/agent.py`、`backend/main.py`、`backend/cli.py`、`backend/storage.py`、`backend/selftest_api.py`。

**Interfaces:** `TurnService.submit(conversation_id: str, request_id: str, text: str) -> OperationResult`、`get(turn_id: str) -> OperationResult`、`cancel(turn_id: str) -> OperationResult`。HTTP 和 CLI 使用相同服务，Repository 为唯一持久真相。

- [ ] 编写失败测试：同 key 同输入只执行一次、同 key 不同输入 409、两连接版本冲突、CLI/Web 同时改状态、提交前取消、提交后取消、提交后回复前崩溃。
- [ ] 执行 `cd Demos/examples && uv run --project .. python -m study_planner.backend.selftest_turns`，确认新操作契约尚不可用。
- [ ] 拆开分析与应用，将模型调用移出事务，应用时检查 revision 与取消状态。最多重新分析一次；再次冲突返回 409。所有业务写接口复用幂等边界。
- [ ] 实现操作查询、取消和重启恢复；断开不撤销已提交动作，回复失败保存事实回复；analyzing 租约过期进入 failed。
- [ ] 重跑 turns/API 自检；模拟慢模型时读取状态仍成功；SSE 队列满或连接断开后仍可查询最终提交结果。
- [ ] 本地提交：`study_planner: 统一操作幂等与提交恢复流程`。

## T4：类型化约束与证据化长期记忆（R2、R4 / P1，依赖 T3）

**Files:** 新增 `backend/memory_policy.py`、`backend/selftest_memory.py`；修改 `backend/memory_types.py`、`backend/memory.py`、`backend/agent.py`、`backend/planner.py`、`backend/review.py`、`backend/llm.py`。

**Interfaces:** `MemoryCandidate` 表达 key/scope/value、kind、证据片段、来源和有效期；`resolve_daily_minutes(day: date) -> int` 直接查询 constraints；记忆确认/更正走 Repository 原子更新与 expected_revision。

- [ ] 编写失败测试：30→60 分钟、今天 15 分钟、周末 120 分钟、时间过期、引述他人、否定、模型推测、相同证据重复提及、同 scope 并发更正、删除记忆同步撤销约束、迁移预算未确认阻止排期。
- [ ] 执行 `cd Demos/examples && uv run --project .. python -m study_planner.backend.selftest_memory`，确认旧字符串规则不符合断言。
- [ ] 实现 default/weekday/weekend/date 作用域和日期优先级；排期读取类型化事实，全新空库默认值显式为 60，迁移旧预算待确认时返回 BUDGET_CONFIRMATION_REQUIRED；constraint 为唯一硬约束源，fact memory 只引用；未知范围、指代不明、推测和无证据内容进入 pending。
- [ ] 用事实 key 代替前 12 字槽位；确认/更正同时关闭旧版本并激活新版本；复盘输出候选，不按 importance/mentions 直接成为确定约束。
- [ ] 重跑 memory、turns、API 自检，确认过期或 pending 内容不会改变计划，已有任务与预算回归行为仍正确。
- [ ] 本地提交：`study_planner: 引入带证据和作用域的长期记忆`。

## T5：有预算的会话工作记忆与整理（R3、R4、R6 / P1，依赖 T3/T4）

**Files:** 新增 `backend/context.py`、`backend/maintenance.py`；扩展 `backend/selftest_memory.py`；修改 `backend/agent.py`、`backend/llm.py`、`backend/main.py`、`backend/cli.py`。

**Interfaces:** `ContextBuilder.build(turn_id: str, purpose: str) -> ContextBundle`，purpose 仅 decision/reply；Clock 注入 UTC now 与个人时区业务日；作业以类型/来源范围/revision 幂等，发布使用输入 revision 校验。

- [ ] 编写失败测试：超长中文、多轮“继续”、无 tokenizer、context window 小于配置、重启跨日、重复 rollover、摘要超长/失败、作业执行中来源删除、同会话并发 turn。
- [ ] 执行 memory/turns 自检，确认原全量快照路径不能满足预算要求。
- [ ] 让两次模型调用共用 ContextBuilder，删除全量档案旁路；包含系统指令和封装开销计数，保留当前输入与硬约束，按方案定义顺序裁剪；必需内容过大返回 422。
- [ ] 保存完整消息；工作预算达到 80% 时生成摘要，保留最近至少 4 个完整轮次，输出不超过 800 tokens；失败采用受预算限制的近期视图，原文不丢。
- [ ] 实现数据库租约 runner、最多 3 次退避重试、按日维护与 restart 恢复；摘要只读取原消息，不递归摘要旧摘要。
- [ ] 重跑测试，检查模型捕获到的真实 messages 均在预算内，pending 的其他 turn 输入不进入当前轮上下文，过期作业不能发布。
- [ ] 本地提交：`study_planner: 实现可恢复且受预算限制的会话记忆`。

## T6：词法召回与用户记忆管理（R4、R5 / P1，依赖 T4/T5）

**Files:** 新增 `backend/retrieval.py`、`backend/deletion.py`、`backend/selftest_retrieval.py`；修改 `backend/main.py`、`backend/repositories.py`、`frontend/src/api.js`、`frontend/src/stores/planner.js`、`frontend/src/components/SidePanel.vue`、`frontend/src/components/ChatPanel.vue`。

**Interfaces:** `search(query: str, limit: int = 5) -> list[RecallHit]` 返回 ID、证据和分项得分；过滤优先于排序；新增会话分页、turn 查询、记忆分页/确认/纠正/删除 API，字段按方案第 7 节。

- [ ] 编写失败测试：active/expired/superseded/deleted/pending 过滤、无匹配、重复证据、修改 revision 冲突、删除关联源消息/摘要/turn 结果/assistant 副本、在途回复停止和旧作业发布竞争、旧快照 revision 回退。
- [ ] 运行 retrieval/API 自检，确认接口与过滤尚未实现。
- [ ] 实现词片召回、稳定排序和预算选择，去掉按召回次数自增强。删除必须按方案的先持久化 intent、再提交 DB 协议执行，事务内清理索引与来源依赖闭包、关闭关联 constraint、递增 privacy_revision；审计只保留 ID/动作。T6 先实现该安全删除协议，T7 复用它做备份恢复，不能在 T7 前允许不安全删除。
- [ ] 前端恢复会话，持久复用未决操作的 request_id；停止操作明确展示取消是否成功；记忆面板支持查看证据、确认、更正、删除影响预览和分页。
- [ ] 重跑离线自检；`cd Demos/examples/study_planner/frontend && npm run build`。浏览器验证刷新恢复、重连查询、纠正后预算生效、删除后搜索及新回复不再读取已删证据；独立保留的领域记录按影响预览展示。
- [ ] 本地提交：`study_planner: 增加可解释召回和记忆纠正删除`。

## T7：备份恢复与日常版本切换（R1、R5、R6 / P1，依赖 T1–T6）

**Files:** 新增 `backend/backup.py`；扩展 storage/turns 自检；修改 README、`Demos/.env.example` 和健康诊断。

**Interfaces:** `python -m study_planner.backend.backup --backup PATH`、`--restore PATH --dry-run`；恢复只在停止写入后执行，校验 schema/完整性/删除清单，不能直接覆盖运行中的数据库。

- [ ] 编写失败测试：写入时在线备份、损坏备份、版本不兼容、恢复旧备份后重放删除清单、备份失败告警、磁盘写入失败、健康诊断不泄露正文或密钥。
- [ ] 运行 storage/turns 自检，确认新恢复流程尚不可用。
- [ ] 使用 SQLite Backup API 实现每日快照与保留最近 7 份；复用 T6 的 deletions.jsonl intent 与 fsync 协议，不随旧备份覆盖该文件；恢复前先按源 ID 重新计算依赖闭包并重放所有 intent，清单缺失/损坏则阻止开放读取。加入 intent 落盘前/落盘后 DB 提交前/DB 提交后三个崩溃点验证。
- [ ] 在临时目录按 dry-run→停写→导入→切换→新写入→备份→删除→恢复旧快照→重放删除的顺序演练；记录 RPO/RTO 与恢复限制。
- [ ] 跑完整回归：study_planner 内核/API/storage/turns/memory/retrieval、chat_cli 自检和前端 build；核对 1 万条经验/10 万条消息/5 并发的 p95，与环境配置一起记录，不将目标值写成实测。
- [ ] 更新 README 运行方式、迁移注意事项、恢复操作、降级语义、配置与备份限制；本地提交：`study_planner: 补齐备份恢复和单机运行验收`。

## T8：用评测决定是否增加语义检索（R4 / P2，可选，依赖 T7）

**Files:** 扩展 `backend/retrieval.py`、`backend/maintenance.py`、`backend/selftest_retrieval.py`；新增 `backend/eval_memory.py` 和不含个人数据的固定黄金集；按所选 provider 修改 `Demos/pyproject.toml` 与锁文件。

**Interfaces:** 检索接口不变；embedding provider 可关闭，记录模型/维度/hash/revision，更新与删除立即使旧向量失效。远程 provider 及外发范围需在启用前单独确认。

- [ ] 先固化至少 40 个离线黄金样本，报告词法 Recall@5、MRR、同义表达子集、关键有效性过滤和延迟；包括“无相关记忆”的负样本。
- [ ] 加入 embedding 缺失、provider 超时、模型换版、维度不一致、正文更新/删除和重启重建的失败测试。
- [ ] 只在词法基线存在明确语义缺口时，实现 SQLite 向量缓存 + NumPy 相似度 + 排名融合，按 1 万条活跃经验的范围测试；检索失败降级词法，业务写入仍成功。
- [ ] 在相同黄金集比较启用前后；只有同义表达 Recall@5 提高、失效内容过滤保持 100%、性能与外发范围符合评审结果才默认启用，否则保留词法主路径。
- [ ] 单独本地提交评测与索引实现；不把 P2 的外部服务选择阻塞到 P0/P1。

## 执行与交付约定

本轮产出方案和计划，不修改产品代码。建议评审后按 T1→T7 顺序逐项实施，完成一个可验证任务即提交；T8 以评测结论决定。未执行测试和性能目标不写成已通过。设计确认后再选择当前会话原生执行或有独立审阅的代理工作流，不因文档模板默认创建用户聊天或改变现有分支。
