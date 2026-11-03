CREATE TABLE profile (
 id INTEGER PRIMARY KEY CHECK(id=1), timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
 state_revision INTEGER NOT NULL DEFAULT 0, privacy_revision INTEGER NOT NULL DEFAULT 0,
 budget_status TEXT NOT NULL DEFAULT 'unset', maintenance_day TEXT NOT NULL DEFAULT ''
);
INSERT INTO profile(id) VALUES(1);
CREATE TABLE imports(id TEXT PRIMARY KEY, report TEXT NOT NULL);
CREATE TABLE goals(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE plan_items(id TEXT PRIMARY KEY, goal_id TEXT NOT NULL REFERENCES goals(id), day TEXT NOT NULL, done INTEGER NOT NULL, payload TEXT NOT NULL);
CREATE INDEX items_day ON plan_items(day);
CREATE TABLE reviews(day TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE conversations(id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
CREATE TABLE turns(
 id TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE, input_hash TEXT NOT NULL,
 conversation_id TEXT NOT NULL REFERENCES conversations(id), command TEXT NOT NULL DEFAULT 'chat',
 status TEXT NOT NULL, base_revision INTEGER NOT NULL DEFAULT 0, state_revision INTEGER NOT NULL DEFAULT 0,
 facts TEXT NOT NULL DEFAULT '', reply TEXT NOT NULL DEFAULT '', error_code TEXT NOT NULL DEFAULT '',
 lease_until REAL NOT NULL DEFAULT 0, created_at TEXT NOT NULL, degraded INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE messages(
 id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id), seq INTEGER NOT NULL,
 turn_id TEXT NOT NULL REFERENCES turns(id), role TEXT NOT NULL CHECK(role IN ('user','assistant')),
 content TEXT NOT NULL, created_at TEXT NOT NULL, day TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0,
 UNIQUE(conversation_id,seq)
);
CREATE INDEX messages_turn ON messages(turn_id);
CREATE TABLE constraints(
 id TEXT PRIMARY KEY, key TEXT NOT NULL, scope TEXT NOT NULL, value INTEGER NOT NULL CHECK(value BETWEEN 15 AND 480),
 unit TEXT NOT NULL DEFAULT 'minute', effective_from TEXT NOT NULL, effective_to TEXT,
 revision INTEGER NOT NULL DEFAULT 1, source_message_id TEXT REFERENCES messages(id), status TEXT NOT NULL DEFAULT 'active'
);
CREATE UNIQUE INDEX constraint_current ON constraints(key,scope) WHERE status='active';
CREATE TABLE memories(
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, category TEXT NOT NULL, content TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('pending','active','superseded','expired','deleted','rejected')),
 constraint_id TEXT REFERENCES constraints(id), importance INTEGER NOT NULL DEFAULT 3 CHECK(importance BETWEEN 1 AND 5),
 revision INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, confirmed_at TEXT,
 effective_to TEXT, superseded_by TEXT NOT NULL DEFAULT '', source_type TEXT NOT NULL
);
CREATE INDEX memories_status ON memories(status,created_at);
CREATE TABLE memory_sources(
 memory_id TEXT NOT NULL REFERENCES memories(id), source_id TEXT NOT NULL,
 excerpt TEXT NOT NULL, extractor TEXT NOT NULL, PRIMARY KEY(memory_id,source_id)
);
CREATE INDEX sources_id ON memory_sources(source_id);
CREATE TABLE memory_terms(memory_id TEXT NOT NULL REFERENCES memories(id), term TEXT NOT NULL, PRIMARY KEY(term,memory_id));
CREATE TABLE artifact_sources(
 artifact_type TEXT NOT NULL, artifact_id TEXT NOT NULL, source_id TEXT NOT NULL,
 PRIMARY KEY(artifact_type,artifact_id,source_id)
);
CREATE INDEX artifacts_source ON artifact_sources(source_id);
CREATE TABLE summaries(
 id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id), start_seq INTEGER NOT NULL,
 end_seq INTEGER NOT NULL, privacy_revision INTEGER NOT NULL, content TEXT NOT NULL,
 token_count INTEGER NOT NULL, version TEXT NOT NULL, valid INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE jobs(
 id TEXT PRIMARY KEY, job_key TEXT NOT NULL UNIQUE, kind TEXT NOT NULL, payload TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'queued', lease_until REAL NOT NULL DEFAULT 0,
 attempts INTEGER NOT NULL DEFAULT 0, next_run REAL NOT NULL DEFAULT 0, error_code TEXT NOT NULL DEFAULT ''
);
CREATE TABLE deletions(id TEXT PRIMARY KEY, applied_at TEXT NOT NULL);
PRAGMA user_version=1;
