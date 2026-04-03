# gdrag v3 — Context for Claude Code

## What this project is

Advanced RAG (Retrieval-Augmented Generation) system for AI agents with multi-agent support. Provides:
- Semantic search across PostgreSQL + Qdrant + Memgraph
- Session memory (per-agent context persistence)
- Cross-encoder re-ranking, context compression, attention-focused retrieval
- **v3: Multi-agent registry, task queue, inter-agent communication, state persistence**
- FastAPI REST API + MCP server (for Cline, Roo, Claude Code)

## Quick start

```bash
make setup    # First time: .env + Docker + pip + migrations
make api      # Start FastAPI (http://localhost:8000/docs)
make mcp      # Start MCP server
make test     # Run full test suite
```

## Architecture

```
src/
  core/                    # Business logic (18 modules)
    # v2 — RAG Pipeline
    config.py              # Pydantic config — AppConfig, EmbeddingConfig, etc.
    embeddings.py          # Dual provider: LocalEmbedder, OpenAIEmbedder, DualEmbedder
    chunker.py             # DocumentChunker: sliding_window / semantic / paragraph / hybrid
    reranker.py            # CrossEncoderReranker + SimpleReranker fallback
    compressor.py          # ContextCompressor: extractive summarization + dedup
    attention.py           # AttentionManager: temporal decay + diversity filter
    session_manager.py     # SessionManager: CRUD + context + compression (multi-tenancy v3)
    vector_store.py        # EnhancedVectorStore wrapping qdrant-client (multi-tenancy v3)
    graph_store.py         # EnhancedGraphStore wrapping neo4j (Memgraph)
    relational_store.py    # EnhancedRelationalStore wrapping psycopg2 (multi-tenancy v3)
    pipeline.py            # QueryPipeline: orchestrates full RAG flow
    # v3 — Multi-Agent
    agent_registry.py      # AgentRegistry: register, heartbeat, health, cleanup
    tenant_manager.py      # TenantManager: isolation, permissions, quotas, RLS
    state_manager.py       # StateManager: snapshots, decisions, auto-save
    event_bus.py           # EventBus: Redis Streams pub/sub with consumer groups
    task_queue.py          # TaskQueue: Redis sorted sets priority queue
    agent_comm.py          # AgentCommunication: request-response, broadcast, handoff
    knowledge_manager.py   # KnowledgeManager: visibility levels, promotion, access control
  api/
    main.py                # FastAPI app: mounts v2 + v3 routers
    v2/
      router.py            # All /api/v2/* endpoints
      middleware.py        # Request logging + session tracking
      dependencies.py      # get_pipeline(), get_config(), get_agent_id()
    v3/
      __init__.py          # Exports: tasks_router, knowledge_router
      agents.py            # /api/v3/agents/* endpoints (register, heartbeat, list, health)
      tasks.py             # /api/v3/tasks/* endpoints (enqueue, status, cancel, result, purge)
      knowledge.py         # /api/v3/knowledge/* endpoints (ingest, promote, search, revoke)
      dependencies.py      # TaskQueue + KnowledgeManager singletons (in-memory stubs)
  mcp/v2/server.py         # FastMCP server with 21 tools (10 v2 + 11 v3)
  models/
    schemas.py             # v2 Pydantic models
    agent.py               # AgentRegistration, AgentInfo, HealthStatus
    communication.py       # ContextRequest, ContextResponse, SharedContext
    events.py              # Event, StreamName, EVENT_TYPES
    tasks.py               # Task, TaskStatus, TaskType, TaskPriority
    tenant.py              # Tenant, TenantPermissions, TenantQuotas, QuotaUsage
    state.py               # AgentState, Decision
  workers/
    task_worker.py         # TaskWorker: polling, handler dispatch, retry
config/
  settings.yaml            # Main config (overridden by env vars)
  agents.yaml              # Agent permissions + rate limits
  embedding_models.yaml    # Model catalog with dimensions, cost, use cases
migrations/
  001_session_memory.sql   # Core schema (sessions, knowledge)
  002_enhanced_knowledge.sql  # Indexes, embedding cache, cleanup
  003_multi_tenancy.sql    # Tenants table, agent_id columns, quota_usage
  004_agent_registry.sql   # Agents table, agent_heartbeats, GIN indexes
  005_state_persistence.sql  # Agent_snapshots, decisions tables
tests/
  test_config.py           # Config loading tests
  test_chunker.py          # Chunking strategy tests
  test_schemas.py          # Pydantic model validation tests
  test_event_bus.py        # EventBus Redis Streams tests (39 tests)
  test_task_queue.py       # TaskQueue priority queue tests (48 tests)
  test_agent_registry.py   # AgentRegistry CRUD + heartbeat tests
  test_tenant_manager.py   # TenantManager isolation + quotas tests
  test_agent_comm.py       # AgentCommunication patterns tests
  test_knowledge_manager.py  # KnowledgeManager visibility + promotion tests
  test_state_manager.py    # StateManager snapshots + decisions tests
  test_integration_v3.py   # End-to-end multi-agent flow tests
```

## Key patterns

**Configuration**: `load_config()` reads `config/settings.yaml` then overrides with env vars. All env vars documented in `.env.example`.

**Pipeline singleton**: `QueryPipeline` is instantiated once per process. In tests, mock via `patch("src.api.v2.router.get_pipeline")`.

**Lazy loading**: Models (sentence-transformers, cross-encoder) load on first use. Keeps startup fast.

**Graceful degradation**: Memgraph optional — graph_store returns `[]` if unavailable. Redis optional — EventBus/TaskQueue fall back to in-memory. SimpleReranker if cross-encoder fails.

**Multi-tenancy**: v3 adds `agent_id` ownership to sessions, knowledge entries, and vector stores. TenantManager enforces isolation. RLS policies in PostgreSQL.

**Agent auth**: v2 uses `X-Agent-ID` header. v1 uses Bearer token. Agent permissions in `config/agents.yaml`.

## MCP Tools (21 total)

### v2 — RAG Pipeline (10 tools)
`rag_query`, `semantic_search`, `graph_search`, `structured_query`, `session_create`, `session_context`, `session_compress`, `ingest_knowledge`, `get_health`, `get_stats`

### v3 — Multi-Agent (11 tools)
`agent_register`, `agent_heartbeat`, `agent_list`, `task_enqueue`, `task_status`, `task_cancel`, `knowledge_promote`, `context_request`, `context_share`, `state_save`, `state_load`

## Infrastructure

| Service | Port | Purpose | Profile |
|---------|------|---------|---------|
| PostgreSQL 16 | 5434 | Sessions, knowledge, agents, state | default |
| Qdrant | 6333 | Vector embeddings | default |
| Memgraph | 7687 | Knowledge graph | graph |
| Redis 7.2 | 6379 | EventBus, TaskQueue, cache | cache |

## Running tests

```bash
make test          # All tests
make test-file FILE=tests/test_event_bus.py
make test-unit     # Unit only (no DB required)
```

Tests use `unittest.mock` and `fakeredis` — no live PostgreSQL, Qdrant, or Redis needed.

## Adding a new core module

1. Create `src/core/mymodule.py`
2. Add config to `src/core/config.py`
3. Update `AppConfig` in `config.py`
4. Add settings to `config/settings.yaml`
5. Write tests in `tests/test_mymodule.py`

## Adding a new API endpoint

1. Add route to `src/api/v2/router.py` or `src/api/v3/`
2. Add Pydantic models in `src/models/`
3. Call core services via dependencies
4. Add test

## Adding a new MCP tool

1. Add function to `src/mcp/v2/server.py` with `@mcp.tool()` decorator
2. Tool calls core modules directly
3. Return string

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | localhost | PostgreSQL host |
| `POSTGRES_PORT` | 5434 | PostgreSQL port |
| `POSTGRES_DB` | gdrag | Database name |
| `POSTGRES_USER` | gdrag | Database user |
| `POSTGRES_PASSWORD` | — | Required |
| `QDRANT_HOST` | localhost | Qdrant host |
| `OPENAI_API_KEY` | — | For openai provider |
| `EMBEDDING_PROVIDER` | local | local/openai/both |
| `GDRAG_API_PORT` | 8000 | API port |
