# gdrag v3

**Advanced RAG system for AI agents** — multi-agent registry, task queue, inter-agent communication, state persistence, session memory, context compression, cross-encoder re-ranking, and dual embeddings over PostgreSQL + Qdrant + Memgraph + Redis.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Version](https://img.shields.io/badge/version-3.0.0-orange)](CHANGELOG.md)

---

## What is gdrag?

gdrag is a production-ready Retrieval-Augmented Generation (RAG) backend designed to give AI agents (Cline, Claude Code, GPT-based tools) a **persistent, intelligent memory layer** with **multi-agent coordination**.

### v2 — RAG Pipeline
1. **Embed** — dual-provider (local sentence-transformers or OpenAI)
2. **Search** — semantic (Qdrant), graph (Memgraph), and structured (PostgreSQL FTS)
3. **Re-rank** — cross-encoder for precision over top-k candidates
4. **Compress** — automatic context summarisation to stay within token budgets
5. **Attend** — temporal decay scoring so recent, diverse results surface first
6. **Remember** — per-agent session memory persisted across queries

### v3 — Multi-Agent System
7. **Register** — agent registry with heartbeat monitoring and health checks
8. **Isolate** — multi-tenancy with RLS, namespaces, quotas, and permissions
9. **Queue** — priority task queue with Redis sorted sets
10. **Communicate** — inter-agent request-response, broadcast, and handoff
11. **Persist** — cognitive state snapshots and decision history
12. **Integrate** — EventBus (Redis Streams) connecting all modules

---

## Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- (Optional) OpenAI API key for cloud embeddings
- (Optional) Redis for v3 multi-agent features

### One-command setup

```bash
git clone https://github.com/tanobuffone/gdrag.git
cd gdrag
cp .env.example .env        # fill in passwords / API keys
make setup                  # venv + Docker + pip + migrations
make api                    # → http://localhost:8000/docs
```

### Docker services

```bash
docker compose up -d                    # PostgreSQL + Qdrant (core)
docker compose --profile cache up -d    # + Redis (v3 multi-agent)
docker compose --profile graph up -d    # + Memgraph (knowledge graph)
docker compose --profile full up -d     # everything
```

---

## MCP Tools (21)

### v2 — RAG Pipeline (10 tools)

| Tool | Description |
|------|-------------|
| `rag_query` | Full RAG query with session support |
| `semantic_search` | Pure vector similarity search |
| `graph_search` | Knowledge graph traversal (Memgraph) |
| `structured_query` | PostgreSQL full-text search |
| `session_create` | Create an agent session |
| `session_context` | Retrieve current session context |
| `session_compress` | Compress session history |
| `ingest_knowledge` | Ingest a document into the knowledge base |
| `get_health` | Check service health |
| `get_stats` | Get system statistics |

### v3 — Multi-Agent (11 tools)

| Tool | Description |
|------|-------------|
| `agent_register` | Register a new agent with capabilities |
| `agent_heartbeat` | Send agent keepalive heartbeat |
| `agent_list` | List active agents |
| `task_enqueue` | Enqueue async task with priority |
| `task_status` | Check task status |
| `task_cancel` | Cancel pending task |
| `knowledge_promote` | Promote knowledge from private to shared |
| `context_request` | Request context from another agent |
| `context_share` | Broadcast context to all agents |
| `state_save` | Save agent cognitive state snapshot |
| `state_load` | Load agent state from snapshot |

---

## REST API

Base URL: `http://localhost:8000`
Docs: `http://localhost:8000/docs`

### v2 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v2/query` | Full RAG query with re-ranking |
| `POST` | `/api/v2/query/session` | Query with session context |
| `POST` | `/api/v2/ingest` | Ingest document |
| `POST` | `/api/v2/sessions` | Create session |
| `GET` | `/api/v2/sessions/{id}` | Get session |
| `DELETE` | `/api/v2/sessions/{id}` | Delete session |
| `GET` | `/api/v2/health` | Health check |
| `GET` | `/api/v2/stats` | System stats |

### v3 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v3/agents/register` | Register agent |
| `POST` | `/api/v3/agents/{id}/heartbeat` | Agent heartbeat |
| `GET` | `/api/v3/agents` | List agents |
| `GET` | `/api/v3/agents/{id}` | Get agent info |
| `GET` | `/api/v3/agents/{id}/health` | Agent health |
| `POST` | `/api/v3/tasks/enqueue` | Enqueue task |
| `GET` | `/api/v3/tasks/{id}/status` | Task status |
| `POST` | `/api/v3/tasks/{id}/cancel` | Cancel task |
| `GET` | `/api/v3/tasks/{id}/result` | Task result |
| `POST` | `/api/v3/tasks/purge` | Purge completed tasks |
| `POST` | `/api/v3/knowledge/ingest` | Ingest knowledge |
| `POST` | `/api/v3/knowledge/{id}/promote` | Promote knowledge |
| `GET` | `/api/v3/knowledge/search` | Search knowledge |
| `DELETE` | `/api/v3/knowledge/{id}/access` | Revoke access |

---

## Project Structure

```
gdrag/
├── src/
│   ├── core/                        # 18 modules
│   │   ├── config.py, embeddings.py, chunker.py     # v2
│   │   ├── reranker.py, compressor.py, attention.py  # v2
│   │   ├── session_manager.py, vector_store.py       # v2 + v3 multi-tenancy
│   │   ├── graph_store.py, relational_store.py       # v2 + v3 multi-tenancy
│   │   ├── pipeline.py                                # v2
│   │   ├── agent_registry.py, tenant_manager.py      # v3
│   │   ├── state_manager.py, event_bus.py            # v3
│   │   ├── task_queue.py, agent_comm.py              # v3
│   │   └── knowledge_manager.py                       # v3
│   ├── api/
│   │   ├── main.py                  # FastAPI app (v2 + v3)
│   │   ├── v2/                      # router, middleware, dependencies
│   │   └── v3/                      # agents, tasks, knowledge endpoints
│   ├── mcp/v2/server.py             # 21 MCP tools
│   ├── models/                      # 7 Pydantic model files
│   └── workers/task_worker.py       # Async task worker
├── config/                          # settings, agents, embedding_models
├── migrations/                      # 001-005 SQL
├── tests/                           # 11 test files
├── CLAUDE.md, README.md, CHANGELOG.md
└── docker-compose.yml, Makefile, requirements.txt
```

---

## Infrastructure

| Service | Port | Purpose | Profile |
|---------|------|---------|---------|
| PostgreSQL 16 | 5434 | Sessions, knowledge, agents, state | default |
| Qdrant | 6333 | Vector embeddings | default |
| Memgraph | 7687 | Knowledge graph | graph |
| Redis 7.2 | 6379 | EventBus, TaskQueue, cache | cache |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | localhost | PostgreSQL host |
| `POSTGRES_PORT` | 5434 | PostgreSQL port |
| `POSTGRES_DB` | gdrag | Database name |
| `POSTGRES_USER` | gdrag | Database user |
| `POSTGRES_PASSWORD` | — | **Required** |
| `QDRANT_HOST` | localhost | Qdrant host |
| `OPENAI_API_KEY` | — | For openai provider |
| `EMBEDDING_PROVIDER` | local | local/openai/both |

---

## Embedding Models

| Model | Dims | Size | Quality | Best for |
|-------|------|------|---------|----------|
| `all-MiniLM-L6-v2` *(default)* | 384 | 22 MB | Good | Fast iteration |
| `all-mpnet-base-v2` | 768 | 420 MB | Very good | General purpose |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.2 GB | Excellent | Production |
| `text-embedding-3-small` | 1536 | Cloud | Great | OpenAI users |
| `text-embedding-3-large` | 3072 | Cloud | Best | Highest accuracy |

---

## Development

```bash
make test           # Full test suite
make test-unit      # Unit tests only
make lint           # ruff linter
make format         # ruff auto-format
make coverage       # HTML coverage report
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE) — Federico Buffone / Family Capital
