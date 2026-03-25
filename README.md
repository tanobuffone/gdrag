# gdrag v2

**Advanced RAG system for AI agents** — session memory, context compression, cross-encoder re-ranking, and dual embeddings over PostgreSQL + Qdrant + Memgraph.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-orange)](CHANGELOG.md)

---

## What is gdrag?

gdrag is a production-ready Retrieval-Augmented Generation (RAG) backend designed to give AI agents (Cline, Claude Code, GPT-based tools) a **persistent, intelligent memory layer**. Unlike simple vector stores, gdrag orchestrates an entire pipeline:

1. **Embed** — dual-provider (local sentence-transformers or OpenAI)
2. **Search** — semantic (Qdrant), graph (Memgraph), and structured (PostgreSQL FTS)
3. **Re-rank** — cross-encoder for precision over top-k candidates
4. **Compress** — automatic context summarisation to stay within token budgets
5. **Attend** — temporal decay scoring so recent, diverse results surface first
6. **Remember** — per-agent session memory persisted across queries

Both a **REST API** (FastAPI) and an **MCP server** (Model Context Protocol) are provided, so any agent that supports MCP can use gdrag as a drop-in memory backend.

---

## Architecture

```
Agent / User
     │
     ├── REST  →  FastAPI (src/api/)          ← /docs at :8000
     └── MCP   →  FastMCP server (src/mcp/)
                        │
                   QueryPipeline (src/core/pipeline.py)
                        │
          ┌─────────────┼─────────────┐
          │             │             │
    SessionManager   Embedder      Chunker
          │             │             │
     ┌────┴────┐   ┌────┴────┐        │
     │ Qdrant  │   │Memgraph │   PostgreSQL
     │(vectors)│   │ (graph) │  (structured+FTS)
     └────┬────┘   └────┬────┘        │
          └─────────────┴─────────────┘
                        │
               CrossEncoderReranker
                        │
               AttentionManager
                        │
               ContextCompressor
                        │
                   Response
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- (Optional) OpenAI API key for cloud embeddings

### One-command setup

```bash
git clone https://github.com/tanobuffone/gdrag.git
cd gdrag
cp .env.example .env        # fill in passwords / API keys
make setup                  # venv + Docker + pip + migrations
make api                    # → http://localhost:8000/docs
```

### Manual setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start infrastructure
docker compose up -d postgres qdrant

# Apply migrations
psql -U gdrag -d gdrag -f migrations/001_session_memory.sql
psql -U gdrag -d gdrag -f migrations/002_enhanced_knowledge.sql

# Start API
uvicorn src.api.main:app --reload --port 8000
```

---

## Configuration

### `config/settings.yaml`

```yaml
embedding:
  provider: local           # local | openai | both
  local_model: all-MiniLM-L6-v2

chunking:
  strategy: hybrid          # sliding_window | semantic | paragraph | hybrid
  chunk_size: 512
  chunk_overlap: 128

reranking:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k_initial: 50
  top_k_final: 10

compression:
  enabled: true
  strategy: hybrid          # summarize | extract | hybrid
  max_tokens: 4000

attention:
  decay_factor: 0.95
  recency_weight: 0.4
  relevance_weight: 0.4
  diversity_weight: 0.2

session:
  max_tokens: 8000
  session_ttl_hours: 24
  auto_compress: true
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `gdrag` | Database name |
| `POSTGRES_USER` | `gdrag` | Database user |
| `POSTGRES_PASSWORD` | — | **Required** |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant REST port |
| `OPENAI_API_KEY` | — | Required only for `provider: openai` |
| `GDRAG_API_PORT` | `8000` | API listen port |
| `EMBEDDING_PROVIDER` | `local` | Override embedding provider |

See [.env.example](.env.example) for the full list.

---

## REST API

Base URL: `http://localhost:8000/api/v2`
Interactive docs: `http://localhost:8000/docs`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Full RAG query with re-ranking and compression |
| `POST` | `/query/session` | Query with automatic session context injection |
| `POST` | `/ingest` | Ingest a document (chunks + optional graph concepts) |
| `POST` | `/sessions` | Create an agent session |
| `GET` | `/sessions/{id}` | Get session info |
| `GET` | `/sessions` | List all sessions |
| `DELETE` | `/sessions/{id}` | Delete a session |
| `GET` | `/health` | Service health check |
| `GET` | `/stats` | System statistics |

### Example: query with session

```bash
curl -X POST http://localhost:8000/api/v2/query \
  -H "Content-Type: application/json" \
  -H "X-Agent-ID: cline" \
  -d '{
    "query": "What is machine learning?",
    "session_id": "my-session",
    "domain": "academic",
    "limit": 10,
    "use_compression": true
  }'
```

### Example: ingest a document

```bash
curl -X POST http://localhost:8000/api/v2/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Machine learning is a branch of AI...",
    "title": "Introduction to ML",
    "domain": "academic",
    "source": "manual"
  }'
```

---

## MCP Server

gdrag exposes 10 MCP tools usable by any MCP-compatible agent.

### Registration (Claude Code / Cline)

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "gdrag": {
      "command": "python3",
      "args": ["-m", "src.mcp.v2.server"],
      "cwd": "/path/to/gdrag",
      "env": { "PYTHONPATH": "/path/to/gdrag" }
    }
  }
}
```

### Available tools

| Tool | Description |
|------|-------------|
| `rag_query` | Full RAG query with session support |
| `semantic_search` | Pure vector similarity search |
| `graph_search` | Knowledge graph traversal (Memgraph) |
| `structured_query` | PostgreSQL full-text search |
| `session_create` | Create an agent session |
| `session_context` | Retrieve current session context |
| `session_compress` | Compress session history to free tokens |
| `ingest_knowledge` | Ingest a document into the knowledge base |
| `get_health` | Check service health |
| `get_stats` | Get system statistics |

---

## Infrastructure

Managed via `docker-compose.yml`:

```bash
docker compose up -d                    # core: PostgreSQL + Qdrant
docker compose --profile graph up -d   # + Memgraph
docker compose --profile cache up -d   # + Redis
docker compose --profile full up -d    # everything
```

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL 15 | 5432 | Sessions, knowledge, agent configs |
| Qdrant 1.7 | 6333 / 6334 | Vector embeddings (REST / gRPC) |
| Memgraph 2.11 | 7687 | Knowledge graph (optional) |
| Redis 7.2 | 6379 | Embedding cache (optional) |

---

## Project Structure

```
gdrag/
├── config/
│   ├── settings.yaml          # Main configuration
│   ├── agents.yaml            # Agent permissions & rate limits
│   └── embedding_models.yaml  # Model catalog with benchmarks
├── migrations/
│   ├── 001_session_memory.sql # Core schema
│   └── 002_enhanced_knowledge.sql  # FTS, cache, cleanup functions
├── src/
│   ├── api/
│   │   ├── main.py            # FastAPI app (mounts v1 + v2)
│   │   └── v2/
│   │       ├── router.py      # All /api/v2/* endpoints
│   │       ├── middleware.py  # Request logging + session tracking
│   │       └── dependencies.py  # FastAPI dependency injection
│   ├── core/                  # Business logic — no FastAPI here
│   │   ├── config.py          # Pydantic config models
│   │   ├── embeddings.py      # LocalEmbedder / OpenAIEmbedder / DualEmbedder
│   │   ├── chunker.py         # 4 chunking strategies
│   │   ├── reranker.py        # CrossEncoderReranker + SimpleReranker fallback
│   │   ├── compressor.py      # Extractive / abstractive compression
│   │   ├── attention.py       # Temporal decay + diversity scoring
│   │   ├── session_manager.py # Session CRUD + context compression
│   │   ├── vector_store.py    # Qdrant wrapper
│   │   ├── graph_store.py     # Memgraph / neo4j wrapper
│   │   ├── relational_store.py  # PostgreSQL wrapper (FTS + cleanup)
│   │   └── pipeline.py        # QueryPipeline — orchestrates everything
│   ├── mcp/v2/server.py       # FastMCP server with 10 tools
│   └── models/schemas.py      # All Pydantic request/response models
├── tests/
│   ├── test_config.py
│   ├── test_chunker.py
│   └── test_schemas.py
├── Makefile                   # Developer task automation
├── docker-compose.yml
├── requirements.txt
└── start.sh                   # Quick-start script
```

---

## Development

```bash
make test           # Full test suite
make test-unit      # Unit tests only (no live DB required)
make lint           # ruff linter
make format         # ruff auto-format
make type-check     # mypy
make coverage       # HTML coverage report → htmlcov/index.html
make clean          # Remove __pycache__, build artifacts
```

Tests use `unittest.mock` — no live PostgreSQL, Qdrant, or Memgraph needed for unit tests.

See [CLAUDE.md](CLAUDE.md) for contributor context, and [CONTRIBUTING.md](CONTRIBUTING.md) for branch/commit conventions.

---

## Agent Configuration

`config/agents.yaml` defines permissions and rate limits per agent:

```yaml
agents:
  cline:
    permissions: [read, write, ingest]
    rate_limit:
      requests_per_hour: 1000
      requests_per_minute: 30
  claude:
    permissions: [read, write, ingest, admin]
    rate_limit:
      requests_per_hour: 2000
      requests_per_minute: 60
```

The `X-Agent-ID` header identifies the calling agent on every API request.

---

## Embedding Models

| Model | Dims | Size | Quality | Best for |
|-------|------|------|---------|----------|
| `all-MiniLM-L6-v2` *(default)* | 384 | 22 MB | Good | Fast iteration |
| `all-mpnet-base-v2` | 768 | 420 MB | Very good | General purpose |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.2 GB | Excellent | Production |
| `BAAI/bge-m3` | 1024 | 2.2 GB | SOTA | Multilingual |
| `text-embedding-3-small` | 1536 | Cloud | Great | OpenAI users |
| `text-embedding-3-large` | 3072 | Cloud | Best | Highest accuracy |

Switch models in `config/settings.yaml` or via `EMBEDDING_PROVIDER` env var. Changing a model requires re-ingesting existing documents.

---

## v1 → v2 Migration

v1 endpoints remain available at `/api/v1/*`. To migrate:

1. Run `migrations/001_session_memory.sql` and `migrations/002_enhanced_knowledge.sql`
2. Update agent clients to use `/api/v2/*` endpoints
3. Add the `X-Agent-ID` header to all requests
4. Optionally register the MCP server for zero-config tool access

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

---

## License

[MIT](LICENSE) — Federico Buffone / Family Capital
