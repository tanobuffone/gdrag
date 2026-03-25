# gdrag v2 — Context for Claude Code

## What this project is

Advanced RAG (Retrieval-Augmented Generation) system for AI agents. Provides:
- Semantic search across PostgreSQL + Qdrant + Memgraph
- Session memory (per-agent context persistence)
- Cross-encoder re-ranking
- Context compression
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
  core/           # Business logic (11 modules, no FastAPI)
    config.py     # Pydantic config — AppConfig, EmbeddingConfig, etc.
    embeddings.py # Dual provider: LocalEmbedder, OpenAIEmbedder, DualEmbedder
    chunker.py    # DocumentChunker: sliding_window / semantic / paragraph / hybrid
    reranker.py   # CrossEncoderReranker + SimpleReranker fallback
    compressor.py # ContextCompressor: extractive summarization + dedup
    attention.py  # AttentionManager: temporal decay + diversity filter
    session_manager.py  # SessionManager: CRUD + context + compression
    vector_store.py     # EnhancedVectorStore wrapping qdrant-client
    graph_store.py      # EnhancedGraphStore wrapping neo4j (Memgraph)
    relational_store.py # EnhancedRelationalStore wrapping psycopg2
    pipeline.py   # QueryPipeline: orchestrates full RAG flow
  api/
    main.py       # FastAPI app: mounts v1 + v2 routers, lifespan, CORS
    v2/
      router.py   # All /api/v2/* endpoints
      middleware.py       # Request logging + session tracking
      dependencies.py     # get_pipeline(), get_config(), get_agent_id()
  mcp/v2/server.py        # FastMCP server with 11 tools
  models/schemas.py       # All Pydantic models
config/
  settings.yaml           # Main config (overridden by env vars)
  agents.yaml             # Agent permissions + rate limits
  embedding_models.yaml   # Model catalog with dimensions, cost, use cases
migrations/
  001_session_memory.sql  # Core schema (sessions, knowledge, agents)
  002_enhanced_knowledge.sql  # Indexes, embedding cache, cleanup functions
tests/                    # pytest, all mocked (no live DB needed)
```

## Key patterns

**Configuration**: `load_config()` reads `config/settings.yaml` then overrides with env vars. All env vars are documented in `.env.example`.

**Pipeline singleton**: `QueryPipeline` is instantiated once per process. In tests, mock it via `patch("src.api.v2.router.get_pipeline")`.

**Lazy loading**: Models (sentence-transformers, cross-encoder) load on first use, not at import time. This keeps startup fast.

**Graceful degradation**: Memgraph is optional — graph_store returns `[]` if unavailable. SimpleReranker is used if cross-encoder fails to load.

**Agent auth**: v2 uses `X-Agent-ID` header (no JWT). v1 uses Bearer token (hash-based). Agent permissions live in `config/agents.yaml`.

## Running tests

```bash
# All tests (unit only — no DB required)
make test

# Specific file
make test-file FILE=tests/test_reranker.py

# Unit tests only (skip slow/integration)
make test-unit
```

Tests use `unittest.mock` to stub external services. No live PostgreSQL, Qdrant, or Memgraph needed for unit tests.

## Adding a new core module

1. Create `src/core/mymodule.py` with a class following existing patterns
2. Add config class to `src/core/config.py` (Pydantic BaseModel)
3. Add to `AppConfig` in `config.py`
4. Update `QueryPipeline.__init__` in `pipeline.py` to instantiate it
5. Add settings to `config/settings.yaml`
6. Write tests in `tests/test_mymodule.py`

## Adding a new API endpoint

1. Add route to `src/api/v2/router.py`
2. Add request/response models (Pydantic) in the same file or `src/models/schemas.py`
3. Call `get_pipeline()` dependency for access to all core services
4. Add test to `tests/test_api_v2.py`

## Adding a new MCP tool

1. Add tool function to `src/mcp/v2/server.py` using `@mcp.tool()` decorator
2. Tool should call `pipeline` methods — same pipeline as API
3. Return a string (MCP tools return text)

## Infrastructure

| Service | Port | Purpose |
|---|---|---|
| PostgreSQL | 5432 | Sessions, knowledge entries, agents |
| Qdrant | 6333 | Vector embeddings (REST) |
| Qdrant | 6334 | Vector embeddings (gRPC) |
| Memgraph | 7687 | Knowledge graph (optional) |
| Redis | 6379 | Embedding cache (optional) |

All services run in Docker. Managed via `docker-compose.yml`.

## Environment variables

Key variables (see `.env.example` for full list):

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | localhost | PostgreSQL host |
| `POSTGRES_PASSWORD` | gdrag2024 | PostgreSQL password |
| `QDRANT_HOST` | localhost | Qdrant host |
| `OPENAI_API_KEY` | — | Required only for `provider: openai` |
| `EMBEDDING_PROVIDER` | local | Override embedding provider |
| `GDRAG_API_PORT` | 8000 | API listen port |

## MCP registration (for Claude Code)

Add to `~/.claude/settings.json` under `mcpServers`:

```json
"gdrag": {
  "command": "python3",
  "args": ["-m", "src.mcp.v2.server"],
  "cwd": "/home/gdrick/gdrag",
  "env": {
    "PYTHONPATH": "/home/gdrick/gdrag"
  }
}
```

Available tools: `rag_query`, `semantic_search`, `graph_search`, `structured_query`,
`session_create`, `session_context`, `session_compress`, `ingest_knowledge`,
`get_health`, `get_stats`.
