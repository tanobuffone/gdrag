# gdrag v2 — Plan de Desarrollo para Eficiencia Máxima

**Fecha:** 2026-03-24
**Estado:** En implementación
**Versión:** 2.0.0

---

## Resumen ejecutivo

gdrag v2 es un sistema RAG avanzado para agentes IA. El código core está **~90% implementado** (11 módulos core, API v2, MCP v2, modelos, configuración, migraciones). Este plan documenta los gaps restantes y establece un entorno de desarrollo eficiente para completarlo y mantenerlo.

---

## Estado actual del proyecto

### ✅ Completado (Fases 1–7)

| Componente | Archivo | Estado |
|---|---|---|
| Configuración centralizada | `src/core/config.py` | ✅ 334 líneas |
| Embeddings duales | `src/core/embeddings.py` | ✅ 389 líneas |
| Chunking inteligente | `src/core/chunker.py` | ✅ 476 líneas |
| Re-ranking cross-encoder | `src/core/reranker.py` | ✅ 304 líneas |
| Compresión de contexto | `src/core/compressor.py` | ✅ 422 líneas |
| Mecanismo de atención | `src/core/attention.py` | ✅ 306 líneas |
| Gestión de sesiones | `src/core/session_manager.py` | ✅ 359 líneas |
| Store vectorial (Qdrant) | `src/core/vector_store.py` | ✅ 369 líneas |
| Store de grafo (Memgraph) | `src/core/graph_store.py` | ✅ 329 líneas |
| Store relacional (PostgreSQL) | `src/core/relational_store.py` | ✅ 519 líneas |
| Pipeline de orquestación | `src/core/pipeline.py` | ✅ 448 líneas |
| API v2 router | `src/api/v2/router.py` | ✅ Completo |
| API v2 middleware | `src/api/v2/middleware.py` | ✅ Completo |
| MCP v2 server | `src/mcp/v2/server.py` | ✅ 423 líneas |
| Modelos Pydantic | `src/models/schemas.py` | ✅ Completo |
| Schema PostgreSQL | `migrations/001_session_memory.sql` | ✅ 4 tablas |
| Configuración YAML | `config/settings.yaml` | ✅ Completo |
| Config de agentes | `config/agents.yaml` | ✅ 5 agentes |
| Tests básicos | `tests/test_config/chunker/schemas.py` | ✅ 3 archivos |

### ❌ Pendiente (Fase 8–9)

| Prioridad | Archivo | Propósito |
|---|---|---|
| 🔴 Alta | `docker-compose.yml` | Infraestructura de desarrollo (PostgreSQL + Qdrant) |
| 🔴 Alta | `Makefile` | Automatización de tareas dev |
| 🔴 Alta | `src/api/main.py` (fix) | Montar router v2, eliminar placeholders de v1 |
| 🟡 Media | `src/api/v2/dependencies.py` | Dependency injection FastAPI |
| 🟡 Media | `config/embedding_models.yaml` | Config detallada de modelos |
| 🟡 Media | `migrations/002_enhanced_knowledge.sql` | Schema mejorado para FTS e índices |
| 🟢 Tests | `tests/test_embeddings.py` | Validar embeddings local/OpenAI |
| 🟢 Tests | `tests/test_reranker.py` | Validar re-ranking |
| 🟢 Tests | `tests/test_session.py` | Validar gestión de sesiones |
| 🟢 Tests | `tests/test_api_v2.py` | Validar endpoints v2 |
| 🔵 Dev | `pyproject.toml` | Configuración pytest, linting, black |
| 🔵 Dev | `CLAUDE.md` | Contexto para Claude Code |
| 🔵 Dev | MCP registration | Registrar en Claude Code settings |

---

## Plan de implementación detallado

### Fase 8A: Infraestructura de desarrollo

**Objetivo:** Poder levantar todo el stack con un solo comando.

#### `docker-compose.yml`
```
Servicios:
  - postgres:15-alpine  → puerto 5432, DB: gdrag, user: gdrag
  - qdrant/qdrant:v1.7.0 → puerto 6333 (REST) + 6334 (gRPC)
  - memgraph/memgraph   → puerto 7687 (Bolt) — opcional, perfil "graph"

Volúmenes persistentes:
  - postgres_data
  - qdrant_data
  - memgraph_data

Healthchecks: en todos los servicios
Networks: gdrag-network (bridge)
```

**Uso:**
```bash
# Stack básico (PostgreSQL + Qdrant)
docker compose up -d

# Stack completo con Memgraph
docker compose --profile graph up -d
```

#### `Makefile`
Tareas a incluir:
- `make setup` — crear .env, levantar Docker, instalar deps, correr migraciones
- `make up` / `make down` — gestión del stack
- `make migrate` — correr todas las migraciones SQL
- `make test` / `make test-watch` — pytest con cobertura
- `make lint` / `make format` — ruff + black
- `make api` / `make mcp` — iniciar servidores
- `make health` — verificar todos los servicios
- `make clean` — limpiar artefactos

---

### Fase 8B: Fix API main.py

**Problema:** `src/api/main.py` tiene:
1. Placeholder `generate_embedding()` que retorna vectores aleatorios
2. Placeholder `extract_concepts()` que solo hace word split
3. Router v2 no montado

**Solución:**
- Montar `api_v2_router` desde `src/api/v2/router.py`
- Reemplazar funciones placeholder con imports de los módulos core reales
- Mantener endpoints v1 funcionales (backwards compatibility)

---

### Fase 8C: Archivos faltantes del plan original

#### `src/api/v2/dependencies.py`
FastAPI Dependency Injection para:
- `get_pipeline()` — singleton del QueryPipeline
- `get_config()` — configuración de la app
- `get_agent_id()` — extrae X-Agent-ID del header con validación

#### `config/embedding_models.yaml`
Catálogo de modelos disponibles con:
- Dimensiones, velocidad, calidad, uso de VRAM
- Modelos locales recomendados (MiniLM, MPNet, BGE)
- Modelos OpenAI (text-embedding-3-small/large)

#### `migrations/002_enhanced_knowledge.sql`
Mejoras al schema:
- Índices compuestos para queries frecuentes (domain + created_at)
- Tabla `embedding_cache` para caché persistente
- Tabla `document_chunks` para tracking de chunks por documento
- View `knowledge_summary` para stats rápidas

---

### Fase 8D: Suite de tests

**Filosofía:** Tests rápidos con mocks, sin dependencias externas.

#### `tests/test_embeddings.py`
- `TestLocalEmbedder`: dimensiones correctas, batch consistency, cache hit
- `TestOpenAIEmbedder`: mock API, manejo de errores
- `TestDualEmbedder`: concatenación de vectores
- `TestEmbeddingCache`: LRU eviction, get/set

#### `tests/test_reranker.py`
- `TestCrossEncoderReranker`: scores normalizados 0-1, ordenamiento
- `TestSimpleReranker`: fallback Jaccard
- `TestRerankerFactory`: graceful degradation

#### `tests/test_session.py`
- `TestSessionManager`: CRUD completo con mock DB
- Expiración TTL
- Compresión de historial
- Lista de sesiones por agente

#### `tests/test_api_v2.py`
- `TestQueryEndpoint`: query básico, con sesión, con compresión
- `TestIngestEndpoint`: ingesta, validación, chunking
- `TestSessionEndpoints`: CRUD de sesiones
- `TestHealthEndpoint`: health check format
- Backwards compatibility con v1

---

### Fase 8E: Tooling de desarrollo

#### `pyproject.toml`
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = ["unit", "integration", "slow"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "UP"]

[tool.black]
line-length = 100
```

#### `CLAUDE.md`
Contexto para Claude Code sobre:
- Arquitectura del proyecto
- Patrones de código usados
- Cómo correr tests
- Cómo agregar nuevos módulos core

---

### Fase 9: Registro MCP

#### Configurar en Claude Code settings
```json
{
  "gdrag": {
    "command": "python3",
    "args": ["-m", "src.mcp.v2.server"],
    "cwd": "/home/gdrick/gdrag",
    "env": {
      "PYTHONPATH": "/home/gdrick/gdrag"
    }
  }
}
```

---

## Arquitectura de flujo de trabajo

```
Dev workflow:
  make setup → make up → make test → make api

Query flow:
  Agent → MCP/REST → API v2 → QueryPipeline
    → SessionManager (contexto)
    → Embeddings (local/OpenAI)
    → VectorStore (Qdrant, top 50)
    → GraphStore (Memgraph, conceptos)
    → RelationalStore (PostgreSQL, FTS)
    → CrossEncoderReranker (top 10)
    → AttentionManager (scoring)
    → ContextCompressor (opcional)
    → EnhancedQueryResponse

Ingest flow:
  Document → Chunker → Embeddings → VectorStore + GraphStore + RelationalStore
```

---

## Métricas de calidad objetivo

| Métrica | Objetivo |
|---|---|
| Test coverage | ≥ 80% |
| Query latency (local) | < 500ms p95 |
| Query latency (OpenAI) | < 2000ms p95 |
| Ingest throughput | ≥ 10 docs/seg |
| Session persistence | 100% (no pérdida entre restarts) |

---

## Dependencias externas requeridas

| Servicio | Puerto | Uso |
|---|---|---|
| PostgreSQL 15+ | 5432 | Sesiones, knowledge entries, agentes |
| Qdrant 1.7+ | 6333 | Vector embeddings |
| Memgraph (opcional) | 7687 | Knowledge graph |
| Redis (opcional) | 6379 | Caché de embeddings |

---

## Notas de implementación

1. **sentence-transformers es pesado** (~500MB). El modelo se descarga automáticamente al primer uso. En producción, pre-descargar en el Dockerfile.

2. **torch CPU vs CUDA**: En WSL2 sin GPU, torch usa CPU. Los modelos son pequeños (MiniLM = 22MB) y rápidos en CPU para embeddings de texto.

3. **Memgraph es opcional**: El sistema degrada gracefully si no está disponible. Los endpoints de graph_search retornan `[]` con warning en lugar de error.

4. **networkingMode=mirrored en WSL2**: Todos los puertos Docker (5432, 6333, 7687) son accesibles desde `localhost` tanto en WSL2 como en Windows.

5. **MCP server**: Usa `mcp==1.0.0` con FastMCP. Se inicia via `python -m src.mcp.v2.server` y expone las herramientas por stdio.
