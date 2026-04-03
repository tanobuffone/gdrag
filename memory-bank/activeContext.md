# Active Context — gdrag

## Current Status: v3.0.0 Complete

**Last updated**: 2026-04-03

## What We Just Did

1. Completed full v3 implementation via 13 Kanban tasks
2. Created all missing core modules (agent_registry, tenant_manager, state_manager)
3. Created all v3 API endpoints (agents, tasks, knowledge)
4. Created MCP v3 tools (11 new tools, 21 total)
5. Integrated EventBus with existing core modules
6. Created 8 test files with comprehensive coverage
7. Created memory-bank, updated CLAUDE.md, README.md, CHANGELOG.md
8. Cleaned up redundant Rules/Hooks/Skills/Workflows (7 files removed)

## Current Work

- Documentation complete: README, CHANGELOG, CLAUDE.md all updated to v3
- Memory-bank created for project context persistence
- Next step: git commit + push to GitHub

## Architecture Summary

### Core Modules (18 files in src/core/)
- v2: config, embeddings, chunker, reranker, compressor, attention, session_manager, vector_store, graph_store, relational_store, pipeline
- v3: agent_registry, tenant_manager, state_manager, event_bus, task_queue, agent_comm, knowledge_manager

### API Endpoints
- v2: /api/v2/ — query, sessions, ingest, health, stats
- v3: /api/v3/ — agents, tasks, knowledge (3 routers)

### MCP Tools (21)
- v2: rag_query, semantic_search, graph_search, structured_query, session_create, session_context, session_compress, ingest_knowledge, get_health, get_stats
- v3: agent_register, agent_heartbeat, agent_list, task_enqueue, task_status, task_cancel, knowledge_promote, context_request, context_share, state_save, state_load

### Tests (11 files)
- test_config, test_chunker, test_schemas (v2)
- test_event_bus (39), test_task_queue (48), test_agent_registry, test_tenant_manager, test_agent_comm, test_knowledge_manager, test_state_manager, test_integration_v3

## Docker Services

| Service | Port | Status |
|---------|------|--------|
| PostgreSQL 16 | 5434 | Running |
| Qdrant | 6333 | Running |
| Redis 7.2 | 6379 | Optional (v3) |
| Memgraph | 7687 | Optional (graph) |

## Cline Integration

- Rule: `19-gdrag-v3-multi-agent.md` — protocolo multi-agente
- Rule: `12-gdrag.md` — protocolo base v2
- Skill: `memory-rag/SKILL.md` — todas las herramientas v2+v3
- Skill: `memory-dashboard/SKILL.md` — monitoreo
- Hooks: PreToolUse, PostToolUse, TaskStart, TaskComplete (all bash)
- Workflow: `gdrag-v3-setup.md` — setup multi-agente
