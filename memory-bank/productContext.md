# Product Context — gdrag

## What is gdrag?

gdrag is an advanced RAG (Retrieval-Augmented Generation) system designed as a **persistent, intelligent memory layer** for AI agents. It provides multi-agent coordination, semantic search, session memory, and state persistence.

## Target Users

- **Cline**: Primary user — coding agent that uses gdrag for context persistence and multi-agent orchestration
- **Claude Code**: Uses gdrag via MCP for memory management
- **Other AI agents**: Any MCP-compatible agent can use gdrag as a drop-in memory backend

## Core Value Proposition

1. **Persistent Memory**: Agents remember context across sessions via PostgreSQL + Qdrant
2. **Multi-Agent Coordination**: Registry, task queue, inter-agent communication
3. **Intelligent Retrieval**: Dual embeddings + cross-encoder re-ranking + attention scoring
4. **Multi-Tenancy**: Isolation per agent with RLS, namespaces, quotas
5. **Graceful Degradation**: Works even when Redis/Memgraph are down

## Architecture

```
Agent → MCP/REST → API v2/v3 → Core Modules → Storage
                                   │
                        ┌──────────┼──────────┐
                   PostgreSQL    Qdrant    Redis
                   (structured)  (vectors) (events/tasks)
```

## Version History

- v1 (2026-03-20): Basic RAG with Qdrant
- v2 (2026-03-24): Advanced pipeline — embeddings, chunker, reranker, compressor, attention, sessions
- v3 (2026-04-03): Multi-agent — registry, tenant manager, state manager, event bus, task queue, inter-agent comm

## Key Decisions

- **Dual embeddings**: Local (sentence-transformers) + OpenAI for flexibility
- **Graceful degradation**: Redis optional, Memgraph optional — system works without them
- **Multi-tenancy via RLS**: PostgreSQL Row Level Security for agent isolation
- **MCP as primary interface**: All functionality accessible via MCP tools
- **Kanban for task management**: Multi-agent execution via Cline Kanban worktrees
