# Changelog

Todos los cambios notables en este proyecto se documentan en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [3.0.0] - 2026-04-03

### Added
- **Agent Registry**: Registro, heartbeat, health monitoring y cleanup de agentes (src/core/agent_registry.py)
- **Multi-Tenancy**: Aislamiento por agente con RLS PostgreSQL, namespaces Qdrant, quotas y permisos (src/core/tenant_manager.py)
- **State Persistence**: Snapshots de estado cognitivo, historial de decisiones, auto-save (src/core/state_manager.py)
- **Event Bus**: Pub/sub con Redis Streams, consumer groups, pending/claim (src/core/event_bus.py)
- **Task Queue**: Cola de tareas con prioridades via Redis sorted sets (src/core/task_queue.py)
- **Inter-Agent Communication**: Request-response, broadcast, handoff entre agentes (src/core/agent_comm.py)
- **Knowledge Manager**: Niveles de visibilidad (private/team/shared/public), promoción, access control (src/core/knowledge_manager.py)
- **Task Worker**: Worker asíncrono para procesamiento de tareas (src/workers/task_worker.py)
- **API v3 Agents**: Endpoints para registro, heartbeat, listado, health de agentes (src/api/v3/agents.py)
- **API v3 Tasks**: Endpoints para enqueue, status, cancel, result, purge (src/api/v3/tasks.py)
- **API v3 Knowledge**: Endpoints para ingest, promote, search, revoke access (src/api/v3/knowledge.py)
- **MCP v3**: 11 herramientas multi-agente nuevas (agent_register, agent_heartbeat, agent_list, task_enqueue, task_status, task_cancel, knowledge_promote, context_request, context_share, state_save, state_load)
- **Multi-tenancy en core**: session_manager, vector_store, relational_store con aislamiento por agente
- **Event Integration**: EventBus conectado a session_manager, vector_store, knowledge_manager, agent_registry
- **Models v3**: agent.py, communication.py, events.py, tasks.py, tenant.py, state.py (6 módulos Pydantic)
- **Migrations**: 003_multi_tenancy.sql, 004_agent_registry.sql, 005_state_persistence.sql
- **Tests v3**: test_event_bus.py (39), test_task_queue.py (48), test_agent_registry.py, test_tenant_manager.py, test_agent_comm.py, test_knowledge_manager.py, test_state_manager.py, test_integration_v3.py
- **Protocolo Cline**: Rule 19-gdrag-v3-multi-agent.md, workflow gdrag-v3-setup.md, hooks actualizados

### Changed
- MCP server: de 10 herramientas a 21 (10 v2 + 11 v3)
- main.py: monta routers v2 + v3, placeholders reemplazados con módulos reales
- Hooks PreToolUse/PostToolUse/TaskStart/TaskComplete: actualizados para v3
- Skills memory-rag y memory-dashboard: actualizados con herramientas v3

### Removed
- Rules/09-auto-rag.md (redundante con 12-gdrag.md)
- Hooks/auto-persist.md, context-recovery.md (lógica en hooks bash ejecutables)
- Workflows/session-start.md, session-end.md, auto-persist.md (subsumidos por gdrag-v3-setup.md y hooks)
- Skills/gdrag-agent-ops/ (redundante con memory-rag)

## [2.0.0] - 2026-03-24

### Added
- **Dual Embedding**: Soporte para embeddings locales (sentence-transformers) y OpenAI
- **Chunking Inteligente**: Estrategias sliding window, semántico, párrafo e híbrido
- **Re-ranking**: Cross-encoder para mejorar relevancia de resultados
- **Memoria de Sesión**: Persistencia de contexto entre queries
- **Compresión de Contexto**: Resúmenes automáticos para optimizar tokens
- **Atención Focalizada**: Decaimiento temporal y scoring de relevancia
- **API v2**: Endpoints mejorados con soporte de sesiones
- **MCP v2**: Herramientas MCP con gestión de sesiones
- **Testing**: Tests básicos para config, chunker y schemas
- **Deployment**: script start.sh, .env.example, documentación completa
- **Docker Compose**: Configuración para despliegue con PostgreSQL, Qdrant y Memgraph
- **Integración Cline**: Plan completo de integración con hooks, skills y workflows

### Changed
- Arquitectura completamente rediseñada para soporte multi-agente
- Pipeline de query reescrito con orquestación mejorada
- Configuración centralizada con Pydantic Settings

### Fixed
- Estructura del repositorio corregida

## [1.0.0] - 2026-03-20

### Added
- Implementación inicial del sistema RAG
- Integración básica con Qdrant para almacenamiento vectorial
- API REST básica
- Documentación inicial

---

## Convenciones

- **Added**: Nuevas funcionalidades
- **Changed**: Cambios en funcionalidades existentes
- **Deprecated**: Funcionalidades que serán eliminadas
- **Removed**: Funcionalidades eliminadas
- **Fixed**: Corrección de errores
- **Security**: Cambios relacionados con seguridad
