# Plan de Integración gdrag v2 con Cline - Estado de Implementación

## ✅ Fases Completadas

### Fase 1: Registro del MCP Server gdrag ✅
- [x] Registrar gdrag en configuración MCP de Cline (`mcp_settings.json`)
- [x] Crear script de verificación de servicios (`/home/gdrick/gdrag/scripts/check_services.sh`)
- [x] Verificar servicios dependientes (PostgreSQL, Qdrant, Memgraph)

### Fase 2: Skill de Gestión de Memoria gdrag ✅
- [x] Crear skill memory-rag (`/home/gdrick/.agents/skills/memory-rag/SKILL.md`)
- [x] Crear documentación de ejemplos (`/home/gdrick/.agents/skills/memory-rag/examples.md`)
- [x] Crear skill memory-dashboard (`/home/gdrick/.agents/skills/memory-dashboard/SKILL.md`)

### Fase 3: Hooks de Persistencia Automática ✅
- [x] Crear hook de recuperación de contexto (`/home/gdrick/Documents/Cline/Hooks/context-recovery.md`)
- [x] Crear hook de persistencia automática (`/home/gdrick/Documents/Cline/Hooks/auto-persist.md`)
- [x] Configurar eventos a persistir/ignorar

### Fase 4: Workflow de Gestión de Sesiones ✅
- [x] Crear workflow session-start.md (`/home/gdrick/Cline/Workflows/session-start.md`)
- [x] Crear workflow session-end.md (`/home/gdrick/Cline/Workflows/session-end.md`)
- [x] Crear workflow knowledge-sync.md (`/home/gdrick/Cline/Workflows/knowledge-sync.md`)

### Fase 5: Integración Cross-Project ✅
- [x] Crear skill memory-dashboard
- [x] Documentación completa de integración (este archivo)

## Servicios Requeridos
- PostgreSQL: ws-postgres:5432 (database: gdrag)
- Qdrant: localhost:6333
- Memgraph: localhost:7687 (opcional)

## Estructura de Directorios Cline
- **Skills**: `/home/gdrick/.agents/skills/`
  - `memory-rag/`: Skill principal de gestión RAG
  - `memory-dashboard/`: Skill de visualización de estadísticas
- **Hooks**: `/home/gdrick/Documents/Cline/Hooks/`
  - `auto-persist.md`: Hook PostToolUse para persistencia automática
  - `context-recovery.md`: Hook PreToolUse para recuperación de contexto
- **Workflows**: `/home/gdrick/Cline/Workflows/`
  - `session-start.md`: Inicialización de sesión RAG
  - `session-end.md`: Cierre de sesión RAG
  - `knowledge-sync.md`: Sincronización de memory-bank
- **Rules**: `/home/gdrick/Cline/Rules/`
  - `09-auto-rag.md`: Reglas de persistencia automática (existente)

## Herramientas MCP gdrag Disponibles

| Herramienta | Descripción | Uso Principal |
|-------------|-------------|---------------|
| `rag_query` | Query RAG con soporte de sesión | Búsqueda de contexto relevante |
| `semantic_search` | Búsqueda semántica pura | Búsqueda por significado |
| `graph_search` | Búsqueda en grafo de conocimiento | Relaciones entre conceptos |
| `structured_query` | Búsqueda estructurada PostgreSQL | Búsqueda por texto exacto |
| `session_create` | Crear nueva sesión | Inicio de sesión de trabajo |
| `session_context` | Obtener contexto de sesión | Recuperar historial |
| `session_compress` | Comprimir historial de sesión | Optimizar tokens |
| `ingest_knowledge` | Ingestar conocimiento al sistema | Persistir eventos significativos |
| `get_health` | Verificar salud del sistema | Diagnóstico |
| `get_stats` | Obtener estadísticas del sistema | Monitoreo |

## Flujo de Trabajo Recomendado

### Al Iniciar Sesión
1. Ejecutar workflow `session-start.md`
2. Verificar servicios con `get_health`
3. Crear sesión con `session_create`
4. Recuperar contexto con `rag_query`

### Durante la Tarea
1. Persistir eventos significativos con `ingest_knowledge`
2. Consultar contexto con `rag_query` o `semantic_search`
3. Usar `memory-rag` skill para operaciones avanzadas

### Al Cerrar Sesión
1. Ejecutar workflow `session-end.md`
2. Comprimir sesión con `session_compress`
3. Persistir resumen con `ingest_knowledge`
4. Ejecutar `knowledge-sync.md` si se actualizó memory-bank

## Configuración por Proyecto

Cada proyecto puede tener su propio namespace usando el parámetro `domain`:
- `domain="gdrag"`: Para el proyecto gdrag
- `domain="chorduction"`: Para el proyecto chorduction
- `domain="seshat"`: Para el proyecto seshat
- etc.

## Notas de Implementación

1. **Servicios caídos**: El sistema degrada gracefully si gdrag no está disponible
2. **Persistencia selectiva**: Solo se persisten eventos significativos, no lecturas/navegación
3. **Compresión automática**: Las sesiones se comprimen para optimizar uso de tokens
4. **Cross-project**: El conocimiento se aisla por dominio/proyecto
5. **Integración existente**: Se integra con hooks, skills y workflows existentes de Cline
