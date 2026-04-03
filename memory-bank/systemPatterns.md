# System Patterns — gdrag

## Code Architecture Patterns

### 1. Core Module Pattern
Every core module follows:
```python
class ModuleName:
    def __init__(self, dependency1, dependency2=None):
        self._dep1 = dependency1
        self._dep2 = dependency2

    def _publish_event(self, stream, event):
        """Graceful event publishing — silently fails if Redis down."""
        if self.event_bus is None:
            return
        try:
            self.event_bus.publish(stream, event)
        except Exception:
            pass
```

### 2. Multi-Tenancy Pattern
All v3 modules accept optional `agent_id` parameter:
- `session_manager`: create_session(agent_id), get verifies ownership
- `vector_store`: namespaces per tenant (tenant_{agent_id}_knowledge)
- `relational_store`: RLS policies filter by agent_id

### 3. Graceful Degradation
Every external dependency is optional:
- Redis → EventBus/TaskQueue fall back to in-memory
- Memgraph → graph_store returns []
- CrossEncoder → SimpleReranker fallback

### 4. Lazy Loading
Models (sentence-transformers, cross-encoder) load on first use, not at import time.

### 5. MCP Tool Pattern
```python
@mcp.tool()
def tool_name(param1: str, param2: int = 10) -> str:
    result = _run_async(core_module.method(param1, param2))
    return json.dumps(result, default=str)
```

### 6. API Router Pattern
```python
router = APIRouter(prefix="/api/v3/thing", tags=["v3"])

@router.post("/action")
async def action(dep: Type = Depends(get_dep)):
    return await dep.method()
```

### 7. Test Pattern
```python
@pytest.mark.asyncio
class TestModuleName:
    @pytest.fixture
    def instance(self):
        return ModuleName(mock_dep=MagicMock())

    async def test_method(self, instance):
        result = await instance.method(args)
        assert result.field == expected
```

## Naming Conventions

- Core modules: `snake_case.py` (e.g., `agent_registry.py`)
- Models: `snake_case.py` (e.g., `agent.py`)
- API routers: descriptive names (e.g., `agents.py`, `tasks.py`)
- Tests: `test_{module}.py` matching `src/core/{module}.py`
- Config: YAML files in `config/`
