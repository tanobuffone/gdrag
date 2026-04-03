"""gdrag v2/v3 - Core module for advanced agent memory and context optimization."""

from .config import (
    EmbeddingConfig,
    ChunkConfig,
    RerankConfig,
    CompressionConfig,
    AttentionConfig,
    AppConfig,
)
from .knowledge_manager import (
    KnowledgeManager,
    VectorStoreProtocol,
    GraphStoreProtocol,
    RelationalStoreProtocol,
    TenantManagerProtocol,
    DocumentChunkerProtocol,
)

from .tenant_manager import (
    TenantManager,
    TenantAccessDeniedError,
    QuotaExceededError,
    TenantNotFoundError,
)

from .task_queue import TaskQueue

from .state_manager import StateManager

from .event_bus import (
    EventBus,
    EventHandler,
    DEFAULT_STREAMS,
)

__all__ = [
    "EmbeddingConfig",
    "ChunkConfig",
    "RerankConfig",
    "CompressionConfig",
    "AttentionConfig",
    "AppConfig",
    "TaskQueue",
    "KnowledgeManager",
    "VectorStoreProtocol",
    "GraphStoreProtocol",
    "RelationalStoreProtocol",
    "TenantManagerProtocol",
    "DocumentChunkerProtocol",
    # TenantManager
    "TenantManager",
    "TenantAccessDeniedError",
    "QuotaExceededError",
    "TenantNotFoundError",
    # EventBus
    "EventBus",
    "EventHandler",
    "DEFAULT_STREAMS",
    # StateManager
    "StateManager",
]
