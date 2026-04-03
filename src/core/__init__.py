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

from .task_queue import TaskQueue

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
    # EventBus
    "EventBus",
    "EventHandler",
    "DEFAULT_STREAMS",
]
