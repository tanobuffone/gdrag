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

__all__ = [
    "EmbeddingConfig",
    "ChunkConfig",
    "RerankConfig",
    "CompressionConfig",
    "AttentionConfig",
    "AppConfig",
    "KnowledgeManager",
    "VectorStoreProtocol",
    "GraphStoreProtocol",
    "RelationalStoreProtocol",
    "TenantManagerProtocol",
    "DocumentChunkerProtocol",
]
