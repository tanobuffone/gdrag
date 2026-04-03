"""gdrag v3 - Data models module."""

from .schemas import (
    EmbeddingResult,
    Chunk,
    EnhancedQueryRequest,
    EnhancedQueryResponse,
    RankedResult,
    ResponseMetadata,
    SessionMemory,
    QueryRecord,
    KnowledgeItem,
    ConceptRelation,
    CollectionStats,
    DomainStats,
)

from .tenant import (
    TenantStatus,
    Tenant,
    TenantPermissions,
    TenantQuotas,
    QuotaUsage,
    TenantCreateRequest,
    TenantUpdateRequest,
    TenantResponse,
)

from .communication import (
    # Communication enums
    CommunicationPattern,
    MessageType,
    # Context models
    ContextRequest,
    ContextResponse,
    SharedContext,
    # Message models
    CommunicationMessage,
    # Collaboration models
    CollaborationRequest,
    CollaborationResponse,
)

from .tasks import (
    # Task enums
    TaskStatus,
    TaskPriority,
    # Task models
    Task,
    TaskHandoff,
    TaskEvent,
)

__all__ = [
    # Schema models
    "EmbeddingResult",
    "Chunk",
    "EnhancedQueryRequest",
    "EnhancedQueryResponse",
    "RankedResult",
    "ResponseMetadata",
    "SessionMemory",
    "QueryRecord",
    "KnowledgeItem",
    "ConceptRelation",
    "CollectionStats",
    "DomainStats",
    # Tenant models
    "TenantStatus",
    "Tenant",
    "TenantPermissions",
    "TenantQuotas",
    "QuotaUsage",
    "TenantCreateRequest",
    "TenantUpdateRequest",
    "TenantResponse",
    # Communication models
    "CommunicationPattern",
    "MessageType",
    "ContextRequest",
    "ContextResponse",
    "SharedContext",
    "CommunicationMessage",
    "CollaborationRequest",
    "CollaborationResponse",
    # Task models
    "TaskStatus",
    "TaskPriority",
    "Task",
    "TaskHandoff",
    "TaskEvent",
]
