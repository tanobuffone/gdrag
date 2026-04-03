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
    CommunicationPattern,
    MessageType,
    ContextRequest,
    ContextResponse,
    SharedContext,
    CommunicationMessage,
    CollaborationRequest,
    CollaborationResponse,
)

from .tasks import (
    TaskStatus,
    TaskPriority,
    Task,
    TaskHandoff,
    TaskEvent,
)

from .agent import (
    HealthStatus,
    AgentEndpoint,
    AgentRegistration,
    AgentInfo,
    AgentHeartbeat,
    AgentRegistryStats,
)

from .events import (
    Event,
    EventType,
    StreamName,
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
    # Agent registry models
    "HealthStatus",
    "AgentEndpoint",
    "AgentRegistration",
    "AgentInfo",
    "AgentHeartbeat",
    "AgentRegistryStats",
    # Event models
    "Event",
    "EventType",
    "StreamName",
]
