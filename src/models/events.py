"""gdrag v3 - Event models for Redis Streams messaging.

Defines event structures used by the EventBus for inter-component communication.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ============================================================================
# Stream Names
# ============================================================================

class StreamName(str, Enum):
    """Predefined Redis Stream names for gdrag v3."""
    KNOWLEDGE = "gdrag:knowledge"
    SESSIONS = "gdrag:sessions"
    TASKS = "gdrag:tasks"
    AGENTS = "gdrag:agents"
    CONTEXT = "gdrag:context"


# ============================================================================
# Event Types
# ============================================================================

class EventType(str, Enum):
    """Event type identifiers."""
    # Knowledge events
    KNOWLEDGE_INGESTED = "knowledge.ingested"
    KNOWLEDGE_UPDATED = "knowledge.updated"
    KNOWLEDGE_DELETED = "knowledge.deleted"
    KNOWLEDGE_SHARED = "knowledge.shared"

    # Session events
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    SESSION_EXPIRED = "session.expired"
    SESSION_QUERY = "session.query"
    SESSION_COMPRESSED = "session.compressed"

    # Task events
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    # Agent events
    AGENT_REGISTERED = "agent.registered"
    AGENT_HEARTBEAT = "agent.heartbeat"
    AGENT_ERROR = "agent.error"
    AGENT_OFFLINE = "agent.offline"

    # Context events
    CONTEXT_UPDATED = "context.updated"
    CONTEXT_COMPRESSED = "context.compressed"
    CONTEXT_REQUESTED = "context.requested"


# ============================================================================
# Event Model
# ============================================================================

class Event(BaseModel):
    """Event model for Redis Streams messaging.

    Each event carries a type, payload, and metadata for tracing and routing.
    Events are serialized to flat dictionaries for Redis Streams storage.
    """
    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique event identifier"
    )
    event_type: str = Field(
        description="Event type identifier (e.g., 'knowledge.ingested')"
    )
    source: str = Field(
        default="gdrag",
        description="Source component that generated the event"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event creation timestamp (UTC)"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for request tracing"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload data"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    def to_stream_dict(self) -> Dict[str, str]:
        """Serialize event to flat string dictionary for Redis Streams.

        Redis Streams requires all values to be strings. Nested structures
        are JSON-encoded.

        Returns:
            Dictionary with string keys and values suitable for XADD.
        """
        import json
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id or "",
            "payload": json.dumps(self.payload),
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_stream_dict(cls, data: Dict[bytes, bytes]) -> "Event":
        """Deserialize event from Redis Streams message.

        Args:
            data: Raw message data from Redis (bytes keys/values).

        Returns:
            Reconstructed Event instance.
        """
        import json
        return cls(
            event_id=data.get(b"event_id", b"").decode("utf-8"),
            event_type=data.get(b"event_type", b"").decode("utf-8"),
            source=data.get(b"source", b"gdrag").decode("utf-8"),
            timestamp=datetime.fromisoformat(
                data.get(b"timestamp", b"").decode("utf-8")
            ),
            correlation_id=(
                data.get(b"correlation_id", b"").decode("utf-8") or None
            ),
            payload=json.loads(data.get(b"payload", b"{}").decode("utf-8")),
            metadata=json.loads(data.get(b"metadata", b"{}").decode("utf-8")),
        )
