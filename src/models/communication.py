"""Communication models for inter-agent communication in gdrag v3.

Defines data structures for agent-to-agent communication patterns:
request-response, broadcast, handoff, and collaborative.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ============================================================================
# Communication Pattern Types
# ============================================================================

class CommunicationPattern(str, Enum):
    """Communication patterns supported by AgentCommunication."""
    REQUEST_RESPONSE = "request_response"
    BROADCAST = "broadcast"
    HANDOFF = "handoff"
    COLLABORATIVE = "collaborative"


class MessageType(str, Enum):
    """Types of messages in agent communication."""
    CONTEXT_REQUEST = "context_request"
    CONTEXT_RESPONSE = "context_response"
    CONTEXT_SHARE = "context_share"
    TASK_HANDOFF = "task_handoff"
    BROADCAST = "broadcast"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"


# ============================================================================
# Context Models
# ============================================================================

class ContextRequest(BaseModel):
    """Request for context from another agent."""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    from_agent: str = Field(description="Requesting agent ID")
    to_agent: str = Field(description="Target agent ID")
    query: str = Field(description="Query for context retrieval")
    max_tokens: int = Field(default=2000, description="Maximum tokens in response")
    domain: Optional[str] = Field(default=None, description="Optional domain filter")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ContextResponse(BaseModel):
    """Response containing context data."""
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str = Field(description="Original request ID")
    from_agent: str = Field(description="Responding agent ID")
    to_agent: str = Field(description="Requesting agent ID")
    context: str = Field(description="Context content")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    token_count: int = Field(default=0, description="Approximate token count")
    source: Optional[str] = Field(default=None, description="Context source")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SharedContext(BaseModel):
    """Shared context between agents."""
    context_id: str = Field(default_factory=lambda: str(uuid4()))
    from_agent: str = Field(description="Sharing agent ID")
    to_agent: Optional[str] = Field(
        default=None,
        description="Target agent ID (None for broadcast)"
    )
    content: str = Field(description="Shared context content")
    domain: Optional[str] = Field(default=None, description="Knowledge domain")
    tags: List[str] = Field(default_factory=list, description="Context tags")
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Optional expiration time"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Communication Message Model
# ============================================================================

class CommunicationMessage(BaseModel):
    """Base message for all agent communications."""
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    message_type: MessageType = Field(description="Type of message")
    pattern: CommunicationPattern = Field(description="Communication pattern")
    from_agent: str = Field(description="Sending agent ID")
    to_agent: Optional[str] = Field(
        default=None,
        description="Receiving agent ID (None for broadcast)"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Message payload"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for request-response"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Collaboration Models
# ============================================================================

class CollaborationRequest(BaseModel):
    """Request for collaborative work between agents."""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    from_agent: str = Field(description="Initiating agent ID")
    to_agents: List[str] = Field(description="Target agent IDs")
    objective: str = Field(description="Collaboration objective")
    context: str = Field(default="", description="Shared context for collaboration")
    max_tokens: int = Field(default=2000, description="Max tokens per response")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CollaborationResponse(BaseModel):
    """Response to a collaboration request."""
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str = Field(description="Original collaboration request ID")
    from_agent: str = Field(description="Responding agent ID")
    contribution: str = Field(description="Agent's contribution")
    status: str = Field(default="completed", description="Response status")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
