"""State Persistence models for gdrag v3.

Defines data structures for agent state snapshots, decision tracking,
and cognitive state persistence across sessions.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ============================================================================
# Agent State Models
# ============================================================================

class AgentState(BaseModel):
    """Snapshot of an agent's cognitive state.

    Captures the full state of an agent at a point in time, including
    active sessions, recent query history, attention focus, pending tasks,
    and performance metrics. Used for state recovery and continuity.
    """
    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique agent identifier (e.g., 'cline', 'claude-prod')"
    )
    snapshot_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique snapshot identifier (uuid4)"
    )
    active_sessions: List[str] = Field(
        default_factory=list,
        description="List of currently active session IDs"
    )
    recent_queries: List[str] = Field(
        default_factory=list,
        description="Recent query texts for context continuity"
    )
    attention_focus: Optional[str] = Field(
        default=None,
        description="Current area of focus or topic the agent is attending to"
    )
    pending_tasks: List[str] = Field(
        default_factory=list,
        description="List of pending task IDs assigned to the agent"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance metrics (queries_processed, avg_latency_ms, error_rate, etc.)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the snapshot was created"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional state metadata"
    )


# ============================================================================
# Decision Tracking Models
# ============================================================================

class Decision(BaseModel):
    """Record of an agent decision.

    Tracks decisions made by agents including the context, reasoning,
    and outcome for audit trails and learning from past decisions.
    """
    decision_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique decision identifier (uuid4)"
    )
    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Agent that made the decision"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual information available at decision time"
    )
    decision: str = Field(
        ...,
        min_length=1,
        description="The decision made (action, route, response, etc.)"
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Explanation of why this decision was made"
    )
    outcome: Optional[str] = Field(
        default=None,
        description="Result or outcome of the decision (filled after execution)"
    )
    outcome_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Quantitative metrics about the outcome (latency, accuracy, etc.)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the decision was recorded"
    )
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the decision outcome was recorded"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional decision metadata"
    )

    def resolve(self, outcome: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Mark this decision as resolved with an outcome.

        Args:
            outcome: The result or outcome of the decision.
            metrics: Optional quantitative metrics about the outcome.
        """
        self.outcome = outcome
        self.outcome_metrics = metrics or {}
        self.resolved_at = datetime.now(timezone.utc)

    def is_resolved(self) -> bool:
        """Check if this decision has been resolved."""
        return self.outcome is not None
