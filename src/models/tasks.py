"""Task models for gdrag v3.

Defines data structures for task queue processing and agent task handoff,
tracking, and lifecycle management between agents in the multi-agent system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ============================================================================
# Task Status (combined queue + agent communication)
# ============================================================================

class TaskStatus(str, Enum):
    """Task lifecycle status.

    Supports both queue-based processing and agent-to-agent handoff.
    """
    # Queue states (used by TaskWorker)
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    # Agent communication states (used by AgentCommunication)
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    # Terminal states
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Task Model
# ============================================================================

class Task(BaseModel):
    """A unit of work that can be processed by queue or handed off between agents.

    Attributes:
        task_id: Unique identifier for the task.
        queue: Queue name this task belongs to (for TaskWorker).
        name: Human-readable task name/function identifier.
        title: Task title (for agent communication display).
        description: Detailed task description (for agent handoff).
        payload: Arbitrary data needed to execute the task.
        priority: Task priority (int for queue, enum for agent comm).
        status: Current task status.
        context: Context data for agent handoff.
        domain: Knowledge domain.
        tags: Task tags for categorization.
        result: Task execution result (set on completion).
        error: Error message (set on failure).
        retries: Number of retry attempts made.
        max_retries: Maximum retries allowed before permanent failure.
        worker_id: ID of the worker currently processing this task.
        created_by: Agent that created the task (for handoff).
        assigned_to: Agent currently assigned to the task.
        parent_task_id: Parent task ID for subtasks.
        created_at: Timestamp when task was created.
        started_at: Timestamp when task execution began.
        updated_at: Timestamp of last update.
        completed_at: Timestamp when task finished (success or failure).
        metadata: Additional metadata for the task.
    """

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    # Queue fields
    queue: str = Field(default="default", description="Queue name")
    name: str = Field(default="", description="Task name / function identifier")
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Task execution data"
    )
    priority_int: int = Field(
        default=0, description="Task priority for queue (higher = sooner)"
    )
    # Agent communication fields
    title: str = Field(default="", description="Task title for agent display")
    description: str = Field(default="", description="Detailed task description")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    context: str = Field(default="", description="Context data for handoff")
    domain: Optional[str] = Field(default=None, description="Knowledge domain")
    tags: List[str] = Field(default_factory=list)
    # Common fields
    result: Optional[Any] = Field(default=None, description="Task result on success")
    error: Optional[str] = Field(default=None, description="Error message on failure")
    retries: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts allowed")
    worker_id: Optional[str] = Field(
        default=None, description="Worker processing this task"
    )
    created_by: str = Field(default="", description="Agent that created the task")
    assigned_to: Optional[str] = Field(
        default=None, description="Agent currently assigned to the task"
    )
    parent_task_id: Optional[str] = Field(
        default=None, description="Parent task ID for subtasks"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Task Handoff Model (for AgentCommunication)
# ============================================================================

class TaskHandoff(BaseModel):
    """Record of a task handoff between agents."""
    handoff_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str = Field(description="Task being handed off")
    from_agent: str = Field(description="Agent handing off")
    to_agent: str = Field(description="Agent receiving handoff")
    reason: str = Field(default="", description="Reason for handoff")
    context: str = Field(
        default="",
        description="Additional context for receiving agent"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Task Event Model (for AgentCommunication)
# ============================================================================

class TaskEvent(BaseModel):
    """Event emitted during task lifecycle."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str = Field(description="Associated task ID")
    event_type: str = Field(description="Event type")
    agent_id: str = Field(description="Agent that triggered event")
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
