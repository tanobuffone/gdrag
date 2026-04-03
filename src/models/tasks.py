"""Task models for gdrag v3 task queue.

Defines Task and TaskStatus data structures used by the TaskQueue.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Possible states for a task in the queue."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """A unit of work in the task queue.

    Attributes:
        task_id: Unique identifier for the task.
        queue: Queue name this task belongs to.
        name: Human-readable task name/function identifier.
        payload: Arbitrary data needed to execute the task.
        priority: Task priority (higher values = higher priority, default 0).
        status: Current task status.
        result: Task execution result (set on completion).
        error: Error message (set on failure).
        retries: Number of retry attempts made.
        max_retries: Maximum retries allowed before permanent failure.
        worker_id: ID of the worker currently processing this task.
        created_at: Timestamp when task was created.
        started_at: Timestamp when task execution began.
        completed_at: Timestamp when task finished (success or failure).
        metadata: Additional metadata for the task.
    """

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    queue: str = Field(default="default", description="Queue name")
    name: str = Field(description="Task name / function identifier")
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Task execution data"
    )
    priority: int = Field(default=0, description="Task priority (higher = sooner)")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    result: Optional[Any] = Field(default=None, description="Task result on success")
    error: Optional[str] = Field(default=None, description="Error message on failure")
    retries: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts allowed")
    worker_id: Optional[str] = Field(
        default=None, description="Worker processing this task"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
