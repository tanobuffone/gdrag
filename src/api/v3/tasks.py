"""gdrag v3 Tasks API.

Provides endpoints for asynchronous task management:
- POST /tasks/enqueue   - Enqueue a new task
- GET  /tasks/{id}/status - Get task status
- POST /tasks/{id}/cancel - Cancel a task
- GET  /tasks/{id}/result - Get task result
- POST /tasks/purge     - Purge completed/cancelled tasks
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .dependencies import TaskQueue, TaskStatus, get_task_queue

logger = logging.getLogger(__name__)

tasks_router = APIRouter(prefix="/api/v3", tags=["gdrag v3 - Tasks"])


# ============================================================================
# Request / Response models
# ============================================================================

class EnqueueRequest(BaseModel):
    """Request body to enqueue a task."""
    task_type: str = Field(..., description="Task type identifier", examples=["ingest", "query", "reindex"])
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task payload")


class PurgeRequest(BaseModel):
    """Request body to purge tasks."""
    status: Optional[str] = Field(
        default=None,
        description="Purge only tasks with this status (pending, running, completed, failed, cancelled). "
                    "Omit to purge ALL tasks.",
    )


class TaskResponse(BaseModel):
    """Standard task response."""
    task_id: str
    task_type: str
    status: str
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class PurgeResponse(BaseModel):
    """Response after purging tasks."""
    purged: int
    status: str = "ok"


# ============================================================================
# Endpoints
# ============================================================================

@tasks_router.post("/tasks/enqueue", response_model=TaskResponse, status_code=201)
async def enqueue_task(
    request: EnqueueRequest,
    queue: TaskQueue = Depends(get_task_queue),
) -> TaskResponse:
    """Enqueue a new asynchronous task.

    Returns the created task with its unique ID and initial status.
    """
    task = queue.enqueue(task_type=request.task_type, payload=request.payload)
    logger.info("Task enqueued: %s (type=%s)", task.task_id, task.task_type)
    return TaskResponse(**task.to_dict())


@tasks_router.get("/tasks/{task_id}/status", response_model=TaskResponse)
async def get_task_status(
    task_id: str,
    queue: TaskQueue = Depends(get_task_queue),
) -> TaskResponse:
    """Get the current status of a task."""
    task = queue.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return TaskResponse(**task.to_dict())


@tasks_router.post("/tasks/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(
    task_id: str,
    queue: TaskQueue = Depends(get_task_queue),
) -> TaskResponse:
    """Cancel a pending or running task.

    Returns 404 if the task does not exist, or 409 if it cannot be cancelled
    (already completed / failed / cancelled).
    """
    task = queue.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    if not queue.cancel(task_id):
        raise HTTPException(
            status_code=409,
            detail=f"Task '{task_id}' cannot be cancelled (status={task.status.value})",
        )
    # Re-fetch after cancel
    task = queue.get(task_id)
    return TaskResponse(**task.to_dict())


@tasks_router.get("/tasks/{task_id}/result", response_model=Dict[str, Any])
async def get_task_result(
    task_id: str,
    queue: TaskQueue = Depends(get_task_queue),
) -> Dict[str, Any]:
    """Get the result of a completed task.

    Returns 404 if the task does not exist, or 409 if the task has not yet
    completed.
    """
    task = queue.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    result = queue.get_result(task_id)
    if result is None:
        raise HTTPException(
            status_code=409,
            detail=f"Task '{task_id}' has no result (status={task.status.value})",
        )
    return result


@tasks_router.post("/tasks/purge", response_model=PurgeResponse)
async def purge_tasks(
    request: PurgeRequest = PurgeRequest(),
    queue: TaskQueue = Depends(get_task_queue),
) -> PurgeResponse:
    """Purge tasks from the queue.

    Optionally filter by status. If no status is given, ALL tasks are removed.
    """
    purge_status: Optional[TaskStatus] = None
    if request.status is not None:
        try:
            purge_status = TaskStatus(request.status)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status '{request.status}'. "
                       f"Valid values: {[s.value for s in TaskStatus]}",
            )
    count = queue.purge(purge_status)
    return PurgeResponse(purged=count)
