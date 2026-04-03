"""FastAPI dependency injection for gdrag v3.

Provides TaskQueue and KnowledgeManager as injectable dependencies.
Stub implementations provided; replace with real backends in production.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import Request


# ============================================================================
# Task types
# ============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task:
    """Represents an asynchronous task in the queue."""

    def __init__(self, task_type: str, payload: Dict[str, Any]) -> None:
        self.task_id: str = uuid.uuid4().hex[:16]
        self.task_type: str = task_type
        self.payload: Dict[str, Any] = payload
        self.status: TaskStatus = TaskStatus.PENDING
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "payload": self.payload,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# ============================================================================
# TaskQueue - in-memory stub
# ============================================================================

class TaskQueue:
    """In-memory task queue (stub).

    Replace with Redis/Celery/RabbitMQ backed implementation for production.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}

    def enqueue(self, task_type: str, payload: Dict[str, Any]) -> Task:
        task = Task(task_type, payload)
        self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            task.status = TaskStatus.CANCELLED
            task.updated_at = datetime.utcnow()
            return True
        return False

    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if task.status != TaskStatus.COMPLETED:
            return None
        return task.result

    def purge(self, status: Optional[TaskStatus] = None) -> int:
        if status is None:
            count = len(self._tasks)
            self._tasks.clear()
            return count
        to_remove = [tid for tid, t in self._tasks.items() if t.status == status]
        for tid in to_remove:
            del self._tasks[tid]
        return len(to_remove)

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in self._tasks.values():
            key = t.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts


# ============================================================================
# KnowledgeManager - in-memory stub
# ============================================================================

class KnowledgeEntry:
    """A single knowledge entry."""

    def __init__(
        self,
        content: str,
        title: str = "",
        domain: Optional[str] = None,
        source: str = "api",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.doc_id: str = uuid.uuid4().hex[:16]
        self.content: str = content
        self.title: str = title
        self.domain: Optional[str] = domain
        self.source: str = source
        self.metadata: Dict[str, Any] = metadata or {}
        self.promoted: bool = False
        self.access_control: Dict[str, Any] = {"public": True}
        self.created_at: datetime = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "domain": self.domain,
            "source": self.source,
            "promoted": self.promoted,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class KnowledgeManager:
    """In-memory knowledge manager (stub).

    Replace with a real persistence layer (PostgreSQL + Qdrant) for production.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, KnowledgeEntry] = {}

    def ingest(
        self,
        content: str,
        title: str = "",
        domain: Optional[str] = None,
        source: str = "api",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeEntry:
        entry = KnowledgeEntry(content, title, domain, source, metadata)
        self._entries[entry.doc_id] = entry
        return entry

    def get(self, doc_id: str) -> Optional[KnowledgeEntry]:
        return self._entries.get(doc_id)

    def promote(self, doc_id: str) -> bool:
        entry = self._entries.get(doc_id)
        if entry is None:
            return False
        entry.promoted = True
        return True

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> List[KnowledgeEntry]:
        query_lower = query.lower()
        results: List[KnowledgeEntry] = []
        for entry in self._entries.values():
            if domain and entry.domain != domain:
                continue
            if query_lower in entry.content.lower() or query_lower in entry.title.lower():
                results.append(entry)
            if len(results) >= limit:
                break
        return results

    def revoke_access(self, doc_id: str) -> bool:
        entry = self._entries.get(doc_id)
        if entry is None:
            return False
        entry.access_control["public"] = False
        return True


# ============================================================================
# FastAPI dependencies
# ============================================================================

_task_queue: Optional[TaskQueue] = None
_knowledge_manager: Optional[KnowledgeManager] = None


def _get_or_create_task_queue() -> TaskQueue:
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue


def _get_or_create_knowledge_manager() -> KnowledgeManager:
    global _knowledge_manager
    if _knowledge_manager is None:
        _knowledge_manager = KnowledgeManager()
    return _knowledge_manager


def get_task_queue(request: Request) -> TaskQueue:
    """Dependency: returns the TaskQueue singleton."""
    queue: Optional[TaskQueue] = getattr(request.app.state, "task_queue", None)
    if queue is not None:
        return queue
    return _get_or_create_task_queue()


def get_knowledge_manager(request: Request) -> KnowledgeManager:
    """Dependency: returns the KnowledgeManager singleton."""
    manager: Optional[KnowledgeManager] = getattr(
        request.app.state, "knowledge_manager", None
    )
    if manager is not None:
        return manager
    return _get_or_create_knowledge_manager()


# ============================================================================
# AgentRegistry dependency
# ============================================================================

_agent_registry_singleton: Optional["AgentRegistry"] = None


def get_agent_registry(request: Request) -> "AgentRegistry":
    """Return the shared AgentRegistry singleton.

    Creates the instance on first call using the app's AppConfig.
    """
    global _agent_registry_singleton
    if _agent_registry_singleton is None:
        from ..core.agent_registry import AgentRegistry as _AR
        from ..core.config import AppConfig

        config = AppConfig()
        _agent_registry_singleton = _AR(config)
    return _agent_registry_singleton
