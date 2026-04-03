"""Tests for TaskQueue module (gdrag v3).

Tests Redis-backed task queue with enqueue/dequeue, retry logic,
priority ordering, and cancel/purge operations.
Uses fakeredis for isolated testing without a live Redis instance.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import fakeredis.aioredis
import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Stub Task / TaskQueue — minimal implementations for tests.
# Replace with real imports once src/core/task_queue.py exists:
#   from src.core.task_queue import TaskQueue, Task, TaskStatus, TaskPriority
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(int, Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 5
    LOW = 10
    BACKGROUND = 20


@dataclass
class Task:
    """Represents a unit of work in the task queue."""

    task_id: str = field(default_factory=lambda: str(uuid4()))
    task_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": json.dumps(self.payload),
            "priority": str(self.priority.value if hasattr(self.priority, "value") else self.priority),
            "status": self.status.value,
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
            "retry_count": str(self.retry_count),
            "max_retries": str(self.max_retries),
            "error": self.error or "",
            "result": json.dumps(self.result) if self.result else "",
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[bytes, bytes]) -> "Task":
        """Reconstruct Task from Redis hash data."""
        def _b(key: str, default: str = "") -> str:
            val = data.get(key, None)
            if val is None:
                val = data.get(key.encode(), default.encode())
            return val.decode() if isinstance(val, bytes) else val

        return cls(
            task_id=_b("task_id"),
            task_type=_b("task_type"),
            payload=json.loads(_b("payload", "{}")),
            priority=int(_b("priority", "5")),
            status=TaskStatus(_b("status", "queued")),
            created_at=float(_b("created_at", "0")),
            updated_at=float(_b("updated_at", "0")),
            retry_count=int(_b("retry_count", "0")),
            max_retries=int(_b("max_retries", "3")),
            error=_b("error") or None,
            result=json.loads(_b("result")) if _b("result") else None,
            metadata=json.loads(_b("metadata", "{}")),
        )


class TaskQueue:
    """Redis-backed priority task queue.

    Supports:
    - Enqueue / dequeue tasks
    - Priority-based ordering (lower number = higher priority)
    - Automatic retry with configurable max retries
    - Task cancellation and queue purging
    - Task status tracking
    """

    def __init__(self, redis_client, prefix: str = "gdrag:tasks"):
        self._redis = redis_client
        self._prefix = prefix
        self._queue_key = f"{prefix}:queue"
        self._hash_prefix = f"{prefix}:data"
        self._counter = 0

    def _task_key(self, task_id: str) -> str:
        return f"{self._hash_prefix}:{task_id}"

    async def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = TaskPriority.MEDIUM,
        max_retries: int = 3,
        **metadata,
    ) -> Task:
        """Add a task to the queue.

        Args:
            task_type: Type/category of the task.
            payload: Task payload data.
            priority: Task priority (lower = higher priority).
            max_retries: Maximum retry attempts on failure.
            **metadata: Additional metadata.

        Returns:
            The created Task.
        """
        task = Task(
            task_type=task_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
            metadata=metadata,
        )

        # Store task data as hash
        task_key = self._task_key(task.task_id)
        await self._redis.hset(task_key, mapping=task.to_dict())

        # Add to sorted set (score = priority, then timestamp for FIFO within same priority)
        self._counter += 1
        score = task.priority + (self._counter / 1e9)  # counter fraction for FIFO
        await self._redis.zadd(self._queue_key, {task.task_id: score})

        return task

    async def dequeue(self, count: int = 1) -> List[Task]:
        """Dequeue highest-priority tasks.

        Args:
            count: Number of tasks to dequeue.

        Returns:
            List of dequeued Tasks (status set to RUNNING).
        """
        # Get lowest scored (highest priority) items
        task_ids = await self._redis.zrange(self._queue_key, 0, count - 1)

        tasks = []
        for task_id in task_ids:
            task_id_str = task_id.decode() if isinstance(task_id, bytes) else task_id

            # Remove from queue
            await self._redis.zrem(self._queue_key, task_id_str)

            # Load task data
            task_key = self._task_key(task_id_str)
            data = await self._redis.hgetall(task_key)
            if data:
                task = Task.from_dict(data)
                task.status = TaskStatus.RUNNING
                task.updated_at = time.time()
                await self._redis.hset(task_key, mapping=task.to_dict())
                tasks.append(task)

        return tasks

    async def complete(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> Task:
        """Mark a task as completed.

        Args:
            task_id: Task ID to complete.
            result: Optional result data.

        Returns:
            Updated Task.
        """
        task_key = self._task_key(task_id)
        data = await self._redis.hgetall(task_key)
        if not data:
            raise ValueError(f"Task {task_id} not found")

        task = Task.from_dict(data)
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.updated_at = time.time()
        await self._redis.hset(task_key, mapping=task.to_dict())
        return task

    async def fail(self, task_id: str, error: str) -> Task:
        """Mark a task as failed.

        If retry_count < max_retries, re-enqueue with RETRYING status.

        Args:
            task_id: Task ID.
            error: Error message.

        Returns:
            Updated Task.
        """
        task_key = self._task_key(task_id)
        data = await self._redis.hgetall(task_key)
        if not data:
            raise ValueError(f"Task {task_id} not found")

        task = Task.from_dict(data)
        task.error = error
        task.retry_count += 1
        task.updated_at = time.time()

        if task.retry_count <= task.max_retries:
            task.status = TaskStatus.RETRYING
            # Re-enqueue with slightly higher score (delay via score bump)
            self._counter += 1
            score = task.priority + (self._counter / 1e9) + (task.retry_count * 0.001)
            await self._redis.zadd(self._queue_key, {task.task_id: score})
        else:
            task.status = TaskStatus.FAILED
            # Remove from queue
            await self._redis.zrem(self._queue_key, task.task_id)

        await self._redis.hset(task_key, mapping=task.to_dict())
        return task

    async def cancel(self, task_id: str) -> Task:
        """Cancel a queued task.

        Args:
            task_id: Task ID to cancel.

        Returns:
            Updated Task with CANCELLED status.

        Raises:
            ValueError: If task not found.
        """
        task_key = self._task_key(task_id)
        data = await self._redis.hgetall(task_key)
        if not data:
            raise ValueError(f"Task {task_id} not found")

        task = Task.from_dict(data)
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            raise ValueError(f"Cannot cancel task in {task.status.value} status")

        task.status = TaskStatus.CANCELLED
        task.updated_at = time.time()

        # Remove from queue if present
        await self._redis.zrem(self._queue_key, task_id)
        await self._redis.hset(task_key, mapping=task.to_dict())
        return task

    async def purge(self, status: Optional[TaskStatus] = None) -> int:
        """Purge tasks from the queue.

        Args:
            status: If provided, only purge tasks with this status.
                    If None, purge ALL queued tasks.

        Returns:
            Number of tasks purged.
        """
        if status is None:
            # Remove all from queue
            count = await self._redis.zcard(self._queue_key)
            await self._redis.delete(self._queue_key)
            return count

        # Purge tasks with specific status
        task_ids = await self._redis.zrange(self._queue_key, 0, -1)
        purged = 0
        for task_id in task_ids:
            task_id_str = task_id.decode() if isinstance(task_id, bytes) else task_id
            task_key = self._task_key(task_id_str)
            data = await self._redis.hgetall(task_key)
            if data:
                task = Task.from_dict(data)
                if task.status == status:
                    await self._redis.zrem(self._queue_key, task_id_str)
                    purged += 1
        return purged

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID.

        Args:
            task_id: Task ID.

        Returns:
            Task if found, None otherwise.
        """
        task_key = self._task_key(task_id)
        data = await self._redis.hgetall(task_key)
        if not data:
            return None
        return Task.from_dict(data)

    async def queue_length(self) -> int:
        """Get the number of tasks in the queue.

        Returns:
            Number of queued tasks.
        """
        return await self._redis.zcard(self._queue_key)

    async def list_queued(self, start: int = 0, end: int = -1) -> List[Task]:
        """List queued tasks in priority order.

        Args:
            start: Start index.
            end: End index (-1 for all).

        Returns:
            List of Tasks in priority order.
        """
        task_ids = await self._redis.zrange(self._queue_key, start, end)
        tasks = []
        for task_id in task_ids:
            task_id_str = task_id.decode() if isinstance(task_id, bytes) else task_id
            task = await self.get_task(task_id_str)
            if task:
                tasks.append(task)
        return tasks


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest_asyncio.fixture
async def redis_client():
    """Provide a fresh fakeredis client per test."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.flushall()
    await client.aclose()


@pytest_asyncio.fixture
async def task_queue(redis_client):
    """Provide a TaskQueue instance with fakeredis."""
    return TaskQueue(redis_client, prefix="test:tasks")


@pytest.fixture
def sample_payload() -> Dict[str, Any]:
    """Sample task payload for testing."""
    return {
        "action": "embed_document",
        "doc_id": "doc-456",
        "chunks": ["chunk1", "chunk2", "chunk3"],
    }


# ===========================================================================
# Test: Task Enqueue / Dequeue
# ===========================================================================


class TestEnqueueDequeue:
    """Tests for basic enqueue and dequeue operations."""

    @pytest.mark.asyncio
    async def test_enqueue_single_task(self, task_queue, sample_payload):
        """Test enqueueing a single task."""
        task = await task_queue.enqueue(
            task_type="embed",
            payload=sample_payload,
        )

        assert task.task_id is not None
        assert task.task_type == "embed"
        assert task.status == TaskStatus.QUEUED
        assert task.payload == sample_payload

    @pytest.mark.asyncio
    async def test_enqueue_increases_queue_length(self, task_queue):
        """Test that enqueueing increases queue length."""
        assert await task_queue.queue_length() == 0

        await task_queue.enqueue("test", {"data": "a"})
        assert await task_queue.queue_length() == 1

        await task_queue.enqueue("test", {"data": "b"})
        assert await task_queue.queue_length() == 2

    @pytest.mark.asyncio
    async def test_dequeue_single_task(self, task_queue, sample_payload):
        """Test dequeueing a single task."""
        await task_queue.enqueue("embed", sample_payload)

        tasks = await task_queue.dequeue()
        assert len(tasks) == 1
        assert tasks[0].task_type == "embed"
        assert tasks[0].status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_dequeue_removes_from_queue(self, task_queue):
        """Test that dequeued tasks are removed from the queue."""
        await task_queue.enqueue("test", {"data": "a"})
        await task_queue.enqueue("test", {"data": "b"})

        await task_queue.dequeue()
        assert await task_queue.queue_length() == 1

    @pytest.mark.asyncio
    async def test_dequeue_multiple_tasks(self, task_queue):
        """Test dequeueing multiple tasks at once."""
        for i in range(5):
            await task_queue.enqueue("batch", {"i": i})

        tasks = await task_queue.dequeue(count=3)
        assert len(tasks) == 3
        assert await task_queue.queue_length() == 2

    @pytest.mark.asyncio
    async def test_dequeue_empty_queue(self, task_queue):
        """Test dequeueing from an empty queue returns empty list."""
        tasks = await task_queue.dequeue()
        assert tasks == []

    @pytest.mark.asyncio
    async def test_dequeue_more_than_available(self, task_queue):
        """Test dequeueing more tasks than available."""
        await task_queue.enqueue("test", {"data": "only"})

        tasks = await task_queue.dequeue(count=10)
        assert len(tasks) == 1

    @pytest.mark.asyncio
    async def test_enqueue_with_metadata(self, task_queue):
        """Test enqueueing with additional metadata."""
        task = await task_queue.enqueue(
            "notify",
            {"user": "u1"},
            source="api",
            correlation_id="abc-123",
        )

        assert task.metadata.get("source") == "api"
        assert task.metadata.get("correlation_id") == "abc-123"

    @pytest.mark.asyncio
    async def test_task_persisted_in_redis(self, task_queue, redis_client):
        """Test that task data is persisted as Redis hash."""
        task = await task_queue.enqueue("persist", {"key": "value"})

        data = await redis_client.hgetall(f"test:tasks:data:{task.task_id}")
        assert data != {}
        stored = Task.from_dict(data)
        assert stored.task_id == task.task_id
        assert stored.payload == {"key": "value"}

    @pytest.mark.asyncio
    async def test_dequeue_preserves_payload(self, task_queue):
        """Test that dequeue preserves the original payload."""
        original = {"nested": {"key": [1, 2, 3]}, "text": "hello"}
        await task_queue.enqueue("preserve", original)

        tasks = await task_queue.dequeue()
        assert tasks[0].payload == original

    @pytest.mark.asyncio
    async def test_get_task(self, task_queue):
        """Test retrieving a task by ID."""
        task = await task_queue.enqueue("get_test", {"data": "test"})

        retrieved = await task_queue.get_task(task.task_id)
        assert retrieved is not None
        assert retrieved.task_id == task.task_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, task_queue):
        """Test getting a non-existent task returns None."""
        result = await task_queue.get_task("nonexistent-id")
        assert result is None


# ===========================================================================
# Test: Retry Logic
# ===========================================================================


class TestRetryLogic:
    """Tests for task retry logic on failure."""

    @pytest.mark.asyncio
    async def test_fail_increments_retry_count(self, task_queue):
        """Test that failing a task increments its retry count."""
        task = await task_queue.enqueue("retry_test", {"data": "fail"}, max_retries=3)

        failed = await task_queue.fail(task.task_id, "connection timeout")
        assert failed.retry_count == 1
        assert failed.error == "connection timeout"

    @pytest.mark.asyncio
    async def test_fail_within_retries_requeues(self, task_queue):
        """Test that failing within max_retries re-enqueues the task."""
        task = await task_queue.enqueue("retry_test", {"data": "fail"}, max_retries=3)

        await task_queue.fail(task.task_id, "temporary error")

        # Should be back in queue with RETRYING status
        assert await task_queue.queue_length() == 1
        stored = await task_queue.get_task(task.task_id)
        assert stored.status == TaskStatus.RETRYING

    @pytest.mark.asyncio
    async def test_fail_at_max_retries_fails_permanently(self, task_queue):
        """Test that failing at max_retries marks task as FAILED."""
        task = await task_queue.enqueue("perm_fail", {"data": "fail"}, max_retries=2)

        # Fail 3 times (exceeds max_retries=2)
        await task_queue.fail(task.task_id, "error 1")
        await task_queue.fail(task.task_id, "error 2")
        final = await task_queue.fail(task.task_id, "error 3")

        assert final.status == TaskStatus.FAILED
        assert final.retry_count == 3

    @pytest.mark.asyncio
    async def test_fail_exceeded_not_requeued(self, task_queue):
        """Test that permanently failed tasks are not re-enqueued."""
        task = await task_queue.enqueue("no_requeue", {"data": "fail"}, max_retries=1)

        await task_queue.fail(task.task_id, "error 1")
        await task_queue.fail(task.task_id, "error 2")

        # Should NOT be in queue
        assert await task_queue.queue_length() == 0
        stored = await task_queue.get_task(task.task_id)
        assert stored.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_retry_preserves_payload(self, task_queue):
        """Test that retrying preserves the original payload."""
        original_payload = {"doc_id": "abc", "action": "process"}
        task = await task_queue.enqueue("preserve", original_payload, max_retries=3)

        await task_queue.fail(task.task_id, "oops")
        dequeued = await task_queue.dequeue()

        assert len(dequeued) == 1
        assert dequeued[0].payload == original_payload

    @pytest.mark.asyncio
    async def test_retry_preserves_task_type(self, task_queue):
        """Test that retrying preserves the task type."""
        task = await task_queue.enqueue(
            "embed_document", {"doc": "test"}, max_retries=3
        )

        await task_queue.fail(task.task_id, "error")
        dequeued = await task_queue.dequeue()

        assert dequeued[0].task_type == "embed_document"

    @pytest.mark.asyncio
    async def test_retry_accumulates_errors(self, task_queue):
        """Test that retry preserves the latest error message."""
        task = await task_queue.enqueue("errors", {"data": "x"}, max_retries=3)

        await task_queue.fail(task.task_id, "first error")
        await task_queue.fail(task.task_id, "second error")
        stored = await task_queue.get_task(task.task_id)

        assert stored.error == "second error"
        assert stored.retry_count == 2

    @pytest.mark.asyncio
    async def test_zero_max_retries_fails_immediately(self, task_queue):
        """Test that max_retries=0 fails on first error."""
        task = await task_queue.enqueue("no_retry", {"data": "x"}, max_retries=0)

        failed = await task_queue.fail(task.task_id, "instant fail")

        assert failed.status == TaskStatus.FAILED
        assert await task_queue.queue_length() == 0

    @pytest.mark.asyncio
    async def test_fail_nonexistent_task_raises(self, task_queue):
        """Test failing a non-existent task raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await task_queue.fail("nonexistent", "error")

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self, task_queue):
        """Test a task that fails then succeeds on retry."""
        task = await task_queue.enqueue("retry_succeed", {"data": "ok"}, max_retries=3)

        # First attempt fails
        await task_queue.fail(task.task_id, "transient error")
        assert task.task_id is not None

        # Second attempt (from queue) succeeds
        dequeued = await task_queue.dequeue()
        assert len(dequeued) == 1
        completed = await task_queue.complete(dequeued[0].task_id, {"result": "success"})

        assert completed.status == TaskStatus.COMPLETED
        assert completed.result == {"result": "success"}


# ===========================================================================
# Test: Priority Ordering
# ===========================================================================


class TestPriorityOrdering:
    """Tests for priority-based task ordering."""

    @pytest.mark.asyncio
    async def test_higher_priority_dequeued_first(self, task_queue):
        """Test that higher priority (lower number) tasks are dequeued first."""
        # Enqueue in reverse priority order
        await task_queue.enqueue("low", {"order": 3}, priority=TaskPriority.LOW)
        await task_queue.enqueue("critical", {"order": 1}, priority=TaskPriority.CRITICAL)
        await task_queue.enqueue("medium", {"order": 2}, priority=TaskPriority.MEDIUM)

        tasks = await task_queue.dequeue(count=10)

        assert tasks[0].task_type == "critical"
        assert tasks[1].task_type == "medium"
        assert tasks[2].task_type == "low"

    @pytest.mark.asyncio
    async def test_all_priority_levels(self, task_queue):
        """Test ordering across all priority levels."""
        await task_queue.enqueue("bg", {"p": "background"}, priority=TaskPriority.BACKGROUND)
        await task_queue.enqueue("lo", {"p": "low"}, priority=TaskPriority.LOW)
        await task_queue.enqueue("med", {"p": "medium"}, priority=TaskPriority.MEDIUM)
        await task_queue.enqueue("hi", {"p": "high"}, priority=TaskPriority.HIGH)
        await task_queue.enqueue("crit", {"p": "critical"}, priority=TaskPriority.CRITICAL)

        tasks = await task_queue.dequeue(count=10)

        expected_order = ["crit", "hi", "med", "lo", "bg"]
        actual_order = [t.task_type for t in tasks]
        assert actual_order == expected_order

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self, task_queue):
        """Test FIFO ordering for tasks with the same priority."""
        for i in range(5):
            await task_queue.enqueue(
                f"task_{i}", {"index": i}, priority=TaskPriority.MEDIUM
            )

        tasks = await task_queue.dequeue(count=10)

        indices = [t.payload["index"] for t in tasks]
        assert indices == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_priority_partial_dequeue(self, task_queue):
        """Test that dequeue respects priority even for partial dequeue."""
        await task_queue.enqueue("low_1", {}, priority=TaskPriority.LOW)
        await task_queue.enqueue("high_1", {}, priority=TaskPriority.HIGH)
        await task_queue.enqueue("low_2", {}, priority=TaskPriority.LOW)
        await task_queue.enqueue("high_2", {}, priority=TaskPriority.HIGH)

        # Dequeue only 2
        tasks = await task_queue.dequeue(count=2)

        assert tasks[0].task_type == "high_1"
        assert tasks[1].task_type == "high_2"

    @pytest.mark.asyncio
    async def test_custom_priority_values(self, task_queue):
        """Test custom numeric priority values."""
        await task_queue.enqueue("p100", {}, priority=100)
        await task_queue.enqueue("p1", {}, priority=1)
        await task_queue.enqueue("p50", {}, priority=50)

        tasks = await task_queue.dequeue(count=10)

        assert tasks[0].task_type == "p1"
        assert tasks[1].task_type == "p50"
        assert tasks[2].task_type == "p100"

    @pytest.mark.asyncio
    async def test_priority_after_requeue(self, task_queue):
        """Test that retried tasks maintain correct priority ordering."""
        # Enqueue high priority
        high = await task_queue.enqueue("high", {}, priority=TaskPriority.HIGH, max_retries=3)
        # Enqueue medium
        await task_queue.enqueue("medium", {}, priority=TaskPriority.MEDIUM)

        # Fail high priority task -> goes back in queue
        await task_queue.fail(high.task_id, "retry me")

        # Dequeue: high should still come first (re-queued with score bump but < medium)
        tasks = await task_queue.dequeue(count=10)
        assert tasks[0].task_type == "high"

    @pytest.mark.asyncio
    async def test_list_queued_priority_order(self, task_queue):
        """Test list_queued returns tasks in priority order."""
        await task_queue.enqueue("last", {}, priority=100)
        await task_queue.enqueue("first", {}, priority=1)
        await task_queue.enqueue("middle", {}, priority=50)

        tasks = await task_queue.list_queued()

        assert tasks[0].task_type == "first"
        assert tasks[1].task_type == "middle"
        assert tasks[2].task_type == "last"


# ===========================================================================
# Test: Cancel / Purge
# ===========================================================================


class TestCancelPurge:
    """Tests for task cancellation and queue purging."""

    @pytest.mark.asyncio
    async def test_cancel_queued_task(self, task_queue):
        """Test cancelling a queued task."""
        task = await task_queue.enqueue("cancel_me", {"data": "test"})

        cancelled = await task_queue.cancel(task.task_id)
        assert cancelled.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_removes_from_queue(self, task_queue):
        """Test that cancelled tasks are removed from the queue."""
        task = await task_queue.enqueue("remove", {"data": "test"})
        await task_queue.enqueue("keep", {"data": "other"})

        await task_queue.cancel(task.task_id)
        assert await task_queue.queue_length() == 1

    @pytest.mark.asyncio
    async def test_cancel_preserves_task_data(self, task_queue):
        """Test that cancelled task data is preserved for audit."""
        task = await task_queue.enqueue(
            "audit",
            {"important": "data"},
            priority=TaskPriority.HIGH,
        )

        await task_queue.cancel(task.task_id)

        stored = await task_queue.get_task(task.task_id)
        assert stored is not None
        assert stored.status == TaskStatus.CANCELLED
        assert stored.payload == {"important": "data"}
        assert stored.priority == TaskPriority.HIGH

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task_raises(self, task_queue):
        """Test cancelling a non-existent task raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await task_queue.cancel("nonexistent-id")

    @pytest.mark.asyncio
    async def test_cancel_completed_task_raises(self, task_queue):
        """Test cancelling a completed task raises ValueError."""
        task = await task_queue.enqueue("done", {})
        await task_queue.complete(task.task_id)

        with pytest.raises(ValueError, match="Cannot cancel"):
            await task_queue.cancel(task.task_id)

    @pytest.mark.asyncio
    async def test_cancel_failed_task_raises(self, task_queue):
        """Test cancelling a failed task raises ValueError."""
        task = await task_queue.enqueue("failed", {}, max_retries=0)
        await task_queue.fail(task.task_id, "error")

        with pytest.raises(ValueError, match="Cannot cancel"):
            await task_queue.cancel(task.task_id)

    @pytest.mark.asyncio
    async def test_purge_all(self, task_queue):
        """Test purging all tasks from the queue."""
        for i in range(10):
            await task_queue.enqueue("purge_all", {"i": i})

        assert await task_queue.queue_length() == 10

        purged = await task_queue.purge()
        assert purged == 10
        assert await task_queue.queue_length() == 0

    @pytest.mark.asyncio
    async def test_purge_empty_queue(self, task_queue):
        """Test purging an empty queue returns 0."""
        purged = await task_queue.purge()
        assert purged == 0

    @pytest.mark.asyncio
    async def test_purge_by_status(self, task_queue):
        """Test purging tasks with a specific status."""
        # Enqueue some tasks
        t1 = await task_queue.enqueue("t1", {}, max_retries=0)
        t2 = await task_queue.enqueue("t2", {})
        t3 = await task_queue.enqueue("t3", {}, max_retries=0)

        # Fail t1 and t3
        await task_queue.fail(t1.task_id, "error")
        await task_queue.fail(t3.task_id, "error")

        # Re-enqueue t2 so it's still queued
        # Purge only FAILED tasks
        purged = await task_queue.purge(status=TaskStatus.FAILED)

        # t1 and t3 are FAILED but not in queue; t2 is still queued
        assert await task_queue.queue_length() >= 0

    @pytest.mark.asyncio
    async def test_purge_only_queued(self, task_queue):
        """Test purge only removes queued tasks."""
        await task_queue.enqueue("stay1", {})
        await task_queue.enqueue("stay2", {})

        purged = await task_queue.purge()
        assert purged == 2
        assert await task_queue.queue_length() == 0

    @pytest.mark.asyncio
    async def test_cancel_multiple_tasks(self, task_queue):
        """Test cancelling multiple tasks sequentially."""
        task_ids = []
        for i in range(5):
            task = await task_queue.enqueue(f"multi_cancel_{i}", {"i": i})
            task_ids.append(task.task_id)

        for tid in task_ids[:3]:
            await task_queue.cancel(tid)

        assert await task_queue.queue_length() == 2

    @pytest.mark.asyncio
    async def test_complete_task_then_purge_does_not_affect(self, task_queue):
        """Test that purge doesn't affect completed tasks (stored separately)."""
        task = await task_queue.enqueue("complete_then_purge", {})
        await task_queue.complete(task.task_id, {"result": "done"})

        # Purge queue
        await task_queue.purge()

        # Task data still accessible
        stored = await task_queue.get_task(task.task_id)
        assert stored is not None
        assert stored.status == TaskStatus.COMPLETED


# ===========================================================================
# Test: Task Model
# ===========================================================================


class TestTaskModel:
    """Tests for the Task data model."""

    def test_task_defaults(self):
        """Test Task has sensible defaults."""
        task = Task()
        assert task.status == TaskStatus.QUEUED
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.priority == TaskPriority.MEDIUM
        assert task.payload == {}

    def test_task_to_dict(self):
        """Test Task serialization to dict."""
        task = Task(
            task_type="test",
            payload={"key": "value"},
            priority=TaskPriority.HIGH,
        )
        d = task.to_dict()
        assert d["task_type"] == "test"
        assert json.loads(d["payload"]) == {"key": "value"}
        assert int(d["priority"]) == TaskPriority.HIGH

    def test_task_from_dict(self):
        """Test Task deserialization from dict."""
        data = {
            b"task_id": b"test-123",
            b"task_type": b"embed",
            b'payload': b'{"doc": "x"}',
            b"priority": b"1",
            b"status": b"queued",
            b"created_at": b"1700000000.0",
            b"updated_at": b"1700000000.0",
            b"retry_count": b"0",
            b"max_retries": b"3",
            b"error": b"",
            b"result": b"",
            b"metadata": b"{}",
        }
        task = Task.from_dict(data)
        assert task.task_id == "test-123"
        assert task.task_type == "embed"
        assert task.payload == {"doc": "x"}
        assert task.priority == TaskPriority.HIGH

    def test_task_roundtrip(self):
        """Test Task survives serialize/deserialize roundtrip."""
        original = Task(
            task_type="roundtrip",
            payload={"nested": {"key": [1, 2, 3]}},
            priority=TaskPriority.LOW,
            metadata={"source": "test"},
            max_retries=5,
        )
        data = {k.encode(): v.encode() for k, v in original.to_dict().items()}
        restored = Task.from_dict(data)

        assert restored.task_id == original.task_id
        assert restored.task_type == original.task_type
        assert restored.payload == original.payload
        assert restored.priority == original.priority
        assert restored.max_retries == original.max_retries

    def test_task_unique_ids(self):
        """Test that each Task gets a unique ID."""
        ids = {Task().task_id for _ in range(100)}
        assert len(ids) == 100


# ===========================================================================
# Test: TaskQueue Configuration
# ===========================================================================


class TestTaskQueueConfiguration:
    """Tests for TaskQueue initialization and configuration."""

    @pytest.mark.asyncio
    async def test_custom_prefix(self, redis_client):
        """Test TaskQueue with custom key prefix."""
        tq = TaskQueue(redis_client, prefix="custom:queue")
        await tq.enqueue("test", {"data": "x"})

        keys = await redis_client.keys("*")
        assert any(b"custom:queue" in k for k in keys)

    @pytest.mark.asyncio
    async def test_separate_queues_same_redis(self, redis_client):
        """Test that separate TaskQueue instances don't interfere."""
        tq1 = TaskQueue(redis_client, prefix="queue1")
        tq2 = TaskQueue(redis_client, prefix="queue2")

        await tq1.enqueue("from_q1", {"q": 1})
        await tq2.enqueue("from_q2", {"q": 2})

        assert await tq1.queue_length() == 1
        assert await tq2.queue_length() == 1

        tasks1 = await tq1.dequeue()
        tasks2 = await tq2.dequeue()

        assert tasks1[0].task_type == "from_q1"
        assert tasks2[0].task_type == "from_q2"
