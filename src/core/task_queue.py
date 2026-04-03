"""Redis-backed task queue for gdrag v3.

Provides async task queue operations with priority support via Redis sorted sets.
Supports enqueue, dequeue, completion, failure, retry, cancellation, and purging.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as aioredis

from ..models.tasks import Task, TaskStatus

logger = logging.getLogger(__name__)

# Redis key prefixes
QUEUE_KEY = "gdrag:queue:{queue}"
TASK_KEY = "gdrag:task:{task_id}"
RESULT_KEY = "gdrag:result:{task_id}"
PROCESSING_KEY = "gdrag:processing:{worker_id}"


class TaskQueue:
    """Async Redis-backed task queue with priority support.

    Uses Redis sorted sets for priority-based ordering and hashes for task data.
    Tasks are serialized as JSON and stored in Redis hashes.

    Args:
        redis_url: Redis connection URL (e.g., "redis://localhost:6379/0").
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis_url = redis_url
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        """Lazily initialize and return the Redis connection."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
            )
        return self._redis

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    # ─── CORE OPERATIONS ───────────────────────────────────────────────────

    async def enqueue(self, task: Task) -> str:
        """Add a task to its queue with priority ordering.

        Args:
            task: The Task to enqueue.

        Returns:
            The task_id of the enqueued task.
        """
        redis_client = await self._get_redis()

        task.status = TaskStatus.QUEUED
        task_data = task.model_dump_json()

        task_key = TASK_KEY.format(task_id=task.task_id)
        queue_key = QUEUE_KEY.format(queue=task.queue)

        # Store task data as a hash
        await redis_client.hset(
            task_key,
            mapping={
                "data": task_data,
                "queue": task.queue,
                "status": task.status.value,
            },
        )

        # Add to sorted set with priority as score (higher priority = higher score)
        # Use negative score so higher priority values dequeue first (ZRANGEBYSCORE)
        await redis_client.zadd(queue_key, {task.task_id: task.priority})

        logger.info(
            "Task %s enqueued to queue '%s' with priority %d",
            task.task_id,
            task.queue,
            task.priority,
        )
        return task.task_id

    async def dequeue(
        self,
        worker_id: str,
        queues: Optional[List[str]] = None,
        timeout: float = 0,
    ) -> Optional[Task]:
        """Dequeue the highest-priority task from the specified queues.

        Checks queues in order and returns the first available task.
        If `timeout > 0`, blocks until a task is available or timeout expires.

        Args:
            worker_id: Identifier of the worker taking the task.
            queues: List of queue names to check. Defaults to ["default"].
            timeout: Block timeout in seconds (0 = no blocking).

        Returns:
            A Task instance if one was available, otherwise None.
        """
        redis_client = await self._get_redis()

        if queues is None:
            queues = ["default"]

        if timeout > 0:
            return await self._dequeue_blocking(redis_client, worker_id, queues, timeout)

        return await self._dequeue_immediate(redis_client, worker_id, queues)

    async def _dequeue_immediate(
        self,
        redis_client: aioredis.Redis,
        worker_id: str,
        queues: List[str],
    ) -> Optional[Task]:
        """Non-blocking dequeue: pop the highest-priority task."""
        for queue_name in queues:
            queue_key = QUEUE_KEY.format(queue=queue_name)

            # ZPOPMAX returns the member with the highest score (= highest priority)
            result = await redis_client.zpopmax(queue_key, count=1)

            if not result:
                continue

            task_id, _score = result[0]

            task = await self._claim_task(redis_client, task_id, worker_id)
            if task is not None:
                return task

        return None

    async def _dequeue_blocking(
        self,
        redis_client: aioredis.Redis,
        worker_id: str,
        queues: List[str],
        timeout: float,
    ) -> Optional[Task]:
        """Blocking dequeue using BZPOPMAX on sorted sets."""
        queue_keys = [QUEUE_KEY.format(queue=q) for q in queues]

        # BZPOPMAX returns (key, member, score)
        result = await redis_client.bzpopmax(queue_keys, timeout=timeout)

        if result is None:
            return None

        _key, task_id, _score = result

        return await self._claim_task(redis_client, task_id, worker_id)

    async def _claim_task(
        self,
        redis_client: aioredis.Redis,
        task_id: str,
        worker_id: str,
    ) -> Optional[Task]:
        """Move a task to running status and assign it to a worker."""
        task_key = TASK_KEY.format(task_id=task_id)
        task_data_raw = await redis_client.hget(task_key, "data")

        if task_data_raw is None:
            logger.warning("Task %s data not found in Redis", task_id)
            return None

        task = Task.model_validate_json(task_data_raw)
        task.status = TaskStatus.RUNNING
        task.worker_id = worker_id
        task.started_at = datetime.utcnow()

        await redis_client.hset(
            task_key,
            mapping={
                "data": task.model_dump_json(),
                "status": task.status.value,
                "worker_id": worker_id,
            },
        )

        # Track active tasks per worker
        processing_key = PROCESSING_KEY.format(worker_id=worker_id)
        await redis_client.sadd(processing_key, task_id)

        logger.info("Task %s claimed by worker '%s'", task_id, worker_id)
        return task

    async def complete(self, task_id: str, result: Any = None) -> None:
        """Mark a task as completed and store its result.

        Args:
            task_id: The ID of the task to complete.
            result: The result data to store.
        """
        redis_client = await self._get_redis()

        task = await self._get_task(redis_client, task_id)
        if task is None:
            logger.error("Cannot complete task %s: not found", task_id)
            return

        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = datetime.utcnow()

        await self._update_task(redis_client, task)
        await self._remove_from_processing(redis_client, task)

        # Store result separately for retrieval
        result_key = RESULT_KEY.format(task_id=task_id)
        await redis_client.set(
            result_key,
            json.dumps(result, default=str),
            ex=86400,  # expire after 24 hours
        )

        logger.info("Task %s completed", task_id)

    async def fail(self, task_id: str, error: str) -> None:
        """Mark a task as failed with an error message.

        Args:
            task_id: The ID of the task that failed.
            error: Error message describing the failure.
        """
        redis_client = await self._get_redis()

        task = await self._get_task(redis_client, task_id)
        if task is None:
            logger.error("Cannot fail task %s: not found", task_id)
            return

        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = datetime.utcnow()

        await self._update_task(redis_client, task)
        await self._remove_from_processing(redis_client, task)

        logger.info("Task %s failed: %s", task_id, error)

    async def retry(self, task_id: str) -> Optional[str]:
        """Re-enqueue a failed task if retries remain.

        Increments the retry counter and puts the task back in its queue.

        Args:
            task_id: The ID of the task to retry.

        Returns:
            The task_id if re-enqueued, None if max retries exceeded or task not found.
        """
        redis_client = await self._get_redis()

        task = await self._get_task(redis_client, task_id)
        if task is None:
            logger.error("Cannot retry task %s: not found", task_id)
            return None

        if task.retries >= task.max_retries:
            logger.warning(
                "Task %s exceeded max retries (%d/%d), not retrying",
                task_id,
                task.retries,
                task.max_retries,
            )
            return None

        task.retries += 1
        task.status = TaskStatus.RETRYING
        task.error = None
        task.worker_id = None
        task.started_at = None
        task.completed_at = None

        await self._update_task(redis_client, task)
        await self._remove_from_processing(redis_client, task)

        # Re-enqueue
        queue_key = QUEUE_KEY.format(queue=task.queue)
        await redis_client.zadd(queue_key, {task.task_id: task.priority})

        # Update status to queued
        task.status = TaskStatus.QUEUED
        await self._update_task(redis_client, task)

        logger.info(
            "Task %s re-enqueued for retry (%d/%d)",
            task_id,
            task.retries,
            task.max_retries,
        )
        return task_id

    # ─── STATUS AND RESULT RETRIEVAL ────────────────────────────────────────

    async def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the current status of a task.

        Args:
            task_id: The task ID to look up.

        Returns:
            The TaskStatus if the task exists, otherwise None.
        """
        redis_client = await self._get_redis()
        task_key = TASK_KEY.format(task_id=task_id)

        status_raw = await redis_client.hget(task_key, "status")
        if status_raw is None:
            return None

        return TaskStatus(status_raw)

    async def get_result(self, task_id: str) -> Optional[Any]:
        """Retrieve the result of a completed task.

        Args:
            task_id: The task ID.

        Returns:
            The stored result if available, otherwise None.
        """
        redis_client = await self._get_redis()
        result_key = RESULT_KEY.format(task_id=task_id)

        result_raw = await redis_client.get(result_key)
        if result_raw is None:
            # Fallback: check task data
            task = await self._get_task(redis_client, task_id)
            if task is not None and task.status == TaskStatus.COMPLETED:
                return task.result
            return None

        return json.loads(result_raw)

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve full task details.

        Args:
            task_id: The task ID.

        Returns:
            The Task instance if found, otherwise None.
        """
        redis_client = await self._get_redis()
        return await self._get_task(redis_client, task_id)

    # ─── CANCELLATION AND PURGING ──────────────────────────────────────────

    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending or queued task.

        Only tasks in PENDING, QUEUED, or RETRYING status can be cancelled.

        Args:
            task_id: The ID of the task to cancel.

        Returns:
            True if the task was cancelled, False otherwise.
        """
        redis_client = await self._get_redis()

        task = await self._get_task(redis_client, task_id)
        if task is None:
            logger.error("Cannot cancel task %s: not found", task_id)
            return False

        if task.status not in (
            TaskStatus.PENDING,
            TaskStatus.QUEUED,
            TaskStatus.RETRYING,
        ):
            logger.warning(
                "Cannot cancel task %s: status is '%s'",
                task_id,
                task.status.value,
            )
            return False

        # Remove from queue sorted set
        queue_key = QUEUE_KEY.format(queue=task.queue)
        await redis_client.zrem(queue_key, task_id)

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()

        await self._update_task(redis_client, task)

        logger.info("Task %s cancelled", task_id)
        return True

    async def purge(
        self,
        queue: str,
        status: Union[TaskStatus, str] = TaskStatus.COMPLETED,
    ) -> int:
        """Remove tasks from a queue with a given status.

        Args:
            queue: Queue name to purge.
            status: Task status to purge (default: COMPLETED).

        Returns:
            Number of tasks purged.
        """
        redis_client = await self._get_redis()

        if isinstance(status, str):
            status = TaskStatus(status)

        queue_key = QUEUE_KEY.format(queue=queue)
        task_ids = await redis_client.zrange(queue_key, 0, -1)

        purged = 0
        for task_id in task_ids:
            task = await self._get_task(redis_client, task_id)
            if task is not None and task.status == status:
                # Remove from queue
                await redis_client.zrem(queue_key, task_id)
                # Delete task data
                task_key = TASK_KEY.format(task_id=task_id)
                await redis_client.delete(task_key)
                # Delete result if any
                result_key = RESULT_KEY.format(task_id=task_id)
                await redis_client.delete(result_key)
                purged += 1

        # Also scan task keys directly for completed tasks not in the sorted set
        # (they may have been removed from the set but still have data)
        async for key in redis_client.scan_iter(match="gdrag:task:*"):
            task_data_raw = await redis_client.hget(key, "data")
            if task_data_raw is None:
                continue
            try:
                task = Task.model_validate_json(task_data_raw)
                if task.queue == queue and task.status == status:
                    await redis_client.delete(key)
                    result_key = RESULT_KEY.format(task_id=task.task_id)
                    await redis_client.delete(result_key)
                    purged += 1
            except Exception:
                continue

        logger.info(
            "Purged %d tasks with status '%s' from queue '%s'",
            purged,
            status.value,
            queue,
        )
        return purged

    # ─── QUEUE INFO ─────────────────────────────────────────────────────────

    async def queue_size(self, queue: str) -> int:
        """Get the number of tasks in a queue.

        Args:
            queue: Queue name.

        Returns:
            Number of tasks in the queue.
        """
        redis_client = await self._get_redis()
        queue_key = QUEUE_KEY.format(queue=queue)
        return await redis_client.zcard(queue_key)

    async def list_queues(self) -> List[str]:
        """List all active queue names.

        Returns:
            List of queue names that have tasks.
        """
        redis_client = await self._get_redis()
        keys = []
        async for key in redis_client.scan_iter(match="gdrag:queue:*"):
            # Extract queue name from key
            name = key.replace("gdrag:queue:", "")
            keys.append(name)
        return keys

    # ─── PRIVATE HELPERS ────────────────────────────────────────────────────

    async def _get_task(
        self, redis_client: aioredis.Redis, task_id: str
    ) -> Optional[Task]:
        """Load a Task from Redis by ID."""
        task_key = TASK_KEY.format(task_id=task_id)
        task_data_raw = await redis_client.hget(task_key, "data")
        if task_data_raw is None:
            return None
        try:
            return Task.model_validate_json(task_data_raw)
        except Exception as exc:
            logger.error("Failed to deserialize task %s: %s", task_id, exc)
            return None

    async def _update_task(
        self, redis_client: aioredis.Redis, task: Task
    ) -> None:
        """Persist an updated Task back to Redis."""
        task_key = TASK_KEY.format(task_id=task.task_id)
        await redis_client.hset(
            task_key,
            mapping={
                "data": task.model_dump_json(),
                "status": task.status.value,
                "queue": task.queue,
                "worker_id": task.worker_id or "",
            },
        )

    async def _remove_from_processing(
        self, redis_client: aioredis.Redis, task: Task
    ) -> None:
        """Remove a task from its worker's processing set."""
        if task.worker_id:
            processing_key = PROCESSING_KEY.format(worker_id=task.worker_id)
            await redis_client.srem(processing_key, task.task_id)
