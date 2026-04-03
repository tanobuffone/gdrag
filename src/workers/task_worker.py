"""Async task worker for gdrag v3.

Provides TaskWorker that polls a TaskQueue, dispatches work to
registered handlers based on task_type, and manages retries,
timeouts, and graceful shutdown.

Supported task types:
    - ingest:   Document ingestion (chunking, embedding, storage)
    - query:    Query execution (retrieval, reranking, compression)
    - compress: Context / session compression
    - export:   Knowledge export operations
    - reindex:  Reindexing of stored data
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TaskWorkerStatus(str, Enum):
    """Lifecycle states of a TaskWorker."""
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class TaskRequest:
    """A unit of work pulled from the queue.

    Attributes:
        task_id:     Unique identifier for the task.
        task_type:   Handler category (ingest, query, compress, export, reindex).
        payload:     Arbitrary data forwarded to the handler.
        max_retries: Maximum retry attempts after the first failure.
        timeout_s:   Per-attempt timeout in seconds.  0 means no timeout.
        priority:    Higher priority tasks are processed first.
        created_at:  Timestamp when the task was created (epoch seconds).
        metadata:    Optional extra metadata.
    """

    task_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: str(uuid4()))
    max_retries: int = 3
    timeout_s: float = 300.0
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of processing a single task.

    Attributes:
        task_id:        The task that produced this result.
        success:        Whether the task completed without error.
        result:         The handler return value (on success).
        error:          Exception string (on failure).
        attempts:       Total attempts made.
        elapsed_s:      Wall-clock seconds spent processing.
    """

    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    attempts: int = 1
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Protocols / type aliases
# ---------------------------------------------------------------------------

TaskHandler = Callable[[Dict[str, Any]], Awaitable[Any]]
"""Async callable that receives the task payload and returns a result."""

# A simple queue interface – any object with async ``get()`` and ``put()``
# methods plus a ``qsize()`` method will work.
#
# If *queues* is a list, the worker round-robins across them; this lets you
# prioritise certain task types (e.g. ingest vs. query).
TaskQueue = Any


# ---------------------------------------------------------------------------
# Built-in handler registry
# ---------------------------------------------------------------------------

_BUILTIN_HANDLERS: Dict[str, TaskHandler] = {}


def register_handler(task_type: str) -> Callable[[TaskHandler], TaskHandler]:
    """Decorator to register a handler for a given *task_type*.

    Usage::

        @register_handler("ingest")
        async def handle_ingest(payload):
            ...
    """

    def decorator(fn: TaskHandler) -> TaskHandler:
        _BUILTIN_HANDLERS[task_type] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# TaskWorker
# ---------------------------------------------------------------------------

class TaskWorker:
    """Async worker that pulls tasks from one or more queues and processes them.

    The worker runs a polling loop (``start()``) that picks the next available
    task, dispatches it to the appropriate handler, and handles retries /
    timeouts.

    Parameters
    ----------
    worker_id : str
        Human-readable identifier for this worker (used in logs).
    queues : list[TaskQueue]
        One or more async queues to poll.  The worker round-robins across
        them so higher-priority queues can be placed first.
    handler : dict[str, TaskHandler] | None
        Map of ``task_type -> async callable``.  Built-in handlers are
        merged automatically; explicit *handler* entries take precedence.
    poll_interval : float
        Seconds to sleep when no task is available.
    """

    SUPPORTED_TASK_TYPES = ("ingest", "query", "compress", "export", "reindex")

    def __init__(
        self,
        worker_id: str,
        queues: List[TaskQueue],
        handler: Optional[Dict[str, TaskHandler]] = None,
        poll_interval: float = 0.5,
    ) -> None:
        self.worker_id = worker_id
        self.queues = list(queues)
        self.poll_interval = poll_interval

        # Merge handlers: user-supplied override built-ins
        self._handlers: Dict[str, TaskHandler] = dict(_BUILTIN_HANDLERS)
        if handler:
            self._handlers.update(handler)

        # Runtime state
        self._status = TaskWorkerStatus.STOPPED
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._current_task: Optional[TaskRequest] = None
        self._processed: int = 0
        self._errors: int = 0
        self._queue_idx: int = 0
        self._stop_event = asyncio.Event()

    # ---- public properties ---------------------------------------------------

    @property
    def status(self) -> TaskWorkerStatus:
        return self._status

    @property
    def processed(self) -> int:
        """Total tasks successfully processed."""
        return self._processed

    @property
    def errors(self) -> int:
        """Total tasks that failed (after all retries)."""
        return self._errors

    # ---- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the worker polling loop.

        This method is non-blocking; it schedules the internal loop as a
        background ``asyncio.Task`` and returns immediately.
        """
        if self._status == TaskWorkerStatus.RUNNING:
            logger.warning("[%s] Worker already running", self.worker_id)
            return

        self._stop_event.clear()
        self._status = TaskWorkerStatus.RUNNING
        self._task = asyncio.create_task(self._run_loop(), name=f"task-worker-{self.worker_id}")
        logger.info("[%s] Worker started", self.worker_id)

    async def stop(self, timeout: Optional[float] = 10.0) -> None:
        """Signal the worker to stop and wait for it to finish.

        If the worker is currently processing a task, it will wait up to
        *timeout* seconds for it to complete before returning.
        """
        if self._status in (TaskWorkerStatus.STOPPED, TaskWorkerStatus.STOPPING):
            return

        logger.info("[%s] Stopping worker …", self.worker_id)
        self._status = TaskWorkerStatus.STOPPING
        self._stop_event.set()

        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("[%s] Worker did not stop within %ss – cancelling", self.worker_id, timeout)
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        self._status = TaskWorkerStatus.STOPPED
        logger.info("[%s] Worker stopped (processed=%d, errors=%d)", self.worker_id, self._processed, self._errors)

    # ---- main loop -----------------------------------------------------------

    async def _run_loop(self) -> None:
        """Internal polling loop."""
        try:
            while not self._stop_event.is_set():
                task_request = await self._poll_queues()
                if task_request is None:
                    # Nothing available – sleep and retry
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=self.poll_interval,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                self._current_task = task_request
                result = await self._process_with_retries(task_request)
                self._current_task = None

                if result.success:
                    self._processed += 1
                else:
                    self._errors += 1
                    logger.error("[%s] Task %s failed: %s", self.worker_id, result.task_id, result.error)

        except asyncio.CancelledError:
            logger.debug("[%s] Run loop cancelled", self.worker_id)
        except Exception as exc:
            self._status = TaskWorkerStatus.ERROR
            logger.exception("[%s] Unexpected error in run loop: %s", self.worker_id, exc)
            raise

    # ---- queue polling -------------------------------------------------------

    async def _poll_queues(self) -> Optional[TaskRequest]:
        """Round-robin poll across queues.

        Tries each queue once per call (starting from the last index) and
        returns the first available task.  Returns ``None`` if all queues
        are empty.
        """
        n = len(self.queues)
        for _ in range(n):
            q = self.queues[self._queue_idx]
            self._queue_idx = (self._queue_idx + 1) % n

            try:
                # Support both asyncio.Queue and custom queue-like objects
                if hasattr(q, "get_nowait"):
                    task = q.get_nowait()
                elif hasattr(q, "get"):
                    task = await asyncio.wait_for(q.get(), timeout=0.1)
                else:
                    continue

                if task is not None:
                    return task  # type: ignore[return-value]
            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                continue
            except Exception as exc:
                logger.warning("[%s] Error polling queue %s: %s", self.worker_id, q, exc)
                continue

        return None

    # ---- processing with retries --------------------------------------------

    async def _process_with_retries(self, task: TaskRequest) -> TaskResult:
        """Execute the task, retrying on failure up to *max_retries* times."""
        attempts = 0
        last_error: Optional[str] = None
        start = time.monotonic()

        while attempts <= task.max_retries:
            attempts += 1
            logger.debug(
                "[%s] Processing task %s (type=%s, attempt=%d/%d)",
                self.worker_id,
                task.task_id,
                task.task_type,
                attempts,
                task.max_retries + 1,
            )

            try:
                result_value = await self.process_task(task)
                elapsed = time.monotonic() - start
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    result=result_value,
                    attempts=attempts,
                    elapsed_s=elapsed,
                )
            except asyncio.TimeoutError:
                last_error = f"Timeout after {task.timeout_s}s (attempt {attempts})"
                logger.warning("[%s] %s", self.worker_id, last_error)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("[%s] Task %s attempt %d failed: %s", self.worker_id, task.task_id, attempts, last_error)

            # Backoff before retry (exponential with cap)
            if attempts <= task.max_retries:
                backoff = min(2 ** (attempts - 1), 30.0)
                logger.debug("[%s] Retrying task %s in %.1fs …", self.worker_id, task.task_id, backoff)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass

                # If we were asked to stop, don't retry
                if self._stop_event.is_set():
                    break

        elapsed = time.monotonic() - start
        return TaskResult(
            task_id=task.task_id,
            success=False,
            error=last_error,
            attempts=attempts,
            elapsed_s=elapsed,
        )

    # ---- single task execution ----------------------------------------------

    async def process_task(self, task: TaskRequest) -> Any:
        """Dispatch a single task to its handler.

        The handler receives ``task.payload`` as its sole argument and must
        return (or await-return) a result.

        Raises
        ------
        ValueError
            If *task.task_type* has no registered handler.
        asyncio.TimeoutError
            If the handler does not finish within *task.timeout_s*.
        """
        handler = self._handlers.get(task.task_type)
        if handler is None:
            raise ValueError(
                f"No handler registered for task_type={task.task_type!r}. "
                f"Available: {sorted(self._handlers.keys())}"
            )

        if task.timeout_s and task.timeout_s > 0:
            return await asyncio.wait_for(handler(task.payload), timeout=task.timeout_s)
        else:
            return await handler(task.payload)

    # ---- stats ---------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot of the worker's statistics."""
        return {
            "worker_id": self.worker_id,
            "status": self._status.value,
            "processed": self._processed,
            "errors": self._errors,
            "current_task": self._current_task.task_id if self._current_task else None,
            "handler_types": sorted(self._handlers.keys()),
        }


# ---------------------------------------------------------------------------
# Built-in handler stubs (replace with real implementations)
# ---------------------------------------------------------------------------

async def _handle_ingest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stub for ingest task – delegates to pipeline.ingest_document."""
    content: str = payload.get("content", "")
    title: str = payload.get("title", "")
    domain: Optional[str] = payload.get("domain")
    source: str = payload.get("source", "worker")
    doc_id: Optional[str] = payload.get("doc_id")

    # Import lazily to avoid circular imports
    from ..core.config import load_config
    from ..core.pipeline import QueryPipeline

    config = load_config()
    pipeline = QueryPipeline(config)
    return pipeline.ingest_document(content, title, domain, source, doc_id)


async def _handle_query(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stub for query task – delegates to pipeline.execute."""
    from ..core.config import load_config
    from ..core.pipeline import QueryPipeline
    from ..models.schemas import EnhancedQueryRequest

    config = load_config()
    pipeline = QueryPipeline(config)
    request = EnhancedQueryRequest(
        query=payload.get("query", ""),
        domain=payload.get("domain"),
        session_id=payload.get("session_id"),
        limit=payload.get("limit", 10),
        compress_results=payload.get("compress_results", False),
    )
    response = pipeline.execute(request)
    return response.model_dump()


async def _handle_compress(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stub for compress task."""
    from ..core.compressor import ContextCompressor
    from ..core.config import CompressionConfig

    compressor = ContextCompressor(CompressionConfig())
    text: str = payload.get("text", "")
    max_tokens: int = payload.get("max_tokens", 500)
    summary = compressor.summarize_text(text, max_tokens)
    return {"summary": summary, "original_length": len(text)}


async def _handle_export(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stub for export task."""
    # Placeholder – real implementation would export data to a file / API
    return {"exported": True, "items": payload.get("items", [])}


async def _handle_reindex(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stub for reindex task."""
    # Placeholder – real implementation would rebuild indexes
    return {"reindexed": True, "scope": payload.get("scope", "all")}


# Register built-in handlers
_BUILTIN_HANDLERS.update({
    "ingest": _handle_ingest,
    "query": _handle_query,
    "compress": _handle_compress,
    "export": _handle_export,
    "reindex": _handle_reindex,
})
