"""gdrag v3 - EventBus based on Redis Streams.

Provides async publish/subscribe messaging between gdrag components
using Redis Streams with consumer group support for reliable delivery.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

import redis.asyncio as redis

from src.models.events import Event, StreamName

logger = logging.getLogger(__name__)

# Type alias for event handler callbacks
EventHandler = Callable[[str, Event], Awaitable[None]]

# Default stream names used by gdrag
DEFAULT_STREAMS: List[str] = [
    StreamName.KNOWLEDGE.value,
    StreamName.SESSIONS.value,
    StreamName.TASKS.value,
    StreamName.AGENTS.value,
    StreamName.CONTEXT.value,
]


class EventBus:
    """Async event bus backed by Redis Streams.

    Provides reliable pub/sub messaging with consumer group support,
    message acknowledgment, pending message inspection, and message claiming.

    Supports graceful degradation when Redis is unavailable: publish calls
    become no-ops and a warning is logged.

    Usage:
        bus = EventBus("redis://localhost:6379/0")
        await bus.connect()

        # Publish
        event = Event(event_type="knowledge.ingested", payload={"doc_id": "123"})
        msg_id = await bus.publish("gdrag:knowledge", event)

        # Subscribe with consumer group
        await bus.create_consumer_group("gdrag:knowledge", "processors")
        await bus.subscribe(
            stream="gdrag:knowledge",
            consumer_group="processors",
            consumer_name="worker-1",
            handler=my_handler,
        )

        await bus.close()
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        """Initialize EventBus with Redis connection URL.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0").
        """
        self._redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def connect(self) -> None:
        """Establish connection to Redis.

        Creates an async Redis client using redis[hiredis] for optimal performance.
        If Redis is unavailable, the EventBus enters degraded mode (publish is no-op).
        """
        if self._redis is not None:
            return
        try:
            self._redis = redis.from_url(
                self._redis_url,
                decode_responses=False,  # Keep bytes for binary safety
            )
            # Verify connection
            await self._redis.ping()
            logger.info("EventBus connected to Redis at %s", self._redis_url)
        except Exception as exc:
            logger.warning(
                "EventBus could not connect to Redis at %s: %s — "
                "operating in degraded mode (events will be dropped)",
                self._redis_url, exc,
            )
            self._redis = None

    @property
    def is_connected(self) -> bool:
        """Return True if the EventBus has an active Redis connection."""
        return self._redis is not None

    async def close(self) -> None:
        """Close Redis connection and stop all subscription loops."""
        self._running = False
        # Cancel all background listener tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        logger.info("EventBus disconnected")

    @property
    def redis(self) -> redis.Redis:
        """Get the underlying Redis client.

        Returns:
            Async Redis client instance.

        Raises:
            RuntimeError: If not connected.
        """
        if self._redis is None:
            raise RuntimeError("EventBus not connected. Call connect() first.")
        return self._redis

    # ────────────────────────────────────────────────────────────────────────
    # Publish
    # ────────────────────────────────────────────────────────────────────────

    async def publish(self, stream: str, event: Event) -> Optional[str]:
        """Publish an event to a Redis Stream.

        If the EventBus is not connected (degraded mode), the event is silently
        dropped and None is returned.

        Args:
            stream: Stream name (e.g., "gdrag:knowledge").
            event: Event to publish.

        Returns:
            Message ID assigned by Redis (e.g., "1700000000000-0"), or None if
            Redis is unavailable.
        """
        if self._redis is None:
            logger.debug(
                "EventBus not connected — dropping event %s for stream %s",
                event.event_type, stream,
            )
            return None

        try:
            data = event.to_stream_dict()
            msg_id = await self._redis.xadd(
                stream,
                data,
                maxlen=10000,
                approximate=True,
            )
            decoded_id = msg_id.decode("utf-8") if isinstance(msg_id, bytes) else msg_id
            logger.debug("Published event %s to %s (id=%s)", event.event_type, stream, decoded_id)
            return decoded_id
        except Exception as exc:
            logger.warning(
                "Failed to publish event %s to %s: %s",
                event.event_type, stream, exc,
            )
            return None

    async def publish_safe(self, stream: str, event: Event) -> Optional[str]:
        """Publish an event with full error suppression.

        Same as publish() but never raises. Suitable for fire-and-forget
        scenarios where event publishing should not break the caller.

        Args:
            stream: Stream name.
            event: Event to publish.

        Returns:
            Message ID or None on failure.
        """
        try:
            return await self.publish(stream, event)
        except Exception:
            return None

    # ────────────────────────────────────────────────────────────────────────
    # Consumer Groups
    # ────────────────────────────────────────────────────────────────────────

    async def create_consumer_group(
        self, stream: str, group: str, start_id: str = "0"
    ) -> bool:
        """Create a consumer group on a stream.

        If the stream does not exist, it will be created. If the group
        already exists, this is a no-op.

        Args:
            stream: Stream name.
            group: Consumer group name.
            start_id: Start reading from this ID. Use "0" to read from
                      beginning, "$" for new messages only.

        Returns:
            True if group was created, False if it already existed.
        """
        if self._redis is None:
            logger.debug("EventBus not connected — skipping consumer group creation")
            return False

        try:
            await self._redis.xgroup_create(
                name=stream,
                groupname=group,
                id=start_id,
                mkstream=True,
            )
            logger.info("Created consumer group '%s' on stream '%s'", group, stream)
            return True
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug("Consumer group '%s' already exists on '%s'", group, stream)
                return False
            raise

    # ────────────────────────────────────────────────────────────────────────
    # Subscribe (consume)
    # ────────────────────────────────────────────────────────────────────────

    async def subscribe(
        self,
        stream: str,
        consumer_group: str,
        consumer_name: str,
        handler: EventHandler,
        count: int = 10,
        block_ms: int = 5000,
    ) -> None:
        """Subscribe to a stream and process messages with a handler.

        Starts an async loop that reads messages from the stream using
        the specified consumer group. Messages are automatically acknowledged
        after successful handler execution.

        Args:
            stream: Stream name to subscribe to.
            consumer_group: Consumer group name.
            consumer_name: Unique name for this consumer within the group.
            handler: Async callback receiving (message_id, Event).
            count: Max messages to fetch per read.
            block_ms: Block timeout in milliseconds for XREADGROUP.
        """
        if self._redis is None:
            logger.warning("EventBus not connected — cannot subscribe to %s", stream)
            return

        self._running = True
        logger.info(
            "Subscribing consumer '%s' to stream '%s' (group='%s')",
            consumer_name, stream, consumer_group,
        )

        while self._running:
            try:
                messages = await self._redis.xreadgroup(
                    groupname=consumer_group,
                    consumername=consumer_name,
                    streams={stream: ">"},
                    count=count,
                    block=block_ms,
                )

                if not messages:
                    continue

                for stream_name, stream_messages in messages:
                    for msg_id, raw_data in stream_messages:
                        decoded_id = msg_id.decode("utf-8") if isinstance(msg_id, bytes) else msg_id
                        try:
                            event = Event.from_stream_dict(raw_data)
                            await handler(decoded_id, event)
                            await self.ack(stream, consumer_group, decoded_id)
                        except Exception:
                            logger.exception(
                                "Error processing message %s on %s", decoded_id, stream
                            )

            except asyncio.CancelledError:
                logger.info("Subscription cancelled for consumer '%s'", consumer_name)
                break
            except Exception:
                logger.exception(
                    "Error reading from stream '%s' (consumer='%s')",
                    stream, consumer_name,
                )
                await asyncio.sleep(1)  # Back off on error

    def subscribe_background(
        self,
        stream: str,
        consumer_group: str,
        consumer_name: str,
        handler: EventHandler,
        count: int = 10,
        block_ms: int = 5000,
    ) -> Optional[asyncio.Task]:
        """Start a subscription in a background task.

        Args:
            stream: Stream name to subscribe to.
            consumer_group: Consumer group name.
            consumer_name: Unique name for this consumer within the group.
            handler: Async callback receiving (message_id, Event).
            count: Max messages to fetch per read.
            block_ms: Block timeout in milliseconds.

        Returns:
            The asyncio.Task running the subscription loop, or None if not connected.
        """
        if self._redis is None:
            logger.debug("EventBus not connected — cannot start background subscription")
            return None

        task = asyncio.create_task(
            self.subscribe(stream, consumer_group, consumer_name, handler, count, block_ms)
        )
        self._tasks.append(task)
        return task

    # ────────────────────────────────────────────────────────────────────────
    # Acknowledgment
    # ────────────────────────────────────────────────────────────────────────

    async def ack(self, stream: str, group: str, message_id: str) -> int:
        """Acknowledge one or more messages.

        Args:
            stream: Stream name.
            group: Consumer group name.
            message_id: Message ID to acknowledge.

        Returns:
            Number of messages successfully acknowledged.
        """
        result = await self.redis.xack(stream, group, message_id)
        return int(result)

    # ────────────────────────────────────────────────────────────────────────
    # Pending Messages
    # ────────────────────────────────────────────────────────────────────────

    async def pending(
        self,
        stream: str,
        group: str,
        count: int = 100,
        min_idle_ms: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get pending (unacknowledged) messages for a consumer group.

        Args:
            stream: Stream name.
            group: Consumer group name.
            count: Max number of pending messages to return.
            min_idle_ms: Minimum idle time in milliseconds.

        Returns:
            List of pending message info dicts with keys:
            - message_id: str
            - consumer: str
            - idle_ms: int
            - delivery_count: int
        """
        result = await self.redis.xpending_range(
            name=stream,
            groupname=group,
            min="-",
            max="+",
            count=count,
            consumername=None,
        )

        pending_list = []
        for entry in result:
            pending_list.append({
                "message_id": entry["message_id"].decode("utf-8")
                if isinstance(entry["message_id"], bytes) else entry["message_id"],
                "consumer": entry["consumer"].decode("utf-8")
                if isinstance(entry["consumer"], bytes) else entry["consumer"],
                "idle_ms": entry["time_since_delivered"],
                "delivery_count": entry["times_delivered"],
            })

        return pending_list

    # ────────────────────────────────────────────────────────────────────────
    # Claim (recover stale messages)
    # ────────────────────────────────────────────────────────────────────────

    async def claim(
        self,
        stream: str,
        group: str,
        min_idle_ms: int,
        consumer_name: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Claim ownership of idle pending messages.

        Transfers messages that have been idle for at least min_idle_ms
        to a different consumer for reprocessing.

        Args:
            stream: Stream name.
            group: Consumer group name.
            min_idle_ms: Minimum idle time in milliseconds to claim.
            consumer_name: Consumer to claim messages for. Defaults to "claimer".
            message_ids: Specific message IDs to claim. If None, claims all
                         idle messages matching min_idle_ms.

        Returns:
            List of claimed message dicts with keys:
            - message_id: str
            - event: Event (reconstructed)
        """
        target_consumer = consumer_name or "claimer"

        if message_ids is None:
            # Get pending messages that are idle long enough
            pending = await self.redis.xpending_range(
                name=stream,
                groupname=group,
                min=min_idle_ms,
                max="+",
                count=100,
            )
            message_ids = [
                entry["message_id"].decode("utf-8")
                if isinstance(entry["message_id"], bytes) else entry["message_id"]
                for entry in pending
            ]

        if not message_ids:
            return []

        # Convert to bytes for XCLAIM
        byte_ids = [mid.encode("utf-8") for mid in message_ids]

        claimed = await self.redis.xclaim(
            name=stream,
            groupname=group,
            consumername=target_consumer,
            min_idle_time=min_idle_ms,
            message_ids=byte_ids,
        )

        result = []
        for msg_id, raw_data in claimed:
            decoded_id = msg_id.decode("utf-8") if isinstance(msg_id, bytes) else msg_id
            event = Event.from_stream_dict(raw_data)
            result.append({
                "message_id": decoded_id,
                "event": event,
            })

        logger.info(
            "Claimed %d messages on '%s' (group='%s', min_idle=%dms)",
            len(result), stream, group, min_idle_ms,
        )
        return result

    # ────────────────────────────────────────────────────────────────────────
    # Stream Info
    # ────────────────────────────────────────────────────────────────────────

    async def stream_info(self, stream: str) -> Dict[str, Any]:
        """Get information about a stream.

        Args:
            stream: Stream name.

        Returns:
            Dict with stream metadata (length, first/last entry, etc.).
        """
        info = await self.redis.xinfo_stream(stream)
        return {
            "length": info.get("length", 0),
            "first_entry_id": info.get("first-entry", (None, None))[0],
            "last_entry_id": info.get("last-entry", (None, None))[0],
            "groups": info.get("groups", 0),
        }

    async def stream_length(self, stream: str) -> int:
        """Get the number of entries in a stream.

        Args:
            stream: Stream name.

        Returns:
            Number of messages in the stream.
        """
        return await self.redis.xlen(stream)

    # ────────────────────────────────────────────────────────────────────────
    # Convenience: Stop
    # ────────────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal all subscription loops to stop."""
        self._running = False
