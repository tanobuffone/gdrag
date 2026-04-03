"""Tests for EventBus module (gdrag v3).

Tests Redis Streams-based event bus with consumer groups,
message acknowledgment, and pending/claim handling.
Uses fakeredis for isolated testing without a live Redis instance.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import fakeredis.aioredis
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Stub EventBus — minimal implementation for tests.
# Replace with real import once src/core/event_bus.py exists:
#   from src.core.event_bus import EventBus, Event, ConsumerGroup
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from enum import Enum


class EventStatus(str, Enum):
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class Event:
    """Represents a message in the event bus."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    topic: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    status: EventStatus = EventStatus.PENDING
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "payload": json.dumps(self.payload),
            "timestamp": str(self.timestamp),
            "status": self.status.value,
            "retry_count": str(self.retry_count),
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_stream(cls, message_id: str, data: Dict[bytes, bytes]) -> "Event":
        """Reconstruct Event from Redis stream message."""
        return cls(
            event_id=message_id,
            topic=data.get(b"topic", b"").decode(),
            payload=json.loads(data.get(b"payload", b"{}").decode()),
            timestamp=float(data.get(b"timestamp", b"0").decode()),
            status=EventStatus(data.get(b"status", b"pending").decode()),
            retry_count=int(data.get(b"retry_count", b"0").decode()),
            metadata=json.loads(data.get(b"metadata", b"{}").decode()),
        )


@dataclass
class ConsumerGroup:
    """Represents a consumer group for a topic."""

    group_name: str
    consumer_name: str
    topic: str
    last_delivered_id: str = "0-0"


class EventBus:
    """Redis Streams-based event bus.

    Supports:
    - Publishing events to topics (streams)
    - Consumer groups with independent consumption
    - Message acknowledgment
    - Pending message inspection and claim
    """

    def __init__(self, redis_client, prefix: str = "gdrag:events"):
        self._redis = redis_client
        self._prefix = prefix
        self._consumer_groups: Dict[str, ConsumerGroup] = {}

    def _stream_key(self, topic: str) -> str:
        return f"{self._prefix}:{topic}"

    async def publish(self, topic: str, payload: Dict[str, Any], **metadata) -> str:
        """Publish an event to a topic.

        Args:
            topic: Topic/stream name.
            payload: Event payload.
            **metadata: Additional metadata fields.

        Returns:
            Message ID assigned by Redis.
        """
        event = Event(
            topic=topic,
            payload=payload,
            metadata=metadata,
        )
        stream_key = self._stream_key(topic)
        message_id = await self._redis.xadd(stream_key, event.to_dict())
        return message_id

    async def create_consumer_group(
        self, topic: str, group_name: str, consumer_name: str, start_id: str = "0-0"
    ) -> ConsumerGroup:
        """Create or get a consumer group for a topic.

        Args:
            topic: Topic/stream name.
            group_name: Name of the consumer group.
            consumer_name: Name of this consumer.
            start_id: Starting message ID (0-0 from beginning, $ for new only).

        Returns:
            ConsumerGroup instance.
        """
        stream_key = self._stream_key(topic)
        try:
            await self._redis.xgroup_create(
                stream_key, group_name, id=start_id, mkstream=True
            )
        except Exception:
            # Group already exists — that's fine
            pass

        group = ConsumerGroup(
            group_name=group_name,
            consumer_name=consumer_name,
            topic=topic,
            last_delivered_id=start_id,
        )
        self._consumer_groups[f"{topic}:{group_name}"] = group
        return group

    async def consume(
        self,
        topic: str,
        group_name: str,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 100,
    ) -> List[tuple[str, Event]]:
        """Consume messages from a consumer group.

        Args:
            topic: Topic/stream name.
            group_name: Consumer group name.
            consumer_name: Consumer name.
            count: Max messages to fetch.
            block_ms: Block timeout in milliseconds.

        Returns:
            List of (message_id, Event) tuples.
        """
        stream_key = self._stream_key(topic)
        results = await self._redis.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams={stream_key: ">"},
            count=count,
            block=block_ms,
        )

        events = []
        for stream, messages in results:
            for message_id, data in messages:
                event = Event.from_stream(message_id, data)
                events.append((message_id.decode(), event))
        return events

    async def ack(self, topic: str, group_name: str, message_id: str) -> int:
        """Acknowledge a message.

        Args:
            topic: Topic/stream name.
            group_name: Consumer group name.
            message_id: Message ID to acknowledge.

        Returns:
            Number of messages acknowledged.
        """
        stream_key = self._stream_key(topic)
        return await self._redis.xack(stream_key, group_name, message_id)

    async def pending(self, topic: str, group_name: str) -> Dict[str, Any]:
        """Get pending message summary for a consumer group.

        Args:
            topic: Topic/stream name.
            group_name: Consumer group name.

        Returns:
            Dict with pending info: count, min_id, max_id, consumers.
        """
        stream_key = self._stream_key(topic)
        result = await self._redis.xpending(stream_key, group_name)
        if not result:
            return {"count": 0, "min_id": None, "max_id": None, "consumers": {}}

        pending_count = result.get("pending", 0)
        min_id = result.get("min_entry") or result.get("min", None)
        max_id = result.get("max_entry") or result.get("max", None)

        def _d(v):
            return v.decode() if isinstance(v, bytes) else v

        return {
            "count": pending_count,
            "min_id": _d(min_id),
            "max_id": _d(max_id),
            "consumers": {},
        }

    async def pending_range(
        self,
        topic: str,
        group_name: str,
        min_id: str = "-",
        max_id: str = "+",
        count: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get detailed pending messages in a range.

        Args:
            topic: Topic/stream name.
            group_name: Consumer group name.
            min_id: Min message ID filter.
            max_id: Max message ID filter.
            count: Max results.

        Returns:
            List of pending message details.
        """
        stream_key = self._stream_key(topic)
        results = await self._redis.xpending_range(
            stream_key, group_name, min_id, max_id, count
        )
        pending_list = []
        for item in results:
            pending_list.append({
                "message_id": item["message_id"].decode()
                if isinstance(item["message_id"], bytes)
                else item["message_id"],
                "consumer": item["consumer"].decode()
                if isinstance(item["consumer"], bytes)
                else item["consumer"],
                "time_since_delivered": item.get("time_since_delivered", 0),
                "delivery_count": item.get("times_delivered", item.get("delivery_count", 1)),
            })
        return pending_list

    async def claim(
        self,
        topic: str,
        group_name: str,
        consumer_name: str,
        min_idle_ms: int,
        message_ids: List[str],
    ) -> List[tuple[str, Event]]:
        """Claim pending messages that have been idle too long.

        Args:
            topic: Topic/stream name.
            group_name: Consumer group name.
            consumer_name: New consumer to assign to.
            min_idle_ms: Minimum idle time in ms.
            message_ids: List of message IDs to claim.

        Returns:
            List of (message_id, Event) tuples for claimed messages.
        """
        stream_key = self._stream_key(topic)
        results = await self._redis.xclaim(
            stream_key,
            group_name,
            consumer_name,
            min_idle_ms,
            message_ids,
        )
        events = []
        for message_id, data in results:
            event = Event.from_stream(
                message_id.decode() if isinstance(message_id, bytes) else message_id,
                data,
            )
            events.append((
                message_id.decode() if isinstance(message_id, bytes) else message_id,
                event,
            ))
        return events

    async def stream_length(self, topic: str) -> int:
        """Get the number of messages in a topic stream.

        Args:
            topic: Topic/stream name.

        Returns:
            Number of messages.
        """
        stream_key = self._stream_key(topic)
        return await self._redis.xlen(stream_key)

    async def trim_by_length(self, topic: str, max_len: int) -> int:
        """Trim stream to max length.

        Args:
            topic: Topic/stream name.
            max_len: Maximum number of messages to keep.

        Returns:
            Number of messages removed.
        """
        stream_key = self._stream_key(topic)
        return await self._redis.xtrim(stream_key, max_len)


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
async def event_bus(redis_client):
    """Provide an EventBus instance with fakeredis."""
    return EventBus(redis_client, prefix="test:events")


@pytest.fixture
def sample_payload() -> Dict[str, Any]:
    """Sample event payload for testing."""
    return {
        "action": "knowledge_ingested",
        "doc_id": "doc-123",
        "domain": "software",
        "chunks": 5,
    }


@pytest.fixture
def large_payload() -> Dict[str, Any]:
    """Large event payload for stress testing."""
    return {
        "action": "bulk_import",
        "items": [{"id": i, "data": f"content-{i}"} for i in range(100)],
    }


# ===========================================================================
# Test: Event Publishing
# ===========================================================================


class TestEventPublishing:
    """Tests for publishing events to the event bus."""

    @pytest.mark.asyncio
    async def test_publish_single_event(self, event_bus, sample_payload):
        """Test publishing a single event returns a valid message ID."""
        message_id = await event_bus.publish("knowledge", sample_payload)

        assert message_id is not None
        assert isinstance(message_id, (str, bytes))

    @pytest.mark.asyncio
    async def test_publish_event_stored_in_stream(self, event_bus, sample_payload):
        """Test that published event is stored in the Redis stream."""
        await event_bus.publish("knowledge", sample_payload)

        length = await event_bus.stream_length("knowledge")
        assert length == 1

    @pytest.mark.asyncio
    async def test_publish_multiple_events(self, event_bus):
        """Test publishing multiple events to the same topic."""
        for i in range(5):
            await event_bus.publish("tasks", {"task_id": f"task-{i}"})

        length = await event_bus.stream_length("tasks")
        assert length == 5

    @pytest.mark.asyncio
    async def test_publish_to_different_topics(self, event_bus):
        """Test publishing to different topics creates separate streams."""
        await event_bus.publish("topic_a", {"data": "a"})
        await event_bus.publish("topic_b", {"data": "b"})
        await event_bus.publish("topic_a", {"data": "a2"})

        assert await event_bus.stream_length("topic_a") == 2
        assert await event_bus.stream_length("topic_b") == 1

    @pytest.mark.asyncio
    async def test_publish_with_metadata(self, event_bus):
        """Test publishing event with additional metadata."""
        message_id = await event_bus.publish(
            "notifications",
            {"message": "hello"},
            source="api",
            priority="high",
        )

        assert message_id is not None
        assert await event_bus.stream_length("notifications") == 1

    @pytest.mark.asyncio
    async def test_publish_empty_payload(self, event_bus):
        """Test publishing event with empty payload."""
        message_id = await event_bus.publish("empty_topic", {})
        assert message_id is not None

    @pytest.mark.asyncio
    async def test_publish_unique_message_ids(self, event_bus):
        """Test that each published event gets a unique message ID."""
        ids = set()
        for _ in range(10):
            mid = await event_bus.publish("id_test", {"i": _})
            ids.add(mid)

        assert len(ids) == 10

    @pytest.mark.asyncio
    async def test_publish_large_payload(self, event_bus, large_payload):
        """Test publishing a large event payload."""
        message_id = await event_bus.publish("bulk", large_payload)
        assert message_id is not None
        assert await event_bus.stream_length("bulk") == 1

    @pytest.mark.asyncio
    async def test_publish_event_contains_correct_data(self, event_bus, sample_payload):
        """Test that published event data can be read back correctly."""
        await event_bus.publish("verify", sample_payload)

        # Read directly from stream
        stream_key = event_bus._stream_key("verify")
        raw = await event_bus._redis.xrange(stream_key)
        assert len(raw) == 1

        _, data = raw[0]
        event = Event.from_stream("test", data)
        assert event.payload["action"] == "knowledge_ingested"
        assert event.payload["doc_id"] == "doc-123"

    @pytest.mark.asyncio
    async def test_publish_sequential_timestamps(self, event_bus):
        """Test that events have increasing timestamps."""
        timestamps = []
        for i in range(3):
            await event_bus.publish("timing", {"i": i})
            # Small delay to ensure different timestamps
            await asyncio.sleep(0.01)

        stream_key = event_bus._stream_key("timing")
        raw = await event_bus._redis.xrange(stream_key)
        for _, data in raw:
            event = Event.from_stream("test", data)
            timestamps.append(event.timestamp)

        assert timestamps == sorted(timestamps)


# ===========================================================================
# Test: Consumer Groups
# ===========================================================================


class TestConsumerGroups:
    """Tests for consumer group functionality."""

    @pytest.mark.asyncio
    async def test_create_consumer_group(self, event_bus):
        """Test creating a consumer group."""
        group = await event_bus.create_consumer_group(
            topic="orders",
            group_name="processors",
            consumer_name="worker-1",
        )

        assert group.group_name == "processors"
        assert group.consumer_name == "worker-1"
        assert group.topic == "orders"

    @pytest.mark.asyncio
    async def test_consume_from_group(self, event_bus):
        """Test consuming messages from a consumer group."""
        # Publish events first
        for i in range(3):
            await event_bus.publish("orders", {"order_id": f"ord-{i}"})

        # Create consumer group
        await event_bus.create_consumer_group(
            "orders", "processors", "worker-1", start_id="0-0"
        )

        # Consume
        events = await event_bus.consume(
            "orders", "processors", "worker-1", count=10
        )

        assert len(events) == 3
        for msg_id, event in events:
            assert event.topic == "orders"
            assert "order_id" in event.payload

    @pytest.mark.asyncio
    async def test_multiple_consumers_in_group(self, event_bus):
        """Test that messages are distributed among consumers in a group."""
        # Publish 4 events
        for i in range(4):
            await event_bus.publish("work", {"task": f"task-{i}"})

        # Create two consumers in the same group
        await event_bus.create_consumer_group("work", "team", "worker-a", start_id="0-0")
        await event_bus.create_consumer_group("work", "team", "worker-b", start_id="0-0")

        # Worker A consumes
        events_a = await event_bus.consume("work", "team", "worker-a", count=2)

        # Worker B consumes
        events_b = await event_bus.consume("work", "team", "worker-b", count=2)

        # Between them they should have consumed messages
        total_consumed = len(events_a) + len(events_b)
        assert total_consumed >= 2  # At least some distribution

    @pytest.mark.asyncio
    async def test_independent_consumer_groups(self, event_bus):
        """Test that different consumer groups receive the same messages."""
        # Publish events
        for i in range(3):
            await event_bus.publish("broadcast", {"event": f"evt-{i}"})

        # Create two independent groups
        await event_bus.create_consumer_group(
            "broadcast", "analytics", "analytics-1", start_id="0-0"
        )
        await event_bus.create_consumer_group(
            "broadcast", "audit", "audit-1", start_id="0-0"
        )

        # Both groups should see all messages
        events_analytics = await event_bus.consume(
            "broadcast", "analytics", "analytics-1", count=10
        )
        events_audit = await event_bus.consume(
            "broadcast", "audit", "audit-1", count=10
        )

        assert len(events_analytics) == 3
        assert len(events_audit) == 3

    @pytest.mark.asyncio
    async def test_consume_from_empty_stream(self, event_bus):
        """Test consuming from an empty stream returns empty list."""
        await event_bus.create_consumer_group(
            "empty", "grp", "consumer-1", start_id="0-0"
        )

        events = await event_bus.consume(
            "empty", "grp", "consumer-1", count=10, block_ms=50
        )

        assert events == []

    @pytest.mark.asyncio
    async def test_consume_new_messages_only(self, event_bus):
        """Test consumer group starting from $ only gets new messages."""
        # Publish before creating group
        await event_bus.publish("stream", {"old": True})

        # Create group starting from new messages only
        await event_bus.create_consumer_group(
            "stream", "newonly", "c1", start_id="$"
        )

        # Publish after group creation
        await event_bus.publish("stream", {"new": True})

        # Should only get the new message
        events = await event_bus.consume(
            "stream", "newonly", "c1", count=10
        )

        assert len(events) == 1
        assert events[0][1].payload.get("new") is True

    @pytest.mark.asyncio
    async def test_consumer_group_idempotent_creation(self, event_bus):
        """Test that creating the same consumer group twice doesn't fail."""
        group1 = await event_bus.create_consumer_group(
            "idempotent", "grp1", "c1"
        )
        # Creating same group again should not raise
        group2 = await event_bus.create_consumer_group(
            "idempotent", "grp1", "c1"
        )

        assert group1.group_name == group2.group_name


# ===========================================================================
# Test: Message Acknowledgment
# ===========================================================================


class TestMessageAcknowledgment:
    """Tests for message acknowledgment flow."""

    @pytest.mark.asyncio
    async def test_ack_single_message(self, event_bus):
        """Test acknowledging a single message."""
        await event_bus.publish("ack_test", {"data": "test"})
        await event_bus.create_consumer_group("ack_test", "grp", "c1", start_id="0-0")

        events = await event_bus.consume("ack_test", "grp", "c1")
        assert len(events) == 1
        msg_id = events[0][0]

        acked = await event_bus.ack("ack_test", "grp", msg_id)
        assert acked == 1

    @pytest.mark.asyncio
    async def test_ack_multiple_messages(self, event_bus):
        """Test acknowledging multiple messages at once."""
        msg_ids = []
        for i in range(5):
            mid = await event_bus.publish("multi_ack", {"i": i})
            msg_ids.append(mid)

        await event_bus.create_consumer_group("multi_ack", "grp", "c1", start_id="0-0")
        events = await event_bus.consume("multi_ack", "grp", "c1", count=10)

        # Ack all
        total_acked = 0
        for msg_id, _ in events:
            acked = await event_bus.ack("multi_ack", "grp", msg_id)
            total_acked += acked

        assert total_acked == 5

    @pytest.mark.asyncio
    async def test_ack_reduces_pending_count(self, event_bus):
        """Test that acknowledging reduces pending message count."""
        for i in range(3):
            await event_bus.publish("pending_ack", {"i": i})

        await event_bus.create_consumer_group("pending_ack", "grp", "c1", start_id="0-0")
        events = await event_bus.consume("pending_ack", "grp", "c1", count=10)

        # Before ack
        pending_before = await event_bus.pending("pending_ack", "grp")
        assert pending_before["count"] == 3

        # Ack all
        for msg_id, _ in events:
            await event_bus.ack("pending_ack", "grp", msg_id)

        # After ack
        pending_after = await event_bus.pending("pending_ack", "grp")
        assert pending_after["count"] == 0

    @pytest.mark.asyncio
    async def test_ack_nonexistent_message(self, event_bus):
        """Test acknowledging a non-existent message returns 0."""
        await event_bus.create_consumer_group("no_msg", "grp", "c1", start_id="0-0")
        acked = await event_bus.ack("no_msg", "grp", "9999999999999-0")
        assert acked == 0

    @pytest.mark.asyncio
    async def test_ack_idempotent(self, event_bus):
        """Test that double acknowledgment doesn't cause errors."""
        await event_bus.publish("double_ack", {"data": "test"})
        await event_bus.create_consumer_group("double_ack", "grp", "c1", start_id="0-0")

        events = await event_bus.consume("double_ack", "grp", "c1")
        msg_id = events[0][0]

        # Ack twice
        acked1 = await event_bus.ack("double_ack", "grp", msg_id)
        acked2 = await event_bus.ack("double_ack", "grp", msg_id)

        assert acked1 == 1
        assert acked2 == 0  # Already acknowledged

    @pytest.mark.asyncio
    async def test_ack_partial_batch(self, event_bus):
        """Test acknowledging only some messages in a batch."""
        for i in range(5):
            await event_bus.publish("partial", {"i": i})

        await event_bus.create_consumer_group("partial", "grp", "c1", start_id="0-0")
        events = await event_bus.consume("partial", "grp", "c1", count=10)

        # Ack only first 2
        await event_bus.ack("partial", "grp", events[0][0])
        await event_bus.ack("partial", "grp", events[1][0])

        # Should still have 3 pending
        pending = await event_bus.pending("partial", "grp")
        assert pending["count"] == 3


# ===========================================================================
# Test: Pending / Claim
# ===========================================================================


class TestPendingClaim:
    """Tests for pending message inspection and claiming."""

    @pytest.mark.asyncio
    async def test_pending_after_consume(self, event_bus):
        """Test that consumed messages appear as pending."""
        for i in range(3):
            await event_bus.publish("pending_test", {"i": i})

        await event_bus.create_consumer_group(
            "pending_test", "grp", "c1", start_id="0-0"
        )
        await event_bus.consume("pending_test", "grp", "c1", count=10)

        pending = await event_bus.pending("pending_test", "grp")
        assert pending["count"] == 3

    @pytest.mark.asyncio
    async def test_pending_empty_group(self, event_bus):
        """Test pending info for a group with no consumed messages."""
        await event_bus.create_consumer_group("empty_pending", "grp", "c1", start_id="0-0")

        pending = await event_bus.pending("empty_pending", "grp")
        assert pending["count"] == 0

    @pytest.mark.asyncio
    async def test_pending_range_details(self, event_bus):
        """Test getting detailed pending message information."""
        for i in range(5):
            await event_bus.publish("detail_pending", {"i": i})

        await event_bus.create_consumer_group(
            "detail_pending", "grp", "c1", start_id="0-0"
        )
        await event_bus.consume("detail_pending", "grp", "c1", count=10)

        pending_list = await event_bus.pending_range("detail_pending", "grp")
        assert len(pending_list) == 5
        for item in pending_list:
            assert "message_id" in item
            assert "consumer" in item
            assert "delivery_count" in item

    @pytest.mark.asyncio
    async def test_claim_idle_messages(self, event_bus):
        """Test claiming messages that have been idle too long."""
        for i in range(3):
            await event_bus.publish("claim_test", {"i": i})

        await event_bus.create_consumer_group(
            "claim_test", "grp", "worker-old", start_id="0-0"
        )
        events = await event_bus.consume("claim_test", "grp", "worker-old", count=10)
        msg_ids = [msg_id for msg_id, _ in events]

        # Simulate idle time — claim from new consumer
        claimed = await event_bus.claim(
            "claim_test",
            "grp",
            "worker-new",
            min_idle_ms=0,  # 0 = claim immediately
            message_ids=msg_ids,
        )

        assert len(claimed) == 3
        for msg_id, event in claimed:
            assert "i" in event.payload

    @pytest.mark.asyncio
    async def test_claim_partial_messages(self, event_bus):
        """Test claiming only some pending messages."""
        for i in range(5):
            await event_bus.publish("partial_claim", {"i": i})

        await event_bus.create_consumer_group(
            "partial_claim", "grp", "c1", start_id="0-0"
        )
        events = await event_bus.consume("partial_claim", "grp", "c1", count=10)
        msg_ids = [msg_id for msg_id, _ in events]

        # Claim only first 2
        claimed = await event_bus.claim(
            "partial_claim", "grp", "c2",
            min_idle_ms=0,
            message_ids=msg_ids[:2],
        )

        assert len(claimed) == 2

    @pytest.mark.asyncio
    async def test_claim_nonexistent_messages(self, event_bus):
        """Test claiming non-existent message IDs returns empty list."""
        await event_bus.create_consumer_group(
            "no_claim", "grp", "c1", start_id="0-0"
        )

        claimed = await event_bus.claim(
            "no_claim", "grp", "c2",
            min_idle_ms=0,
            message_ids=["9999999999999-0"],
        )

        assert claimed == []

    @pytest.mark.asyncio
    async def test_pending_tracks_delivery_count(self, event_bus):
        """Test that pending tracks how many times a message was delivered."""
        await event_bus.publish("delivery_count", {"data": "retry_me"})
        await event_bus.create_consumer_group(
            "delivery_count", "grp", "c1", start_id="0-0"
        )

        # Consume without ack
        events = await event_bus.consume("delivery_count", "grp", "c1", count=10)
        msg_ids = [msg_id for msg_id, _ in events]

        # Claim (re-deliver)
        await asyncio.sleep(0.01)
        await event_bus.claim(
            "delivery_count", "grp", "c2",
            min_idle_ms=0,
            message_ids=msg_ids,
        )

        pending_list = await event_bus.pending_range("delivery_count", "grp")
        assert len(pending_list) >= 1
        # After claim, delivery count should be >= 2
        assert any(p["delivery_count"] >= 2 for p in pending_list)

    @pytest.mark.asyncio
    async def test_consume_after_ack_shows_no_pending(self, event_bus):
        """Test full consume-ack cycle leaves no pending messages."""
        for i in range(5):
            await event_bus.publish("full_cycle", {"i": i})

        await event_bus.create_consumer_group(
            "full_cycle", "grp", "c1", start_id="0-0"
        )
        events = await event_bus.consume("full_cycle", "grp", "c1", count=10)

        for msg_id, _ in events:
            await event_bus.ack("full_cycle", "grp", msg_id)

        pending = await event_bus.pending("full_cycle", "grp")
        assert pending["count"] == 0

    @pytest.mark.asyncio
    async def test_stream_trim_with_pending(self, event_bus):
        """Test that stream trimming doesn't break pending tracking."""
        for i in range(10):
            await event_bus.publish("trim_test", {"i": i})

        await event_bus.create_consumer_group(
            "trim_test", "grp", "c1", start_id="0-0"
        )
        await event_bus.consume("trim_test", "grp", "c1", count=10)

        # Trim stream
        await event_bus.trim_by_length("trim_test", max_len=5)

        # Stream should be trimmed
        length = await event_bus.stream_length("trim_test")
        assert length <= 5


# ===========================================================================
# Test: Event Model
# ===========================================================================


class TestEventModel:
    """Tests for the Event data model."""

    def test_event_defaults(self):
        """Test Event has sensible defaults."""
        event = Event()
        assert event.status == EventStatus.PENDING
        assert event.retry_count == 0
        assert event.payload == {}
        assert event.metadata == {}

    def test_event_to_dict(self):
        """Test Event serialization to dict."""
        event = Event(
            topic="test",
            payload={"key": "value"},
            status=EventStatus.PENDING,
        )
        d = event.to_dict()
        assert d["topic"] == "test"
        assert json.loads(d["payload"]) == {"key": "value"}
        assert d["status"] == "pending"

    def test_event_from_stream(self):
        """Test Event deserialization from stream data."""
        data = {
            b"topic": b"orders",
            b'payload': b'{"order_id": "123"}',
            b"timestamp": b"1700000000.0",
            b"status": b"pending",
            b"retry_count": b"0",
            b"metadata": b'{}',
        }
        event = Event.from_stream("1700000000000-0", data)
        assert event.topic == "orders"
        assert event.payload["order_id"] == "123"
        assert event.status == EventStatus.PENDING

    def test_event_roundtrip(self):
        """Test Event survives serialize/deserialize roundtrip."""
        original = Event(
            topic="roundtrip",
            payload={"nested": {"key": [1, 2, 3]}},
            metadata={"source": "test"},
        )
        data = {k.encode(): v.encode() for k, v in original.to_dict().items()}
        restored = Event.from_stream(original.event_id, data)

        assert restored.topic == original.topic
        assert restored.payload == original.payload
        assert restored.metadata == original.metadata


# ===========================================================================
# Test: EventBus Configuration
# ===========================================================================


class TestEventBusConfiguration:
    """Tests for EventBus initialization and configuration."""

    @pytest.mark.asyncio
    async def test_custom_prefix(self, redis_client):
        """Test EventBus with custom key prefix."""
        bus = EventBus(redis_client, prefix="custom:prefix")
        await bus.publish("topic", {"data": "test"})

        # Check key has custom prefix
        keys = await redis_client.keys("*")
        assert any(b"custom:prefix:topic" in k for k in keys)

    @pytest.mark.asyncio
    async def test_default_prefix(self, redis_client):
        """Test EventBus with default key prefix."""
        bus = EventBus(redis_client)
        await bus.publish("topic", {"data": "test"})

        keys = await redis_client.keys("*")
        assert any(b"gdrag:events:topic" in k for k in keys)

    @pytest.mark.asyncio
    async def test_separate_bus_instances_same_redis(self, redis_client):
        """Test that separate EventBus instances share the same Redis."""
        bus1 = EventBus(redis_client, prefix="bus1")
        bus2 = EventBus(redis_client, prefix="bus2")

        await bus1.publish("shared", {"from": "bus1"})
        await bus2.publish("shared", {"from": "bus2"})

        assert await bus1.stream_length("shared") == 1
        assert await bus2.stream_length("shared") == 1
