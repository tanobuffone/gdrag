"""Tests for Pydantic schemas."""

import pytest
from datetime import datetime

from src.models.schemas import (
    Chunk,
    EnhancedQueryRequest,
    KnowledgeItem,
    QueryRecord,
    RankedResult,
    SessionMemory,
)


class TestChunk:
    """Tests for Chunk model."""

    def test_create_chunk(self):
        """Test creating a chunk."""
        chunk = Chunk(
            doc_id="doc1",
            content="This is test content",
            chunk_index=0,
            start_char=0,
            end_char=20,
            token_count=5,
        )
        assert chunk.doc_id == "doc1"
        assert chunk.chunk_index == 0

    def test_chunk_with_metadata(self):
        """Test chunk with metadata."""
        chunk = Chunk(
            doc_id="doc1",
            content="Test",
            chunk_index=0,
            start_char=0,
            end_char=4,
            token_count=1,
            metadata={"domain": "test"},
        )
        assert chunk.metadata["domain"] == "test"


class TestRankedResult:
    """Tests for RankedResult model."""

    def test_create_result(self):
        """Test creating a ranked result."""
        result = RankedResult(
            content="Test content",
            title="Test Title",
            domain="test",
            source="test",
            semantic_score=0.9,
            final_score=0.85,
            doc_id="doc1",
        )
        assert result.semantic_score == 0.9
        assert result.final_score == 0.85


class TestSessionMemory:
    """Tests for SessionMemory model."""

    def test_create_session(self):
        """Test creating a session."""
        session = SessionMemory(
            session_id="session1",
            agent_id="agent1",
        )
        assert session.session_id == "session1"
        assert session.agent_id == "agent1"
        assert len(session.query_history) == 0

    def test_session_expiration(self):
        """Test session expiration check."""
        session = SessionMemory(
            session_id="session1",
            agent_id="agent1",
        )
        # New session should not be expired
        assert not session.is_expired(ttl_hours=24)


class TestQueryRecord:
    """Tests for QueryRecord model."""

    def test_create_record(self):
        """Test creating a query record."""
        record = QueryRecord(
            query="test query",
        )
        assert record.query == "test query"
        assert record.query_id is not None


class TestEnhancedQueryRequest:
    """Tests for EnhancedQueryRequest model."""

    def test_create_request(self):
        """Test creating a query request."""
        request = EnhancedQueryRequest(
            query="test query",
            limit=10,
        )
        assert request.query == "test query"
        assert request.limit == 10
        assert request.use_context is True

    def test_request_validation(self):
        """Test request validation."""
        with pytest.raises(ValueError):
            EnhancedQueryRequest(query="")  # Empty query


class TestKnowledgeItem:
    """Tests for KnowledgeItem model."""

    def test_create_item(self):
        """Test creating a knowledge item."""
        item = KnowledgeItem(
            title="Test",
            content="Test content",
            source="test",
        )
        assert item.title == "Test"
        assert item.source_type == "document"