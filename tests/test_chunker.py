"""Tests for document chunker module."""

import pytest

from src.core.chunker import DocumentChunker, TokenCounter
from src.core.config import ChunkConfig, ChunkStrategy


class TestTokenCounter:
    """Tests for TokenCounter."""

    def test_count_empty(self):
        """Test counting empty text."""
        assert TokenCounter.count("") == 0

    def test_count_words(self):
        """Test counting words."""
        text = "this is a test"
        count = TokenCounter.count(text)
        assert count > 0

    def test_split_by_tokens(self):
        """Test splitting by tokens."""
        text = "word " * 100
        chunks = TokenCounter.split_by_tokens(text, max_tokens=50)
        assert len(chunks) > 1


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_chunk_empty(self):
        """Test chunking empty text."""
        chunker = DocumentChunker()
        chunks = chunker.chunk("", doc_id="test")
        assert chunks == []

    def test_chunk_sliding_window(self):
        """Test sliding window chunking."""
        config = ChunkConfig(
            strategy=ChunkStrategy.SLIDING_WINDOW,
            chunk_size=100,
            chunk_overlap=20,
        )
        chunker = DocumentChunker(config)

        text = "This is a test sentence. " * 50
        chunks = chunker.chunk(text, doc_id="test")

        assert len(chunks) > 0
        assert all(c.doc_id == "test" for c in chunks)

    def test_chunk_paragraph(self):
        """Test paragraph chunking."""
        config = ChunkConfig(
            strategy=ChunkStrategy.PARAGRAPH,
            chunk_size=200,
        )
        chunker = DocumentChunker(config)

        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunker.chunk(text, doc_id="test")

        assert len(chunks) > 0

    def test_merge_small_chunks(self):
        """Test merging small chunks."""
        config = ChunkConfig(min_chunk_size=50)
        chunker = DocumentChunker(config)

        # Create small chunks
        text = "Short. " * 10
        chunks = chunker.chunk(text, doc_id="test")

        # Verify small chunks are merged
        for chunk in chunks:
            assert chunk.token_count >= config.min_chunk_size or chunk == chunks[-1]