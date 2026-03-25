"""Intelligent document chunking for gdrag v2.

Supports multiple chunking strategies: sliding window, semantic,
paragraph-based, and hybrid approaches.
"""

import logging
import re
from typing import List, Optional, Tuple

from ..core.config import ChunkConfig, ChunkStrategy
from ..models.schemas import Chunk

logger = logging.getLogger(__name__)


class TokenCounter:
    """Simple token counter using word-based approximation."""

    @staticmethod
    def count(text: str) -> int:
        """Count approximate tokens in text.

        Uses word count * 1.3 as approximation (accounts for subword tokens).
        """
        words = text.split()
        return int(len(words) * 1.3)

    @staticmethod
    def split_by_tokens(text: str, max_tokens: int) -> List[str]:
        """Split text into chunks by token limit.

        Args:
            text: Text to split.
            max_tokens: Maximum tokens per chunk.

        Returns:
            List of text chunks.
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0

        for word in words:
            word_tokens = int(len(word) / 4) + 1  # Rough estimate

            if current_tokens + word_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class SentenceSplitter:
    """Split text into sentences."""

    # Pattern to split on sentence boundaries
    SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$", re.MULTILINE
    )

    @classmethod
    def split(cls, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        # Split on sentence boundaries
        sentences = cls.SENTENCE_PATTERN.split(text)

        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences


class DocumentChunker:
    """Main chunking class supporting multiple strategies."""

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.token_counter = TokenCounter()
        self.sentence_splitter = SentenceSplitter()

    def chunk(self, text: str, doc_id: str = "") -> List[Chunk]:
        """Chunk document using configured strategy.

        Args:
            text: Document text to chunk.
            doc_id: Document identifier.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        strategy = self.config.strategy

        if strategy == ChunkStrategy.SLIDING_WINDOW:
            chunks = self._chunk_sliding_window(text, doc_id)
        elif strategy == ChunkStrategy.SEMANTIC:
            chunks = self._chunk_semantic(text, doc_id)
        elif strategy == ChunkStrategy.PARAGRAPH:
            chunks = self._chunk_paragraph(text, doc_id)
        elif strategy == ChunkStrategy.HYBRID:
            chunks = self._chunk_hybrid(text, doc_id)
        else:
            chunks = self._chunk_sliding_window(text, doc_id)

        # Merge small chunks
        chunks = self.merge_small_chunks(chunks)

        # Add overlap if needed
        if self.config.chunk_overlap > 0 and strategy != ChunkStrategy.SLIDING_WINDOW:
            chunks = self.add_overlap(chunks)

        return chunks

    def _chunk_sliding_window(self, text: str, doc_id: str) -> List[Chunk]:
        """Chunk using sliding window with overlap.

        Args:
            text: Text to chunk.
            doc_id: Document ID.

        Returns:
            List of chunks.
        """
        words = text.split()
        chunks = []

        # Calculate word-based chunk size (approximate)
        words_per_token = 0.75  # Rough estimate
        chunk_size_words = int(self.config.chunk_size * words_per_token)
        overlap_words = int(self.config.chunk_overlap * words_per_token)

        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Find character positions
            start_char = len(" ".join(words[:start]))
            if start > 0:
                start_char += 1  # Account for space
            end_char = start_char + len(chunk_text)

            chunk = Chunk(
                doc_id=doc_id,
                content=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                token_count=self.token_counter.count(chunk_text),
            )
            chunks.append(chunk)

            # Move window
            start = end - overlap_words
            chunk_index += 1

            # Prevent infinite loop
            if start >= end:
                break

        return chunks

    def _chunk_semantic(self, text: str, doc_id: str) -> List[Chunk]:
        """Chunk based on semantic boundaries.

        Uses sentence similarity to find natural breakpoints.

        Args:
            text: Text to chunk.
            doc_id: Document ID.

        Returns:
            List of chunks.
        """
        sentences = self.sentence_splitter.split(text)

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_tokens = 0
        chunk_index = 0
        start_char = 0

        for sentence in sentences:
            sentence_tokens = self.token_counter.count(sentence)

            # Check if adding this sentence exceeds limit
            if (
                current_tokens + sentence_tokens > self.config.chunk_size
                and current_sentences
            ):
                # Create chunk from current sentences
                chunk_text = " ".join(current_sentences)
                end_char = start_char + len(chunk_text)

                chunk = Chunk(
                    doc_id=doc_id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=current_tokens,
                )
                chunks.append(chunk)

                # Reset for next chunk
                start_char = end_char + 1
                current_sentences = []
                current_tokens = 0
                chunk_index += 1

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Handle remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            end_char = start_char + len(chunk_text)

            chunk = Chunk(
                doc_id=doc_id,
                content=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                token_count=current_tokens,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_paragraph(self, text: str, doc_id: str) -> List[Chunk]:
        """Chunk by paragraphs.

        Args:
            text: Text to chunk.
            doc_id: Document ID.

        Returns:
            List of chunks.
        """
        # Split by double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\s*\n", text)

        chunks = []
        current_paragraphs = []
        current_tokens = 0
        chunk_index = 0
        start_char = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            paragraph_tokens = self.token_counter.count(paragraph)

            # Check if adding this paragraph exceeds limit
            if (
                current_tokens + paragraph_tokens > self.config.chunk_size
                and current_paragraphs
            ):
                # Create chunk
                chunk_text = "\n\n".join(current_paragraphs)
                end_char = start_char + len(chunk_text)

                chunk = Chunk(
                    doc_id=doc_id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=current_tokens,
                )
                chunks.append(chunk)

                # Reset
                start_char = end_char + 2  # Account for paragraph separator
                current_paragraphs = []
                current_tokens = 0
                chunk_index += 1

            current_paragraphs.append(paragraph)
            current_tokens += paragraph_tokens

        # Handle remaining paragraphs
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            end_char = start_char + len(chunk_text)

            chunk = Chunk(
                doc_id=doc_id,
                content=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                token_count=current_tokens,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_hybrid(self, text: str, doc_id: str) -> List[Chunk]:
        """Hybrid chunking combining semantic and paragraph strategies.

        First splits by paragraphs, then applies semantic chunking within
        paragraphs that exceed the size limit.

        Args:
            text: Text to chunk.
            doc_id: Document ID.

        Returns:
            List of chunks.
        """
        # First split by paragraphs
        paragraphs = re.split(r"\n\s*\n", text)

        all_chunks = []
        chunk_index = 0
        global_start = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_tokens = self.token_counter.count(paragraph)

            if para_tokens <= self.config.chunk_size:
                # Paragraph fits in one chunk
                chunk = Chunk(
                    doc_id=doc_id,
                    content=paragraph,
                    chunk_index=chunk_index,
                    start_char=global_start,
                    end_char=global_start + len(paragraph),
                    token_count=para_tokens,
                )
                all_chunks.append(chunk)
                chunk_index += 1
            else:
                # Paragraph too large, apply semantic chunking
                sub_chunks = self._chunk_semantic(paragraph, doc_id)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_index = chunk_index
                    sub_chunk.start_char += global_start
                    sub_chunk.end_char += global_start
                    all_chunks.append(sub_chunk)
                    chunk_index += 1

            global_start += len(paragraph) + 2  # +2 for paragraph separator

        return all_chunks

    def merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks that are smaller than minimum size.

        Args:
            chunks: List of chunks to merge.

        Returns:
            List of merged chunks.
        """
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            if current.token_count < self.config.min_chunk_size:
                # Merge with next chunk
                current = Chunk(
                    chunk_id=current.chunk_id,
                    doc_id=current.doc_id,
                    content=f"{current.content} {next_chunk.content}",
                    chunk_index=current.chunk_index,
                    start_char=current.start_char,
                    end_char=next_chunk.end_char,
                    token_count=current.token_count + next_chunk.token_count,
                    metadata=current.metadata,
                )
            else:
                merged.append(current)
                current = next_chunk

        # Don't forget the last chunk
        merged.append(current)

        # Re-index chunks
        for i, chunk in enumerate(merged):
            chunk.chunk_index = i

        return merged

    def add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Add context overlap between chunks.

        Args:
            chunks: List of chunks.

        Returns:
            List of chunks with overlap added.
        """
        if len(chunks) <= 1:
            return chunks

        overlapped = []

        for i, chunk in enumerate(chunks):
            content = chunk.content

            # Add overlap from previous chunk
            if i > 0:
                prev_words = chunks[i - 1].content.split()
                overlap_count = min(
                    len(prev_words),
                    int(self.config.chunk_overlap * 0.75),  # Convert tokens to words
                )
                if overlap_count > 0:
                    prefix = " ".join(prev_words[-overlap_count:])
                    content = f"{prefix} {content}"

            # Add overlap from next chunk
            if i < len(chunks) - 1:
                next_words = chunks[i + 1].content.split()
                overlap_count = min(
                    len(next_words),
                    int(self.config.chunk_overlap * 0.75),
                )
                if overlap_count > 0:
                    suffix = " ".join(next_words[:overlap_count])
                    content = f"{content} {suffix}"

            overlapped_chunk = Chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=content,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                token_count=self.token_counter.count(content),
                metadata=chunk.metadata,
            )
            overlapped.append(overlapped_chunk)

        return overlapped