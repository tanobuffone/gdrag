"""Tests for configuration module."""

import pytest
from pathlib import Path

from src.core.config import (
    AppConfig,
    AttentionConfig,
    ChunkConfig,
    ChunkStrategy,
    CompressionConfig,
    EmbeddingConfig,
    EmbeddingProvider,
    RerankConfig,
    load_config,
)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        assert config.provider == EmbeddingProvider.LOCAL
        assert config.local_model == "all-MiniLM-L6-v2"
        assert config.batch_size == 32
        assert config.cache_embeddings is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            batch_size=64,
        )
        assert config.provider == EmbeddingProvider.OPENAI
        assert config.batch_size == 64


class TestChunkConfig:
    """Tests for ChunkConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkConfig()
        assert config.strategy == ChunkStrategy.SLIDING_WINDOW
        assert config.chunk_size == 512
        assert config.chunk_overlap == 128

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            ChunkConfig(chunk_size=10)  # Below minimum


class TestRerankConfig:
    """Tests for RerankConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RerankConfig()
        assert config.enabled is True
        assert config.top_k_initial == 50
        assert config.top_k_final == 10


class TestAppConfig:
    """Tests for AppConfig."""

    def test_default_values(self):
        """Test default application configuration."""
        config = AppConfig()
        assert config.app_name == "gdrag"
        assert config.version == "2.0.0"
        assert config.api_port == 8000

    def test_component_configs(self):
        """Test component configuration initialization."""
        config = AppConfig()
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.chunking, ChunkConfig)
        assert isinstance(config.reranking, RerankConfig)
        assert isinstance(config.compression, CompressionConfig)
        assert isinstance(config.attention, AttentionConfig)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default(self):
        """Test loading default configuration."""
        config = load_config()
        assert isinstance(config, AppConfig)
        assert config.app_name == "gdrag"