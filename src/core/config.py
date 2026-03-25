"""Centralized configuration for gdrag v2.

Defines all configuration models using Pydantic for validation
and type safety throughout the application.
"""

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    LOCAL = "local"
    OPENAI = "openai"
    BOTH = "both"


class ChunkStrategy(str, Enum):
    """Available chunking strategies."""
    SLIDING_WINDOW = "sliding_window"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    HYBRID = "hybrid"


class CompressionStrategy(str, Enum):
    """Available compression strategies."""
    SUMMARIZE = "summarize"
    EXTRACT = "extract"
    HYBRID = "hybrid"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers."""
    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.LOCAL,
        description="Embedding provider to use"
    )
    local_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Local sentence-transformers model name"
    )
    openai_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name"
    )
    dimensions: int = Field(
        default=384,
        description="Embedding dimensions (local default)"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding generation"
    )
    cache_embeddings: bool = Field(
        default=True,
        description="Whether to cache generated embeddings"
    )


class ChunkConfig(BaseModel):
    """Configuration for document chunking."""
    strategy: ChunkStrategy = Field(
        default=ChunkStrategy.SLIDING_WINDOW,
        description="Chunking strategy to use"
    )
    chunk_size: int = Field(
        default=512,
        ge=50,
        le=4096,
        description="Target chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=128,
        ge=0,
        le=512,
        description="Overlap between chunks in tokens"
    )
    min_chunk_size: int = Field(
        default=100,
        ge=10,
        description="Minimum chunk size in tokens"
    )
    preserve_sentences: bool = Field(
        default=True,
        description="Avoid splitting sentences when possible"
    )
    semantic_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for semantic chunking"
    )


class RerankConfig(BaseModel):
    """Configuration for re-ranking."""
    enabled: bool = Field(
        default=True,
        description="Whether re-ranking is enabled"
    )
    model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking"
    )
    top_k_initial: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Number of results to retrieve before re-ranking"
    )
    top_k_final: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results after re-ranking"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for re-ranking"
    )


class CompressionConfig(BaseModel):
    """Configuration for context compression."""
    enabled: bool = Field(
        default=True,
        description="Whether compression is enabled"
    )
    strategy: CompressionStrategy = Field(
        default=CompressionStrategy.HYBRID,
        description="Compression strategy to use"
    )
    max_summary_tokens: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum tokens for summaries"
    )
    compression_ratio: float = Field(
        default=0.3,
        ge=0.1,
        le=0.9,
        description="Target compression ratio (0.3 = 30% of original)"
    )
    preserve_entities: bool = Field(
        default=True,
        description="Preserve named entities during compression"
    )


class AttentionConfig(BaseModel):
    """Configuration for focused attention mechanism."""
    enabled: bool = Field(
        default=True,
        description="Whether attention focusing is enabled"
    )
    decay_factor: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Temporal decay factor per time unit"
    )
    recency_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for recency in attention scoring"
    )
    relevance_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for relevance in attention scoring"
    )
    diversity_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for diversity in attention scoring"
    )
    max_attention_items: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum items to track in attention"
    )


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    # PostgreSQL
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="gdrag")
    postgres_user: str = Field(default="gdrag")
    postgres_password: str = Field(default="")

    # Qdrant
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="gdrag_knowledge")

    # Memgraph
    memgraph_host: str = Field(default="localhost")
    memgraph_port: int = Field(default=7687)
    memgraph_user: str = Field(default="admin")
    memgraph_password: str = Field(default="memgraph2024")


class SessionConfig(BaseModel):
    """Session memory configuration."""
    max_tokens: int = Field(
        default=8000,
        ge=1000,
        le=32000,
        description="Maximum tokens per session context"
    )
    max_history_items: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum query history items per session"
    )
    session_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Session time-to-live in hours"
    )
    auto_compress: bool = Field(
        default=True,
        description="Automatically compress old session history"
    )


class AppConfig(BaseModel):
    """Main application configuration."""
    # App metadata
    app_name: str = Field(default="gdrag")
    version: str = Field(default="2.0.0")
    debug: bool = Field(default=False)

    # Component configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkConfig = Field(default_factory=ChunkConfig)
    reranking: RerankConfig = Field(default_factory=RerankConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)

    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)

    # Paths
    config_dir: Path = Field(default=Path("config"))
    data_dir: Path = Field(default=Path("data"))


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from YAML file with environment variable override.
    
    Args:
        config_path: Path to YAML config file. If None, uses default locations.
    
    Returns:
        Loaded and validated AppConfig instance.
    """
    import os

    config_data = {}

    # Try to load from YAML file
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}
    else:
        # Try default locations
        default_paths = [
            Path("config/settings.yaml"),
            Path("/etc/gdrag/settings.yaml"),
        ]
        for path in default_paths:
            if path.exists():
                with open(path, "r") as f:
                    config_data = yaml.safe_load(f) or {}
                break

    # Override with environment variables
    env_mappings = {
        "GDRAG_DEBUG": ("debug", bool),
        "GDRAG_API_HOST": ("api_host", str),
        "GDRAG_API_PORT": ("api_port", int),
        "POSTGRES_HOST": ("database.postgres_host", str),
        "POSTGRES_PORT": ("database.postgres_port", int),
        "POSTGRES_DB": ("database.postgres_db", str),
        "POSTGRES_USER": ("database.postgres_user", str),
        "POSTGRES_PASSWORD": ("database.postgres_password", str),
        "QDRANT_HOST": ("database.qdrant_host", str),
        "QDRANT_PORT": ("database.qdrant_port", int),
        "MEMGRAPH_HOST": ("database.memgraph_host", str),
        "MEMGRAPH_PORT": ("database.memgraph_port", int),
        "MEMGRAPH_PASSWORD": ("database.memgraph_password", str),
        "OPENAI_API_KEY": (None, str),  # Special handling
    }

    for env_var, (config_key, value_type) in env_mappings.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            if config_key:
                # Convert type
                if value_type == bool:
                    env_value = env_value.lower() in ("true", "1", "yes")
                elif value_type == int:
                    env_value = int(env_value)

                # Set nested key
                keys = config_key.split(".")
                current = config_data
                for key in keys[:-1]:
                    current = current.setdefault(key, {})
                current[keys[-1]] = env_value

    return AppConfig(**config_data)