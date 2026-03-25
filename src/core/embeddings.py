"""Dual embedding provider for gdrag v2.

Supports local embeddings (sentence-transformers) and OpenAI API,
with caching and batch processing capabilities.
"""

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from ..core.config import EmbeddingConfig, EmbeddingProvider
from ..models.schemas import EmbeddingResult

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""

    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, List[float]] = {}
        self._max_size = max_size

    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        key = self._hash_text(text)
        return self._cache.get(key)

    def set(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for text."""
        if len(self._cache) >= self._max_size:
            # Remove oldest 20% of entries
            keys_to_remove = list(self._cache.keys())[: self._max_size // 5]
            for key in keys_to_remove:
                del self._cache[key]

        key = self._hash_text(text)
        self._cache[key] = embedding

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class EmbeddingProviderBase(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name."""
        pass


class LocalEmbedder(EmbeddingProviderBase):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._dimension = config.dimensions

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading local model: {self.config.local_model}")
                self._model = SentenceTransformer(self.config.local_model)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded with dimension: {self._dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._dimension

    def get_model_name(self) -> str:
        """Get model name."""
        return self.config.local_model


class OpenAIEmbedder(EmbeddingProviderBase):
    """OpenAI API embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._client = None
        self._dimension = 1536  # text-embedding-3-small default

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI()
                logger.info("OpenAI client initialized")
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAI embeddings. "
                    "Install with: pip install openai"
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to initialize OpenAI client. "
                    f"Ensure OPENAI_API_KEY is set. Error: {e}"
                )

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._get_client()

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            try:
                response = self._client.embeddings.create(
                    model=self.config.openai_model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}")
                raise

        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def get_model_name(self) -> str:
        """Get model name."""
        return self.config.openai_model


class DualEmbedder(EmbeddingProviderBase):
    """Combined local + OpenAI embedding provider.

    Produces embeddings from both providers and concatenates them,
    or uses a specified provider.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.local = LocalEmbedder(config)
        self.openai = OpenAIEmbedder(config)
        self._cache = EmbeddingCache() if config.cache_embeddings else None

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # Check cache for each text
        results: Dict[int, List[float]] = {}
        texts_to_embed: List[Tuple[int, str]] = []

        for i, text in enumerate(texts):
            if self._cache:
                cached = self._cache.get(text)
                if cached is not None:
                    results[i] = cached
                    continue
            texts_to_embed.append((i, text))

        # Generate embeddings for uncached texts
        if texts_to_embed:
            indices = [i for i, _ in texts_to_embed]
            text_list = [t for _, t in texts_to_embed]

            local_embeddings = self.local.embed_batch(text_list)
            openai_embeddings = self.openai.embed_batch(text_list)

            # Combine embeddings based on provider config
            for idx, local_emb, openai_emb in zip(
                indices, local_embeddings, openai_embeddings
            ):
                if self.config.provider == EmbeddingProvider.BOTH:
                    # Concatenate embeddings
                    combined = local_emb + openai_emb
                elif self.config.provider == EmbeddingProvider.LOCAL:
                    combined = local_emb
                else:
                    combined = openai_emb

                results[idx] = combined

                # Cache result
                if self._cache:
                    self._cache.set(text_list[indices.index(idx)], combined)

        # Return in original order
        return [results[i] for i in range(len(texts))]

    def get_dimension(self) -> int:
        """Get embedding dimension based on provider config."""
        if self.config.provider == EmbeddingProvider.BOTH:
            return self.local.get_dimension() + self.openai.get_dimension()
        elif self.config.provider == EmbeddingProvider.LOCAL:
            return self.local.get_dimension()
        else:
            return self.openai.get_dimension()

    def get_model_name(self) -> str:
        """Get model name(s)."""
        if self.config.provider == EmbeddingProvider.BOTH:
            return f"{self.local.get_model_name()}+{self.openai.get_model_name()}"
        elif self.config.provider == EmbeddingProvider.LOCAL:
            return self.local.get_model_name()
        else:
            return self.openai.get_model_name()


class EmbeddingFactory:
    """Factory for creating embedding providers."""

    _instances: Dict[str, EmbeddingProviderBase] = {}

    @classmethod
    def get_provider(
        cls, config: EmbeddingConfig, provider: Optional[str] = None
    ) -> EmbeddingProviderBase:
        """Get or create an embedding provider.

        Args:
            config: Embedding configuration.
            provider: Override provider type (local, openai, both).

        Returns:
            Embedding provider instance.
        """
        provider_type = provider or config.provider.value
        cache_key = f"{provider_type}:{config.local_model}:{config.openai_model}"

        if cache_key not in cls._instances:
            if provider_type == "local":
                cls._instances[cache_key] = LocalEmbedder(config)
            elif provider_type == "openai":
                cls._instances[cache_key] = OpenAIEmbedder(config)
            else:
                cls._instances[cache_key] = DualEmbedder(config)

        return cls._instances[cache_key]


# ============================================================================
# Convenience Functions
# ============================================================================

def embed_texts(
    texts: List[str],
    config: Optional[EmbeddingConfig] = None,
    provider: Optional[str] = None,
) -> List[List[float]]:
    """Embed multiple texts.

    Args:
        texts: List of texts to embed.
        config: Embedding configuration (uses defaults if None).
        provider: Override provider type.

    Returns:
        List of embedding vectors.
    """
    if config is None:
        config = EmbeddingConfig()

    embedder = EmbeddingFactory.get_provider(config, provider)
    return embedder.embed_batch(texts)


def embed_query(
    query: str,
    config: Optional[EmbeddingConfig] = None,
    provider: Optional[str] = None,
) -> List[float]:
    """Embed a single query.

    Args:
        query: Query text to embed.
        config: Embedding configuration (uses defaults if None).
        provider: Override provider type.

    Returns:
        Embedding vector.
    """
    if config is None:
        config = EmbeddingConfig()

    embedder = EmbeddingFactory.get_provider(config, provider)
    return embedder.embed(query)


def get_embedding_dimension(
    config: Optional[EmbeddingConfig] = None,
    provider: Optional[str] = None,
) -> int:
    """Get embedding dimension for provider.

    Args:
        config: Embedding configuration (uses defaults if None).
        provider: Override provider type.

    Returns:
        Embedding dimension.
    """
    if config is None:
        config = EmbeddingConfig()

    embedder = EmbeddingFactory.get_provider(config, provider)
    return embedder.get_dimension()


def cache_embedding(
    text: str,
    embedding: List[float],
    config: Optional[EmbeddingConfig] = None,
) -> None:
    """Cache an embedding for later use.

    Args:
        text: Text that was embedded.
        embedding: Embedding vector.
        config: Embedding configuration (uses defaults if None).
    """
    if config is None:
        config = EmbeddingConfig()

    if config.cache_embeddings:
        embedder = EmbeddingFactory.get_provider(config)
        if hasattr(embedder, "_cache") and embedder._cache:
            embedder._cache.set(text, embedding)