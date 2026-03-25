# Implementation Plan: gdrag v2 - Advanced Agent Memory & Context Optimization

[Overview]
Complete redesign of the gdrag RAG system to implement advanced agent memory management, intelligent context window optimization, and focused attention mechanisms. This v2 will transform gdrag from a basic RAG API into a professional-grade agentic memory system capable of managing context efficiently, remembering session history, compressing information, and maintaining focused attention on relevant knowledge.

The implementation adds dual embedding support (local + API), semantic chunking with sliding window, cross-encoder re-ranking, session memory persistence, automatic context compression, and attention-focused retrieval. These features address the core limitations of the current v1 system which lacks intelligent context management and uses placeholder embeddings.

[Types]
New type definitions for the enhanced memory system:

```python
# Embedding Configuration
class EmbeddingConfig(BaseModel):
    provider: Literal["local", "openai", "both"]
    local_model: str = "all-MiniLM-L6-v2"
    openai_model: str = "text-embedding-3-small"
    dimensions: int = 384  # local default
    batch_size: int = 32
    cache_embeddings: bool = True

# Chunk Configuration
class ChunkConfig(BaseModel):
    strategy: Literal["sliding_window", "semantic", "paragraph", "hybrid"]
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 128  # tokens
    min_chunk_size: int = 100
    preserve_sentences: bool = True
    semantic_threshold: float = 0.7

# Re-ranking Configuration
class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_initial: int = 50  # results before re-ranking
    top_k_final: int = 10  # results after re-ranking
    batch_size: int = 32

# Session Memory
class SessionMemory(BaseModel):
    session_id: str
    agent_id: str
    created_at: datetime
    last_active: datetime
    query_history: List[QueryRecord]
    context_summary: Optional[str]
    token_count: int
    max_tokens: int = 8000

class QueryRecord(BaseModel):
    query_id: str
    query: str
    timestamp: datetime
    results_ids: List[str]
    relevance_scores: List[float]
    feedback: Optional[float]  # user/agent feedback

# Context Compression
class CompressionConfig(BaseModel):
    enabled: bool = True
    strategy: Literal["summarize", "extract", "hybrid"]
    max_summary_tokens: int = 500
    compression_ratio: float = 0.3  # target: 30% of original
    preserve_entities: bool = True

# Attention Focus
class AttentionConfig(BaseModel):
    enabled: bool = True
    decay_factor: float = 0.95  # temporal decay
    recency_weight: float = 0.3
    relevance_weight: float = 0.5
    diversity_weight: float = 0.2
    max_attention_items: int = 100

# Enhanced Query Request
class EnhancedQueryRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    session_id: Optional[str] = None
    use_context: bool = True  # include session context
    compress_results: bool = False
    attention_focus: bool = True
    embedding_provider: Optional[Literal["local", "openai", "both"]] = None
    limit: int = Field(10, ge=1, le=100)

# Enhanced Query Response
class EnhancedQueryResponse(BaseModel):
    query: str
    results: List[RankedResult]
    compressed_context: Optional[str]
    session_context_used: bool
    attention_scores: Dict[str, float]
    metadata: ResponseMetadata

class RankedResult(BaseModel):
    content: str
    title: str
    domain: str
    source: str
    semantic_score: float
    rerank_score: float
    attention_score: float
    final_score: float
    chunk_index: int
    doc_id: str

class ResponseMetadata(BaseModel):
    total_results: int
    reranked: bool
    compressed: bool
    session_id: Optional[str]
    embedding_provider: str
    processing_time_ms: float
    tokens_saved: int
```

[Files]
Detailed file structure for gdrag v2:

**New Files to Create:**

- `/home/gdrick/gdrag/src/core/__init__.py` - Core module init
- `/home/gdrick/gdrag/src/core/embeddings.py` - Dual embedding provider (local + OpenAI)
- `/home/gdrick/gdrag/src/core/chunker.py` - Intelligent document chunking
- `/home/gdrick/gdrag/src/core/reranker.py` - Cross-encoder re-ranking
- `/home/gdrick/gdrag/src/core/compressor.py` - Context compression/summarization
- `/home/gdrick/gdrag/src/core/attention.py` - Focused attention mechanism
- `/home/gdrick/gdrag/src/core/session_manager.py` - Session memory persistence
- `/home/gdrick/gdrag/src/core/vector_store.py` - Enhanced Qdrant operations
- `/home/gdrick/gdrag/src/core/graph_store.py` - Enhanced Memgraph operations
- `/home/gdrick/gdrag/src/core/relational_store.py` - Enhanced PostgreSQL operations
- `/home/gdrick/gdrag/src/core/config.py` - Centralized configuration

- `/home/gdrick/gdrag/src/models/__init__.py` - Models module init
- `/home/gdrick/grag/src/models/schemas.py` - All Pydantic models
- `/home/gdrick/gdrag/src/models/database.py` - Database ORM models

- `/home/gdrick/gdrag/src/api/v2/__init__.py` - API v2 module init
- `/home/gdrick/gdrag/src/api/v2/router.py` - Enhanced API router
- `/home/gdrick/grag/src/api/v2/dependencies.py` - FastAPI dependencies
- `/home/gdrick/gdrag/src/api/v2/middleware.py` - Custom middleware

- `/home/gdrick/gdrag/src/mcp/v2/__init__.py` - MCP v2 module init
- `/home/gdrick/gdrag/src/mcp/v2/server.py` - Enhanced MCP server
- `/home/gdrick/gdrag/src/mcp/v2/tools.py` - MCP tool definitions

- `/home/gdrick/gdrag/tests/__init__.py` - Tests module init
- `/home/gdrick/gdrag/tests/test_embeddings.py` - Embedding tests
- `/home/gdrick/gdrag/tests/test_chunker.py` - Chunking tests
- `/home/gdrick/gdrag/tests/test_reranker.py` - Re-ranking tests
- `/home/gdrick/gdrag/tests/test_session.py` - Session tests
- `/home/gdrick/gdrag/tests/test_api_v2.py` - API v2 tests

- `/home/gdrick/gdrag/migrations/001_session_memory.sql` - Session tables
- `/home/gdrick/gdrag/migrations/002_enhanced_knowledge.sql` - Enhanced knowledge schema

- `/home/gdrick/gdrag/config/settings.yaml` - Application settings
- `/home/gdrick/gdrag/config/embedding_models.yaml` - Embedding model configs

**Existing Files to Modify:**

- `/home/gdrick/gdrag/requirements.txt` - Add new dependencies
- `/home/gdrick/gdrag/src/api/main.py` - Refactor to use v2 router
- `/home/gdrick/gdrag/src/mcp/rag_mcp_server.py` - Refactor to use v2 tools
- `/home/gdrick/gdrag/config/agents.yaml` - Add session/attention permissions
- `/home/gdrick/gdrag/README.md` - Update documentation

[Functions]
New and modified functions:

**src/core/embeddings.py:**
- `class EmbeddingProvider` - Abstract base for embeddings
- `class LocalEmbedder(EmbeddingProvider)` - sentence-transformers implementation
- `class OpenAIEmbedder(EmbeddingProvider)` - OpenAI API implementation
- `class DualEmbedder(EmbeddingProvider)` - Combined local + OpenAI
- `embed_texts(texts: List[str], provider: str) -> List[List[float]]` - Main embedding function
- `embed_query(query: str, provider: str) -> List[float]` - Single query embedding
- `get_embedding_dimension(provider: str) -> int` - Get vector dimensions
- `cache_embedding(text: str, embedding: List[float])` - Cache embeddings

**src/core/chunker.py:**
- `class DocumentChunker` - Main chunking class
- `chunk_by_tokens(text: str, chunk_size: int, overlap: int) -> List[Chunk]` - Token-based chunking
- `chunk_by_sentences(text: str, min_size: int) -> List[Chunk]` - Sentence-aware chunking
- `chunk_semantic(text: str, threshold: float) -> List[Chunk]` - Semantic boundary detection
- `chunk_hybrid(text: str, config: ChunkConfig) -> List[Chunk]` - Hybrid strategy
- `merge_small_chunks(chunks: List[Chunk], min_size: int) -> List[Chunk]` - Merge undersized chunks
- `add_overlap(chunks: List[Chunk], overlap: int) -> List[Chunk]` - Add context overlap

**src/core/reranker.py:**
- `class CrossEncoderReranker` - Re-ranking with cross-encoder
- `rerank(query: str, documents: List[str], top_k: int) -> List[RankedDocument]` - Main re-ranking
- `batch_rerank(queries: List[str], documents: List[List[str]]) -> List[List[RankedDocument]]` - Batch re-ranking
- `load_model(model_name: str) -> CrossEncoder` - Load cross-encoder model
- `compute_scores(query: str, documents: List[str]) -> List[float]` - Compute relevance scores

**src/core/compressor.py:**
- `class ContextCompressor` - Context compression
- `summarize_text(text: str, max_tokens: int) -> str` - Generate summary
- `extract_key_points(text: str, num_points: int) -> List[str]` - Extract key information
- `compress_results(results: List[RankedResult], ratio: float) -> List[RankedResult]` - Compress result set
- `merge_similar_results(results: List[RankedResult], threshold: float) -> List[RankedResult]` - Deduplicate

**src/core/attention.py:**
- `class AttentionManager` - Focused attention mechanism
- `calculate_attention_score(item: KnowledgeItem, context: AttentionContext) -> float` - Score attention
- `apply_temporal_decay(score: float, age: float, decay: float) -> float` - Time-based decay
- `diversity_filter(results: List[RankedResult], max_similar: int) -> List[RankedResult]` - Ensure diversity
- `update_attention_weights(session_id: str, feedback: float)` - Learn from feedback
- `get_focused_context(session_id: str, query: str) -> List[KnowledgeItem]` - Get focused context

**src/core/session_manager.py:**
- `class SessionManager` - Session memory management
- `create_session(agent_id: str) -> SessionMemory` - Create new session
- `get_session(session_id: str) -> SessionMemory` - Retrieve session
- `update_session(session_id: str, query: QueryRecord)` - Add query to history
- `get_session_context(session_id: str, max_tokens: int) -> str` - Get relevant context
- `compress_session_history(session_id: str)` - Compress old history
- `delete_session(session_id: str)` - Clean up session
- `list_sessions(agent_id: str) -> List[SessionMemory]` - List agent sessions

**src/core/vector_store.py:**
- `class EnhancedVectorStore` - Enhanced Qdrant operations
- `upsert_chunks(chunks: List[Chunk], embeddings: Dict[str, List[float]])` - Store chunks
- `search_with_rerank(query: str, config: RerankConfig) -> List[RankedResult]` - Search + re-rank
- `search_with_attention(query: str, session_id: str) -> List[RankedResult]` - Attention-focused search
- `delete_document(doc_id: str)` - Remove document
- `get_collection_stats() -> Dict` - Collection statistics

**src/core/graph_store.py:**
- `class EnhancedGraphStore` - Enhanced Memgraph operations
- `extract_and_store_concepts(doc_id: str, content: str)` - Extract concepts from content
- `find_related_concepts(concepts: List[str], depth: int) -> List[ConceptRelation]` - Graph traversal
- `build_knowledge_graph(documents: List[Document])` - Build graph from documents
- `get_concept_context(concept: str) -> str` - Get textual context for concept

**src/core/relational_store.py:**
- `class EnhancedRelationalStore` - Enhanced PostgreSQL operations
- `store_session_memory(session: SessionMemory)` - Persist session
- `query_with_session_filter(query: str, session_id: str) -> List[dict]` - Session-aware query
- `get_knowledge_stats(domain: str) -> Dict` - Knowledge statistics
- `full_text_search(query: str, limit: int) -> List[dict]` - Enhanced full-text search

**Modified Functions in src/api/main.py:**
- `rag_query` → Refactor to use `EnhancedQueryPipeline`
- `semantic_search` → Add re-ranking option
- All endpoints → Add session support headers
- Add new `/api/v2/` endpoints

**Modified Functions in src/mcp/rag_mcp_server.py:**
- `rag_query` → Enhanced with session and attention
- Add new tools: `session_create`, `session_context`, `compress_context`
- `ingest_knowledge` → Add chunking support

[Classes]
New and modified classes:

**New Classes:**

- `EmbeddingProvider` (src/core/embeddings.py) - Abstract base
  - Methods: `embed()`, `embed_batch()`, `get_dimension()`
  - Inheritance: `LocalEmbedder`, `OpenAIEmbedder`, `DualEmbedder`

- `DocumentChunker` (src/core/chunker.py)
  - Methods: `chunk()`, `merge()`, `add_overlap()`
  - Config: `ChunkConfig`

- `CrossEncoderReranker` (src/core/reranker.py)
  - Methods: `rerank()`, `batch_rerank()`, `load_model()`
  - Config: `RerankConfig`

- `ContextCompressor` (src/core/compressor.py)
  - Methods: `summarize()`, `extract()`, `compress_results()`
  - Config: `CompressionConfig`

- `AttentionManager` (src/core/attention.py)
  - Methods: `calculate_score()`, `apply_decay()`, `diversity_filter()`
  - Config: `AttentionConfig`

- `SessionManager` (src/core/session_manager.py)
  - Methods: `create()`, `get()`, `update()`, `compress()`, `delete()`
  - Dependencies: PostgreSQL, Redis (optional cache)

- `EnhancedVectorStore` (src/core/vector_store.py)
  - Methods: `upsert_chunks()`, `search_with_rerank()`, `search_with_attention()`
  - Wraps: qdrant-client

- `EnhancedGraphStore` (src/core/graph_store.py)
  - Methods: `extract_concepts()`, `find_related()`, `build_graph()`
  - Wraps: neo4j driver

- `EnhancedRelationalStore` (src/core/relational_store.py)
  - Methods: `store_session()`, `query_with_filter()`, `full_text_search()`
  - Wraps: psycopg2

- `QueryPipeline` (src/core/pipeline.py) - Orchestrates full RAG flow
  - Methods: `execute()`, `execute_with_session()`, `execute_compressed()`

**Modified Classes:**

- `FastAPI app` (src/api/main.py)
  - Add v2 router
  - Add session middleware
  - Update CORS for new endpoints

- `FastMCP server` (src/mcp/rag_mcp_server.py)
  - Add enhanced tools
  - Add session management tools

[Dependencies]
New dependencies to add to requirements.txt:

```txt
# Embeddings (dual support)
sentence-transformers==2.3.1  # Local embeddings
openai==1.12.0  # OpenAI API embeddings
tiktoken==0.6.0  # Token counting

# Re-ranking
# Note: sentence-transformers includes CrossEncoder

# Chunking
nltk==3.8.1  # Sentence tokenization
spacy==3.7.4  # Advanced NLP (optional)

# Compression/Summarization
transformers==4.37.2  # For local summarization models
torch==2.2.0  # Required by transformers

# Caching
redis==5.0.1  # Optional: for embedding/result caching

# Monitoring enhancements
prometheus-client==0.19.0  # Already present
structlog==24.1.0  # Structured logging

# Testing
pytest==8.0.0
pytest-asyncio==0.23.0
httpx==0.26.0  # Already present
fakeredis==2.21.0  # Mock Redis for tests
```

Version constraints:
- sentence-transformers >= 2.3.0 (for CrossEncoder support)
- openai >= 1.0.0 (new API structure)
- transformers >= 4.35.0 (for recent model support)
- torch >= 2.0.0 (performance improvements)

[Testing]
Testing strategy for the enhanced system:

**Test Files:**

1. `tests/test_embeddings.py`
   - Test local embedder produces correct dimensions
   - Test OpenAI embedder with mock API
   - Test dual embedder combines results
   - Test embedding caching
   - Test batch vs single embedding consistency

2. `tests/test_chunker.py`
   - Test sliding window produces correct overlap
   - Test semantic chunking respects boundaries
   - Test hybrid chunking combines strategies
   - Test min/max chunk size enforcement
   - Test sentence preservation

3. `tests/test_reranker.py`
   - Test re-ranking improves relevance
   - Test batch re-ranking efficiency
   - Test score normalization
   - Test top_k filtering

4. `tests/test_session.py`
   - Test session creation and retrieval
   - Test session history tracking
   - Test session context generation
   - Test session compression
   - Test session expiration

5. `tests/test_api_v2.py`
   - Test enhanced query endpoint
   - Test session-aware queries
   - Test compression endpoint
   - Test attention-focused retrieval
   - Test backward compatibility with v1

**Validation Strategies:**

- Unit tests for each core module
- Integration tests for full pipeline
- Mock external services (OpenAI, Qdrant, PostgreSQL, Memgraph)
- Performance benchmarks for embedding and re-ranking
- Relevance evaluation with sample queries

**Test Commands:**
```bash
# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/test_embeddings.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

[Implementation Order]
Recommended implementation sequence:

1. **Phase 1: Core Infrastructure** (Foundation)
   - Create `src/core/config.py` with all configuration models
   - Create `src/models/schemas.py` with all Pydantic models
   - Implement `src/core/embeddings.py` with dual provider support
   - Implement `src/core/chunker.py` with sliding window strategy
   - Update `requirements.txt` with new dependencies

2. **Phase 2: Enhanced Storage Layer** (Data Layer)
   - Implement `src/core/vector_store.py` with chunked storage
   - Implement `src/core/graph_store.py` with concept extraction
   - Implement `src/core/relational_store.py` with session support
   - Create database migrations for session tables

3. **Phase 3: Intelligence Layer** (Processing)
   - Implement `src/core/reranker.py` with cross-encoder
   - Implement `src/core/compressor.py` with summarization
   - Implement `src/core/attention.py` with focus mechanism

4. **Phase 4: Session Management** (Memory)
   - Implement `src/core/session_manager.py`
   - Integrate session context into query pipeline
   - Add session persistence to PostgreSQL

5. **Phase 5: Query Pipeline** (Orchestration)
   - Create `src/core/pipeline.py` orchestrating all components
   - Implement full RAG flow: chunk → embed → store → retrieve → rerank → compress
   - Add attention-focused retrieval path

6. **Phase 6: API v2** (Interface)
   - Create `src/api/v2/` with enhanced router
   - Implement all new endpoints
   - Add session middleware
   - Ensure backward compatibility with v1

7. **Phase 7: MCP v2** (Agent Interface)
   - Refactor MCP server with enhanced tools
   - Add session management tools
   - Update tool descriptions and schemas

8. **Phase 8: Testing & Documentation** (Quality)
   - Write comprehensive test suite
   - Update README with v2 documentation
   - Add usage examples
   - Performance benchmarks

9. **Phase 9: Migration & Deployment** (Finalization)
   - Create migration scripts from v1
   - Update configuration files
   - Document upgrade path