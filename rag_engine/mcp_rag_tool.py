"""
mcp_rag_tool.py — Phase 6: MCP tool exposing the full RAG pipeline.

Registers as ``rag_retrieve`` in :class:`MCPToolRegistry` via the
re-export shim in ``tools/rag_retrieve_tool.py``.

Pipeline: chunk → embed → index (dense + sparse) → hybrid retrieve → rerank.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml # type: ignore

from evaluation_core import ensure_deadline
from tools.mcp_base import MCPToolBase

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(domain: str, template_name: str) -> str:
    """Load a YAML prompt template.  Returns empty string on failure."""
    path = _PROMPTS_DIR / domain / f"{template_name}.yaml"
    if not path.exists():
        return ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("template", "")
    except Exception:
        logger.warning("Failed to load prompt %s", path, exc_info=True)
        return ""


class RAGTool(MCPToolBase):
    """MCP tool exposing the full hybrid RAG pipeline.

    Accepts a query and an optional list of inline documents (for ad-hoc
    indexing) or a path to a pre-built index directory.  Returns top-k
    reranked results.
    """

    name = "rag_retrieve"
    description = (
        "Central RAG engine: hybrid dense+sparse retrieval with "
        "RRF fusion and cross-encoder reranking over medical literature."
    )
    triggers = [
        "retrieve",
        "search",
        "find",
        "evidence",
        "literature",
        "papers",
        "documents",
        "rag",
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Medical research query",
            },
            "documents": {
                "type": "array",
                "items": {"type": "object"},
                "description": (
                    "Optional inline documents to index on-the-fly. "
                    "Each item: {text: str, metadata: dict}"
                ),
            },
            "index_path": {
                "type": "string",
                "description": "Path to pre-built index directory",
            },
            "top_k": {
                "type": "integer",
                "default": 10,
                "description": "Number of results to return",
            },
            "domain": {
                "type": "string",
                "enum": ["pubmed", "fda", "clinical_trials", "local"],
                "default": "pubmed",
                "description": "Prompt domain for synthesis templates",
            },
            "rerank": {
                "type": "boolean",
                "default": True,
                "description": "Whether to apply cross-encoder reranking",
            },
        },
        "required": ["query"],
    }

    def __init__(self) -> None:
        self._dense_index = None
        self._sparse_index = None
        self._embedder = None
        self._retriever = None
        self._reranker = None
        self._chunker = None

    # ------------------------------------------------------------------ #
    # Lazy initialisation
    # ------------------------------------------------------------------ #

    def _ensure_embedder(self):
        if self._embedder is None:
            from rag_engine.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _ensure_chunker(self):
        if self._chunker is None:
            from rag_engine.chunker import SemanticChunker
            self._chunker = SemanticChunker(
                embedder=self._ensure_embedder(),
                max_chunk_tokens=512,
            )
        return self._chunker

    def _ensure_indexes(self, dimension: int):
        """Create fresh dense + sparse indexes."""
        if self._dense_index is None:
            from rag_engine.dense_index import DenseIndex
            self._dense_index = DenseIndex(dimension=dimension)
        if self._sparse_index is None:
            from rag_engine.sparse_index import BM25Index
            self._sparse_index = BM25Index()

    def _ensure_retriever(self, dimension: int):
        self._ensure_indexes(dimension)
        if self._retriever is None:
            from rag_engine.hybrid_retriever import HybridRetriever
            self._retriever = HybridRetriever(
                dense_index=self._dense_index,
                sparse_index=self._sparse_index,
                embedder=self._ensure_embedder(),
                rrf_k=60,
            )
        return self._retriever

    def _ensure_reranker(self):
        if self._reranker is None:
            from rag_engine.reranker import Reranker
            self._reranker = Reranker()
        return self._reranker

    # ------------------------------------------------------------------ #
    # MCPToolBase.call
    # ------------------------------------------------------------------ #

    def call(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        query = input_dict["query"]
        documents = input_dict.get("documents", [])
        index_path = input_dict.get("index_path")
        top_k = input_dict.get("top_k", 10)
        domain = input_dict.get("domain", "pubmed")
        do_rerank = input_dict.get("rerank", True)
        deadline_at = input_dict.get("_runtime_deadline_at_monotonic")

        try:
            ensure_deadline(deadline_at)
            embedder = self._ensure_embedder()

            # --- Load or build index ---
            if index_path:
                return self._query_existing_index(
                    query,
                    index_path,
                    top_k,
                    domain,
                    do_rerank,
                    start,
                    deadline_at=deadline_at,
                )

            if not documents:
                elapsed = time.time() - start
                return self._success(
                    results=[],
                    retrieval_time_sec=elapsed,
                    message="No documents or index_path provided.",
                )

            # --- Inline indexing: chunk → embed → index ---
            chunker = self._ensure_chunker()

            all_chunks: List[str] = []
            all_meta: List[Dict[str, Any]] = []
            for doc in documents:
                ensure_deadline(deadline_at)
                text = doc.get("text", "")
                meta = doc.get("metadata", {})
                chunks = chunker.chunk(
                    text,
                    deadline_at=deadline_at,
                    client_max_attempts=(
                        1 if deadline_at is not None else None
                    ),
                )
                for chunk in chunks:
                    all_chunks.append(chunk.text)
                    all_meta.append({**meta, "chunk_index": chunk.index})

            if not all_chunks:
                elapsed = time.time() - start
                return self._success(
                    results=[],
                    retrieval_time_sec=elapsed,
                    message="No text found in provided documents.",
                )

            # Embed all chunks
            vectors = embedder.embed_batch(
                all_chunks,
                deadline_at=deadline_at,
                client_max_attempts=(
                    1 if deadline_at is not None else None
                ),
            )
            dimension = len(vectors[0])

            # Build indexes
            self._dense_index = None
            self._sparse_index = None
            self._retriever = None
            self._ensure_indexes(dimension)
            self._dense_index.add_documents(vectors, all_chunks, all_meta)
            self._sparse_index.add_documents(all_chunks, all_meta)

            # Retrieve
            retriever = self._ensure_retriever(dimension)
            hybrid_results = retriever.retrieve(
                query,
                top_k=top_k * 3,
                deadline_at=deadline_at,
            )

            # Rerank
            if do_rerank and hybrid_results:
                ensure_deadline(deadline_at)
                reranker = self._ensure_reranker()
                reranked = reranker.rerank(query, hybrid_results, top_k=top_k)
                ensure_deadline(deadline_at)
                final_results = [
                    {
                        "text": r.text,
                        "score": r.score,
                        "doc_id": r.doc_id,
                        "metadata": r.metadata,
                        "original_rank": r.original_rank,
                    }
                    for r in reranked
                ]
            else:
                final_results = [
                    {
                        "text": r.text,
                        "score": r.score,
                        "doc_id": r.doc_id,
                        "metadata": r.metadata,
                    }
                    for r in hybrid_results[:top_k]
                ]

            elapsed = time.time() - start
            return self._success(
                results=final_results,
                retrieval_time_sec=elapsed,
                chunks_indexed=len(all_chunks),
                domain=domain,
            )

        except Exception as exc:
            error_type = type(exc).__name__
            logger.error("rag_retrieve failed error_type=%s", error_type)
            return self._error(
                f"rag_retrieve_failed:{error_type}",
                time.time() - start,
            )

    # ------------------------------------------------------------------ #
    # Existing index path
    # ------------------------------------------------------------------ #

    def _query_existing_index(
        self,
        query: str,
        index_path: str,
        top_k: int,
        domain: str,
        do_rerank: bool,
        start: float,
        *,
        deadline_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Query a pre-built index on disk."""
        from rag_engine.dense_index import DenseIndex
        from rag_engine.sparse_index import BM25Index

        path = Path(index_path)
        ensure_deadline(deadline_at)

        # Load dense index
        hnsw_path = path / "hnsw.index"
        docs_path = path / "documents.json"
        bm25_path = path / "bm25.pkl"

        if not hnsw_path.exists():
            return self._error(
                f"Dense index not found at {hnsw_path}",
                time.time() - start,
            )

        embedder = self._ensure_embedder()
        dim = embedder.dimension or 3072

        dense = DenseIndex(dimension=dim)
        dense.load(index_path)

        # Dense search
        query_vector = embedder.embed(
            query,
            deadline_at=deadline_at,
            client_max_attempts=(1 if deadline_at is not None else None),
        )
        dense_results = dense.search(query_vector, top_k=top_k * 3)

        # Sparse search (if BM25 index exists)
        sparse_results = []
        if bm25_path.exists():
            sparse = BM25Index()
            sparse.load(str(bm25_path))
            sparse_results = sparse.search(query, top_k=top_k * 3)

        # Fuse if both available, otherwise use dense only
        if sparse_results:
            from rag_engine.hybrid_retriever import HybridRetriever

            hybrid = HybridRetriever(
                dense_index=dense,
                sparse_index=sparse,
                embedder=embedder,
                rrf_k=60,
            )
            hybrid_results = hybrid.retrieve(
                query,
                top_k=top_k * 3,
                deadline_at=deadline_at,
            )
        else:
            # Convert dense results to a compatible format
            from rag_engine.hybrid_retriever import RetrievalResult

            hybrid_results = [
                RetrievalResult(
                    doc_id=r.doc_id,
                    score=r.score,
                    text=r.text,
                    metadata=r.metadata,
                )
                for r in dense_results
            ]

        # Rerank
        if do_rerank and hybrid_results:
            ensure_deadline(deadline_at)
            reranker = self._ensure_reranker()
            reranked = reranker.rerank(query, hybrid_results, top_k=top_k)
            ensure_deadline(deadline_at)
            final_results = [
                {
                    "text": r.text,
                    "score": r.score,
                    "doc_id": r.doc_id,
                    "metadata": r.metadata,
                    "original_rank": r.original_rank,
                }
                for r in reranked
            ]
        else:
            final_results = [
                {
                    "text": r.text,
                    "score": r.score,
                    "doc_id": r.doc_id,
                    "metadata": r.metadata,
                }
                for r in hybrid_results[:top_k]
            ]

        elapsed = time.time() - start
        return self._success(
            results=final_results,
            retrieval_time_sec=elapsed,
            index_path=index_path,
            domain=domain,
        )
