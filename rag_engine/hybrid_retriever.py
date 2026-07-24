"""
hybrid_retriever.py — Phase 6: Reciprocal Rank Fusion (RRF) of dense + sparse results.

Combines :class:`DenseIndex` (FAISS HNSW) and :class:`BM25Index` (BM25Okapi)
retrieval results using RRF with a default ``k=60``.

RRF formula::

    score(d) = Σ  1 / (k + rank_i(d))

where ``rank_i(d)`` is the rank of document *d* in result list *i*.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag_engine.dense_index import DenseIndex, DenseResult
from rag_engine.sparse_index import BM25Index, SparseResult
from rag_engine.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A fused retrieval result."""

    doc_id: int
    score: float
    text: str
    metadata: Dict[str, Any]
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


class HybridRetriever:
    """RRF-based hybrid retriever combining dense and sparse indexes.

    Parameters
    ----------
    dense_index : DenseIndex
        FAISS HNSW dense vector index.
    sparse_index : BM25Index
        BM25 sparse term-frequency index.
    embedder : Embedder
        Embedding interface for encoding queries.
    rrf_k : int
        RRF smoothing constant (default 60).
    dense_weight : float
        Relative weight for dense results in RRF (default 1.0).
    sparse_weight : float
        Relative weight for sparse results in RRF (default 1.0).
    """

    def __init__(
        self,
        dense_index: Optional[DenseIndex],
        sparse_index: BM25Index,
        embedder: Optional[Embedder],
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        self._dense = dense_index
        self._sparse = sparse_index
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents using RRF fusion of dense + sparse results.

        Parameters
        ----------
        query : str
            User query.
        top_k : int
            Number of final results to return.

        Returns
        -------
        list[RetrievalResult]
            Sorted by descending RRF score.
        """
        # Fetch more candidates than final top_k for better fusion
        fetch_k = top_k * 3

        # Dense retrieval degrades to sparse-only when embeddings are unavailable.
        dense_results: List[DenseResult] = []
        if self._dense is not None and self._embedder is not None:
            try:
                query_vector = self._embedder.embed(query)
                dense_results = self._dense.search(query_vector, top_k=fetch_k)
            except Exception as exc:
                logger.warning(
                    "Dense retrieval unavailable; continuing with BM25: %s", exc
                )

        # Sparse retrieval
        sparse_results = self._sparse.search(query, top_k=fetch_k)

        # RRF fusion
        fused = self._rrf_fuse(dense_results, sparse_results)

        # Sort by fused score descending
        fused.sort(key=lambda r: r.score, reverse=True)

        return fused[:top_k]

    # ------------------------------------------------------------------ #
    # RRF fusion
    # ------------------------------------------------------------------ #

    def _rrf_fuse(
        self,
        dense_results: List[DenseResult],
        sparse_results: List[SparseResult],
    ) -> List[RetrievalResult]:
        """Fuse dense and sparse result lists using Reciprocal Rank Fusion."""
        k = self._rrf_k

        # Map doc_id → accumulated score + metadata
        scores: Dict[int, float] = {}
        texts: Dict[int, str] = {}
        meta: Dict[int, Dict[str, Any]] = {}
        dense_ranks: Dict[int, int] = {}
        sparse_ranks: Dict[int, int] = {}

        # Dense contribution
        for rank, result in enumerate(dense_results):
            doc_id = result.doc_id
            rrf_score = self._dense_weight / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
            texts[doc_id] = result.text
            meta[doc_id] = result.metadata
            dense_ranks[doc_id] = rank + 1

        # Sparse contribution
        for rank, result in enumerate(sparse_results):
            doc_id = result.doc_id
            rrf_score = self._sparse_weight / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
            texts.setdefault(doc_id, result.text)
            meta.setdefault(doc_id, result.metadata)
            sparse_ranks[doc_id] = rank + 1

        # Build result list
        fused: List[RetrievalResult] = []
        for doc_id, score in scores.items():
            fused.append(
                RetrievalResult(
                    doc_id=doc_id,
                    score=score,
                    text=texts[doc_id],
                    metadata=meta[doc_id],
                    dense_rank=dense_ranks.get(doc_id),
                    sparse_rank=sparse_ranks.get(doc_id),
                )
            )

        return fused
