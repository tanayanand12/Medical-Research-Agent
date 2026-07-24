"""
reranker.py — Phase 6: Cross-encoder reranking via sentence-transformers.

Reranks hybrid retrieval results using a cross-encoder model that scores
each ``(query, document)`` pair directly, producing higher-precision
rankings than bi-encoder similarity alone.

The default model is ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — a
lightweight cross-encoder suitable for medical document reranking.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder # type: ignore

    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False
    logger.debug(
        "sentence_transformers not installed — Reranker will use score passthrough"
    )


@dataclass
class RerankResult:
    """A reranked retrieval result."""

    doc_id: int
    score: float
    text: str
    metadata: Dict[str, Any]
    original_rank: int


class Reranker:
    """Cross-encoder reranker for retrieval results.

    Parameters
    ----------
    model_name : str
        HuggingFace cross-encoder model name.
    device : str
        Torch device (``"cpu"`` or ``"cuda"``).
    batch_size : int
        Batch size for cross-encoder inference.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._device = device
        self._batch_size = batch_size
        self._model: Optional[Any] = None  # lazy-loaded

    def _ensure_model(self) -> None:
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return
        if not _CROSS_ENCODER_AVAILABLE:
            return
        self._model = CrossEncoder(self._model_name, device=self._device)
        logger.info("Reranker: loaded model %s on %s", self._model_name, self._device)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    # @timed("Reranker.rerank Node") 
    def rerank(
        self,
        query: str,
        results: List[Any],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank retrieval results using the cross-encoder.

        Parameters
        ----------
        query : str
            User query.
        results : list
            Retrieval results with ``.text``, ``.doc_id``, ``.metadata``
            attributes (e.g. :class:`RetrievalResult`).
        top_k : int, optional
            Number of results to return.  Defaults to all.

        Returns
        -------
        list[RerankResult]
            Sorted by descending cross-encoder score.
        """
        if not results:
            return []

        if top_k is None:
            top_k = len(results)

        if _CROSS_ENCODER_AVAILABLE:
            return self._cross_encoder_rerank(query, results, top_k)
        return self._passthrough_rerank(results, top_k)

    # ------------------------------------------------------------------ #
    # Cross-encoder path
    # ------------------------------------------------------------------ #

    def _cross_encoder_rerank(
        self,
        query: str,
        results: List[Any],
        top_k: int,
    ) -> List[RerankResult]:
        """Score (query, doc) pairs with the cross-encoder."""
        self._ensure_model()

        pairs = [[query, r.text] for r in results]
        scores = self._model.predict(pairs, batch_size=self._batch_size)

        scored = list(zip(range(len(results)), results, scores))
        scored.sort(key=lambda x: x[2], reverse=True)

        reranked: List[RerankResult] = []
        for orig_rank, result, score in scored[:top_k]:
            reranked.append(
                RerankResult(
                    doc_id=result.doc_id,
                    score=float(score),
                    text=result.text,
                    metadata=result.metadata,
                    original_rank=orig_rank + 1,
                )
            )
        return reranked

    # ------------------------------------------------------------------ #
    # Passthrough fallback (no cross-encoder installed)
    # ------------------------------------------------------------------ #

    def _passthrough_rerank(
        self,
        results: List[Any],
        top_k: int,
    ) -> List[RerankResult]:
        """Passthrough when sentence-transformers is not available.

        Preserves the original ordering and scores.
        """
        logger.warning(
            "Reranker: sentence_transformers not installed — "
            "using original scores as passthrough"
        )
        reranked: List[RerankResult] = []
        for rank, result in enumerate(results[:top_k]):
            reranked.append(
                RerankResult(
                    doc_id=result.doc_id,
                    score=result.score,
                    text=result.text,
                    metadata=result.metadata,
                    original_rank=rank + 1,
                )
            )
        return reranked
