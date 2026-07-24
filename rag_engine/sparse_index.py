"""
sparse_index.py — Phase 6: BM25 sparse retrieval index.

Uses ``rank_bm25.BM25Okapi`` for term-frequency-based document retrieval.
Documents are tokenised on whitespace + lowercased.  The index is built
in-memory and can be serialised/deserialised via pickle for persistence.
"""

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.debug("rank_bm25 not installed — BM25Index will be unavailable")


@dataclass
class SparseResult:
    """A single BM25 retrieval result."""

    doc_id: int
    score: float
    text: str
    metadata: Dict[str, Any]


class BM25Index:
    """In-memory BM25 sparse retrieval index.

    Parameters
    ----------
    tokenizer : callable, optional
        Custom tokeniser function ``str → list[str]``.
        Defaults to lowercased whitespace splitting with punctuation removal.
    """

    def __init__(self, tokenizer=None) -> None:
        if not _BM25_AVAILABLE:
            raise ImportError(
                "rank_bm25 is required for BM25Index. "
                "Install with: pip install rank_bm25"
            )
        self._tokenizer = tokenizer or self._default_tokenizer
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._tokenized_corpus: List[List[str]] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to the index (rebuilds BM25 from scratch).

        Parameters
        ----------
        documents : list[str]
            Document texts.
        metadata : list[dict], optional
            One metadata dict per document.  Defaults to empty dicts.
        """
        if metadata is None:
            metadata = [{} for _ in documents]
        if len(metadata) != len(documents):
            raise ValueError("metadata length must match documents length")

        self._documents.extend(documents)
        self._metadata.extend(metadata)

        self._tokenized_corpus = [self._tokenizer(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info("BM25Index: indexed %d documents", len(self._documents))

    def search(self, query: str, top_k: int = 10) -> List[SparseResult]:
        """Retrieve the top-k documents for *query*.

        Returns
        -------
        list[SparseResult]
            Sorted by descending BM25 score.
        """
        if self._bm25 is None or not self._documents:
            return []

        tokenized_query = self._tokenizer(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results: List[SparseResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append(
                SparseResult(
                    doc_id=idx,
                    score=float(scores[idx]),
                    text=self._documents[idx],
                    metadata=self._metadata[idx],
                )
            )
        return results

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Persist index to disk."""
        data = {
            "documents": self._documents,
            "metadata": self._metadata,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh)
        logger.info("BM25Index: saved %d docs to %s", len(self._documents), path)

    def load(self, path: str) -> None:
        """Load index from disk."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._documents = data["documents"]
        self._metadata = data["metadata"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25Index: loaded %d docs from %s", len(self._documents), path)

    # ------------------------------------------------------------------ #
    # Tokeniser
    # ------------------------------------------------------------------ #

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()
