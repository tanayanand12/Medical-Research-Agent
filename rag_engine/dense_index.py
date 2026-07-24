"""
dense_index.py — Phase 6: FAISS HNSW dense vector index.

Default parameters: ``IndexHNSWFlat`` with **M=16**, **ef_construction=200**.
All parameters are configurable at construction time or via ``models.yaml``
(the ``hnsw_index`` section).

The index stores dense embeddings produced by :class:`Embedder`
and supports nearest-neighbour search.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import-untyped]

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.debug("faiss not installed — DenseIndex will be unavailable")


def _load_hnsw_config() -> Dict[str, Any]:
    """Load HNSW parameters from models.yaml if available."""
    try:
        import yaml

        config_path = Path(__file__).parent.parent / "models.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            return raw.get("hnsw_index", {})
    except Exception:
        pass
    return {}


@dataclass
class DenseResult:
    """A single dense retrieval result."""

    doc_id: int
    score: float
    text: str
    metadata: Dict[str, Any]


class DenseIndex:
    """FAISS HNSW dense vector index.

    Parameters
    ----------
    dimension : int
        Embedding vector dimension (e.g. 3072 for ``text-embedding-3-large``).
    M : int
        HNSW graph degree — number of bi-directional links per node.
        Higher values give better recall at the cost of memory.
    ef_construction : int
        Size of the dynamic candidate list during index construction.
        Higher values give better recall at the cost of build time.
    ef_search : int
        Size of the dynamic candidate list during search.
        Higher values give better recall at the cost of query latency.
    """

    def __init__(
        self,
        dimension: int,
        M: Optional[int] = None,
        ef_construction: Optional[int] = None,
        ef_search: Optional[int] = None,
    ) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for DenseIndex. "
                "Install with: pip install faiss-cpu"
            )

        # Load defaults from models.yaml, then override with explicit args
        cfg = _load_hnsw_config()
        self._dimension = dimension
        self._M = M or cfg.get("M", 16)
        self._ef_construction = ef_construction or cfg.get("ef_construction", 200)
        self._ef_search = ef_search or cfg.get("ef_search", 64)

        self._index = faiss.IndexHNSWFlat(self._dimension, self._M)
        self._index.hnsw.efConstruction = self._ef_construction
        self._index.hnsw.efSearch = self._ef_search

        self._documents: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

        logger.info(
            "DenseIndex: dim=%d M=%d ef_construction=%d ef_search=%d",
            self._dimension,
            self._M,
            self._ef_construction,
            self._ef_search,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_documents(
        self,
        vectors: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add document vectors to the HNSW index.

        Parameters
        ----------
        vectors : list[list[float]]
            Dense embedding vectors, one per document.
        documents : list[str]
            Raw document texts (stored for result retrieval).
        metadata : list[dict], optional
            One metadata dict per document.
        """
        if metadata is None:
            metadata = [{} for _ in documents]
        if len(vectors) != len(documents):
            raise ValueError("vectors length must match documents length")
        if len(metadata) != len(documents):
            raise ValueError("metadata length must match documents length")

        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._dimension:
            raise ValueError(
                f"Expected vectors of dimension {self._dimension}, "
                f"got shape {arr.shape}"
            )

        self._index.add(arr)
        self._documents.extend(documents)
        self._metadata.extend(metadata)

        logger.info(
            "DenseIndex: added %d vectors (total: %d)",
            len(vectors),
            self._index.ntotal,
        )

    def search(self, query_vector: List[float], top_k: int = 10) -> List[DenseResult]:
        """Find the top-k nearest neighbours for *query_vector*.

        Parameters
        ----------
        query_vector : list[float]
            Query embedding vector.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[DenseResult]
            Sorted by ascending L2 distance (lower = more similar).
        """
        if self._index.ntotal == 0:
            return []

        top_k = min(top_k, self._index.ntotal)
        query_arr = np.array([query_vector], dtype=np.float32)
        distances, indices = self._index.search(query_arr, top_k)

        results: List[DenseResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            results.append(
                DenseResult(
                    doc_id=int(idx),
                    score=float(dist),
                    text=self._documents[idx],
                    metadata=self._metadata[idx],
                )
            )
        return results

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        return self._dimension

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, directory: str) -> None:
        """Save FAISS index and document store to *directory*."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "hnsw.index"))
        with open(path / "documents.json", "w", encoding="utf-8") as fh:
            json.dump(
                {"documents": self._documents, "metadata": self._metadata},
                fh,
                ensure_ascii=False,
            )
        logger.info("DenseIndex: saved to %s (%d vectors)", directory, self.size)

    def load(self, directory: str) -> None:
        """Load FAISS index and document store from *directory*."""
        path = Path(directory)

        self._index = faiss.read_index(str(path / "hnsw.index"))
        with open(path / "documents.json", "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._documents = data["documents"]
        self._metadata = data.get("metadata", [{} for _ in self._documents])
        logger.info("DenseIndex: loaded from %s (%d vectors)", directory, self.size)
