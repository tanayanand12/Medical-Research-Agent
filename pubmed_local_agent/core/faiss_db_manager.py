"""
core.faiss_db_manager
~~~~~~~~~~~~~~~~~~~~~

Simple, exact‑recall vector store around FAISS IndexFlatL2.

Key features
------------
* Exact L2 search (IndexFlatL2) – no training, deterministic results [1].
* Saves index + pickle'd metadata to `pubmed_faiss_index/<db_name>.*`.
* Converts NumPy float32 rows to FAISS‑accepted layout automatically.
* Returns *distances*, not similarities, so callers can re‑rank if needed.

References
----------
[1] Faiss wiki – "Guidelines to choose an index", Flat indexes guarantee
    exact results.  :contentReference[oaicite:4]{index=4}
[2] Faiss tips – .write_index / .read_index for persistence. :contentReference[oaicite:5]{index=5}
[3] Faiss docs – vectors must be float32.  :contentReference[oaicite:6]{index=6}
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissVectorDB:
    """
    Minimal FAISS wrapper for exact L2 similarity search.

    Parameters
    ----------
    dimension : int
        Vector dimensionality (3072 for `text‑embedding‑3‑large`).
    """

    def __init__(self, dimension: int = 3072) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # exact search  :contentReference[oaicite:7]{index=7}
        self.metadata: List[Dict] = []            # parallels index order

    # ------------------------------------------------------------------ #
    #  CRUD operations                                                   #
    # ------------------------------------------------------------------ #
    def add_documents(self, docs: List[Dict]) -> bool:
        """
        Append a list of *chunk dicts* to the index.

        Each dict **must** include:
            • "embedding" : np.ndarray[float32, shape=(d,)]
            • "text"      : str
        Additional keys are stored as metadata.

        Returns
        -------
        bool
            True on success, False if no vectors were added.
        """
        if not docs:
            logger.warning("add_documents called with empty list")
            return False

        vecs = np.vstack([doc["embedding"] for doc in docs]).astype("float32")
        if vecs.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vecs.shape[1]}"
            )

        self.index.add(vecs)  # IndexFlatL2 adds sequentially
        for doc in docs:
            meta = {k: v for k, v in doc.items() if k != "embedding"}
            self.metadata.append(meta)

        logger.info("Added %d vectors to FAISS index (ntotal=%d)", len(docs), self.index.ntotal)
        return True

    # ------------------------------------------------------------------ #
    #  Similarity search                                                 #
    # ------------------------------------------------------------------ #
    def similarity_search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        metadata_filter: Optional[Dict] = None,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Exact k‑NN search (L2). For cosine, query + index vectors must be
        pre‑normalised [3].

        Parameters
        ----------
        query_vec : np.ndarray[float32, shape=(d,)]
        k : int
            Number of hits to return.
        metadata_filter : dict, optional
            Simple equality filter, e.g. {"paper_year": "2025"}.

        Returns
        -------
        docs : List[Dict]
            Metadata dicts (original fields + 'text').
        distances : List[float]
            Raw L2 distances from FAISS.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty.")
            return [], []

        query_vec = query_vec.astype("float32").reshape(1, -1)
        dists, idxs = self.index.search(query_vec, k * 4)  # over‑fetch for filter
        results: List[Dict] = []
        distances: List[float] = []

        for dist, idx in zip(dists[0], idxs[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if metadata_filter:
                if any(meta.get(k) != v for k, v in metadata_filter.items()):
                    continue
            results.append(meta)
            distances.append(float(dist))
            if len(results) >= k:
                break
        return results, distances

    # ------------------------------------------------------------------ #
    #  Persistence                                                       #
    # ------------------------------------------------------------------ #
    def save(self, db_name: str = "index") -> None:
        """
        Persist index + metadata to ./pubmed_faiss_index/<db_name>.*

        Files written
        -------------
        <db_name>.index      – FAISS binary index
        <db_name>.meta.pkl   – pickled metadata list
        """
        base = Path("pubmed_faiss_index")
        base.mkdir(exist_ok=True)
        idx_path = base / f"{db_name}.index"
        meta_path = base / f"{db_name}.meta.pkl"

        faiss.write_index(self.index, str(idx_path))            # :contentReference[oaicite:8]{index=8}
        with meta_path.open("wb") as f:
            pickle.dump(self.metadata, f)

        logger.info("Saved FAISS index to %s (meta → %s)", idx_path, meta_path)

    def load(self, db_name: str = "index") -> None:
        """
        Load index + metadata from ./pubmed_faiss_index/<db_name>.*

        Raises
        ------
        FileNotFoundError
            If either index or metadata file is missing.
        """
        base = Path("pubmed_faiss_index")
        idx_path = base / f"{db_name}.index"
        meta_path = base / f"{db_name}.meta.pkl"

        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"{db_name} not found in {base}")

        self.index = faiss.read_index(str(idx_path))            # :contentReference[oaicite:9]{index=9}
        with meta_path.open("rb") as f:
            self.metadata = pickle.load(f)

        self.dimension = self.index.d
        logger.info("Loaded FAISS index (%d vectors) from %s", self.index.ntotal, idx_path)


# ───────────────────────────────────────────────────────────────────── #
# Smoke‑test                                                           #
# ───────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":  # pragma: no cover
    import numpy.random as npr

    db = FaissVectorDB(dimension=8)
    fake_docs = [
        {
            "embedding": npr.rand(8).astype("float32"),
            "text": "Hello world",
            "paper_id": "0001",
            "paper_year": "2024",
        }
        for _ in range(10)
    ]
    db.add_documents(fake_docs)
    q = npr.rand(8).astype("float32")
    hits, dists = db.similarity_search(q, k=3)
    print("Top‑3:", [h["paper_id"] for h in hits], dists)

    db.save("demo")
    db2 = FaissVectorDB()
    db2.load("demo")
    hits2, _ = db2.similarity_search(q, k=3)
    print("Round‑trip OK:", hits2[0]["paper_id"] == hits[0]["paper_id"])
