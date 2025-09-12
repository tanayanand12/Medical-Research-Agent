"""
Validate FaissVectorDB add → search → save → load cycle.

Uses the official FAISS IO helpers :contentReference[oaicite:8]{index=8}&#8203;:contentReference[oaicite:9]{index=9}.
"""

import numpy as np
import tempfile
from core.faiss_db_manager import FaissVectorDB

def test_faiss_roundtrip(tmp_path):
    db = FaissVectorDB(dimension=8)
    vecs = [np.random.rand(8).astype("float32") for _ in range(5)]
    docs = [{"embedding": v, "text": f"doc{i}"} for i, v in enumerate(vecs)]
    db.add_documents(docs)

    q = vecs[0]
    hits, dists = db.similarity_search(q, k=1)
    assert hits[0]["text"] == "doc0"

    # Save/load
    db_path = tmp_path / "demo"
    db.save(db_path.name)
    db2 = FaissVectorDB()
    db2.load(db_path.name)
    hits2, _ = db2.similarity_search(q, k=1)
    assert hits2[0]["text"] == "doc0"
