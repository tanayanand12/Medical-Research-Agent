"""
High‑level smoke test: Vectorizer → FaissDB → retrieval.

Skipped if OPENAI_API_KEY missing, shows pytest.skip pattern :contentReference[oaicite:10]{index=10}.
"""

import os
import pytest
from core.vectorizer import Vectorizer
from core.faiss_db_manager import FaissVectorDB

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires real OpenAI key"
)
def test_end_to_end(sample_corpus):
    vec = Vectorizer()
    docs = vec.embed_corpus(sample_corpus)

    db = FaissVectorDB(dimension=vec.EMBED_DIM if hasattr(vec, 'EMBED_DIM') else 3072)
    db.add_documents(docs)

    q_vec = vec._openai_embed(["glioblastoma"])[0]
    hits, dists = db.similarity_search(q_vec, k=3)
    assert hits, "Should retrieve at least one chunk"
