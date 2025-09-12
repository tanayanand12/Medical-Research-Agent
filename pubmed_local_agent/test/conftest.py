"""
Pytest fixtures shared across the suite.

• tmp_path fixture is built‑in and provides an isolated filesystem sandbox
  (docs :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}).
• Monkeypatch OpenAI embeddings so tests run offline.
"""

import numpy as np
import pytest
from unittest.mock import patch

from core.vectorizer import EMBED_DIM


@pytest.fixture
def fake_embeddings():
    """Return deterministic vectors to replace the OpenAI API call."""
    def _rand_vec(texts):
        return [np.full(EMBED_DIM, hash(t) % 97, dtype=np.float32) for t in texts]
    # Tenacity retries will succeed immediately with mock :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
    with patch("core.vectorizer.Vectorizer._openai_embed", side_effect=_rand_vec):
        yield


@pytest.fixture
def sample_corpus():
    return {
        "PMID1": {
            "paper_title": "Sample Title",
            "paper_year": "2024",
            "content": "Glioblastoma example text " * 100,
        }
    }
