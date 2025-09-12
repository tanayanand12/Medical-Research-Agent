"""
core.vectorizer
~~~~~~~~~~~~~~~

• Splits long paper texts into ~700‑token chunks with 50‑token overlap.
• Embeds chunks using OpenAI `text-embedding-3-large` (3 072‑dim) [1].
• Returns a list of dicts compatible with FaissDB.add_documents().

References
----------
[1] OpenAI, "New embedding models and API updates" 2024‑12‑18.  
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
#  Constants                                                            #
# --------------------------------------------------------------------- #
EMBED_MODEL = "text-embedding-3-large"            # OpenAI doc :contentReference[oaicite:4]{index=4}
EMBED_DIM = 3072
MAX_TOKENS = 8192                                 # hard model limit :contentReference[oaicite:5]{index=5}
CHUNK_SIZE = 5000                                 # from RAG benchmarks :contentReference[oaicite:6]{index=6}
CHUNK_OVERLAP = CHUNK_SIZE // 10                  # 10% overlap (500 tokens)
BATCH_SIZE = 16


class Vectorizer:
    """
    Token‑aware chunker + batch embedder suitable for FAISS ingestion.
    """

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY missing. Check your .env.")
        self.client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # fast default :contentReference[oaicite:7]{index=7}
        logger.info("Vectorizer initialised with %s", EMBED_MODEL)

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #
    def embed_corpus(self, papers: Dict[str, Dict]) -> List[Dict]:
        """
        Convert each paper’s `content` into multiple embedded chunks.

        Returns
        -------
        List[Dict]
            Each dict → {'embedding': np.ndarray, 'text': str, **metadata}
        """
        chunk_records: List[Dict] = []

        for pmid, paper in papers.items():
            text_chunks = self._chunk_text(paper["content"])
            meta_base = {
                "paper_id": pmid,
                "paper_title": paper["paper_title"],
                "paper_year": paper["paper_year"],
            }
            for i, chunk in enumerate(text_chunks):
                chunk_records.append(
                    {
                        "chunk_id": f"{pmid}_{i}",
                        "text": chunk,
                        **meta_base,
                    }
                )

        logger.info("Total chunks to embed: %d", len(chunk_records))
        embeddings = self._batched_embeddings([c["text"] for c in chunk_records])

        # Attach vectors
        for rec, vec in zip(chunk_records, embeddings):
            rec["embedding"] = vec.astype("float32")

        return chunk_records

    # ------------------------------------------------------------------ #
    #  Token‑aware chunker                                               #
    # ------------------------------------------------------------------ #
    def _chunk_text(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)
        n = len(tokens)
        if n <= CHUNK_SIZE:
            return [text]

        chunks = []
        start = 0
        while start < n:
            end = min(start + CHUNK_SIZE, n)
            chunk_tokens = tokens[start:end]
            chunks.append(self.encoding.decode(chunk_tokens))
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ------------------------------------------------------------------ #
    #  Batched embeddings with retry                                     #
    # ------------------------------------------------------------------ #
    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(
            (APIError, APITimeoutError, RateLimitError, APIConnectionError, BadRequestError)
        ),
    )
    def _openai_embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        """
        Single OpenAI embeddings.create call; wrapped by @retry for robustness.
        """
        resp = self.client.embeddings.create(input=list(texts), model=EMBED_MODEL)
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

    def _batched_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            # Ensure token count ≤ MAX_TOKENS
            safe_batch = [t if self._tokens(t) < MAX_TOKENS else t[:4096] for t in batch]
            vecs = self._openai_embed(safe_batch)
            embeddings.extend(vecs)
            if i + BATCH_SIZE < len(texts):
                time.sleep(0.4)  # gentle pacing
        return embeddings

    # ------------------------------------------------------------------ #
    #  Utility                                                           #
    # ------------------------------------------------------------------ #
    def _tokens(self, text: str) -> int:
        """Quick token count using tiktoken :contentReference[oaicite:8]{index=8}."""
        return len(self.encoding.encode(text))


# ──────────────────────────────────────────────────────────────────────
# Smoke‑test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    sample_paper = {
        "paper_title": "Glioblastoma Immunotherapy Overview",
        "paper_year": "2024",
        "content": "Glioblastoma is a malignant brain tumour. " * 400,  # ~4k tokens
    }
    vec = Vectorizer()
    docs = vec.embed_corpus({"PMID0001": sample_paper})
    print(docs[0].keys(), len(docs))
