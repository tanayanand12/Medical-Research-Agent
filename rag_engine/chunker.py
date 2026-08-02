"""
chunker.py — Phase 6: Semantic chunking with recursive fallback.

Two strategies:
1. **Semantic chunking** — embed adjacent sentences and split where cosine
   similarity drops below a threshold (embedding boundary detection).
2. **Recursive fallback** — split by paragraphs → sentences → fixed-size
   token windows when embeddings are unavailable or the corpus is small.

All embedding calls route through :class:`Embedder` (LLM-agnostic).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from evaluation_core import RuntimeDeadlineExceeded, ensure_deadline

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    """Hybrid semantic + recursive text chunker.

    Parameters
    ----------
    embedder : Embedder, optional
        Used for semantic boundary detection.  When *None* the chunker
        falls back to the recursive strategy automatically.
    similarity_threshold : float
        Cosine-similarity threshold below which a boundary is placed
        between adjacent sentences (default 0.5).
    max_chunk_tokens : int
        Soft ceiling on chunk size in whitespace-delimited tokens
        (default 512).
    min_chunk_tokens : int
        Minimum chunk size — very short chunks are merged with their
        neighbour (default 30).
    """

    def __init__(
        self,
        embedder=None,
        similarity_threshold: float = 0.5,
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 30,
    ) -> None:
        self._embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def chunk(
        self,
        text: str,
        *,
        deadline_at: Optional[float] = None,
        client_max_attempts: Optional[int] = None,
    ) -> List[Chunk]:
        """Split *text* into chunks.

        Tries semantic chunking first (if an embedder is available),
        then falls back to recursive splitting.
        """
        if not text or not text.strip():
            return []
        ensure_deadline(deadline_at)

        if self._embedder is not None:
            try:
                chunks = self._semantic_chunk(
                    text,
                    deadline_at=deadline_at,
                    client_max_attempts=client_max_attempts,
                )
                if chunks:
                    return chunks
            except RuntimeDeadlineExceeded:
                raise
            except Exception:
                logger.warning(
                    "Semantic chunking failed — falling back to recursive",
                    exc_info=True,
                )

        return self._recursive_chunk(text)

    # ------------------------------------------------------------------ #
    # Semantic chunking
    # ------------------------------------------------------------------ #

    def _semantic_chunk(
        self,
        text: str,
        *,
        deadline_at: Optional[float] = None,
        client_max_attempts: Optional[int] = None,
    ) -> List[Chunk]:
        """Embed sentences, detect boundaries via cosine-similarity drops."""
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return [Chunk(text=text.strip(), index=0, start_char=0, end_char=len(text))]

        embeddings = self._embedder.embed_batch(
            [s for s in sentences],
            deadline_at=deadline_at,
            client_max_attempts=client_max_attempts,
        )
        if len(embeddings) < 2:
            return [Chunk(text=text.strip(), index=0, start_char=0, end_char=len(text))]

        # Cosine similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find boundary positions where similarity drops below threshold
        boundaries = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)
        boundaries.append(len(sentences))

        # Build chunks from boundary groups
        chunks: List[Chunk] = []
        char_offset = 0
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            chunk_text = " ".join(sentences[start:end]).strip()
            if not chunk_text:
                continue

            chunk_start = text.find(sentences[start], char_offset)
            if chunk_start == -1:
                chunk_start = char_offset
            chunk_end = chunk_start + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=chunk_start,
                    end_char=chunk_end,
                )
            )
            char_offset = chunk_end

        # Merge tiny chunks and split oversized ones
        chunks = self._enforce_size_limits(chunks)
        # Re-index
        for i, c in enumerate(chunks):
            c.index = i
        return chunks

    # ------------------------------------------------------------------ #
    # Recursive fallback
    # ------------------------------------------------------------------ #

    def _recursive_chunk(self, text: str) -> List[Chunk]:
        """Split by paragraphs → sentences → fixed-size token windows."""
        # Level 1: paragraphs
        paragraphs = self._split_paragraphs(text)
        pieces: List[str] = []
        for para in paragraphs:
            token_count = len(para.split())
            if token_count <= self.max_chunk_tokens:
                pieces.append(para)
            else:
                # Level 2: sentences
                sentences = self._split_sentences(para)
                buf: List[str] = []
                buf_tokens = 0
                for sent in sentences:
                    sent_tokens = len(sent.split())
                    if buf_tokens + sent_tokens > self.max_chunk_tokens and buf:
                        pieces.append(" ".join(buf))
                        buf = []
                        buf_tokens = 0
                    if sent_tokens > self.max_chunk_tokens:
                        # Level 3: fixed-size token windows
                        pieces.extend(self._fixed_window_split(sent))
                    else:
                        buf.append(sent)
                        buf_tokens += sent_tokens
                if buf:
                    pieces.append(" ".join(buf))

        # Build Chunk objects
        chunks: List[Chunk] = []
        char_offset = 0
        for idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            start = text.find(piece, char_offset)
            if start == -1:
                start = char_offset
            end = start + len(piece)
            chunks.append(Chunk(text=piece, index=idx, start_char=start, end_char=end))
            char_offset = end

        chunks = self._enforce_size_limits(chunks)
        for i, c in enumerate(chunks):
            c.index = i
        return chunks

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split on sentence-ending punctuation, keeping abbreviations intact."""
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in parts if s.strip()]

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split on double newlines."""
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]

    def _fixed_window_split(self, text: str) -> List[str]:
        """Split into fixed-size token windows with no overlap."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.max_chunk_tokens):
            chunk = " ".join(words[i : i + self.max_chunk_tokens])
            chunks.append(chunk)
        return chunks

    def _enforce_size_limits(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge tiny chunks, split oversized ones."""
        if not chunks:
            return chunks

        merged: List[Chunk] = []
        for c in chunks:
            token_count = len(c.text.split())
            if merged and token_count < self.min_chunk_tokens:
                # Merge with previous
                prev = merged[-1]
                prev.text = prev.text + " " + c.text
                prev.end_char = c.end_char
            elif token_count > self.max_chunk_tokens:
                # Split oversized
                windows = self._fixed_window_split(c.text)
                offset = c.start_char
                for w in windows:
                    end = offset + len(w)
                    merged.append(Chunk(text=w, start_char=offset, end_char=end))
                    offset = end
            else:
                merged.append(c)
        return merged

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        a_arr = np.asarray(a, dtype=np.float32)
        b_arr = np.asarray(b, dtype=np.float32)
        dot = np.dot(a_arr, b_arr)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm == 0:
            return 0.0
        return float(dot / norm)
