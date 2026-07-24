"""
embedder.py — Phase 6: LLM-agnostic dense embedding wrapper.

All embedding calls route through :class:`LLMClient` (Phase 1), making the
embedding model swappable via ``models.yaml`` or the ``DEFAULT_EMBEDDING_MODEL``
env var.  No direct ``openai`` or provider-specific imports.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class Embedder:
    """LLM-agnostic embedding interface backed by LLMClient.

    Parameters
    ----------
    model : str, optional
        Embedding model name (e.g. ``"text-embedding-3-large"``).
        When *None* the LLMClient default is used.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model
        self._llm = None  # lazy-loaded

    @property
    def _client(self):
        """Lazy-load LLMClient to avoid import-time side effects."""
        if self._llm is None:
            from llm_client import LLMClient
            self._llm = LLMClient()
        return self._llm

    @property
    def model_name(self) -> str:
        return self._model or self._client.default_embedding_model

    @property
    def dimension(self) -> int:
        """Return the expected embedding dimension from models.yaml config.

        Falls back to 0 if not configured (the actual dimension is known
        after the first ``embed`` call).
        """
        info = self._client._embeddings.get(self.model_name, {})
        return info.get("model_info", {}).get("dimension", 0)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def embed(self, text: str) -> List[float]:
        """Embed a single text string.

        Returns
        -------
        list[float]
            Embedding vector.
        """
        return self._client.embed(text, model=self._model)

    # def embed_batch(self, texts: List[str]) -> List[List[float]]:
    #     """Embed a batch of texts.

    #     Currently calls :meth:`embed` in a loop.  A future optimisation
    #     could batch requests at the LiteLLM level.

    #     Returns
    #     -------
    #     list[list[float]]
    #         One embedding vector per input text.
    #     """
    #     vectors: List[List[float]] = []
    #     for text in texts:
    #         vectors.append(self.embed(text))
    #     return vectors

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts via a single batched LLMClient call.

        Delegates entirely to :meth:`LLMClient.embed_batch`, which handles
        provider-specific batching, Ollama fallback, cost tracking, and logging.
        ``batch_size`` is resolved inside LLMClient based on provider limits.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text, in input order.
        """
        if not texts:
            return []
        return self._client.embed_batch(texts, model=self._model)