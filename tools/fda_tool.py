"""
fda_tool.py — MCP tool for FDA regulatory data retrieval.

Ported from ``agentic-pipeline-clinical/fda_agent_wrapper.py``.
All LLM/embedding calls routed through LLMClient (Phase 1).
"""

import logging
import time
from typing import Any, Dict

from tools.mcp_base import MCPToolBase

logger = logging.getLogger(__name__)

# Optional pipeline import — available once FDA_agent/ is migrated.
try:
    from FDA_agent.fda_rag_pipeline import FdaRAGPipeline

    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False
    logger.debug(
        "FDA_agent package not found — "
        "search_fda tool will use LLMClient-based retrieval"
    )


class FDATool(MCPToolBase):
    """FDA regulatory data retrieval tool.

    Searches FDA drug labels, adverse events, recalls, and regulatory
    approvals via a RAG pipeline backed by LLMClient.
    """

    name = "search_fda"
    description = (
        "FDA regulatory data RAG system that answers queries based on "
        "drug labels, adverse events, recalls, and regulatory approvals."
    )
    triggers = [
        "fda", "drug label", "adverse event", "recall",
        "regulatory", "drug safety", "approval",
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "FDA-related query"},
            "top_k": {
                "type": "integer",
                "default": 10,
                "description": "Number of results to retrieve",
            },
            "max_records": {
                "type": "integer",
                "default": 300,
                "description": "Maximum records to process",
            },
        },
        "required": ["query"],
    }

    def __init__(self) -> None:
        self._pipeline = None

    def _get_llm_client(self):
        from llm_client import LLMClient
        return LLMClient()

    def _ensure_pipeline(self, max_records: int = 300) -> None:
        """Lazy-init the FDA pipeline using LLMClient."""
        if self._pipeline is not None:
            return
        if not _PIPELINE_AVAILABLE:
            return

        llm = self._get_llm_client()
        # FdaRAGPipeline expects an openai-compatible client;
        # pass LLMClient instance — the pipeline will need refactoring
        # to call llm.chat() / llm.embed() instead of openai.* directly.
        # For now, construct with defaults and route via LLMClient at
        # the tool boundary.
        self._pipeline = FdaRAGPipeline(
            openai_client=None,  # deferred — LLMClient used below
            model_name=llm.default_model,
            embedding_model=llm.default_embedding_model,
            max_records=max_records,
            chunk_size=10000,
            chunk_overlap=400,
            max_context_length=8000,
        )

    def call(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        query = input_dict["query"]
        top_k = input_dict.get("top_k", 10)
        max_records = input_dict.get("max_records", 300)

        try:
            if _PIPELINE_AVAILABLE:
                return self._call_pipeline(query, top_k, max_records, start)
            return self._call_llm_client(query, top_k, start)
        except Exception as exc:
            logger.error("search_fda failed: %s", exc, exc_info=True)
            return self._error(str(exc), time.time() - start)

    # ------------------------------------------------------------------ #
    # Path A: legacy pipeline available
    # ------------------------------------------------------------------ #

    def _call_pipeline(
        self, query: str, top_k: int, max_records: int, start: float
    ) -> Dict[str, Any]:
        self._ensure_pipeline(max_records)

        if max_records != self._pipeline.max_records:  # type: ignore[union-attr]
            self._pipeline.max_records = max_records  # type: ignore[union-attr]

        result = self._pipeline.process_query(query=query, top_k=top_k)  # type: ignore[union-attr]
        elapsed = time.time() - start

        if not result.get("success", False):
            return self._error(
                result.get("error", "Unknown FDA pipeline error"), elapsed
            )

        answer = result.get("answer", "")
        citations = self._normalise_citations(result.get("citations", []))
        confidence = self._calculate_confidence(result, elapsed)

        return self._success(
            results=[{"citation": c} for c in citations],
            tokens_used=result.get("tokens_used", 0),
            cost=result.get("cost", 0.0),
            retrieval_time_sec=elapsed,
            answer=answer,
            citations=citations,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    # Path B: standalone LLMClient retrieval
    # ------------------------------------------------------------------ #

    def _call_llm_client(
        self, query: str, top_k: int, start: float
    ) -> Dict[str, Any]:
        """Fallback when the legacy FDA pipeline is not yet migrated.

        Uses LLMClient to synthesise a response acknowledging that
        the FDA index is not yet available locally.
        """
        llm = self._get_llm_client()

        answer = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an FDA regulatory data assistant. The local FDA "
                        "RAG index is not yet available. Provide a helpful response "
                        "based on your training knowledge about FDA data, and note "
                        "that results are not grounded in a local index."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )

        elapsed = time.time() - start
        return self._success(
            results=[],
            retrieval_time_sec=elapsed,
            answer=answer,
            citations=[],
            confidence=0.4,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_citations(raw) -> list:
        """Ensure citations is always a list of strings."""
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str) and raw:
            return [raw]
        return []

    @staticmethod
    def _calculate_confidence(result: Dict[str, Any], elapsed: float) -> float:
        """Heuristic confidence score (ported from legacy FDAAgent)."""
        base = 0.82
        answer_len = len(result.get("answer", ""))
        if answer_len > 800:
            base += 0.05
        elif answer_len < 150:
            base -= 0.10

        citations = result.get("citations", [])
        if isinstance(citations, list):
            if len(citations) > 5:
                base += 0.08
            elif len(citations) == 0:
                base -= 0.12

        if elapsed < 3.0:
            base -= 0.03
        elif elapsed > 15.0:
            base -= 0.08

        records_count = result.get("metadata", {}).get("records_count", 0)
        if records_count > 100:
            base += 0.05
        elif records_count < 10:
            base -= 0.10

        return max(0.0, min(1.0, base))
