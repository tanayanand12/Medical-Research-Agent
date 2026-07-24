"""
clinical_trials_tool.py — MCP tool for clinical trials data retrieval.

Ported from ``agentic-pipeline-clinical/clinical_trials_agent_wrapper.py``.
All LLM/embedding calls routed through LLMClient (Phase 1).
"""

import logging
import time
from typing import Any, Dict

from tools.mcp_base import MCPToolBase

logger = logging.getLogger(__name__)

# Optional pipeline import — available once clinical_trials_agent1/ is migrated.
try:
    from clinical_trials_agent1.clinical_trials_rag_pipeline import (
        ClinicalTrialsRAGPipeline,
    )

    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False
    logger.debug(
        "clinical_trials_agent1 package not found — "
        "search_clinical_trials tool will use LLMClient-based retrieval"
    )


class ClinicalTrialsTool(MCPToolBase):
    """Clinical trials data retrieval tool.

    Searches clinical trials data and research studies via a RAG
    pipeline backed by LLMClient.
    """

    name = "search_clinical_trials"
    description = (
        "Clinical trials RAG system that answers queries based on "
        "clinical trials data and research studies."
    )
    triggers = [
        "clinical trial", "trial", "phase", "randomized",
        "placebo", "endpoint", "enrollment", "NCT",
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Clinical trials query"},
            "top_k": {
                "type": "integer",
                "default": 10,
                "description": "Number of results to retrieve",
            },
            "max_trials": {
                "type": "integer",
                "default": 25,
                "description": "Maximum trials to process",
            },
        },
        "required": ["query"],
    }

    def __init__(self) -> None:
        self._pipeline = None

    def _get_llm_client(self):
        from llm_client import LLMClient
        return LLMClient()

    def _ensure_pipeline(self, max_trials: int = 25) -> None:
        """Lazy-init the clinical-trials pipeline using LLMClient."""
        if self._pipeline is not None:
            return
        if not _PIPELINE_AVAILABLE:
            return

        llm = self._get_llm_client()
        self._pipeline = ClinicalTrialsRAGPipeline(
            openai_client=None,  # deferred — LLMClient used at tool boundary
            model_name=llm.default_model,
            embedding_model=llm.default_embedding_model,
            max_trials=max_trials,
            max_context_length=8000,
            chunk_size=1000,
            chunk_overlap=200,
        )

    def call(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        query = input_dict["query"]
        top_k = input_dict.get("top_k", 10)
        max_trials = input_dict.get("max_trials", 25)

        try:
            if _PIPELINE_AVAILABLE:
                return self._call_pipeline(query, top_k, max_trials, start)
            return self._call_llm_client(query, top_k, start)
        except Exception as exc:
            logger.error("search_clinical_trials failed: %s", exc, exc_info=True)
            return self._error(str(exc), time.time() - start)

    # ------------------------------------------------------------------ #
    # Path A: legacy pipeline available
    # ------------------------------------------------------------------ #

    def _call_pipeline(
        self, query: str, top_k: int, max_trials: int, start: float
    ) -> Dict[str, Any]:
        self._ensure_pipeline(max_trials)

        if max_trials != self._pipeline.max_trials:  # type: ignore[union-attr]
            self._pipeline.max_trials = max_trials  # type: ignore[union-attr]

        result = self._pipeline.process_query(query=query, top_k=top_k)  # type: ignore[union-attr]
        elapsed = time.time() - start

        if "error" in result:
            return self._error(result["error"], elapsed)

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
        """Fallback when the legacy clinical-trials pipeline is not migrated."""
        llm = self._get_llm_client()

        answer = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical trials research assistant. The local "
                        "clinical-trials RAG index is not yet available. Provide a "
                        "helpful response based on your training knowledge about "
                        "clinical trials, and note that results are not grounded "
                        "in a local index."
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
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str) and raw:
            return [raw]
        return []

    @staticmethod
    def _calculate_confidence(result: Dict[str, Any], elapsed: float) -> float:
        """Heuristic confidence (ported from legacy ClinicalTrialsAgent)."""
        base = 0.80
        answer_len = len(result.get("answer", ""))
        if answer_len > 500:
            base += 0.05
        elif answer_len < 100:
            base -= 0.10

        citations = result.get("citations", [])
        if isinstance(citations, list):
            if len(citations) > 3:
                base += 0.10
            elif len(citations) == 0:
                base -= 0.15

        if elapsed < 2.0:
            base -= 0.05
        elif elapsed > 10.0:
            base -= 0.10

        return max(0.0, min(1.0, base))
