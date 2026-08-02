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
            error_type = type(exc).__name__
            logger.error(
                "search_clinical_trials failed error_type=%s", error_type
            )
            return self._error(
                f"search_clinical_trials_failed:{error_type}",
                time.time() - start,
            )

    # ------------------------------------------------------------------ #
    # Path A: legacy pipeline available
    # ------------------------------------------------------------------ #

    def _call_pipeline(
        self, query: str, top_k: int, max_trials: int, start: float
    ) -> Dict[str, Any]:
        # The legacy answer-only pipeline cannot expose its exact synthesis
        # context, so invoking it would perform unverifiable work.
        return self._error(
            "clinical_trials_pipeline_exact_context_unavailable",
            time.time() - start,
        )

    # ------------------------------------------------------------------ #
    # Path B: standalone LLMClient retrieval
    # ------------------------------------------------------------------ #

    def _call_llm_client(
        self, query: str, top_k: int, start: float
    ) -> Dict[str, Any]:
        """Fail closed when no evidence-producing backend is available."""
        return self._error(
            "clinical_trials_retrieval_backend_unavailable",
            time.time() - start,
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
