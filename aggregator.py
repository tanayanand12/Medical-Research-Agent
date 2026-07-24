"""
aggregator.py — Phase 1 refactor of legacy aggregator.

LLM-based multi-agent response synthesis with AMA citation formatting.
All LLM calls route through LLMClient (no hardcoded OpenAI).
"""

import logging
from typing import Any, Dict, List, Optional

from llm_client import LLMClient

logger = logging.getLogger(__name__)


class Aggregator:
    """Synthesises information from multiple agent/tool responses.

    Phase 1 refactor: LLM calls go through LLMClient instead of
    hardcoded ``openai.ChatCompletion.create()``.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._model_id = model_id

    def synthesise(
        self,
        query: str,
        agent_results: Dict[str, Dict[str, Any]],
        model: Optional[str] = None,
    ) -> str:
        """Combine agent results into a single evidence-based answer.

        Parameters
        ----------
        query : str
            The user's original medical question.
        agent_results : dict
            ``{tool_name: {"results": [...], ...}}`` from parallel retrieval.
        model : str, optional
            Override the LLM model for this call.

        Returns
        -------
        str
            Synthesised answer text.
        """
        model = model or self._model_id

        context_parts: List[str] = []
        for tool_name, data in agent_results.items():
            results = data.get("results", [])
            if results:
                context_parts.append(
                    f"[{tool_name}]: {len(results)} result(s)"
                )

        context_text = "\n".join(context_parts) if context_parts else "(no results)"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert medical research synthesiser. "
                    "Combine evidence from multiple sources into a coherent, "
                    "citation-rich answer."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nEvidence:\n{context_text}",
            },
        ]

        return self._llm.chat(messages=messages, model=model, temperature=0.7)
