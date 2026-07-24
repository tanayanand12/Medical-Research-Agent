"""
fallback.py — Phase 1 refactor of legacy fallback mechanism.

Coherence evaluation and fallback regeneration.
All LLM calls route through LLMClient (no hardcoded OpenAI).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from llm_client import LLMClient

logger = logging.getLogger(__name__)


class FallbackMechanism:
    """Evaluate answer coherence and regenerate if needed.

    Phase 1 refactor: LLM calls go through LLMClient instead of
    hardcoded ``openai.chat.completions.create()``.
    """

    coherence_threshold: float = 0.6

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._model = model
        logger.info("FallbackMechanism initialised")

    def evaluate_coherence(
        self,
        query: str,
        answer: str,
        sources: Optional[List[str]] = None,
    ) -> Tuple[float, str]:
        """Score coherence of a synthesised answer.

        Parameters
        ----------
        query : str
            Original user query.
        answer : str
            Synthesised answer to evaluate.
        sources : list[str], optional
            Tool names that contributed evidence.

        Returns
        -------
        tuple[float, str]
            ``(coherence_score, explanation)`` where score is in ``[0, 1]``.
        """
        sources_text = ", ".join(sources) if sources else "none"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical research quality evaluator. "
                    "Score the coherence of the answer on a scale of 0.0 to 1.0. "
                    "Respond with JSON: {\"score\": float, \"explanation\": str}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n"
                    f"Sources used: {sources_text}\n"
                    f"Answer:\n{answer}\n\n"
                    "Rate the coherence, clinical accuracy, and grounding."
                ),
            },
        ]

        try:
            raw = self._llm.chat(
                messages=messages,
                model=self._model,
                temperature=0.1,
                max_tokens=256,
            )
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0.5))
            explanation = parsed.get("explanation", "")
            return score, explanation

        except Exception as exc:
            logger.warning(
                "Coherence evaluation failed, defaulting to 0.7: %s", exc,
            )
            return 0.7, f"Evaluation error (defaulting): {exc}"

    def regenerate(
        self,
        query: str,
        original_answer: str,
        reason: str,
    ) -> str:
        """Regenerate an answer after coherence failure.

        Parameters
        ----------
        query : str
            Original user query.
        original_answer : str
            The answer that failed coherence evaluation.
        reason : str
            Explanation of why the original answer was rejected.

        Returns
        -------
        str
            Regenerated answer text.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evidence-based medical research assistant. "
                    "The previous answer was judged incoherent. "
                    "Produce a clearer, more grounded response."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Previous answer (rejected — {reason}):\n{original_answer}\n\n"
                    "Please provide an improved, coherent answer."
                ),
            },
        ]

        return self._llm.chat(
            messages=messages,
            model=self._model,
            temperature=0.7,
            max_tokens=1000,
        )
