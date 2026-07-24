"""
query_classifier.py — Phase 1 port of legacy query_classifier.py.

Medical query classification module for filtering out non-medical research
queries before passing them to the agent pipeline.

All LLM calls route through LLMClient (no hardcoded OpenAI).
"""

import json
import logging
from typing import Dict, Optional, Tuple

from llm_client import LLMClient

logger = logging.getLogger(__name__)

# System prompt for classification — unchanged from legacy
CLASSIFICATION_SYSTEM_PROMPT = """
You are a query classifier specializing in distinguishing medical research questions from other domains.

TASK:
Determine if the input query is related to medical research including:
- Clinical studies and trials
- Medical treatments and interventions
- Disease mechanisms and pathology
- Drug development and pharmacology
- Medical devices and diagnostics
- Public health research
- Medical literature and publication analysis
- Health systems and policy research
- Medical education and training research
- Patient-centered outcomes research

IMPORTANT: Medical device queries (like TR Band, VasoStat, hemostasis techniques)
should be classified as medical research queries, even if they involve markets,
business aspects, or usage statistics.

Respond ONLY with a JSON-formatted classification:
{
    "is_medical_research": true/false,
    "confidence": 0.0-1.0,
    "domain": "medical_research" OR [specific non-medical domain],
    "reason": "Brief explanation of classification"
}
"""


class QueryClassifier:
    """Classifier for determining if a query is related to medical research.

    Uses LLMClient (LiteLLM-backed) instead of hardcoded OpenAI.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        """Initialise classifier.

        Parameters
        ----------
        llm_client : LLMClient, optional
            Shared LLMClient instance.  If *None*, creates/reuses the
            singleton.
        """
        self._llm = llm_client or LLMClient()

    # ---- Phase 4 interface ------------------------------------------------

    def classify_with_reason(self, query: str) -> Tuple[bool, float, str]:
        """Classify a query and return structured result.

        This is the interface expected by the LangGraph classify_intent node.

        Parameters
        ----------
        query : str
            The user query to classify.

        Returns
        -------
        tuple[bool, float, str]
            (is_medical, confidence, reason)
        """
        is_medical, details = self.is_medical_research_query(query)
        confidence = details.get("confidence", 0.5)
        reason = details.get("reason", "")
        return is_medical, confidence, reason

    # ---- legacy-compatible interface --------------------------------------

    def is_medical_research_query(self, query: str) -> Tuple[bool, Dict]:
        """Determine if a query is related to medical research.

        Parameters
        ----------
        query : str
            The user query to classify.

        Returns
        -------
        tuple[bool, dict]
            Boolean indicating if query is medical, and classification
            details dict.
        """
        user_message = (
            f"USER QUERY: {query}\n\n"
            "Carefully analyze whether this query is related to medical research.\n"
            "Consider questions about medical devices, procedures, or healthcare "
            "systems as medical research queries.\n\n"
            "Respond ONLY with a JSON-formatted classification."
        )

        try:
            raw = self._llm.chat(
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=256,
            )

            classification = self._parse_classification(raw)
            is_medical = classification.get("is_medical_research", True)

            logger.info(
                "Query classified: is_medical=%s, query=%s", is_medical, query
            )
            return is_medical, classification

        except Exception as e:
            logger.error(
                "Error in LLM classification: %s", str(e), exc_info=True
            )
            # Default to medical to avoid false negatives
            fallback = {
                "is_medical_research": True,
                "confidence": 0.99,
                "domain": "medical_research",
                "reason": f"Error in classification process: {str(e)}",
            }
            return True, fallback

    def get_non_medical_response(self, query: str, classification: Dict) -> Dict:
        """Generate a response for non-medical queries.

        Parameters
        ----------
        query : str
            The original query.
        classification : dict
            Classification details from ``is_medical_research_query``.

        Returns
        -------
        dict
            Response with a generic out-of-domain message.
        """
        domain = classification.get("domain", "non-medical")
        confidence = classification.get("confidence", 0.0)

        response = {
            "answer": (
                "I'm a medical research assistant specialized in answering "
                "questions about medical academic research, clinical studies, "
                "and scientific literature. "
                f"Your question appears to be about {domain}, which is outside "
                "my area of expertise. "
                "I can help with questions about medical research papers, "
                "clinical trials, treatment efficacy, disease mechanisms, "
                "drug development, and other topics in the medical research domain."
            ),
            "citations": [],
            "confidence": confidence,
            "classification": classification,
        }

        logger.info("Generated non-medical response for query: %s", query)
        return response

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _parse_classification(raw: str) -> Dict:
        """Parse JSON classification from LLM output.

        Falls back to a safe default if parsing fails.
        """
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error(
                "Failed to parse LLM classification response: %s", raw
            )
            return {
                "is_medical_research": True,
                "confidence": 0.5,
                "domain": "medical_research",
                "reason": "Classification parsing error, defaulting to medical research",
            }
