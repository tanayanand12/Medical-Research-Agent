"""
Node 8: format_response — AMA citation formatting + disclaimers.

Formats final answer with:
1. AMA-style citations for all referenced papers
2. Clinical disclaimer
3. [FALLBACK] tag if fallback was triggered
4. Early-exit response if query was non-medical
"""

from agent_state import AgentState
from citation_formatter import format_citations_to_ama
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


CLINICAL_DISCLAIMER = """[DISCLAIMER: This response is AI-generated and not a substitute for professional medical advice.
Always consult with a qualified healthcare provider before making any medical decisions.
This information is for educational purposes only.]"""

FALLBACK_TAG = "\n\n[NOTE: This response was regenerated using fallback mechanism due to coherence concerns.]"

NON_MEDICAL_RESPONSE = """I cannot provide information on this topic as it falls outside the scope of medical research.
Please consult with appropriate professionals or resources for this question."""


def format_response(state: AgentState) -> AgentState:
    """
    Format final response for user delivery.

    Handles three cases:
    1. Non-medical query: Return rejection message
    2. Medical query with results: Format with citations and disclaimer
    3. Medical query with partial/no results: Return best-effort answer with warning

    Args:
        state: Current agent state

    Returns:
        Updated state with output_answer, output_citations, output_sources, etc.
    """
    trace_id = state.get("trace_id", "unknown")
    timestamp_end = datetime.utcnow()

    try:
        # Case 1: Non-medical query
        if not state.get("is_medical_query", False):
            state["output_answer"] = NON_MEDICAL_RESPONSE
            state["output_sources"] = []
            state["output_citations"] = []
            state["output_disclaimer"] = CLINICAL_DISCLAIMER
            state["timestamp_end"] = timestamp_end
            state["execution_time_sec"] = (
                timestamp_end - state["timestamp_start"]
            ).total_seconds()
            logger.info(
                f"[{trace_id}] format_response: Non-medical query. "
                f"Returning rejection response."
            )
            return state

        # Case 2 & 3: Medical query
        # Build answer
        final_answer = state["intermediate_answer"]

        # Add fallback tag if triggered
        if state.get("fallback_triggered", False):
            final_answer += FALLBACK_TAG

        # Prepend disclaimer
        final_answer = f"{CLINICAL_DISCLAIMER}\n\n{final_answer}"

        # Add partial response warning if applicable
        if state.get("is_partial_response", False):
            warning = (
                "\n\n[NOTE: Some data sources could not be retrieved. "
                "This answer may be incomplete.]"
            )
            final_answer += warning

        # Extract citations from retrieval results
        citations = []
        sources = state.get("intermediate_sources", [])

        for tool_name in sources:
            tool_results = state["retrieval_results"].get(tool_name, {})
            results = tool_results.get("results", [])

            for result in results[:5]:  # Use top 5 results per tool
                # Format as AMA citation
                try:
                    ama_citation = format_citations_to_ama(
                        {
                            "title": result.get("title"),
                            "authors": result.get("authors", []),
                            "year": result.get("year"),
                            "journal": result.get("journal", ""),
                            "volume": result.get("volume"),
                            "issue": result.get("issue"),
                            "pages": result.get("pages"),
                            "doi": result.get("doi"),
                            "pmid": result.get("pmid"),
                        }
                    )
                    if ama_citation:
                        citations.append(ama_citation)
                except Exception as cite_error:
                    logger.warning(
                        f"[{trace_id}] Failed to format citation: {str(cite_error)}"
                    )

        # Build citations section
        citations_section = ""
        if citations:
            citations_section = "\n\nReferences:\n" + "\n".join(
                [f"{i+1}. {c}" for i, c in enumerate(citations)]
            )
            final_answer += citations_section

        # Update state
        state["output_answer"] = final_answer
        state["output_sources"] = sources
        state["output_citations"] = citations
        state["output_disclaimer"] = CLINICAL_DISCLAIMER
        state["timestamp_end"] = timestamp_end

        # Calculate total execution time
        execution_time = (timestamp_end - state["timestamp_start"]).total_seconds()
        state["execution_time_sec"] = execution_time

        # Estimate cost (placeholder; will be updated by observability layer)
        state["cost_estimate"] = 0.0

        logger.info(
            f"[{trace_id}] format_response: Formatted answer with {len(citations)} citations. "
            f"Total time: {execution_time:.2f}s. Fallback: {state.get('fallback_triggered', False)}"
        )

    except Exception as e:
        logger.error(
            f"[{trace_id}] format_response: Failed to format response: {str(e)}",
            exc_info=True,
        )
        # Return best-effort answer even if formatting fails
        state["output_answer"] = (
            f"{CLINICAL_DISCLAIMER}\n\n"
            f"{state['intermediate_answer']}\n\n"
            f"[NOTE: Response formatting encountered an error.]"
        )
        state["output_sources"] = []
        state["output_citations"] = []
        state["output_disclaimer"] = CLINICAL_DISCLAIMER
        state["timestamp_end"] = timestamp_end
        state["execution_time_sec"] = (
            timestamp_end - state["timestamp_start"]
        ).total_seconds()
        state["error_occurred"] = True
        state["error_messages"].append(f"Response formatting error: {str(e)}")

    return state
