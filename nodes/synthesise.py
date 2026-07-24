"""
Node 4: synthesise — LLM answer generation with context.

Aggregates retrieved results into a coherent answer using the LLM.
Reuses aggregator.py logic, but routes through LLMClient.
"""

import json
from agent_state import AgentState
from aggregator import Aggregator
from llm_client import LLMClient
import logging
import time

logger = logging.getLogger(__name__)


def synthesise(state: AgentState) -> AgentState:
    """
    Generate answer by synthesizing retrieved documents.

    Reuses legacy aggregator.py; now routes LLM calls through LLMClient.

    Args:
        state: Current agent state

    Returns:
        Updated state with intermediate_answer and synthesis metadata
    """
    trace_id = state.get("trace_id", "unknown")

    try:
        llm_client = LLMClient()

        # Extract model from context; default to LLMClient default
        model_id = state.get("context", {}).get("model_id")

        # Build context string from retrieved results
        context_sections = []
        sources_used = []

        for tool_name in state["discovered_skills"]:
            tool_results = state["retrieval_results"].get(tool_name, {})

            # Skip if tool errored
            if tool_results.get("error"):
                logger.debug(
                    f"[{trace_id}] Skipping {tool_name} (error): {tool_results['error']}"
                )
                continue

            results = tool_results.get("results", [])
            if not results:
                continue

            # Format top results for context
            formatted_results = []
            for result in results[:3]:  # Use top 3 from each tool
                formatted_results.append(
                    json.dumps(
                        {
                            "title": result.get("title"),
                            "authors": result.get("authors"),
                            "year": result.get("year"),
                            "abstract": result.get("abstract", "")[:200],  # Truncate
                            "doi": result.get("doi"),
                        },
                        indent=2,
                    )
                )

            if formatted_results:
                context_sections.append(
                    f"From {tool_name}:\n" + "\n".join(formatted_results)
                )
                sources_used.append(tool_name)

        # Build synthesis prompt
        context_text = "\n\n".join(context_sections) if context_sections else "(No results retrieved)"

        system_prompt = """You are an evidence-based medical research assistant.
Your task is to answer medical/clinical questions using the provided scientific evidence.

Guidelines:
1. Answer directly and clearly
2. Ground your answer in the provided evidence
3. Mention specific findings from the literature
4. If evidence is limited or conflicting, state this clearly
5. Do NOT make up information or cite sources not provided
6. Use professional medical terminology appropriately

You will synthesize multiple sources into a coherent, clinically-informed answer."""

        user_message = f"""Question: {state["input_query"]}

Available Evidence:
{context_text}

Please provide a comprehensive, evidence-based answer to the question."""

        # Call LLM through LLMClient
        start_time = time.time()

        answer = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=model_id,
            temperature=0.7,
            max_tokens=1000,
        )

        elapsed_time = time.time() - start_time

        # Update state
        state["intermediate_answer"] = answer
        state["intermediate_sources"] = sources_used
        state["intermediate_model_used"] = model_id or "default"
        state["synthesis_time_sec"] = elapsed_time

        # Note: Token counts will be tracked by LLMClient metrics
        # For now, we'll set placeholder values
        state["synthesis_tokens_in"] = 0
        state["synthesis_tokens_out"] = 0

        logger.info(
            f"[{trace_id}] synthesise: Generated answer using {len(sources_used)} sources "
            f"({', '.join(sources_used)}). Time: {elapsed_time:.2f}s. "
            f"Answer length: {len(answer)} chars."
        )

    except Exception as e:
        logger.error(
            f"[{trace_id}] synthesise: Failed to generate answer: {str(e)}",
            exc_info=True,
        )
        state["intermediate_answer"] = (
            f"I was unable to synthesize an answer due to an error: {str(e)} "
            f"Please review the evidence below and provide your own analysis."
        )
        state["intermediate_sources"] = []
        state["intermediate_model_used"] = "error"
        state["synthesis_time_sec"] = 0.0
        state["error_occurred"] = True
        state["error_messages"].append(f"Synthesis error: {str(e)}")

    return state
