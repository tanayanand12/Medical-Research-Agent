"""
LangGraph StateGraph for Medical Research Agent.

Defines the complete orchestration graph with 8 nodes and conditional routing.

Nodes:
1. classify_intent — Medical domain filtering
2. discover_skills — Tool selection via semantic matching
3. parallel_retrieve — Concurrent MCP tool invocation
4. synthesise — LLM answer generation
5. score_confidence — Coverage-based confidence calculation
6. evaluate_coherence — Coherence scoring for fallback decision
7. fallback_regen — Fallback answer regeneration
8. format_response — AMA citation formatting + disclaimers

Edges (routing):
- classify_intent → [conditional] → discover_skills OR format_response
- discover_skills → parallel_retrieve
- parallel_retrieve → synthesise
- synthesise → score_confidence
- score_confidence → evaluate_coherence
- evaluate_coherence → [conditional] → fallback_regen OR format_response
- fallback_regen → format_response
- format_response → END
"""

from langgraph.graph import StateGraph, END
from agent_state import AgentState
from edges import after_classify_intent, after_evaluate_coherence

from nodes import (
    classify_intent,
    discover_skills,
    parallel_retrieve,
    synthesise,
    score_confidence,
    evaluate_coherence,
    fallback_regen,
    format_response,
)

import logging

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Build the complete LangGraph StateGraph for medical research agent.

    Returns:
        Compiled StateGraph ready for invocation
    """

    # Initialize StateGraph
    graph = StateGraph(AgentState)

    # Add all 8 nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("discover_skills", discover_skills)
    graph.add_node("parallel_retrieve", parallel_retrieve)
    graph.add_node("synthesise", synthesise)
    graph.add_node("score_confidence", score_confidence)
    graph.add_node("evaluate_coherence", evaluate_coherence)
    graph.add_node("fallback_regen", fallback_regen)
    graph.add_node("format_response", format_response)

    # Set entry point
    graph.set_entry_point("classify_intent")

    # Add edges: linear progression with two conditional branches

    # After classify_intent: branch on is_medical_query
    graph.add_conditional_edges(
        "classify_intent",
        after_classify_intent,
        {
            "discover_skills": "discover_skills",
            "format_response": "format_response",
        },
    )

    # Linear edges: discover_skills → parallel_retrieve → synthesise → score_confidence → evaluate_coherence
    graph.add_edge("discover_skills", "parallel_retrieve")
    graph.add_edge("parallel_retrieve", "synthesise")
    graph.add_edge("synthesise", "score_confidence")
    graph.add_edge("score_confidence", "evaluate_coherence")

    # After evaluate_coherence: branch on should_fallback
    graph.add_conditional_edges(
        "evaluate_coherence",
        after_evaluate_coherence,
        {
            "fallback_regen": "fallback_regen",
            "format_response": "format_response",
        },
    )

    # Linear edge: fallback_regen → format_response
    graph.add_edge("fallback_regen", "format_response")

    # Terminal edge: format_response → END
    graph.add_edge("format_response", END)

    # Compile graph
    compiled_graph = graph.compile()

    logger.info("LangGraph StateGraph compiled successfully")
    return compiled_graph


# Global graph instance (lazy-loaded)
_graph_instance = None


def get_graph() -> StateGraph:
    """
    Get or create the global graph instance.

    Returns:
        Compiled StateGraph
    """
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance


# Visualization helpers
def print_graph_ascii() -> str:
    """
    Print ASCII representation of the graph topology.

    Returns:
        ASCII art string
    """
    graph = get_graph()
    try:
        ascii_art = graph.get_graph().draw_ascii()
        print(ascii_art)
        return ascii_art
    except Exception as e:
        logger.error(f"Failed to print graph ASCII: {str(e)}")
        return "Graph visualization unavailable"


def get_graph_diagram() -> str:
    """
    Get Mermaid diagram definition of the graph.

    Returns:
        Mermaid diagram string
    """
    diagram = """
    graph TD
        A[classify_intent] -->|is_medical=true| B[discover_skills]
        A -->|is_medical=false| H[format_response]
        B --> C[parallel_retrieve]
        C --> D[synthesise]
        D --> E[score_confidence]
        E --> F[evaluate_coherence]
        F -->|should_fallback=true| G[fallback_regen]
        F -->|should_fallback=false| H
        G --> H
        H --> I[END]
    """
    return diagram
