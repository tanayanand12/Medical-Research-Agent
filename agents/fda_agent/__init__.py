"""
agents/fda_agent — FDA regulatory data sub-agent (Phase 7).

Exposes FDAAgentGraph as the primary entry point for the orchestrator.

Architecture
------------
6-node LangGraph subgraph::

    expand_query → fetch → chunk_and_index → retrieve → rerank → synthesise → END

Fetches live records from openFDA (drug/label, drug/event, device/recall,
food/enforcement), builds ephemeral BM25 + HNSW indexes on-the-fly, and
synthesises regulatory answers with inline FDA record ID citations.

Usage
-----
    from agents.fda_agent import FDAAgentGraph

    agent = FDAAgentGraph()
    output = agent.invoke("What are the adverse events for metformin?")
    print(output.answer)
    print(output.citations)  # ['safetyreportid:12345678', 'recall_number:Z-1234-2024']

    # With context overrides
    output = agent.invoke(
        "Recall history for heparin contamination",
        context={"top_k": 15, "max_records": 100},
    )
"""

from agents.fda_agent.graph import FDAAgentGraph, FDAAgentState

__all__ = ["FDAAgentGraph", "FDAAgentState"]