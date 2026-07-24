"""
agents — Phase 7: LangGraph sub-agent migration.

Each sub-agent is a self-contained LangGraph subgraph that retrieves
domain-specific evidence via the central RAG engine and synthesises
an answer through LLMClient.

Public API::

    from agents.pubmed_agent.graph import PubMedAgentGraph
    from agents.fda_agent.graph import FDAAgentGraph
    from agents.clinical_trials_agent.graph import ClinicalTrialsAgentGraph
    from agents.local_agent.graph import LocalAgentGraph
"""

from agents.base import AgentOutput, SubAgentGraph

__all__ = ["AgentOutput", "SubAgentGraph"]
