"""
Local Index sub-agent — Phase 7.

LangGraph subgraph that retrieves and synthesises evidence from local
medical document indexes (institutional papers, uploaded documents,
curated knowledge bases).  Replaces the legacy ``LocalAgent`` wrapper
in ``agentic-pipeline-clinical/local_agent_wrapper.py``.

Legacy behaviour ported
-----------------------
* Query a local FAISS index (or GCP-hosted index) by ``model_id``.
* Fixed confidence 0.8 when results are present.
* Default ``top_k = 5``.

Changes from legacy
-------------------
* Retrieval runs in-process via RAGTool (was HTTP proxy to Cloud Run).
* LLM calls routed through LLMClient (zero hardcoded OpenAI).
* Reranking with ncbi/MedCPT-Cross-Encoder (was absent).
* Query expansion via ``prompts/local/query_expansion.yaml``.
* Synthesis via ``prompts/local/synthesis.yaml``.
"""

from agents.base import SubAgentGraph


class LocalAgentGraph(SubAgentGraph):
    """Local index retrieval and synthesis sub-agent."""

    domain = "local"
    default_top_k = 5
    base_confidence = 0.80
    summary = (
        "Local index RAG sub-agent that retrieves and synthesises "
        "evidence from local medical document indexes using hybrid "
        "retrieval with MedCPT cross-encoder reranking."
    )
