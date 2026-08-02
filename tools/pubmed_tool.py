"""
pubmed_tool.py — MCP tool for PubMed local vector-search retrieval.

Ported from ``agentic-pipeline-clinical/pubmed_local_agent_wrapper.py``.
All LLM/embedding calls routed through LLMClient (Phase 1).
"""

import logging
import os
import time
from typing import Any, Dict, List

from evaluation_core import document_text, stable_document_id
from runtime_verification import build_attempt_event, call_llm_with_metadata
from tools.mcp_base import MCPToolBase

logger = logging.getLogger(__name__)

# Optional pipeline imports — available once pubmed_local_agent/ is migrated.
try:
    from pubmed_local_agent.core.vectorizer import Vectorizer
    from pubmed_local_agent.core.faiss_db_manager import FaissVectorDB
    from pubmed_local_agent.query import PubMedQAEngine

    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False
    logger.debug(
        "pubmed_local_agent package not found — "
        "search_pubmed tool will use LLMClient-based retrieval"
    )


class PubMedTool(MCPToolBase):
    """PubMed local vector-search retrieval tool.

    Searches a local FAISS index of PubMed papers using embedding
    similarity and synthesises an answer via LLMClient.
    """

    name = "search_pubmed"
    description = (
        "Medical research RAG system that answers queries based on "
        "PubMed academic papers using vector similarity search."
    )
    triggers = ["pubmed", "research paper", "medical literature", "journal", "study"]

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Medical research question"},
            "top_k": {
                "type": "integer",
                "default": 8,
                "description": "Number of documents to retrieve",
            },
            "db_name": {
                "type": "string",
                "default": "index",
                "description": "FAISS index name",
            },
        },
        "required": ["query"],
    }

    def __init__(self) -> None:
        self._qa_engine = None
        self._vector_db = None

    def _get_llm_client(self):
        from llm_client import LLMClient
        return LLMClient()

    def _ensure_pipeline(self, db_name: str) -> None:
        """Lazy-init the pipeline components."""
        if self._qa_engine is not None:
            return
        if not _PIPELINE_AVAILABLE:
            return
        self._vector_db = FaissVectorDB(dimension=3072)
        self._qa_engine = PubMedQAEngine()

    def call(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        query = input_dict["query"]
        top_k = input_dict.get("top_k", 8)
        db_name = input_dict.get("db_name", "index")
        deadline_at = input_dict.get("_runtime_deadline_at_monotonic")
        trace_id = str(input_dict.get("trace_id") or "unknown")
        attempt_id = str(
            input_dict.get("attempt_id") or f"{trace_id}:search_pubmed:1"
        )

        try:
            if _PIPELINE_AVAILABLE:
                return self._call_pipeline(query, top_k, db_name, start)
            return self._call_llm_client(
                query,
                top_k,
                db_name,
                start,
                deadline_at=deadline_at,
                trace_id=trace_id,
                attempt_id=attempt_id,
                parent_attempt_id=input_dict.get("parent_attempt_id"),
            )
        except Exception as exc:
            error_type = type(exc).__name__
            logger.error("search_pubmed failed error_type=%s", error_type)
            return self._error(
                f"search_pubmed_failed:{error_type}",
                time.time() - start,
            )

    # ------------------------------------------------------------------ #
    # Path A: legacy pipeline available
    # ------------------------------------------------------------------ #

    def _call_pipeline(
        self, query: str, top_k: int, db_name: str, start: float
    ) -> Dict[str, Any]:
        # The legacy answer-only contract cannot report the exact ordered
        # context supplied to generation. Do not run unbounded generation only
        # to discard its result as unverifiable.
        return self._error(
            "pubmed_pipeline_exact_context_unavailable",
            time.time() - start,
        )

    # ------------------------------------------------------------------ #
    # Path B: standalone LLMClient-based retrieval
    # ------------------------------------------------------------------ #

    def _call_llm_client(
        self,
        query: str,
        top_k: int,
        db_name: str,
        start: float,
        *,
        deadline_at: Any = None,
        trace_id: str = "unknown",
        attempt_id: str = "",
        parent_attempt_id: Any = None,
    ) -> Dict[str, Any]:
        """Fallback when the legacy pipeline modules are not yet migrated.

        Uses LLMClient to embed the query and synthesise an answer.
        Requires a FAISS index on disk.
        """
        llm = self._get_llm_client()

        # Attempt FAISS-based retrieval if index files exist
        index_path = os.path.join("pubmed_faiss_index", f"{db_name}.index")
        docs_path = os.path.join("pubmed_faiss_index", f"{db_name}.documents")

        if os.path.exists(index_path) and os.path.exists(docs_path):
            results = self._faiss_retrieve(
                llm,
                query,
                index_path,
                docs_path,
                top_k,
                deadline_at=deadline_at,
            )
        else:
            return self._error(
                "pubmed_retrieval_backend_unavailable",
                time.time() - start,
            )

        # Synthesise answer from retrieved documents
        context_text = "\n\n".join(
            f"[{i+1}] {doc.get('title', 'Untitled')}\n{doc.get('text', '')}"
            for i, doc in enumerate(results)
        )

        call_result = call_llm_with_metadata(
            llm,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical research assistant. Answer the question "
                        "using ONLY the provided PubMed sources. Cite sources with "
                        "bracketed numbers [1], [2], etc. If the sources do not "
                        "contain relevant information, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Sources:\n{context_text}\n\nQuestion: {query}",
                },
            ],
            temperature=0.3,
            timeout=(
                max(0.1, float(deadline_at) - time.monotonic())
                if deadline_at is not None
                else 300
            ),
            client_max_attempts=1 if deadline_at is not None else None,
            deadline_at=deadline_at,
            _telemetry_stage="mcp_fallback_synthesis",
            _telemetry_attempt_id=attempt_id,
            _telemetry_parent_attempt_id=str(parent_attempt_id or ""),
            _telemetry_repair_status=(
                "retrieval_retry" if parent_attempt_id else "initial"
            ),
        )
        answer = call_result.text
        synthesis_context = [
            {
                "document_id": stable_document_id(document, "pubmed", rank),
                "text": document_text(document),
                "start_char": 0,
                "original_length": len(document_text(document)),
                "truncated": False,
                "citation_marker": rank,
            }
            for rank, document in enumerate(results, 1)
        ]
        event = build_attempt_event(
            trace_id=trace_id,
            attempt_id=attempt_id,
            parent_attempt_id=parent_attempt_id,
            stage="mcp_fallback_synthesis",
            component=self.name,
            status=str(call_result.status or "success"),
            repair_status=(
                "retrieval_retry" if parent_attempt_id else "initial"
            ),
            model=call_result.model,
            model_revision=call_result.model_revision,
            tokens_in=call_result.tokens_in,
            tokens_out=call_result.tokens_out,
            cost_usd=call_result.cost_usd,
            latency_sec=call_result.latency_sec,
            finish_reason=call_result.finish_reason,
            error_type=call_result.error_type,
            provider_metadata=call_result.provider_metadata,
        )

        elapsed = time.time() - start
        return self._success(
            results=results,
            tokens_used=call_result.tokens_in + call_result.tokens_out,
            cost=call_result.cost_usd,
            retrieval_time_sec=elapsed,
            answer=answer,
            citations=[doc.get("citation", "") for doc in results if doc.get("citation")],
            confidence=0.85 if results else 0.0,
            synthesis_context=synthesis_context,
            model_used=(
                f"{call_result.model}@{call_result.model_revision}"
                if call_result.model_revision
                else call_result.model
            ),
            attempt_events=[event],
        )

    def _faiss_retrieve(
        self,
        llm,
        query: str,
        index_path: str,
        docs_path: str,
        top_k: int,
        *,
        deadline_at: Any = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k documents from FAISS using LLMClient embeddings."""
        import json
        try:
            import faiss  # type: ignore
        except ImportError:
            logger.warning("faiss not installed — cannot do local retrieval")
            return []

        query_vec = llm.embed(
            query,
            deadline_at=deadline_at,
            client_max_attempts=1 if deadline_at is not None else None,
        )

        import numpy as np
        index = faiss.read_index(index_path)
        query_array = np.array([query_vec], dtype=np.float32)
        distances, indices = index.search(query_array, top_k)

        with open(docs_path, "r", encoding="utf-8") as fh:
            documents = json.load(fh)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(documents):
                continue
            doc = documents[idx]
            results.append({
                "title": doc.get("title", ""),
                "text": doc.get("text", doc.get("content", "")),
                "authors": doc.get("authors", []),
                "year": doc.get("year", ""),
                "doi": doc.get("doi", ""),
                "pmid": doc.get("pmid", ""),
                "citation": doc.get("citation", ""),
                "score": float(dist),
            })
        return results

    @staticmethod
    def _format_results(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalise legacy pipeline output to list of result dicts."""
        citations = result.get("citations", [])
        if isinstance(citations, list):
            return [{"citation": c} for c in citations]
        return []
