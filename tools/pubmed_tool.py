"""
pubmed_tool.py — MCP tool for PubMed local vector-search retrieval.

Ported from ``agentic-pipeline-clinical/pubmed_local_agent_wrapper.py``.
All LLM/embedding calls routed through LLMClient (Phase 1).
"""

import logging
import os
import time
from typing import Any, Dict, List

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

        try:
            if _PIPELINE_AVAILABLE:
                return self._call_pipeline(query, top_k, db_name, start)
            return self._call_llm_client(query, top_k, db_name, start)
        except Exception as exc:
            logger.error("search_pubmed failed: %s", exc, exc_info=True)
            return self._error(str(exc), time.time() - start)

    # ------------------------------------------------------------------ #
    # Path A: legacy pipeline available
    # ------------------------------------------------------------------ #

    def _call_pipeline(
        self, query: str, top_k: int, db_name: str, start: float
    ) -> Dict[str, Any]:
        self._ensure_pipeline(db_name)

        index_path = os.path.join("pubmed_faiss_index", f"{db_name}.index")
        if not os.path.exists(index_path):
            return self._error(
                f"PubMed index not found at: {index_path}",
                time.time() - start,
            )

        result = self._qa_engine.answer(query, top_k=top_k)  # type: ignore[union-attr]
        if "error" in result:
            return self._error(result["error"], time.time() - start)

        elapsed = time.time() - start
        return self._success(
            results=self._format_results(result),
            tokens_used=result.get("tokens_used", 0),
            cost=result.get("cost", 0.0),
            retrieval_time_sec=elapsed,
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            confidence=0.85,
        )

    # ------------------------------------------------------------------ #
    # Path B: standalone LLMClient-based retrieval
    # ------------------------------------------------------------------ #

    def _call_llm_client(
        self, query: str, top_k: int, db_name: str, start: float
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
            results = self._faiss_retrieve(llm, query, index_path, docs_path, top_k)
        else:
            # No local index — return empty results with explanation
            elapsed = time.time() - start
            return self._success(
                results=[],
                retrieval_time_sec=elapsed,
                answer="PubMed local index not available. Ensure FAISS index is built.",
                citations=[],
                confidence=0.0,
            )

        # Synthesise answer from retrieved documents
        context_text = "\n\n".join(
            f"[{i+1}] {doc.get('title', 'Untitled')}\n{doc.get('text', '')}"
            for i, doc in enumerate(results)
        )

        answer = llm.chat(
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
        )

        elapsed = time.time() - start
        return self._success(
            results=results,
            retrieval_time_sec=elapsed,
            answer=answer,
            citations=[doc.get("citation", "") for doc in results if doc.get("citation")],
            confidence=0.85 if results else 0.0,
        )

    def _faiss_retrieve(
        self,
        llm,
        query: str,
        index_path: str,
        docs_path: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k documents from FAISS using LLMClient embeddings."""
        import json
        try:
            import faiss  # type: ignore
        except ImportError:
            logger.warning("faiss not installed — cannot do local retrieval")
            return []

        query_vec = llm.embed(query)

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
