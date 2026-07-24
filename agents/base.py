"""
base.py — Phase 7: Sub-agent base class and shared types.

Provides the 4-node LangGraph subgraph pattern shared by all
domain-specific agents:

    expand_query → retrieve → rerank → synthesise

All domain agents (PubMed, FDA, Clinical Trials, Local) inherit from
:class:`SubAgentGraph` and override configuration attributes.

Retrieval backbone: ``rag_engine.RAGTool`` (``rag_retrieve``).
Reranking model: ``ncbi/MedCPT-Cross-Encoder``.
LLM calls: routed through ``LLMClient`` (zero hardcoded OpenAI).
Prompts: loaded from ``prompts/<domain>/`` YAML templates.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import yaml # type: ignore
from langgraph.graph import END, StateGraph # type: ignore

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


# agents/base.py — add at module level, outside any class
_RERANKER_INSTANCE = None
_RERANKER_LOCK = threading.Lock()

def get_reranker_singleton(model_name: str = "ncbi/MedCPT-Cross-Encoder"):
    """Load MedCPT cross-encoder once per process, reuse across all agents.
    
    The CrossEncoder takes ~2-3s to load from disk on first call.
    All subsequent calls return the cached instance immediately.
    """
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        with _RERANKER_LOCK:
            if _RERANKER_INSTANCE is None:
                from sentence_transformers import CrossEncoder
                import logging
                logging.getLogger("agents.base").info(
                    "Loading MedCPT CrossEncoder (once per process)..."
                )
                _RERANKER_INSTANCE = CrossEncoder(model_name)
    return _RERANKER_INSTANCE


def serialized_invoke(method):
    """Serialize calls that mutate an agent's ephemeral retrieval indexes."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._invoke_lock:
            return method(self, *args, **kwargs)

    return wrapped

# ====================================================================== #
# State
# ====================================================================== #


class SubAgentState(TypedDict):
    """State contract for the 4-node sub-agent subgraph."""

    # ---- Input ----
    input_query: str
    context: Dict[str, Any]
    domain: str

    # ---- Query expansion ----
    expanded_query: str

    # ---- Retrieval ----
    retrieval_results: List[Dict[str, Any]]
    retrieval_time_sec: float

    # ---- Reranking ----
    reranked_results: List[Dict[str, Any]]

    # ---- Synthesis ----
    answer: str
    citations: List[str]
    confidence: float
    model_used: str

    # ---- Metadata ----
    error: Optional[str]
    execution_time_sec: float


# ====================================================================== #
# Output
# ====================================================================== #


@dataclass
class AgentOutput:
    """Standardised output returned by :meth:`SubAgentGraph.invoke`."""

    answer: str
    citations: List[str]
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    model_used: str = ""
    domain: str = ""
    execution_time_sec: float = 0.0
    error: Optional[str] = None


# ====================================================================== #
# Retrieval-doc adapter
# ====================================================================== #


class _RetrievalDoc:
    """Adapter: present RAGTool result *dicts* as attribute-bearing objects
    compatible with :meth:`Reranker.rerank`."""

    __slots__ = ("text", "doc_id", "score", "metadata")

    def __init__(self, d: Dict[str, Any]) -> None:
        self.text: str = d.get("text", "")
        self.doc_id: int = d.get("doc_id", 0)
        self.score: float = d.get("score", 0.0)
        self.metadata: Dict[str, Any] = d.get("metadata", {})


# ====================================================================== #
# Prompt loader
# ====================================================================== #


def load_prompt(domain: str, template_name: str) -> str:
    """Load a YAML prompt template.  Returns empty string on failure."""
    path = _PROMPTS_DIR / domain / f"{template_name}.yaml"
    if not path.exists():
        logger.warning("Prompt template not found: %s", path)
        return ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("template", "")
    except Exception:
        logger.warning("Failed to load prompt %s", path, exc_info=True)
        return ""


# ====================================================================== #
# Base sub-agent graph
# ====================================================================== #


class SubAgentGraph:
    """Base class for domain-specific LangGraph sub-agent subgraphs.

    Subclasses override class-level configuration:

    * ``domain`` — prompt directory name and RAGTool domain tag.
    * ``default_top_k`` — default retrieval depth.
    * ``base_confidence`` — starting confidence for heuristic scoring.
    * ``summary`` — human-readable capability description.

    Subgraph topology (4 nodes)::

        expand_query → retrieve → rerank → synthesise → END
    """

    # ---- Override in subclasses ----
    domain: str = ""
    default_top_k: int = 10
    base_confidence: float = 0.8
    summary: str = ""

    RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"

    def __init__(self) -> None:
        self._invoke_lock = threading.RLock()
        self._rag_tool: Any = None
        self._reranker: Any = None
        self._llm: Any = None
        self._compiled_graph: Any = None

    # ------------------------------------------------------------------ #
    # Lazy component accessors
    # ------------------------------------------------------------------ #

    @property
    def rag_tool(self):
        """Central RAG engine (``rag_retrieve`` MCP tool)."""
        if self._rag_tool is None:
            from rag_engine.mcp_rag_tool import RAGTool

            self._rag_tool = RAGTool()
        return self._rag_tool

    @property
    def reranker(self):
        """Cross-encoder reranker (``ncbi/MedCPT-Cross-Encoder``)."""
        if self._reranker is None:
            from rag_engine.reranker import Reranker

            self._reranker = Reranker(model_name=self.RERANKER_MODEL)
        return self._reranker

    @property
    def llm(self):
        """LiteLLM router singleton."""
        if self._llm is None:
            from llm_client import LLMClient

            self._llm = LLMClient()
        return self._llm

    # ------------------------------------------------------------------ #
    # Graph construction (lazy)
    # ------------------------------------------------------------------ #

    @property
    def graph(self):
        """Compiled LangGraph subgraph (built on first access)."""
        if self._compiled_graph is None:
            self._compiled_graph = self._build_graph()
        return self._compiled_graph

    def _build_graph(self):
        sg = StateGraph(SubAgentState)

        sg.add_node("expand_query", self._expand_query_node)
        sg.add_node("retrieve", self._retrieve_node)
        sg.add_node("rerank", self._rerank_node)
        sg.add_node("synthesise", self._synthesise_node)

        sg.set_entry_point("expand_query")
        sg.add_edge("expand_query", "retrieve")
        sg.add_edge("retrieve", "rerank")
        sg.add_edge("rerank", "synthesise")
        sg.add_edge("synthesise", END)

        return sg.compile()

    # ------------------------------------------------------------------ #
    # Node: expand_query
    # ------------------------------------------------------------------ #

    def _expand_query_node(self, state: SubAgentState) -> Dict[str, Any]:
        """Expand the user query using domain-specific LLM prompt."""
        query = state["input_query"]
        try:
            template = load_prompt(self.domain, "query_expansion")
            if not template:
                return {"expanded_query": query}

            prompt_text = template.format(query=query)
            expanded = self.llm.chat(
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.3,
                max_tokens=500,
            )
            return {"expanded_query": expanded.strip() if expanded else query}
        except Exception as exc:
            logger.warning(
                "%s: query expansion failed, using original: %s",
                self.domain,
                exc,
            )
            return {"expanded_query": query}

    # ------------------------------------------------------------------ #
    # Node: retrieve
    # ------------------------------------------------------------------ #

    def _retrieve_node(self, state: SubAgentState) -> Dict[str, Any]:
        """Retrieve documents via RAGTool (rerank=False — next node handles that)."""
        query = state.get("expanded_query") or state["input_query"]
        context = state.get("context", {})
        top_k = context.get("top_k", self.default_top_k)
        index_path = context.get("index_path")

        start = time.time()
        try:
            call_input: Dict[str, Any] = {
                "query": query,
                "top_k": top_k * 3,  # over-fetch for reranker
                "domain": self.domain,
                "rerank": False,
            }
            if index_path:
                call_input["index_path"] = index_path
            if "documents" in context:
                call_input["documents"] = context["documents"]

            result = self.rag_tool.call(call_input)
            elapsed = time.time() - start

            return {
                "retrieval_results": result.get("results", []),
                "retrieval_time_sec": elapsed,
                "error": result.get("error"),
            }
        except Exception as exc:
            elapsed = time.time() - start
            logger.error("%s: retrieval failed: %s", self.domain, exc, exc_info=True)
            return {
                "retrieval_results": [],
                "retrieval_time_sec": elapsed,
                "error": str(exc),
            }

    # ------------------------------------------------------------------ #
    # Node: rerank
    # ------------------------------------------------------------------ #

    def _rerank_node(self, state: SubAgentState) -> Dict[str, Any]:
        """Rerank retrieval results with ncbi/MedCPT-Cross-Encoder."""
        results = state.get("retrieval_results", [])
        query = state.get("expanded_query") or state["input_query"]
        context = state.get("context", {})
        top_k = context.get("top_k", self.default_top_k)

        if not results:
            return {"reranked_results": []}

        try:
            docs = [_RetrievalDoc(r) for r in results]
            reranked = self.reranker.rerank(query, docs, top_k=top_k)
            return {
                "reranked_results": [
                    {
                        "text": r.text,
                        "score": r.score,
                        "doc_id": r.doc_id,
                        "metadata": r.metadata,
                        "original_rank": r.original_rank,
                    }
                    for r in reranked
                ]
            }
        except Exception as exc:
            logger.warning(
                "%s: reranking failed, keeping original order: %s",
                self.domain,
                exc,
            )
            return {"reranked_results": results[:top_k]}

    # ------------------------------------------------------------------ #
    # Node: synthesise
    # ------------------------------------------------------------------ #

    def _synthesise_node(self, state: SubAgentState) -> Dict[str, Any]:
        """Synthesise an answer from reranked results via LLMClient."""
        start = time.time()
        query = state["input_query"]
        results = state.get("reranked_results") or state.get(
            "retrieval_results", []
        )

        if not results:
            return {
                "answer": (
                    f"No relevant {self.domain} results found for this query."
                ),
                "citations": [],
                "confidence": 0.0,
                "model_used": self.llm.default_model,
                "execution_time_sec": time.time() - start,
            }

        try:
            sources_text = self._format_sources(results)

            template = load_prompt(self.domain, "synthesis")
            if not template:
                template = (
                    "Answer the question using the provided sources.\n\n"
                    "SOURCES:\n{sources}\n\nQUESTION:\n{query}\n\nANSWER:\n"
                )

            prompt_text = template.format(query=query, sources=sources_text)
            answer = self.llm.chat(
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.7,
                max_tokens=1000,
            )

            citations = self._extract_citations(results)
            confidence = self._calculate_confidence(
                answer=answer,
                citations=citations,
                results=results,
                elapsed=time.time() - start,
            )

            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "model_used": self.llm.default_model,
                "execution_time_sec": time.time() - start,
            }
        except Exception as exc:
            logger.error(
                "%s: synthesis failed: %s", self.domain, exc, exc_info=True
            )
            return {
                "answer": f"Error synthesising {self.domain} response: {exc}",
                "citations": [],
                "confidence": 0.0,
                "model_used": "",
                "error": str(exc),
                "execution_time_sec": time.time() - start,
            }

    # ------------------------------------------------------------------ #
    # Helpers (overridable by subclasses)
    # ------------------------------------------------------------------ #

    def _format_sources(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieval results as numbered source text for synthesis."""
        parts: List[str] = []
        for i, r in enumerate(results, 1):
            text = r.get("text", "")
            meta = r.get("metadata", {})
            header = ""
            if meta.get("title"):
                header = f" — {meta['title']}"
            if meta.get("authors"):
                authors = meta["authors"]
                if isinstance(authors, list):
                    authors = ", ".join(authors)
                header += f" ({authors})"
            parts.append(f"[{i}]{header}\n{text}")
        return "\n\n".join(parts)

    def _extract_citations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Build AMA-style citations from result metadata."""
        citations: List[str] = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            segments: List[str] = []
            if meta.get("authors"):
                authors = meta["authors"]
                if isinstance(authors, list):
                    authors = ", ".join(authors)
                segments.append(authors)
            if meta.get("title"):
                segments.append(meta["title"])
            if meta.get("journal"):
                segments.append(meta["journal"])
            if meta.get("year"):
                segments.append(str(meta["year"]))
            if meta.get("doi"):
                segments.append(f"doi:{meta['doi']}")
            text = ". ".join(segments) + "." if segments else f"Source {i}"
            citations.append(f"{i}. {text}")
        return citations

    def _calculate_confidence(
        self,
        answer: str,
        citations: List[str],
        results: List[Dict[str, Any]],
        elapsed: float,
    ) -> float:
        """Heuristic confidence score.  Override in subclasses for
        domain-specific logic."""
        return self.base_confidence

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @serialized_invoke
    def invoke(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """Run the sub-agent subgraph end-to-end.

        Parameters
        ----------
        query : str
            Medical research question.
        context : dict, optional
            ``top_k``, ``index_path``, ``documents``, and other params.

        Returns
        -------
        AgentOutput
        """
        start = time.time()
        initial_state: Dict[str, Any] = {
            "input_query": query,
            "context": context or {},
            "domain": self.domain,
            "expanded_query": "",
            "retrieval_results": [],
            "retrieval_time_sec": 0.0,
            "reranked_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "model_used": "",
            "error": None,
            "execution_time_sec": 0.0,
        }

        result = self.graph.invoke(initial_state)

        return AgentOutput(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            confidence=result.get("confidence", 0.0),
            sources=result.get("reranked_results", []),
            model_used=result.get("model_used", ""),
            domain=self.domain,
            execution_time_sec=time.time() - start,
            error=result.get("error"),
        )

    def get_summary(self) -> str:
        """Return a human-readable summary of the agent's capabilities."""
        return self.summary
