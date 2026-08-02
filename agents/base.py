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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict

import yaml # type: ignore
from langgraph.graph import END, StateGraph # type: ignore
from evaluation_core import RuntimeDeadlineExceeded, safe_error_type

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from evaluation_core import EvaluationTrace

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


def llm_deadline_kwargs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the request deadline into provider-level LLM controls."""
    deadline_at = (state.get("context") or {}).get(
        "_runtime_deadline_at_monotonic"
    )
    if deadline_at is None:
        return {}
    remaining = float(deadline_at) - time.monotonic()
    if remaining <= 0:
        raise RuntimeDeadlineExceeded(
            "runtime deadline expired before agent LLM call"
        )
    return {
        "timeout": max(0.1, remaining),
        "client_max_attempts": 1,
        "deadline_at": float(deadline_at),
    }


def llm_telemetry_kwargs(
    state: Dict[str, Any], stage: str
) -> Dict[str, Any]:
    """Attach non-provider telemetry labels consumed by ``LLMClient``."""
    context = state.get("context") or {}
    return {
        "_telemetry_stage": stage,
        "_telemetry_attempt_id": str(context.get("attempt_id") or ""),
        "_telemetry_parent_attempt_id": str(
            context.get("parent_attempt_id") or ""
        ),
        "_telemetry_repair_status": str(
            context.get("repair_status") or "initial"
        ),
    }

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
    synthesis_context: List[Dict[str, Any]]
    stage_latency_sec: Dict[str, float]
    token_usage: Dict[str, int]
    cost_breakdown_usd: Dict[str, float]
    attempt_events: List[Dict[str, Any]]

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
    evaluation_trace: Optional["EvaluationTrace"] = None


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
        if getattr(self, "_llm", None) is None:
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
                **llm_deadline_kwargs(state),
                **llm_telemetry_kwargs(state, "agent_query_expansion"),
            )
            return {"expanded_query": expanded.strip() if expanded else query}
        except Exception as exc:
            logger.warning(
                "%s: query expansion failed, using original: %s",
                self.domain,
                safe_error_type(exc),
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
            if context.get("_runtime_deadline_at_monotonic") is not None:
                call_input["_runtime_deadline_at_monotonic"] = context[
                    "_runtime_deadline_at_monotonic"
                ]

            result = self.rag_tool.call(call_input)
            elapsed = time.time() - start

            return {
                "retrieval_results": result.get("results", []),
                "retrieval_time_sec": elapsed,
                "error": result.get("error"),
            }
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(
                "%s: retrieval failed error_type=%s",
                self.domain,
                safe_error_type(exc),
            )
            return {
                "retrieval_results": [],
                "retrieval_time_sec": elapsed,
                "error": f"retrieval_failed:{safe_error_type(exc)}",
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
                "%s: reranking failed, keeping original order error_type=%s",
                self.domain,
                safe_error_type(exc),
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
                "synthesis_context": [],
                "execution_time_sec": time.time() - start,
            }

        try:
            from runtime_verification.telemetry import (
                build_attempt_event,
                call_llm_with_metadata,
            )

            sources_text = self._format_sources(results)
            synthesis_context = self._build_synthesis_context(results)

            template = load_prompt(self.domain, "synthesis")
            if not template:
                template = (
                    "Answer the question using the provided sources.\n\n"
                    "SOURCES:\n{sources}\n\nQUESTION:\n{query}\n\nANSWER:\n"
                )

            prompt_text = template.format(query=query, sources=sources_text)
            call_result = call_llm_with_metadata(
                self.llm,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.7,
                max_tokens=1000,
                **llm_deadline_kwargs(state),
                **llm_telemetry_kwargs(state, "agent_synthesis"),
            )
            answer = call_result.text

            citations = self._extract_citations(results)
            confidence = self._calculate_confidence(
                answer=answer,
                citations=citations,
                results=results,
                elapsed=time.time() - start,
            )

            context = state.get("context", {})
            trace_id = str(context.get("trace_id") or "unknown")
            attempt_id = str(
                context.get("attempt_id")
                or f"{trace_id}:{self.domain}:1"
            )
            model_used = (
                f"{call_result.model}@{call_result.model_revision}"
                if call_result.model_revision
                else call_result.model
            )
            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "model_used": model_used,
                "synthesis_context": synthesis_context,
                "execution_time_sec": time.time() - start,
                "stage_latency_sec": {
                    "generation": call_result.latency_sec,
                },
                "token_usage": {
                    "input": call_result.tokens_in,
                    "output": call_result.tokens_out,
                    "total": call_result.tokens_in + call_result.tokens_out,
                },
                "cost_breakdown_usd": {
                    "generation": call_result.cost_usd,
                },
                "attempt_events": [
                    build_attempt_event(
                        trace_id=trace_id,
                        attempt_id=attempt_id,
                        parent_attempt_id=context.get("parent_attempt_id"),
                        stage="agent_synthesis",
                        component=self.domain,
                        status=str(call_result.status or "success"),
                        repair_status=str(
                            context.get("repair_status") or "initial"
                        ),
                        model=call_result.model,
                        model_revision=call_result.model_revision,
                        tokens_in=call_result.tokens_in,
                        tokens_out=call_result.tokens_out,
                        cost_usd=call_result.cost_usd,
                        latency_sec=call_result.latency_sec,
                        finish_reason=call_result.finish_reason,
                        deadline_exhausted=(
                            call_result.error_type
                            == "RuntimeDeadlineExceeded"
                        ),
                        error_type=call_result.error_type,
                        provider_metadata=call_result.provider_metadata,
                    )
                ],
            }
        except Exception as exc:
            logger.error(
                "%s: synthesis failed error_type=%s",
                self.domain,
                safe_error_type(exc),
            )
            context = state.get("context", {})
            trace_id = str(context.get("trace_id") or "unknown")
            attempt_id = str(
                context.get("attempt_id")
                or f"{trace_id}:{self.domain}:1"
            )
            from runtime_verification.telemetry import build_attempt_event

            return {
                "answer": f"Unable to synthesise the {self.domain} evidence.",
                "citations": [],
                "confidence": 0.0,
                "model_used": "",
                "synthesis_context": [],
                "error": f"synthesis_failed:{safe_error_type(exc)}",
                "execution_time_sec": time.time() - start,
                "attempt_events": [
                    build_attempt_event(
                        trace_id=trace_id,
                        attempt_id=attempt_id,
                        parent_attempt_id=context.get("parent_attempt_id"),
                        stage="agent_synthesis",
                        component=self.domain,
                        status=(
                            "deadline_exhausted"
                            if isinstance(exc, TimeoutError)
                            else "error"
                        ),
                        repair_status=str(
                            context.get("repair_status") or "initial"
                        ),
                        model=str(getattr(self.llm, "default_model", "")),
                        latency_sec=time.time() - start,
                        deadline_exhausted=isinstance(exc, TimeoutError),
                        error_type=type(exc).__name__,
                    )
                ],
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

    def _build_synthesis_context(
        self,
        results: List[Dict[str, Any]],
        citations: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Record the exact ordered source text supplied to synthesis."""
        from evaluation_core import stable_document_id

        manifest: List[Dict[str, Any]] = []
        for rank, result in enumerate(results, 1):
            text = str(result.get("text") or "")
            manifest.append(
                {
                    "document_id": stable_document_id(
                        result, self.domain, rank
                    ),
                    "text": text,
                    "start_char": 0,
                    "original_length": len(text),
                    "truncated": False,
                    "citation_marker": rank,
                }
            )
        return manifest

    def _generation_telemetry(
        self,
        state: Dict[str, Any],
        call_result: Any,
        *,
        stage: str = "agent_synthesis",
    ) -> Dict[str, Any]:
        """Map one structured completion into trace and event metadata."""
        from runtime_verification.telemetry import build_attempt_event

        context = state.get("context", {})
        trace_id = str(context.get("trace_id") or "unknown")
        attempt_id = str(
            context.get("attempt_id") or f"{trace_id}:{self.domain}:1"
        )
        model_used = (
            f"{call_result.model}@{call_result.model_revision}"
            if call_result.model_revision
            else call_result.model
        )
        return {
            "model_used": model_used,
            "stage_latency_sec": {
                "generation": call_result.latency_sec,
            },
            "token_usage": {
                "input": call_result.tokens_in,
                "output": call_result.tokens_out,
                "total": call_result.tokens_in + call_result.tokens_out,
            },
            "cost_breakdown_usd": {
                "generation": call_result.cost_usd,
            },
            "attempt_events": [
                build_attempt_event(
                    trace_id=trace_id,
                    attempt_id=attempt_id,
                    parent_attempt_id=context.get("parent_attempt_id"),
                    stage=stage,
                    component=self.domain,
                    status=str(call_result.status or "success"),
                    repair_status=str(
                        context.get("repair_status") or "initial"
                    ),
                    model=call_result.model,
                    model_revision=call_result.model_revision,
                    tokens_in=call_result.tokens_in,
                    tokens_out=call_result.tokens_out,
                    cost_usd=call_result.cost_usd,
                    latency_sec=call_result.latency_sec,
                    finish_reason=call_result.finish_reason,
                    deadline_exhausted=(
                        call_result.error_type
                        == "RuntimeDeadlineExceeded"
                    ),
                    error_type=call_result.error_type,
                    provider_metadata=call_result.provider_metadata,
                )
            ],
        }

    def _failure_telemetry(
        self,
        state: Dict[str, Any],
        exc: BaseException,
        *,
        stage: str,
        latency_sec: float,
    ) -> Dict[str, Any]:
        """Map a failed provider call into the canonical attempt schema."""
        from llm_client import LLMCallResult
        from runtime_verification.telemetry import build_attempt_event

        context = state.get("context", {})
        trace_id = str(context.get("trace_id") or "unknown")
        attempt_id = str(
            context.get("attempt_id") or f"{trace_id}:{self.domain}:1"
        )
        history_reader = getattr(self.llm, "thread_call_history", None)
        history = (
            list(history_reader() or [])
            if callable(history_reader)
            else list(history_reader or [])
        )
        history_start = context.get("_llm_history_start")
        if (
            isinstance(history_start, int)
            and not isinstance(history_start, bool)
            and history_start >= 0
        ):
            history = history[history_start:]
        call_result = history[-1] if history else None
        if not isinstance(call_result, LLMCallResult):
            call_result = None
        model = (
            call_result.model
            if call_result is not None
            else str(getattr(self.llm, "default_model", "") or "")
        )
        model_revision = (
            call_result.model_revision if call_result is not None else ""
        )
        model_used = (
            f"{model}@{model_revision}" if model_revision else model
        )
        tokens_in = call_result.tokens_in if call_result is not None else 0
        tokens_out = call_result.tokens_out if call_result is not None else 0
        cost_usd = call_result.cost_usd if call_result is not None else 0.0
        actual_latency = (
            call_result.latency_sec
            if call_result is not None
            else latency_sec
        )
        error_type = (
            call_result.error_type
            if call_result is not None and call_result.error_type
            else type(exc).__name__
        )
        deadline_exhausted = error_type == "RuntimeDeadlineExceeded"
        return {
            "model_used": model_used,
            "stage_latency_sec": {"generation": actual_latency},
            "token_usage": {
                "input": tokens_in,
                "output": tokens_out,
                "total": tokens_in + tokens_out,
            },
            "cost_breakdown_usd": {"generation": cost_usd},
            "attempt_events": [
                build_attempt_event(
                    trace_id=trace_id,
                    attempt_id=attempt_id,
                    parent_attempt_id=context.get("parent_attempt_id"),
                    stage=stage,
                    component=self.domain,
                    status=(
                        "deadline_exhausted"
                        if deadline_exhausted
                        else "error"
                    ),
                    repair_status=str(
                        context.get("repair_status") or "initial"
                    ),
                    model=model,
                    model_revision=model_revision,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=cost_usd,
                    latency_sec=actual_latency,
                    finish_reason=(
                        call_result.finish_reason
                        if call_result is not None
                        else "error"
                    ),
                    deadline_exhausted=deadline_exhausted,
                    error_type=error_type,
                    provider_metadata=(
                        call_result.provider_metadata
                        if call_result is not None
                        else None
                    ),
                )
            ],
        }

    def _merge_llm_history_telemetry(
        self, result: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Rebuild attempt telemetry from every physical provider call."""
        from llm_client import LLMCallResult
        from runtime_verification.telemetry import build_attempt_event

        history_reader = getattr(self.llm, "thread_call_history", None)
        if not callable(history_reader):
            return

        history = list(history_reader() or [])
        history_start = context.get("_llm_history_start")
        if (
            isinstance(history_start, int)
            and not isinstance(history_start, bool)
            and history_start >= 0
        ):
            history = history[history_start:]

        calls = [
            call for call in history if isinstance(call, LLMCallResult)
        ]
        if not calls:
            return

        trace_id = str(context.get("trace_id") or "unknown")
        attempt_id = str(
            context.get("attempt_id") or f"{trace_id}:{self.domain}:1"
        )
        events: List[Dict[str, Any]] = []
        tokens_in = 0
        tokens_out = 0
        total_cost = 0.0
        stage_latency = dict(result.get("stage_latency_sec") or {})
        cost_breakdown = dict(result.get("cost_breakdown_usd") or {})

        for index, call in enumerate(calls, 1):
            provider_attempt = int(
                (call.provider_metadata or {}).get("provider_attempt")
                or index
            )
            stage = str(
                (call.provider_metadata or {}).get("telemetry_stage")
                or "agent_llm_call"
            )
            event_id = (
                f"{attempt_id}:{stage}:provider:{provider_attempt}"
                if len(calls) > 1
                else f"{attempt_id}:{stage}"
            )
            deadline_exhausted = (
                call.error_type == "RuntimeDeadlineExceeded"
            )
            events.append(
                build_attempt_event(
                    trace_id=trace_id,
                    attempt_id=attempt_id,
                    parent_attempt_id=context.get("parent_attempt_id"),
                    stage=stage,
                    component=self.domain,
                    status=(
                        "deadline_exhausted"
                        if deadline_exhausted
                        else str(call.status or "success")
                    ),
                    repair_status=str(
                        context.get("repair_status") or "initial"
                    ),
                    model=call.model,
                    model_revision=call.model_revision,
                    tokens_in=call.tokens_in,
                    tokens_out=call.tokens_out,
                    cost_usd=call.cost_usd,
                    latency_sec=call.latency_sec,
                    finish_reason=call.finish_reason,
                    deadline_exhausted=deadline_exhausted,
                    error_type=call.error_type,
                    provider_metadata=call.provider_metadata,
                    event_id=event_id,
                )
            )
            tokens_in += int(call.tokens_in or 0)
            tokens_out += int(call.tokens_out or 0)
            total_cost += float(call.cost_usd or 0.0)
            stage_latency[stage] = float(stage_latency.get(stage) or 0.0) + float(
                call.latency_sec or 0.0
            )
            cost_breakdown[stage] = float(
                cost_breakdown.get(stage) or 0.0
            ) + float(call.cost_usd or 0.0)

        result["attempt_events"] = events
        result["token_usage"] = {
            "input": tokens_in,
            "output": tokens_out,
            "total": tokens_in + tokens_out,
        }
        result["cost_breakdown_usd"] = cost_breakdown
        result["stage_latency_sec"] = stage_latency

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

    def _output_from_result(
        self,
        *,
        result: Dict[str, Any],
        query: str,
        context: Optional[Dict[str, Any]],
        started_at: float,
    ) -> AgentOutput:
        """Create the stable output plus its backward-compatible trace sidecar."""
        elapsed = time.time() - started_at
        output = AgentOutput(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            confidence=result.get("confidence", 0.0),
            sources=result.get("reranked_results", []),
            model_used=result.get("model_used", ""),
            domain=self.domain,
            execution_time_sec=elapsed,
            error=result.get("error"),
        )

        from evaluation_core import build_agent_evaluation_trace

        trace_state = dict(result)
        trace_state["execution_time_sec"] = elapsed
        try:
            output.evaluation_trace = build_agent_evaluation_trace(
                agent_name=str((context or {}).get("agent_name") or self.domain),
                domain=self.domain,
                original_query=query,
                state=trace_state,
                context=context,
            )
        except Exception as exc:
            logger.error(
                "%s: evaluation trace adapter failed; preserving AgentOutput "
                "error_type=%s",
                self.domain,
                safe_error_type(exc),
            )
        return output

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
        execution_context = dict(context or {})
        history_reader = getattr(
            self.llm, "thread_call_history", None
        )
        execution_context["_llm_history_start"] = (
            len(history_reader()) if callable(history_reader) else 0
        )
        initial_state: Dict[str, Any] = {
            "input_query": query,
            "context": execution_context,
            "domain": self.domain,
            "expanded_query": "",
            "retrieval_results": [],
            "retrieval_time_sec": 0.0,
            "reranked_results": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "model_used": "",
            "synthesis_context": [],
            "stage_latency_sec": {},
            "token_usage": {},
            "cost_breakdown_usd": {},
            "attempt_events": [],
            "error": None,
            "execution_time_sec": 0.0,
        }

        result = self.graph.invoke(initial_state)
        self._merge_llm_history_telemetry(result, execution_context)

        return self._output_from_result(
            result=result,
            query=query,
            context=execution_context,
            started_at=start,
        )

    def get_summary(self) -> str:
        """Return a human-readable summary of the agent's capabilities."""
        return self.summary
