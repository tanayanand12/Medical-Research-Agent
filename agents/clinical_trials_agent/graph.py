"""
Clinical Trials sub-agent — Phase 7.

LangGraph subgraph that fetches live trial data from ClinicalTrials.gov,
chunks and indexes it on-the-fly, retrieves via hybrid BM25 + HNSW,
reranks with a cross-encoder, and synthesises a citation-rich answer.

Replaces the legacy ``ClinicalTrialsAgent`` wrapper in
``agentic-pipeline-clinical/clinical_trials_agent_wrapper.py``.

Node topology (6 nodes)::

    expand_query → fetch → chunk_and_index → retrieve → rerank → synthesise → END

Legacy behaviour ported
-----------------------
* ClinicalTrialsFetcherAgent URL generation + 5-URL diversification.
* RAG pipeline over clinical trials with chunk_size=1000, chunk_overlap=200.
* Confidence heuristic based on answer length, citation count, and
  processing time.
* Default ``top_k = 10``, ``max_trials = 25``.

Changes from legacy
-------------------
* LLM calls routed through LLMClient (was hardcoded gpt-4-turbo).
* Embeddings via configurable model (was hardcoded text-embedding-ada-002).
* Retrieval via hybrid BM25 + HNSW with RRF fusion (was legacy RAG pipeline).
* Reranking with cross-encoder (was absent).
* Query expansion via ``prompts/clinical_trials/query_expansion.yaml``.
* Synthesis via ``prompts/clinical_trials/synthesis.yaml``.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph # type: ignore 
from evaluation_core import RuntimeDeadlineExceeded, safe_error_type

from agents.base import (
    AgentOutput,
    SubAgentGraph,
    SubAgentState,
    llm_deadline_kwargs,
    llm_telemetry_kwargs,
    load_prompt,
    serialized_invoke,
)

logger = logging.getLogger(__name__)


# ====================================================================== #
# Extended state — adds fetch-specific fields to the shared contract
# ====================================================================== #


class ClinicalTrialsState(TypedDict):
    """State contract for the 6-node clinical trials subgraph."""

    # ---- Input ----
    input_query: str
    context: Dict[str, Any]
    domain: str

    # ---- Query expansion ----
    expanded_query: str

    # ---- Fetch ----
    fetched_studies: List[Dict[str, Any]]
    fetch_meta: Dict[str, Any]

    # ---- Chunking / indexing ----
    chunks_ready: bool

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
# Sub-agent graph
# ====================================================================== #


class ClinicalTrialsAgentGraph(SubAgentGraph):
    """Clinical trials data retrieval and synthesis sub-agent.

    Unlike the other domain agents that rely on pre-built indexes,
    this agent fetches live data from ClinicalTrials.gov, chunks it
    on-the-fly, builds ephemeral BM25 + HNSW indexes, and runs hybrid
    retrieval with cross-encoder reranking.
    """

    domain = "clinical_trials"
    default_top_k = 10
    base_confidence = 0.80
    summary = (
        "Clinical trials RAG sub-agent that fetches live data from "
        "ClinicalTrials.gov, builds ephemeral indexes, and synthesises "
        "evidence using hybrid retrieval with cross-encoder reranking."
    )

    MAX_TRIALS = 25

    def __init__(self) -> None:
        super().__init__()
        self._fetcher = None
        self._chunker = None
        self._embedder = None
        self._sparse_idx = None
        self._dense_idx = None
        self._hybrid = None

    # ------------------------------------------------------------------ #
    # Lazy component accessors
    # ------------------------------------------------------------------ #

    @property
    def fetcher(self):
        """ClinicalTrials.gov data fetcher."""
        if self._fetcher is None:
            from agents.clinical_trials_agent.data_fetcher import (
                ClinicalTrialsFetcher,
            )

            self._fetcher = ClinicalTrialsFetcher(llm_client=self.llm)
        return self._fetcher

    @property
    def chunker(self):
        """Semantic / recursive chunker."""
        if self._chunker is None:
            from rag_engine.chunker import SemanticChunker

            self._chunker = SemanticChunker(
                max_chunk_tokens=512,
                min_chunk_tokens=30,
            )
        return self._chunker

    @property
    def embedder(self):
        """LLM-agnostic embedder."""
        if self._embedder is None:
            from rag_engine.embedder import Embedder

            self._embedder = Embedder()
        return self._embedder

    # ------------------------------------------------------------------ #
    # Graph construction — overrides base 4-node with 6-node topology
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        sg = StateGraph(ClinicalTrialsState)

        sg.add_node("expand_query", self._expand_query_node)
        sg.add_node("fetch", self._fetch_node)
        sg.add_node("chunk_and_index", self._chunk_and_index_node)
        sg.add_node("retrieve", self._retrieve_node)
        sg.add_node("rerank", self._rerank_node)
        sg.add_node("synthesise", self._synthesise_node)

        sg.set_entry_point("expand_query")
        sg.add_edge("expand_query", "fetch")
        sg.add_edge("fetch", "chunk_and_index")
        sg.add_edge("chunk_and_index", "retrieve")
        sg.add_edge("retrieve", "rerank")
        sg.add_edge("rerank", "synthesise")
        sg.add_edge("synthesise", END)

        return sg.compile()

    # ------------------------------------------------------------------ #
    # Node: expand_query (inherited prompt template, override to use domain)
    # ------------------------------------------------------------------ #

    def _expand_query_node(self, state: ClinicalTrialsState) -> Dict[str, Any]:
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
                "%s: query expansion failed error_type=%s",
                self.domain,
                safe_error_type(exc),
            )
            return {"expanded_query": query}

    # ------------------------------------------------------------------ #
    # Node: fetch — live data from ClinicalTrials.gov
    # ------------------------------------------------------------------ #

    def _fetch_node(self, state: ClinicalTrialsState) -> Dict[str, Any]:
        """Fetch live trial data via the data fetcher."""
        query = state.get("expanded_query") or state["input_query"]
        context = state.get("context", {})
        max_trials = context.get("max_trials", self.MAX_TRIALS)

        try:
            result = self.fetcher.analyze_user_query(
                query,
                llm_kwargs={
                    **llm_deadline_kwargs(state),
                    **llm_telemetry_kwargs(
                        state, "clinical_trials_query_extraction"
                    ),
                },
            )

            if not result.get("success"):
                logger.warning(
                    "ClinicalTrials fetch failed error_type=%s",
                    result.get("error_type") or "FetcherError",
                )
                return {
                    "fetched_studies": [],
                    "fetch_meta": result,
                    "error": "clinical_trials_fetch_failed",
                }

            studies = result.get("data", {}).get("studies", [])
            # Cap to max_trials
            studies = studies[:max_trials]

            logger.info(
                "Fetched %d studies (capped from %d)",
                len(studies),
                result.get("total_count", 0),
            )
            return {
                "fetched_studies": studies,
                "fetch_meta": result.get("query_analysis", {}),
            }
        except Exception as exc:
            logger.error(
                "ClinicalTrials fetch failed error_type=%s",
                safe_error_type(exc),
            )
            return {
                "fetched_studies": [],
                "fetch_meta": {},
                "error": f"fetch_failed:{safe_error_type(exc)}",
            }

    # ------------------------------------------------------------------ #
    # Node: chunk_and_index — build ephemeral BM25 + HNSW indexes
    # ------------------------------------------------------------------ #

    def _chunk_and_index_node(self, state: ClinicalTrialsState) -> Dict[str, Any]:
        """Chunk fetched trial JSON into text, build BM25 + HNSW indexes."""
        self._sparse_idx = None
        self._dense_idx = None
        self._hybrid = None
        studies = state.get("fetched_studies", [])

        if not studies:
            return {"chunks_ready": False}

        try:
            # Flatten each study JSON into readable text
            texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []

            for study in studies:
                text, meta = self._study_to_text(study)
                if text.strip():
                    texts.append(text)
                    metadatas.append(meta)

            if not texts:
                return {"chunks_ready": False, "error": "No text extracted from studies"}

            # Chunk all texts
            all_chunk_texts: List[str] = []
            all_chunk_metas: List[Dict[str, Any]] = []

            for text, meta in zip(texts, metadatas):
                chunks = self.chunker.chunk(text)
                for chunk in chunks:
                    all_chunk_texts.append(chunk.text)
                    all_chunk_metas.append(meta)

            if not all_chunk_texts:
                return {"chunks_ready": False, "error": "Chunking produced no output"}

            logger.info(
                "Chunked %d studies into %d chunks",
                len(texts),
                len(all_chunk_texts),
            )

            # Build BM25 sparse index
            from rag_engine.sparse_index import BM25Index

            self._sparse_idx = BM25Index()
            self._sparse_idx.add_documents(all_chunk_texts, all_chunk_metas)

            # Build HNSW dense index
            try:
                deadline_at = (state.get("context") or {}).get(
                    "_runtime_deadline_at_monotonic"
                )
                embeddings = self.embedder.embed_batch(
                    all_chunk_texts,
                    deadline_at=deadline_at,
                    client_max_attempts=(
                        1 if deadline_at is not None else None
                    ),
                )
            except RuntimeDeadlineExceeded:
                raise
            except Exception as exc:
                embeddings = []
                logger.warning(
                    "Dense indexing unavailable; continuing with BM25 "
                    "error_type=%s",
                    safe_error_type(exc),
                )

            if embeddings:
                from rag_engine.dense_index import DenseIndex

                dim = len(embeddings[0])
                self._dense_idx = DenseIndex(dimension=dim)
                self._dense_idx.add_documents(embeddings, all_chunk_texts, all_chunk_metas)

            # Build hybrid retriever
            from rag_engine.hybrid_retriever import HybridRetriever

            self._hybrid = HybridRetriever(
                dense_index=self._dense_idx,
                sparse_index=self._sparse_idx,
                embedder=self.embedder,
            )

            return {"chunks_ready": True}

        except Exception as exc:
            logger.error(
                "Chunk/index failed error_type=%s",
                safe_error_type(exc),
            )
            return {
                "chunks_ready": False,
                "error": f"chunk_index_failed:{safe_error_type(exc)}",
            }

    # ------------------------------------------------------------------ #
    # Node: retrieve — hybrid BM25 + HNSW with RRF
    # ------------------------------------------------------------------ #

    def _retrieve_node(self, state: ClinicalTrialsState) -> Dict[str, Any]:
        """Retrieve documents using hybrid retriever (RRF fusion)."""
        if not state.get("chunks_ready"):
            return {
                "retrieval_results": [],
                "retrieval_time_sec": 0.0,
                "error": state.get("error", "No index available"),
            }

        query = state.get("expanded_query") or state["input_query"]
        context = state.get("context", {})
        top_k = context.get("top_k", self.default_top_k)

        start = time.time()
        try:
            results = self._hybrid.retrieve(
                query,
                top_k=top_k * 3,
                deadline_at=(state.get("context") or {}).get(
                    "_runtime_deadline_at_monotonic"
                ),
            )
            elapsed = time.time() - start

            retrieval_dicts = [
                {
                    "text": r.text,
                    "score": r.score,
                    "doc_id": r.doc_id,
                    "metadata": r.metadata,
                }
                for r in results
            ]

            logger.info(
                "Hybrid retrieval returned %d results in %.2fs",
                len(retrieval_dicts),
                elapsed,
            )
            return {
                "retrieval_results": retrieval_dicts,
                "retrieval_time_sec": elapsed,
            }
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(
                "Retrieval failed error_type=%s",
                safe_error_type(exc),
            )
            return {
                "retrieval_results": [],
                "retrieval_time_sec": elapsed,
                "error": f"retrieval_failed:{safe_error_type(exc)}",
            }

    # ------------------------------------------------------------------ #
    # Node: rerank — cross-encoder reranking
    # ------------------------------------------------------------------ #

    def _rerank_node(self, state: ClinicalTrialsState) -> Dict[str, Any]:
        """Rerank retrieval results with cross-encoder."""
        results = state.get("retrieval_results", [])
        query = state.get("expanded_query") or state["input_query"]
        context = state.get("context", {})
        top_k = context.get("top_k", self.default_top_k)

        if not results:
            return {"reranked_results": []}

        try:
            from agents.base import _RetrievalDoc

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
                "Reranking failed; keeping original order error_type=%s",
                safe_error_type(exc),
            )
            return {"reranked_results": results[:top_k]}

    # ------------------------------------------------------------------ #
    # Node: synthesise — LLMClient + prompt template
    # ------------------------------------------------------------------ #

    def _synthesise_node(self, state: ClinicalTrialsState) -> Dict[str, Any]:
        """Synthesise an answer from reranked results via LLMClient."""
        start = time.time()
        query = state["input_query"]
        results = state.get("reranked_results") or state.get(
            "retrieval_results", []
        )

        if not results:
            return {
                "answer": (
                    "No relevant clinical trials results found for this query."
                ),
                "citations": [],
                "confidence": 0.0,
                "model_used": self.llm.default_model,
                "synthesis_context": [],
                "execution_time_sec": time.time() - start,
            }

        try:
            sources_text = self._format_sources(results)
            synthesis_context = self._build_synthesis_context(results)

            template = load_prompt(self.domain, "synthesis")
            if not template:
                template = (
                    "Answer the question using the provided sources.\n\n"
                    "SOURCES:\n{sources}\n\nQUESTION:\n{query}\n\nANSWER:\n"
                )

            prompt_text = template.format(query=query, sources=sources_text)
            from runtime_verification import call_llm_with_metadata

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

            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "synthesis_context": synthesis_context,
                "execution_time_sec": time.time() - start,
                **self._generation_telemetry(state, call_result),
            }
        except Exception as exc:
            logger.error(
                "Synthesis failed error_type=%s",
                safe_error_type(exc),
            )
            return {
                "answer": "Unable to synthesise the clinical-trial evidence.",
                "citations": [],
                "confidence": 0.0,
                "synthesis_context": [],
                "error": f"synthesis_failed:{safe_error_type(exc)}",
                "execution_time_sec": time.time() - start,
                **self._failure_telemetry(
                    state,
                    exc,
                    stage="agent_synthesis",
                    latency_sec=time.time() - start,
                ),
            }

    # ------------------------------------------------------------------ #
    # Confidence — ported from legacy ClinicalTrialsAgent._calculate_confidence
    # ------------------------------------------------------------------ #

    def _calculate_confidence(
        self,
        answer: str,
        citations: List[str],
        results: List[Dict[str, Any]],
        elapsed: float,
    ) -> float:
        conf = self.base_confidence

        # Answer length
        if len(answer) > 500:
            conf += 0.05
        elif len(answer) < 100:
            conf -= 0.1

        # Citation count
        if len(citations) > 3:
            conf += 0.1
        elif len(citations) == 0:
            conf -= 0.15

        # Processing time
        if elapsed < 2.0:
            conf -= 0.05
        elif elapsed > 10.0:
            conf -= 0.1

        return max(0.0, min(1.0, conf))

    # ------------------------------------------------------------------ #
    # Study JSON → readable text
    # ------------------------------------------------------------------ #

    @staticmethod
    def _study_to_text(study: Dict[str, Any]) -> tuple:
        """Convert a ClinicalTrials.gov JSON study to readable text + metadata.

        Returns
        -------
        tuple[str, dict]
            ``(text, metadata)``
        """
        protocol = study.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        status_mod = protocol.get("statusModule", {})
        desc = protocol.get("descriptionModule", {})
        design = protocol.get("designModule", {})
        arms = protocol.get("armsInterventionsModule", {})
        outcomes = protocol.get("outcomesModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        contacts = protocol.get("contactsLocationsModule", {})
        sponsor = protocol.get("sponsorCollaboratorsModule", {})

        nct_id = ident.get("nctId", "Unknown")
        title = ident.get("briefTitle", ident.get("officialTitle", "Unknown"))

        parts: List[str] = []
        parts.append(f"Study: {title}")
        parts.append(f"NCT ID: {nct_id}")

        # Status
        overall_status = status_mod.get("overallStatus", "")
        if overall_status:
            parts.append(f"Status: {overall_status}")

        # Phase
        phases = design.get("phases", [])
        if phases:
            parts.append(f"Phase: {', '.join(phases)}")

        # Description
        brief_summary = desc.get("briefSummary", "")
        if brief_summary:
            parts.append(f"Summary: {brief_summary}")
        detailed = desc.get("detailedDescription", "")
        if detailed:
            parts.append(f"Description: {detailed}")

        # Interventions
        interventions = arms.get("interventions", [])
        if interventions:
            intv_strs = []
            for intv in interventions:
                name = intv.get("name", "")
                itype = intv.get("type", "")
                intv_desc = intv.get("description", "")
                intv_strs.append(f"{itype}: {name} - {intv_desc}")
            parts.append("Interventions: " + "; ".join(intv_strs))

        # Outcomes
        primary_outcomes = outcomes.get("primaryOutcomes", [])
        if primary_outcomes:
            outcome_strs = [
                o.get("measure", "") for o in primary_outcomes if o.get("measure")
            ]
            if outcome_strs:
                parts.append("Primary Outcomes: " + "; ".join(outcome_strs))

        secondary_outcomes = outcomes.get("secondaryOutcomes", [])
        if secondary_outcomes:
            outcome_strs = [
                o.get("measure", "") for o in secondary_outcomes if o.get("measure")
            ]
            if outcome_strs:
                parts.append("Secondary Outcomes: " + "; ".join(outcome_strs))

        # Eligibility
        criteria = eligibility.get("eligibilityCriteria", "")
        if criteria:
            parts.append(f"Eligibility: {criteria}")

        # Sponsor
        lead = sponsor.get("leadSponsor", {})
        if lead.get("name"):
            parts.append(f"Sponsor: {lead['name']}")

        # Enrollment
        enrollment_info = design.get("enrollmentInfo", {})
        if enrollment_info.get("count"):
            parts.append(
                f"Enrollment: {enrollment_info['count']} "
                f"({enrollment_info.get('type', '')})"
            )

        text = "\n".join(parts)
        metadata = {
            "nct_id": nct_id,
            "title": title,
            "status": overall_status,
            "source": "ClinicalTrials.gov",
        }
        return text, metadata

    # ------------------------------------------------------------------ #
    # Public API — override to use extended state
    # ------------------------------------------------------------------ #

    @serialized_invoke
    def invoke(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """Run the clinical trials sub-agent end-to-end.

        Parameters
        ----------
        query : str
            Medical research question.
        context : dict, optional
            ``top_k``, ``max_trials``, and other params.

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
            "fetched_studies": [],
            "fetch_meta": {},
            "chunks_ready": False,
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

        return self._output_from_result(
            result=result,
            query=query,
            context=context,
            started_at=start,
        )
