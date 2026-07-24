"""
graph.py — Phase 7: FDA regulatory data sub-agent.

LangGraph subgraph that fetches live regulatory records from openFDA
(drug labels, adverse events, device recalls, food enforcement), chunks
and indexes them on-the-fly, retrieves via hybrid BM25 + HNSW with RRF
fusion, reranks with the shared cross-encoder, and synthesises a
citation-rich regulatory answer.

Replaces the legacy FdaRAGPipeline / FdaFetcherAgent / FdaRAGModule stack
from ``agentic-pipeline-clinical/``.

Node topology (6 nodes)::

    expand_query → fetch → chunk_and_index → retrieve → rerank → synthesise → END

Legacy behaviour ported
-----------------------
* FdaFetcherAgent parallel fetch (ThreadPoolExecutor, 8 workers).
* Record type detection: drug_label, adverse_event, recall / fallback.
* FDA regulatory synthesis with structured 6-section format.
* Confidence heuristic: answer length + citation count + latency.
* Default top_k = 10, MAX_RECORDS = 200.

Changes from legacy
-------------------
* LLM calls routed through LLMClient (was: hardcoded OpenAI gpt-4o client).
* URL generation replaced by LLM concept extraction + deterministic builder.
* Retrieval via hybrid BM25 + HNSW with RRF fusion (was: cosine similarity
  over OpenAI text-embedding-ada-002 vectors).
* Reranking with shared cross-encoder (was: absent in legacy).
* Query expansion via ``prompts/fda/query_expansion.yaml``.
* Synthesis via ``prompts/fda/synthesis.yaml``.
* ``_extract_citations()`` overrides base: uses FDA record IDs, not NCT IDs.
* ``base_confidence = 0.75`` (lower than clinical_trials 0.80, reflecting
  higher heterogeneity of openFDA data across multiple endpoints).
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph  # type: ignore

from agents.base import (  # type: ignore
    AgentOutput,
    SubAgentGraph,
    SubAgentState,
    load_prompt,
    serialized_invoke,
)

logger = logging.getLogger(__name__)


# ====================================================================== #
# State — mirrors ClinicalTrialsState with FDA-specific field names
# ====================================================================== #


class FDAAgentState(TypedDict):
    """State contract for the 6-node FDA regulatory sub-agent.

    All fields must be initialised in invoke() before graph execution.
    Fields are written by individual nodes and consumed by downstream
    nodes; no field should be mutated by more than one node.

    Attributes
    ----------
    input_query : str
        Original user question (immutable throughout execution).
    context : dict
        Caller-supplied overrides: top_k, max_records, etc.
    domain : str
        Agent domain identifier (``"fda"``).
    expanded_query : str
        LLM-expanded query from expand_query node; falls back to input_query.
    fetched_records : list[dict]
        Raw openFDA records from fetch node, capped to MAX_RECORDS.
    fetch_meta : dict
        Query analysis metadata from FDAFetcher.analyze_user_query().
    chunks_ready : bool
        True when BM25 + HNSW indexes are built and hybrid retriever is ready.
    retrieval_results : list[dict]
        Hybrid retrieval results (text, score, doc_id, metadata).
    retrieval_time_sec : float
        Wall-clock seconds for hybrid retrieval.
    reranked_results : list[dict]
        Cross-encoder reranked results, capped to top_k.
    answer : str
        Final synthesised regulatory answer.
    citations : list[str]
        Deduplicated FDA record IDs cited in the answer.
    confidence : float
        Heuristic confidence score in [0.0, 1.0].
    model_used : str
        LLM model identifier used for synthesis.
    error : str | None
        First error message encountered; None on success.
    execution_time_sec : float
        Total wall-clock seconds from invoke() to AgentOutput.
    """

    # ---- Input ----
    input_query: str
    context: Dict[str, Any]
    domain: str

    # ---- Query expansion ----
    expanded_query: str

    # ---- Fetch ----
    fetched_records: List[Dict[str, Any]]
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

    # ---- Metadata ----
    error: Optional[str]
    execution_time_sec: float


# ====================================================================== #
# Sub-agent graph
# ====================================================================== #


class FDAAgentGraph(SubAgentGraph):
    """FDA regulatory data retrieval and synthesis sub-agent.

    Fetches live records from openFDA (drug labels, adverse events, device
    recalls, food enforcement), chunks on-the-fly, builds ephemeral BM25 +
    HNSW indexes, and runs hybrid retrieval with cross-encoder reranking.

    Designed as a drop-in sub-agent for the multi-agent orchestrator; exposes
    the same ``invoke(query, context) → AgentOutput`` interface as all other
    domain agents.

    Attributes
    ----------
    domain : str
        ``"fda"`` — used for prompt loading and orchestrator routing.
    default_top_k : int
        Number of reranked chunks passed to synthesis (default 10).
    base_confidence : float
        Starting confidence before heuristic adjustments (0.75).
    summary : str
        One-line description for orchestrator skill discovery.
    MAX_RECORDS : int
        Hard cap on raw openFDA records ingested per query to bound latency.

    Examples
    --------
    >>> from agents.fda_agent import FDAAgentGraph
    >>> agent = FDAAgentGraph()
    >>> output = agent.invoke("What are the adverse events for metformin?")
    >>> print(output.answer)
    >>> print(output.citations)
    """

    domain = "fda"
    default_top_k = 10
    base_confidence = 0.75
    summary = (
        "FDA regulatory RAG sub-agent that fetches live data from openFDA "
        "(drug labels, adverse events, recalls), builds ephemeral BM25 + HNSW "
        "indexes, and synthesises regulatory evidence with cross-encoder reranking."
    )
    MAX_RECORDS = 200

    def __init__(self) -> None:
        super().__init__()
        # Lazy-initialised RAG components (not instantiated until first call)
        self._fetcher = None
        self._chunker = None
        self._embedder = None
        self._sparse_idx = None
        self._dense_idx = None
        self._hybrid = None

    # ------------------------------------------------------------------ #
    # Lazy component accessors (mirrors ClinicalTrialsAgentGraph pattern)
    # ------------------------------------------------------------------ #

    @property
    def fetcher(self):
        """Lazy-loaded FDAFetcher (concept extraction + deterministic URLs)."""
        if self._fetcher is None:
            from agents.fda_agent.data_fetcher import FDAFetcher
            self._fetcher = FDAFetcher(llm_client=self.llm)
        return self._fetcher

    @property
    def chunker(self):
        """Lazy-loaded SemanticChunker from central rag_engine."""
        if self._chunker is None:
            from rag_engine.chunker import SemanticChunker
            self._chunker = SemanticChunker(
                max_chunk_tokens=512,
                min_chunk_tokens=30,
            )
        return self._chunker

    @property
    def embedder(self):
        """Lazy-loaded LLM-agnostic Embedder from central rag_engine."""
        if self._embedder is None:
            from rag_engine.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    # ------------------------------------------------------------------ #
    # Graph construction — 6-node topology
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        """Construct and compile the 6-node LangGraph subgraph.

        Returns
        -------
        CompiledGraph
            Compiled LangGraph graph ready for invocation.
        """
        sg = StateGraph(FDAAgentState)

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
    # Node 1: expand_query
    # ------------------------------------------------------------------ #

    def _expand_query_node(self, state: FDAAgentState) -> Dict[str, Any]:
        """Expand user query with FDA-domain regulatory terminology.

        Loads ``prompts/fda/query_expansion.yaml`` and calls the LLM to
        rephrase the query with precise regulatory vocabulary (drug names,
        indication terms, adverse event terminology) that improves both
        openFDA URL construction recall and hybrid index retrieval.

        Falls back to the original query on any failure (graceful degradation).

        Parameters
        ----------
        state : FDAAgentState
            Current pipeline state.

        Returns
        -------
        dict
            ``{"expanded_query": str}``
        """
        query = state["input_query"]
        try:
            template = load_prompt(self.domain, "query_expansion")
            if not template:
                logger.debug("No FDA query_expansion template found; using original query")
                return {"expanded_query": query}

            prompt_text = template.format(query=query)
            expanded = self.llm.chat(
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.3,
                max_tokens=500,
            )
            result = expanded.strip() if expanded else query
            logger.info("Expanded FDA query: %s", result[:120])
            return {"expanded_query": result}

        except Exception as exc:
            logger.warning(
                "FDA query expansion failed, using original query: %s", exc
            )
            return {"expanded_query": query}

    # ------------------------------------------------------------------ #
    # Node 2: fetch
    # ------------------------------------------------------------------ #

    def _fetch_node(self, state: FDAAgentState) -> Dict[str, Any]:
        """Fetch live FDA regulatory records via FDAFetcher.

        **Always uses ``input_query``** for fetching — not ``expanded_query``.
        The ``expanded_query`` is a natural-language rephrase used only by the
        retrieve node for index search.  Passing the expanded query to the
        fetcher's concept-extraction step caused JSON artifacts from the
        expansion template to pollute the keyword fallback, producing garbage
        openFDA URLs (e.g. ``search=openfda.brand_name:json``).

        Records are capped to MAX_RECORDS (200) before downstream processing
        to bound chunking and embedding latency.

        Parameters
        ----------
        state : FDAAgentState
            Current pipeline state.

        Returns
        -------
        dict
            ``{"fetched_records": list, "fetch_meta": dict}``
            On failure: also sets ``"error": str``.
        """
        node_start = time.time()
        # ALWAYS use the original query for concept extraction and URL building.
        # expanded_query is reserved for retrieval against the built index.
        query = state["input_query"]
        context = state.get("context", {})
        max_records = context.get("max_records", self.MAX_RECORDS)

        logger.info("[fetch] start — query: %s", query[:120])

        try:
            result = self.fetcher.analyze_user_query(query)

            if not result.get("success"):
                logger.warning("[fetch] failed: %s (%.1fs)", result.get("error"), time.time() - node_start)
                return {
                    "fetched_records": [],
                    "fetch_meta": result,
                    "error": result.get("error", "Fetch failed"),
                }

            records = result.get("data", {}).get("records", [])
            records = records[:max_records]

            logger.info(
                "[fetch] done — %d records (capped from %d total) in %.1fs",
                len(records),
                result.get("total_count", 0),
                time.time() - node_start,
            )
            return {
                "fetched_records": records,
                "fetch_meta": result.get("query_analysis", {}),
            }

        except Exception as exc:
            logger.error("[fetch] error: %s (%.1fs)", exc, time.time() - node_start, exc_info=True)
            return {
                "fetched_records": [],
                "fetch_meta": {},
                "error": str(exc),
            }

    # ------------------------------------------------------------------ #
    # Node 3: chunk_and_index
    # ------------------------------------------------------------------ #

    def _chunk_and_index_node(self, state: FDAAgentState) -> Dict[str, Any]:
        """Convert FDA records to text, chunk, build BM25 + HNSW indexes.

        Each record is converted to human-readable text via ``_record_to_text()``
        which type-dispatches on record structure (drug label, adverse event,
        recall).  The SemanticChunker from rag_engine then splits each text
        block into 512-token chunks.  Chunks are indexed into both a BM25
        sparse index and an HNSW dense index, then wired into a HybridRetriever
        with RRF fusion stored on the instance for use in the retrieve node.

        Parameters
        ----------
        state : FDAAgentState
            Current pipeline state.

        Returns
        -------
        dict
            ``{"chunks_ready": True}`` on success.
            ``{"chunks_ready": False, "error": str}`` on any failure.
        """
        node_start = time.time()
        self._sparse_idx = None
        self._dense_idx = None
        self._hybrid = None
        records = state.get("fetched_records", [])

        if not records:
            logger.warning("[chunk_and_index] no records to process")
            return {"chunks_ready": False}

        logger.info("[chunk_and_index] start — %d records", len(records))

        try:
            # Serialise each record to readable text
            texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []

            for rec in records:
                text, meta = self._record_to_text(rec)
                if text.strip():
                    texts.append(text)
                    metadatas.append(meta)

            logger.info(
                "[chunk_and_index] text extraction done — %d/%d records produced text (%.1fs)",
                len(texts), len(records), time.time() - node_start,
            )

            if not texts:
                return {
                    "chunks_ready": False,
                    "error": "No text extracted from FDA records",
                }

            # Chunk all texts through SemanticChunker
            chunk_start = time.time()
            all_chunk_texts: List[str] = []
            all_chunk_metas: List[Dict[str, Any]] = []

            for text, meta in zip(texts, metadatas):
                chunks = self.chunker.chunk(text)
                for chunk in chunks:
                    all_chunk_texts.append(chunk.text)
                    all_chunk_metas.append(meta)

            logger.info(
                "[chunk_and_index] chunking done — %d chunks from %d texts in %.1fs",
                len(all_chunk_texts), len(texts), time.time() - chunk_start,
            )

            if not all_chunk_texts:
                return {
                    "chunks_ready": False,
                    "error": "SemanticChunker produced no chunks",
                }

            # BM25 sparse index (fast — no network calls)
            bm25_start = time.time()
            from rag_engine.sparse_index import BM25Index  # type: ignore
            self._sparse_idx = BM25Index()
            self._sparse_idx.add_documents(all_chunk_texts, all_chunk_metas)
            logger.info(
                "[chunk_and_index] BM25 index built — %d docs in %.1fs",
                len(all_chunk_texts), time.time() - bm25_start,
            )

            # HNSW dense index (slow — calls embedder which may be local Ollama)
            embed_start = time.time()
            logger.info(
                "[chunk_and_index] embedding %d chunks — this may take several minutes "
                "with a local model...",
                len(all_chunk_texts),
            )
            try:
                embeddings = self.embedder.embed_batch(all_chunk_texts)
            except Exception as exc:
                embeddings = []
                logger.warning(
                    "[chunk_and_index] dense indexing unavailable; "
                    "continuing with BM25: %s",
                    exc,
                )
            logger.info(
                "[chunk_and_index] embedding done — %d vectors in %.1fs",
                len(embeddings) if embeddings else 0,
                time.time() - embed_start,
            )

            if embeddings:
                dense_start = time.time()
                from rag_engine.dense_index import DenseIndex  # type: ignore
                dim = len(embeddings[0])
                self._dense_idx = DenseIndex(dimension=dim)
                self._dense_idx.add_documents(
                    embeddings, all_chunk_texts, all_chunk_metas
                )
                logger.info(
                    "[chunk_and_index] HNSW index built — dim=%d in %.1fs",
                    dim, time.time() - dense_start,
                )
            else:
                logger.warning(
                    "[chunk_and_index] embedder returned no vectors; dense index skipped"
                )

            # Hybrid retriever with RRF fusion
            from rag_engine.hybrid_retriever import HybridRetriever  # type: ignore
            self._hybrid = HybridRetriever(
                dense_index=self._dense_idx,
                sparse_index=self._sparse_idx,
                embedder=self.embedder,
            )

            logger.info(
                "[chunk_and_index] complete — total %.1fs", time.time() - node_start
            )
            return {"chunks_ready": True}

        except Exception as exc:
            logger.error(
                "[chunk_and_index] failed: %s (%.1fs)", exc, time.time() - node_start,
                exc_info=True,
            )
            return {"chunks_ready": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Node 4: retrieve
    # ------------------------------------------------------------------ #

    def _retrieve_node(self, state: FDAAgentState) -> Dict[str, Any]:
        """Retrieve chunks using hybrid BM25 + HNSW retriever with RRF fusion.

        Retrieves top_k * 3 candidates to give the cross-encoder enough
        material for effective reranking.  Gracefully returns empty results
        if chunk_and_index failed (chunks_ready=False).

        Parameters
        ----------
        state : FDAAgentState
            Current pipeline state.

        Returns
        -------
        dict
            ``{"retrieval_results": list, "retrieval_time_sec": float}``
            On failure: also sets ``"error": str``.
        """
        if not state.get("chunks_ready"):
            return {
                "retrieval_results": [],
                "retrieval_time_sec": 0.0,
                "error": state.get("error", "No index available for retrieval"),
            }

        query = state.get("expanded_query") or state["input_query"]
        context = state.get("context", {})
        top_k = context.get("top_k", self.default_top_k)

        start = time.time()
        try:
            results = self._hybrid.retrieve(query, top_k=top_k * 3)
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
                "Hybrid retrieval returned %d FDA results in %.2fs",
                len(retrieval_dicts),
                elapsed,
            )
            return {
                "retrieval_results": retrieval_dicts,
                "retrieval_time_sec": elapsed,
            }

        except Exception as exc:
            elapsed = time.time() - start
            logger.error("FDA retrieval failed: %s", exc, exc_info=True)
            return {
                "retrieval_results": [],
                "retrieval_time_sec": elapsed,
                "error": str(exc),
            }

    # ------------------------------------------------------------------ #
    # Node 5: rerank
    # ------------------------------------------------------------------ #

    def _rerank_node(self, state: FDAAgentState) -> Dict[str, Any]:
        """Rerank retrieval candidates with the shared cross-encoder.

        Uses the cross-encoder reranker from SubAgentGraph.reranker (shared
        across all domain agents; defaults to MedCPT or configured model).
        Falls back to truncated original order on any reranking failure to
        ensure the pipeline never stalls.

        Parameters
        ----------
        state : FDAAgentState
            Current pipeline state.

        Returns
        -------
        dict
            ``{"reranked_results": list}``
        """
        results = state.get("retrieval_results", [])
        query = state.get("expanded_query") or state["input_query"]
        context = state.get("context", {})
        top_k = context.get("top_k", self.default_top_k)

        if not results:
            return {"reranked_results": []}

        try:
            from agents.base import _RetrievalDoc  # type: ignore
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
                "FDA reranking failed, keeping original order: %s", exc
            )
            return {"reranked_results": results[:top_k]}

    # ------------------------------------------------------------------ #
    # Node 6: synthesise
    # ------------------------------------------------------------------ #

    def _synthesise_node(self, state: FDAAgentState) -> Dict[str, Any]:
        """Synthesise a regulatory answer from reranked results via LLMClient.

        Formats reranked chunks into a numbered source block and injects it
        into the ``prompts/fda/synthesis.yaml`` template.  The prompt
        instructs the LLM to produce a structured 6-section regulatory
        analysis with inline FDA record ID citations.

        Falls back to a hardcoded template if the YAML prompt is unavailable
        (graceful degradation for first-run without prompt files).

        Parameters
        ----------
        state : FDAAgentState
            Current pipeline state.

        Returns
        -------
        dict
            answer, citations, confidence, model_used, execution_time_sec.
            On failure: answer contains error description, confidence = 0.0.
        """
        start = time.time()
        query = state["input_query"]
        results = state.get("reranked_results") or state.get(
            "retrieval_results", []
        )

        logger.info(
            "[synthesise] start — %d results, query: %s", len(results), query[:80]
        )

        if not results:
            return {
                "answer": "No relevant FDA regulatory records found for this query.",
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
                    "Answer the regulatory question using only the FDA records below.\n\n"
                    "SOURCES:\n{sources}\n\nQUESTION:\n{query}\n\nANSWER:\n"
                )

            prompt_text = template.format(query=query, sources=sources_text)
            est_tokens = len(prompt_text) // 4
            logger.info(
                "[synthesise] calling LLM (%s) — prompt ~%d tokens, max_tokens=1400 ...",
                self.llm.default_model, est_tokens,
            )

            answer = self.llm.chat(
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.2,
                max_tokens=1400,
            )

            citations = self._extract_citations(results)
            confidence = self._calculate_confidence(
                answer=answer,
                citations=citations,
                results=results,
                elapsed=time.time() - start,
            )

            logger.info(
                "[synthesise] done — %d chars, %d citations, conf=%.2f, %.1fs",
                len(answer), len(citations), confidence, time.time() - start,
            )

            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "model_used": self.llm.default_model,
                "execution_time_sec": time.time() - start,
            }

        except Exception as exc:
            logger.error("[synthesise] failed: %s (%.1fs)", exc, time.time() - start, exc_info=True)
            return {
                "answer": f"Error synthesising FDA regulatory response: {exc}",
                "citations": [],
                "confidence": 0.0,
                "model_used": "",
                "error": str(exc),
                "execution_time_sec": time.time() - start,
            }

    # ------------------------------------------------------------------ #
    # Confidence heuristic — ported from legacy FdaRAGModule._generate()
    # ------------------------------------------------------------------ #

    def _calculate_confidence(
        self,
        answer: str,
        citations: List[str],
        results: List[Dict[str, Any]],
        elapsed: float,
    ) -> float:
        """Heuristic confidence score for FDA regulatory answers.

        Starts at base_confidence (0.75) and applies incremental adjustments
        for answer completeness, citation density, and processing latency.
        FDA base is set lower than clinical_trials (0.80) to reflect the
        higher variability of openFDA data across heterogeneous endpoints.

        Parameters
        ----------
        answer : str
            Generated regulatory answer text.
        citations : list[str]
            Extracted FDA record IDs (recall_number, safetyreportid, etc.).
        results : list[dict]
            Reranked retrieval results (used for future score-based extensions).
        elapsed : float
            Synthesis wall-clock time in seconds.

        Returns
        -------
        float
            Confidence score clamped to [0.0, 1.0].
        """
        conf = self.base_confidence

        # Answer completeness
        if len(answer) > 500:
            conf += 0.05
        elif len(answer) < 100:
            conf -= 0.1

        # Citation density
        if len(citations) > 3:
            conf += 0.1
        elif len(citations) == 0:
            conf -= 0.15

        # Latency heuristic (very fast = likely insufficient context)
        if elapsed < 2.0:
            conf -= 0.05
        elif elapsed > 10.0:
            conf -= 0.1

        return max(0.0, min(1.0, conf))

    # ------------------------------------------------------------------ #
    # Citation extraction — overrides base to use FDA-specific identifiers
    # ------------------------------------------------------------------ #

    def _extract_citations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract rich, human-readable FDA citations from retrieval result metadata.

        Formats citations by record type so each citation carries meaningful
        information for the clinician reader and the orchestrator's coherence
        evaluator.  Raw integer IDs like ``"22"`` or ``"unknown"`` provide no
        information content and will degrade orchestration-level scoring.

        Citation format by record type
        --------------------------------
        - **drug_label**: ``FDA Drug Label | <drug_name> | SPL:<id[:8]>``
        - **adverse_event**: ``FAERS Report | <drug_name> | SAFETYREPORTID:<id>``
        - **recall**: ``FDA Recall | <product[:50]> | <recall_number>``
        - **unknown**: ``openFDA | <source> | <id[:12]>``

        Overrides the base class ``_extract_citations`` which uses NCT IDs
        (ClinicalTrials.gov-specific and meaningless for FDA records).

        Parameters
        ----------
        results : list[dict]
            Reranked retrieval results with ``metadata`` dicts containing
            at minimum: ``record_type``, ``drug_name``, ``record_id``,
            ``source`` (all set by ``_record_to_text``).

        Returns
        -------
        list[str]
            Deduplicated list of formatted citation strings in retrieval order.

        Examples
        --------
        >>> citations = agent._extract_citations(reranked_results)
        >>> citations[0]
        'FDA Drug Label | metformin hydrochloride | SPL:4eb6a025'
        >>> citations[1]
        'FAERS Report | metformin | SAFETYREPORTID:10015976'
        """
        seen: set = set()
        citations: List[str] = []

        for r in results:
            meta = r.get("metadata", {})
            record_type = meta.get("record_type", "unknown")
            drug_name = (meta.get("drug_name") or "").strip()
            record_id = str(meta.get("record_id") or r.get("doc_id") or "unknown")

            if record_type == "drug_label":
                spl_short = record_id[:8] if len(record_id) > 8 else record_id
                citation = f"FDA Drug Label | {drug_name} | SPL:{spl_short}"

            elif record_type == "adverse_event":
                safetyrpt = str(
                    meta.get("safetyreportid") or record_id
                )
                citation = f"FAERS Report | {drug_name} | SAFETYREPORTID:{safetyrpt}"

            elif record_type == "recall":
                recall_num = str(meta.get("recall_number") or record_id)
                product = drug_name[:50] if drug_name else recall_num
                citation = f"FDA Recall | {product} | {recall_num}"

            else:
                source = meta.get("source", "openFDA")
                id_short = record_id[:12]
                citation = f"openFDA | {source} | {id_short}"

            if citation not in seen:
                seen.add(citation)
                citations.append(citation)

        return citations

    # ------------------------------------------------------------------ #
    # FDA record → readable text (type-dispatched)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _record_to_text(record: Dict[str, Any]) -> tuple:
        """Convert a raw openFDA record to readable text and metadata.

        Type-dispatches on record structure (not on a type string field) to
        handle the three primary openFDA record schemas.  Extraction is
        conservative: missing fields are silently skipped rather than raising.

        Record types handled
        --------------------
        - **Drug label** (contains ``indications_and_usage``):
          Extracts brand name, generic name, manufacturer, indications,
          dosage, warnings, adverse reactions, contraindications, drug
          interactions.  Truncates each section to 2000 chars.

        - **Adverse event** (contains ``safetyreportid`` or ``patient``):
          Extracts report ID, seriousness classification, implicated drugs,
          and MedDRA reaction terms.

        - **Recall / enforcement** (contains ``reason_for_recall`` or
          ``product_description``):
          Extracts recall number, product description, reason, corrective
          action, classification, and status.

        - **Fallback**: Compact JSON dump (first 3000 chars) with SHA-256
          hash ID.

        Parameters
        ----------
        record : dict
            Raw openFDA API record dict.

        Returns
        -------
        tuple[str, dict]
            ``(readable_text, metadata)`` where metadata always contains
            ``record_id``, ``record_type``, ``drug_name``, and ``source``.

        Examples
        --------
        >>> text, meta = FDAAgentGraph._record_to_text(drug_label_record)
        >>> meta["record_type"]
        'drug_label'
        >>> meta["source"]
        'openFDA/drug/label'
        """
        openfda = record.get("openfda", {})

        brand_name = (openfda.get("brand_name") or [""])[0]
        generic_name = (openfda.get("generic_name") or [""])[0]
        manufacturer = (openfda.get("manufacturer_name") or [""])[0]
        drug_name = brand_name or generic_name or "Unknown drug"

        # ---- Drug label ----
        if "indications_and_usage" in record:
            spl_id = (openfda.get("spl_id") or ["unknown"])[0]
            parts: List[str] = [
                f"Drug: {drug_name}",
            ]
            if generic_name:
                parts.append(f"Generic name: {generic_name}")
            if manufacturer:
                parts.append(f"Manufacturer: {manufacturer}")

            # Section key → display label pairs
            label_sections = [
                ("indications_and_usage",        "Indications and Usage"),
                ("dosage_and_administration",     "Dosage and Administration"),
                ("warnings_and_cautions",         "Warnings and Cautions"),
                ("warnings",                      "Warnings"),
                ("adverse_reactions",             "Adverse Reactions"),
                ("contraindications",             "Contraindications"),
                ("drug_interactions",             "Drug Interactions"),
                ("boxed_warning",                 "Boxed Warning"),
            ]
            for key, label in label_sections:
                val = record.get(key)
                if val:
                    text_block = val[0] if isinstance(val, list) else str(val)
                    text_block = text_block.strip()
                    if text_block:
                        parts.append(f"{label}: {text_block[:2000]}")

            text = "\n".join(p for p in parts if p)
            metadata = {
                "record_id": spl_id,
                "record_type": "drug_label",
                "drug_name": drug_name,
                "source": "openFDA/drug/label",
            }
            return text, metadata

        # ---- Adverse event ----
        if "safetyreportid" in record or "patient" in record:
            report_id = str(record.get("safetyreportid", "unknown"))
            patient = record.get("patient", {})

            reactions = [
                rxn.get("reactionmeddrapt", "")
                for rxn in patient.get("reaction", [])
                if rxn.get("reactionmeddrapt")
            ]

            drugs_implicated = []
            for drg in patient.get("drug", []):
                name = drg.get("medicinalproduct") or (
                    drg.get("openfda", {}).get("brand_name") or [""]
                )[0]
                if name:
                    drugs_implicated.append(name)

            serious_map = {"1": "Serious", "2": "Not serious"}
            seriousness = serious_map.get(
                str(record.get("serious", "")), "Unknown"
            )
            serious_criteria = record.get("seriousnessdeath", "")

            parts = [
                f"Adverse Event Report: {report_id}",
                f"Seriousness: {seriousness}",
            ]
            if serious_criteria == "1":
                parts.append("Outcome: Death reported")
            if drugs_implicated:
                parts.append(f"Drugs implicated: {', '.join(drugs_implicated)}")
            if reactions:
                parts.append(f"Reactions (MedDRA): {', '.join(reactions)}")

            text = "\n".join(p for p in parts if p)
            metadata = {
                "record_id": report_id,
                "safetyreportid": report_id,
                "record_type": "adverse_event",
                "drug_name": drugs_implicated[0] if drugs_implicated else drug_name,
                "source": "openFDA/drug/event",
            }
            return text, metadata

        # ---- Recall / enforcement ----
        if "reason_for_recall" in record or "product_description" in record:
            recall_number = (
                record.get("recall_number")
                or record.get("enforcement_report_number")
                or record.get("report_number")
                or "unknown"
            )
            parts = [
                f"Recall Number: {recall_number}",
                f"Product: {record.get('product_description', '')}",
                f"Recall Reason: {record.get('reason_for_recall', '')}",
                f"Corrective Action: {record.get('corrective_action', '')}",
                f"Classification: {record.get('classification', '')}",
                f"Status: {record.get('status', '')}",
                f"Firm: {record.get('recalling_firm', '')}",
            ]
            text = "\n".join(
                p for p in parts
                if p and not p.endswith(": ")
            )
            metadata = {
                "record_id": recall_number,
                "recall_number": recall_number,
                "record_type": "recall",
                "drug_name": record.get("product_description", "")[:80],
                "source": "openFDA/recall",
            }
            return text, metadata

        # ---- Fallback: compact JSON dump ----
        fallback_id = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()[:12]
        text = json.dumps(record, ensure_ascii=False)[:3000]
        metadata = {
            "record_id": fallback_id,
            "record_type": "unknown",
            "drug_name": drug_name,
            "source": "openFDA",
        }
        return text, metadata

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @serialized_invoke
    def invoke(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """Run the FDA regulatory sub-agent end-to-end.

        Initialises the full 6-node state, invokes the compiled LangGraph
        graph, and wraps the result in the standardised AgentOutput contract
        consumed by the orchestrator.

        Parameters
        ----------
        query : str
            Medical or regulatory research question.
        context : dict, optional
            Override parameters:
            - ``top_k`` (int): reranked results for synthesis (default 10).
            - ``max_records`` (int): openFDA records cap (default 200).

        Returns
        -------
        AgentOutput
            Standardised output: answer, citations, confidence, sources,
            model_used, domain, execution_time_sec, error.

        Examples
        --------
        >>> output = agent.invoke("What recalls exist for contaminated heparin?")
        >>> output.confidence
        0.8
        >>> output.citations[0]
        'Z-1234-2024'
        """
        start = time.time()
        initial_state: Dict[str, Any] = {
            "input_query": query,
            "context": context or {},
            "domain": self.domain,
            "expanded_query": "",
            "fetched_records": [],
            "fetch_meta": {},
            "chunks_ready": False,
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