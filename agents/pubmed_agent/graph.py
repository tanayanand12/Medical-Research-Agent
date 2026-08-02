"""
graph.py — Phase 7: PubMed sub-agent.

LangGraph subgraph that fetches live papers from NCBI PubMed,
chunks and indexes them on-the-fly, retrieves via hybrid BM25 + HNSW,
reranks with MedCPT cross-encoder, and synthesises AMA-cited answers.

Node topology (6 nodes):
    expand_query → fetch → chunk_and_index → retrieve → rerank → synthesise → END

Legacy behaviour ported
-----------------------
* QueryProcessor query expansion and URL construction strategy.
* PubMedClient parallel NCBI fetching with rate limiting.
* AnswerGenerator synthesis prompt and evidence-quality assessment.
* AMA citation formatting from utils/ama_formatter.py.

Changes from legacy
-------------------
* LLM calls routed through LLMClient (was hardcoded OpenAI).
* Retrieval via hybrid BM25 + HNSW with RRF fusion (was cosine-only).
* Reranking with ncbi/MedCPT-Cross-Encoder (was LLM reranker).
* Query expansion via prompts/pubmed/query_expansion.yaml.
* Synthesis via prompts/pubmed/synthesis.yaml.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph  # type: ignore

from agents.base import (  # type: ignore
    AgentOutput,
    SubAgentGraph,
    llm_deadline_kwargs,
    llm_telemetry_kwargs,
    load_prompt,
    serialized_invoke,
)
from evaluation_core import (
    RuntimeDeadlineExceeded,
    safe_error_type,
    stable_document_id,
)

from utils.perf import timed_node, time_block  

logger = logging.getLogger(__name__)


# ======================================================================
# State contract
# ======================================================================

from typing import TypedDict  # noqa: E402 (after langgraph import)


class PubMedState(TypedDict):
    # Input
    input_query: str
    context: Dict[str, Any]
    domain: str

    # Query expansion
    expanded_query: str

    # Fetch — papers stored as dicts (JSON-serializable for LangGraph)
    fetched_papers: Dict[str, Dict[str, Any]]
    fetch_meta: Dict[str, Any]

    # Chunking / indexing
    chunks_ready: bool

    # Retrieval
    retrieval_results: List[Dict[str, Any]]
    retrieval_time_sec: float

    # Reranking
    reranked_results: List[Dict[str, Any]]

    # Synthesis
    answer: str
    citations: List[str]   # AMA-formatted strings
    confidence: float
    model_used: str
    synthesis_context: List[Dict[str, Any]]
    stage_latency_sec: Dict[str, float]
    token_usage: Dict[str, int]
    cost_breakdown_usd: Dict[str, float]
    attempt_events: List[Dict[str, Any]]

    # Metadata
    error: Optional[str]
    execution_time_sec: float


# ======================================================================
# Sub-agent graph
# ======================================================================


class PubMedAgentGraph(SubAgentGraph):
    """PubMed evidence retrieval and synthesis sub-agent.

    Fetches live papers from NCBI, builds ephemeral BM25 + HNSW indexes,
    runs hybrid retrieval with RRF fusion, reranks with MedCPT, and
    synthesises AMA-cited answers via LLMClient.
    """

    domain = "pubmed"
    default_top_k = 8
    base_confidence = 0.75
    MAX_PAPERS = 50

    summary = (
        "PubMed RAG sub-agent — fetches live NCBI papers, builds ephemeral "
        "BM25 + HNSW indexes, hybrid retrieval with RRF, MedCPT reranking, "
        "AMA citations."
    )

    def __init__(self) -> None:
        super().__init__()
        self._fetcher = None
        self._chunker = None
        self._embedder = None
        self._sparse_idx = None
        self._dense_idx = None
        self._hybrid = None

    # ------------------------------------------------------------------
    # Lazy component accessors
    # ------------------------------------------------------------------

    @property
    def fetcher(self):
        if self._fetcher is None:
            from agents.pubmed_agent.data_fetcher import PubMedFetcher
            self._fetcher = PubMedFetcher(llm_client=self.llm)
        return self._fetcher

    @property
    def chunker(self):
        if self._chunker is None:
            from rag_engine.chunker import SemanticChunker
            self._chunker = SemanticChunker(
                max_chunk_tokens=512,
                min_chunk_tokens=30,
            )
        return self._chunker

    @property
    def embedder(self):
        if self._embedder is None:
            from rag_engine.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        sg = StateGraph(PubMedState)

        sg.add_node("expand_query",    self._expand_query_node)
        sg.add_node("fetch",           self._fetch_node)
        sg.add_node("chunk_and_index", self._chunk_and_index_node)
        sg.add_node("retrieve",        self._retrieve_node)
        sg.add_node("rerank",          self._rerank_node)
        sg.add_node("synthesise",      self._synthesise_node)

        sg.set_entry_point("expand_query")
        sg.add_edge("expand_query",    "fetch")
        sg.add_edge("fetch",           "chunk_and_index")
        sg.add_edge("chunk_and_index", "retrieve")
        sg.add_edge("retrieve",        "rerank")
        sg.add_edge("rerank",          "synthesise")
        sg.add_edge("synthesise",      END)

        return sg.compile()

    # ------------------------------------------------------------------
    # Node: expand_query
    # ------------------------------------------------------------------

    @timed_node("expand_query")
    def _expand_query_node(self, state: PubMedState) -> Dict[str, Any]:
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
                "Query expansion failed error_type=%s",
                safe_error_type(exc),
            )
            return {"expanded_query": query}

    # ------------------------------------------------------------------
    # Node: fetch
    # ------------------------------------------------------------------

    @timed_node("fetch")
    def _fetch_node(self, state: PubMedState) -> Dict[str, Any]:
        query      = state.get("expanded_query") or state["input_query"]
        context    = state.get("context", {})
        max_papers = context.get("max_papers", self.MAX_PAPERS)

        try:
            with time_block("ncbi_fetch", max_papers=max_papers):
                result = self.fetcher.analyze_user_query(
                    query,
                    max_papers=max_papers,
                    llm_kwargs={
                        **llm_deadline_kwargs(state),
                        **llm_telemetry_kwargs(
                            state, "pubmed_query_extraction"
                        ),
                    },
                )

            if not result.get("success"):
                logger.warning(
                    "PubMed fetch failed error_type=%s",
                    result.get("error_type") or "FetcherError",
                )
                return {
                    "fetched_papers": {},
                    "fetch_meta":     result.get("query_analysis", {}),
                    "error":          "pubmed_fetch_failed",
                }

            # Serialize Paper objects → dicts (LangGraph state must be JSON-serializable)
            papers_dict: Dict[str, Dict[str, Any]] = {
                pmid: paper.model_dump()
                for pmid, paper in result["papers"].items()
            }

            logger.info(
                "Fetched %d papers (attempted %d PMIDs)",
                len(papers_dict),
                len(result.get("pmids_fetched", [])),
            )
            return {
                "fetched_papers": papers_dict,
                "fetch_meta":     result.get("query_analysis", {}),
            }
        except Exception as exc:
            logger.error(
                "Fetch node failed error_type=%s",
                safe_error_type(exc),
            )
            return {
                "fetched_papers": {},
                "fetch_meta":     {},
                "error":          f"fetch_failed:{safe_error_type(exc)}",
            }

    # ------------------------------------------------------------------
    # Node: chunk_and_index
    # ------------------------------------------------------------------

    # @timed_node("chunk_and_index")
    # def _chunk_and_index_node(self, state: PubMedState) -> Dict[str, Any]:
    #     papers = state.get("fetched_papers", {})
    #     if not papers:
    #         return {"chunks_ready": False}

    #     try:
    #         all_chunk_texts: List[str] = []
    #         all_chunk_metas: List[Dict[str, Any]] = []

    #         for paper_dict in papers.values():
    #             text, meta = self._paper_to_text(paper_dict)
    #             if not text.strip():
    #                 continue
    #             chunks = self.chunker.chunk(text)
    #             for chunk in chunks:
    #                 all_chunk_texts.append(chunk.text)
    #                 all_chunk_metas.append(meta)

    #         if not all_chunk_texts:
    #             return {"chunks_ready": False, "error": "No text extracted from papers"}

    #         logger.info(
    #             "Chunked %d papers into %d chunks",
    #             len(papers),
    #             len(all_chunk_texts),
    #         )

    #         # Build BM25 sparse index
    #         from rag_engine.sparse_index import BM25Index
    #         self._sparse_idx = BM25Index()
    #         self._sparse_idx.add_documents(all_chunk_texts, all_chunk_metas)

    #         # Build HNSW dense index
    #         embeddings = self.embedder.embed_batch(all_chunk_texts)
    #         if not embeddings:
    #             return {"chunks_ready": False, "error": "Embedding returned empty"}

    #         from rag_engine.dense_index import DenseIndex
    #         dim = len(embeddings[0])
    #         self._dense_idx = DenseIndex(dimension=dim)
    #         self._dense_idx.add_documents(embeddings, all_chunk_texts, all_chunk_metas)

    #         # Build hybrid retriever
    #         from rag_engine.hybrid_retriever import HybridRetriever
    #         self._hybrid = HybridRetriever(
    #             dense_index=self._dense_idx,
    #             sparse_index=self._sparse_idx,
    #             embedder=self.embedder,
    #         )

    #         return {"chunks_ready": True}

    #     except Exception as exc:
    #         logger.error("chunk_and_index failed: %s", exc, exc_info=True)
    #         return {"chunks_ready": False, "error": str(exc)}

    @timed_node("chunk_and_index")
    def _chunk_and_index_node(self, state: PubMedState) -> Dict[str, Any]:
        self._sparse_idx = None
        self._dense_idx = None
        self._hybrid = None
        papers = state.get("fetched_papers", {})
        if not papers:
            return {"chunks_ready": False}

        try:
            all_chunk_texts: List[str] = []
            all_chunk_metas: List[Dict[str, Any]] = []

            with time_block("chunk_papers", n_papers=len(papers)):
                for paper_dict in papers.values():
                    text, meta = self._paper_to_text(paper_dict)
                    if not text.strip():
                        continue
                    chunks = self.chunker.chunk(text)
                    for chunk in chunks:
                        all_chunk_texts.append(chunk.text)
                        all_chunk_metas.append(meta)

            if not all_chunk_texts:
                return {"chunks_ready": False, "error": "No text extracted from papers"}

            logger.info(
                "Chunked %d papers into %d chunks",
                len(papers), len(all_chunk_texts),
            )

            # Build BM25 sparse index
            from rag_engine.sparse_index import BM25Index
            with time_block("build_bm25", n_chunks=len(all_chunk_texts)):
                self._sparse_idx = BM25Index()
                self._sparse_idx.add_documents(all_chunk_texts, all_chunk_metas)

            # Build HNSW dense index
            embeddings = []
            try:
                with time_block("embed_chunks", n_chunks=len(all_chunk_texts)):
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
                logger.warning(
                    "Dense indexing unavailable; continuing with BM25 "
                    "error_type=%s",
                    safe_error_type(exc),
                )

            if embeddings:
                from rag_engine.dense_index import DenseIndex
                with time_block(
                    "build_hnsw", n_vecs=len(embeddings), dim=len(embeddings[0])
                ):
                    dim = len(embeddings[0])
                    self._dense_idx = DenseIndex(dimension=dim)
                    self._dense_idx.add_documents(
                        embeddings, all_chunk_texts, all_chunk_metas
                    )

            from rag_engine.hybrid_retriever import HybridRetriever
            with time_block("build_hybrid"):
                self._hybrid = HybridRetriever(
                    dense_index=self._dense_idx,
                    sparse_index=self._sparse_idx,
                    embedder=self.embedder,
                )

            return {"chunks_ready": True}
        except Exception as exc:
                logger.error(
                    "chunk_and_index failed error_type=%s",
                    safe_error_type(exc),
                )
                return {
                    "chunks_ready": False,
                    "error": f"chunk_index_failed:{safe_error_type(exc)}",
                }

    # ------------------------------------------------------------------
    # Node: retrieve
    # ------------------------------------------------------------------

    @timed_node("retrieve")
    def _retrieve_node(self, state: PubMedState) -> Dict[str, Any]:
        if not state.get("chunks_ready"):
            return {
                "retrieval_results":  [],
                "retrieval_time_sec": 0.0,
                "error": state.get("error", "No index available"),
            }

        query   = state.get("expanded_query") or state["input_query"]
        context = state.get("context", {})
        top_k   = context.get("top_k", self.default_top_k)

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
                    "text":     r.text,
                    "score":    r.score,
                    "doc_id":   r.doc_id,
                    "metadata": r.metadata,
                }
                for r in results
            ]
            logger.info(
                "Hybrid retrieval: %d results in %.2fs", len(retrieval_dicts), elapsed
            )
            return {
                "retrieval_results":  retrieval_dicts,
                "retrieval_time_sec": elapsed,
            }
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(
                "Retrieval failed error_type=%s",
                safe_error_type(exc),
            )
            return {
                "retrieval_results":  [],
                "retrieval_time_sec": elapsed,
                "error": f"retrieval_failed:{safe_error_type(exc)}",
            }

    # ------------------------------------------------------------------
    # Node: rerank
    # ------------------------------------------------------------------

    # @timed_node("rerank")
    # def _rerank_node(self, state: PubMedState) -> Dict[str, Any]:
    #     results = state.get("retrieval_results", [])
    #     query   = state.get("expanded_query") or state["input_query"]
    #     top_k   = state.get("context", {}).get("top_k", self.default_top_k)

    #     if not results:
    #         return {"reranked_results": []}

    #     try:
    #         from agents.base import _RetrievalDoc
    #         docs     = [_RetrievalDoc(r) for r in results]
    #         reranked = self.reranker.rerank(query, docs, top_k=top_k)

    @timed_node("rerank")
    def _rerank_node(self, state: PubMedState) -> Dict[str, Any]:
        results = state.get("retrieval_results", [])
        query   = state.get("expanded_query") or state["input_query"]
        top_k   = state.get("context", {}).get("top_k", self.default_top_k)

        if not results:
            return {"reranked_results": []}

        try:
            from agents.base import _RetrievalDoc
            docs = [_RetrievalDoc(r) for r in results]
            with time_block("medcpt_rerank", n_pairs=len(docs)):
                reranked = self.reranker.rerank(query, docs, top_k=top_k)
            
            return {
                "reranked_results": [
                    {
                        "text":          r.text,
                        "score":         r.score,
                        "doc_id":        r.doc_id,
                        "metadata":      r.metadata,
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

    # ------------------------------------------------------------------
    # Node: synthesise
    # ------------------------------------------------------------------

    # @timed_node("synthesise")
    # def _synthesise_node(self, state: PubMedState) -> Dict[str, Any]:
    #     start   = time.time()
    #     query   = state["input_query"]
    #     results = state.get("reranked_results") or state.get("retrieval_results", [])
    #     papers  = state.get("fetched_papers", {})

    #     if not results:
    #         return {
    #             "answer":             "No relevant PubMed papers found for this query.",
    #             "citations":          [],
    #             "confidence":         0.0,
    #             "model_used":         self.llm.default_model,
    #             "execution_time_sec": time.time() - start,
    #         }

    #     try:
    #         citations    = self._extract_citations(results, papers)
    #         sources_text = self._format_sources(results, citations)

    #         template = load_prompt(self.domain, "synthesis")
    #         if not template:
    #             template = (
    #                 "Answer the question using the provided sources. "
    #                 "Cite each claim with [n].\n\n"
    #                 "SOURCES:\n{sources}\n\nQUESTION:\n{query}\n\nANSWER:\n"
    #             )

    #         prompt_text = template.format(query=query, sources=sources_text)
    #         answer = self.llm.chat(
    #             messages=[{"role": "user", "content": prompt_text}],
    #             temperature=0.2,
    #             max_tokens=1500,
    #         )

    @timed_node("synthesise")
    def _synthesise_node(self, state: PubMedState) -> Dict[str, Any]:
        start   = time.time()
        query   = state["input_query"]
        results = state.get("reranked_results") or state.get("retrieval_results", [])
        papers  = state.get("fetched_papers", {})

        if not results:
            return {
                "answer": "No relevant PubMed papers found for this query.",
                "citations": [],
                "confidence": 0.0,
                "model_used": self.llm.default_model,
                "synthesis_context": [],
                "execution_time_sec": time.time() - start,
            }

        try:
            with time_block("build_citations_and_sources", n_results=len(results)):
                citations    = self._extract_citations(results, papers)
                sources_text = self._format_sources(results, citations)
                synthesis_context = self._build_synthesis_context(
                    results, citations
                )

            template = load_prompt(self.domain, "synthesis")
            if not template:
                template = (
                    "Answer the medical research question using only the "
                    "provided PubMed sources. Cite supported claims with [n]. "
                    "State when the evidence is insufficient.\n\n"
                    "SOURCES:\n{sources}\n\n"
                    "QUESTION:\n{query}\n\nANSWER:\n"
                )

            prompt_text = template.format(query=query, sources=sources_text)
            with time_block("llm_synthesis"):
                from runtime_verification import call_llm_with_metadata

                call_result = call_llm_with_metadata(
                    self.llm,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.2,
                    max_tokens=1500,
                    **llm_deadline_kwargs(state),
                    **llm_telemetry_kwargs(state, "agent_synthesis"),
                )
                answer = call_result.text
            

            confidence = self._calculate_confidence(
                answer=answer,
                citations=citations,
                results=results,
                elapsed=time.time() - start,
            )

            return {
                "answer":             answer,
                "citations":          citations,
                "confidence":         confidence,
                "synthesis_context":  synthesis_context,
                "execution_time_sec": time.time() - start,
                **self._generation_telemetry(state, call_result),
            }
        except Exception as exc:
            logger.error(
                "Synthesis failed error_type=%s",
                safe_error_type(exc),
            )
            return {
                "answer":             "Unable to synthesise the PubMed evidence.",
                "citations":          [],
                "confidence":         0.0,
                "synthesis_context":  [],
                "error":              f"synthesis_failed:{safe_error_type(exc)}",
                "execution_time_sec": time.time() - start,
                **self._failure_telemetry(
                    state,
                    exc,
                    stage="agent_synthesis",
                    latency_sec=time.time() - start,
                ),
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _paper_to_text(paper: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Convert a serialised Paper dict to readable text + citation metadata.

        Returns
        -------
        tuple[str, dict]
            (text, metadata)  — metadata contains all citation fields.
        """
        pmid     = paper.get("pmid", "")
        title    = paper.get("title", "")
        authors  = paper.get("authors", [])
        journal  = paper.get("journal", "")
        year     = paper.get("year", "")
        doi      = paper.get("doi")
        abstract = paper.get("abstract", "")
        full_text = paper.get("full_text")

        if not title and not abstract:
            return "", {}

        parts = [
            f"TITLE: {title}",
            f"AUTHORS: {', '.join(authors[:5])}",
            f"JOURNAL: {journal} ({year})",
            f"PMID: {pmid}",
            f"\nABSTRACT:\n{abstract}",
        ]
        if full_text:
            parts.append(f"\nFULL TEXT (excerpt):\n{full_text[:3000]}")

        metadata = {
            "pmid":    pmid,
            "title":   title,
            "authors": authors,
            "journal": journal,
            "year":    year,
            "doi":     doi,
            "source":  "PubMed",
        }
        return "\n".join(parts), metadata

    @staticmethod
    def _format_author_ama(name: str) -> str:
        """Format single author name in AMA style: LastName Initials."""
        name = name.strip()
        if not name:
            return ""
        if ", " in name:
            parts = name.split(", ", 1)
            initials = "".join(p[0].upper() for p in parts[1].split() if p)
            return f"{parts[0]} {initials}"
        parts = name.split()
        if len(parts) == 1:
            return parts[0]
        initials = "".join(p[0].upper() for p in parts[:-1] if p)
        return f"{parts[-1]} {initials}"

    def _extract_citations(
        self,
        results: List[Dict[str, Any]],
        papers: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Build AMA-formatted citation strings for unique PMIDs in results.

        Ported from legacy utils/ama_formatter.format_paper_citation_ama().
        """
        seen_pmids: List[str] = []
        for r in results:
            pmid = r.get("metadata", {}).get("pmid", "")
            if pmid and pmid not in seen_pmids:
                seen_pmids.append(pmid)

        citations: List[str] = []
        for idx, pmid in enumerate(seen_pmids, start=1):
            paper = papers.get(pmid, {})
            if not paper:
                # Fallback from chunk metadata
                meta = next(
                    (r.get("metadata", {}) for r in results
                     if r.get("metadata", {}).get("pmid") == pmid),
                    {},
                )
                paper = meta

            authors = paper.get("authors", [])
            # AMA: up to 6 authors; >6 → first 3 + "et al"
            if len(authors) > 6:
                author_str = ", ".join(
                    self._format_author_ama(a) for a in authors[:3]
                ) + ", et al"
            else:
                author_str = ", ".join(
                    self._format_author_ama(a) for a in authors
                )

            title   = paper.get("title", "Unknown title").rstrip(".")
            journal = paper.get("journal", "Unknown journal")
            year    = paper.get("year", "n.d.")
            volume  = paper.get("volume") or ""
            issue   = paper.get("issue") or ""
            pages   = paper.get("pages") or ""
            doi     = paper.get("doi") or ""

            vol_issue_pages = ""
            if volume:
                vol_issue_pages = volume
                if issue:
                    vol_issue_pages += f"({issue})"
                if pages:
                    vol_issue_pages += f":{pages}"

            parts = [f"{author_str}. {title}. {journal}. {year}"]
            if vol_issue_pages:
                parts[0] += f";{vol_issue_pages}"
            if doi:
                doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
                parts.append(f"doi:{doi_clean}")
            parts.append(f"PMID: {pmid}")

            citations.append(f"[{idx}] " + ". ".join(parts))

        return citations

    @timed_node("format_sources")
    def _format_sources(
        self,
        results: List[Dict[str, Any]],
        citations: List[str],
    ) -> str:
        """Build numbered source block for the synthesis prompt."""
        # Build pmid → citation index map
        pmid_to_idx: Dict[str, int] = {}
        for i, cit in enumerate(citations, start=1):
            m = re.search(r"PMID:\s*(\d+)", cit)
            if m:
                pmid_to_idx[m.group(1)] = i

        parts: List[str] = []
        for r in results:
            pmid  = r.get("metadata", {}).get("pmid", "")
            idx   = pmid_to_idx.get(pmid, "?")
            score = r.get("score", 0.0)
            title = r.get("metadata", {}).get("title", "")
            year  = r.get("metadata", {}).get("year", "")
            text  = r.get("text", "")[:800]
            parts.append(
                f"[{idx}] {title} ({year}) | relevance: {score:.3f}\n{text}"
            )

        return "\n\n---\n\n".join(parts)

    def _build_synthesis_context(
        self,
        results: List[Dict[str, Any]],
        citations: List[str],
    ) -> List[Dict[str, Any]]:
        """Record the exact PubMed chunks and markers sent to the LLM."""
        pmid_to_marker: Dict[str, int] = {}
        for marker, citation in enumerate(citations, 1):
            match = re.search(r"PMID:\s*(\d+)", citation)
            if match:
                pmid_to_marker[match.group(1)] = marker

        manifest: List[Dict[str, Any]] = []
        for rank, result in enumerate(results, 1):
            text = str(result.get("text") or "")
            included_text = text[:800]
            pmid = str(result.get("metadata", {}).get("pmid") or "")
            manifest.append(
                {
                    "document_id": stable_document_id(
                        result, self.domain, rank
                    ),
                    "text": included_text,
                    "original_length": len(text),
                    "truncated": len(included_text) < len(text),
                    "citation_marker": pmid_to_marker.get(pmid, "?"),
                }
            )
        return manifest

    @timed_node("calculate_confidence")
    def _calculate_confidence(
        self,
        answer: str,
        citations: List[str],
        results: List[Dict[str, Any]],
        elapsed: float,
    ) -> float:
        conf = self.base_confidence  # 0.75

        # Answer length
        if len(answer) > 600:
            conf += 0.05
        elif len(answer) < 150:
            conf -= 0.10

        # Citation support
        if len(citations) >= 5:
            conf += 0.10
        elif len(citations) == 0:
            conf -= 0.20
        elif len(citations) <= 2:
            conf -= 0.05

        # Top reranked score signal
        if results:
            top_score = results[0].get("score", 0.5)
            if top_score > 0.8:
                conf += 0.05
            elif top_score < 0.4:
                conf -= 0.10

        # Processing time heuristic
        if elapsed < 2.0:
            conf -= 0.05
        elif elapsed > 15.0:
            conf -= 0.10

        return max(0.0, min(1.0, conf))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @serialized_invoke
    def invoke(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """Run the PubMed sub-agent end-to-end.

        Parameters
        ----------
        query : str
            Medical research question.
        context : dict, optional
            Accepts: top_k, max_papers, include_fulltext.

        Returns
        -------
        AgentOutput
        """
        start = time.time()
        initial_state: Dict[str, Any] = {
            "input_query":        query,
            "context":            context or {},
            "domain":             self.domain,
            "expanded_query":     "",
            "fetched_papers":     {},
            "fetch_meta":         {},
            "chunks_ready":       False,
            "retrieval_results":  [],
            "retrieval_time_sec": 0.0,
            "reranked_results":   [],
            "answer":             "",
            "citations":          [],
            "confidence":         0.0,
            "model_used":         "",
            "synthesis_context":  [],
            "stage_latency_sec":   {},
            "token_usage":         {},
            "cost_breakdown_usd":  {},
            "attempt_events":      [],
            "error":              None,
            "execution_time_sec": 0.0,
        }

        result = self.graph.invoke(initial_state)

        return self._output_from_result(
            result=result,
            query=query,
            context=context,
            started_at=start,
        )