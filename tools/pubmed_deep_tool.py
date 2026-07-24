"""
pubmed_deep_tool.py — MCP tool for PubMed deep-research retrieval (HTTP proxy).

Ported from ``agentic-pipeline-clinical/pubmed_deep_research_agent_wrapper.py``.
This tool delegates to a deployed PubMed Deep Research service via HTTP —
no direct LLM/embedding calls in the wrapper itself.
"""

import logging
import os
import time
from typing import Any, Dict

import requests
from dotenv import load_dotenv

from tools.mcp_base import MCPToolBase

load_dotenv()

logger = logging.getLogger(__name__)


class PubMedDeepTool(MCPToolBase):
    """PubMed Deep Research retrieval tool (HTTP proxy).

    Delegates to a deployed service that performs real-time PubMed
    literature search, parallel paper fetching, semantic retrieval,
    LLM reranking, and AMA-formatted answer synthesis.
    """

    name = "search_pubmed_deep"
    description = (
        "PubMed Deep Research Agent — real-time medical literature search "
        "and Q&A system with AMA-formatted citations. Ideal for "
        "evidence-based medical research queries."
    )
    triggers = [
        "deep research", "comprehensive", "systematic",
        "evidence", "meta-analysis", "review",
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Medical research question"},
            "max_papers": {
                "type": "integer",
                "default": 50,
                "description": "Maximum papers to analyse",
            },
            "top_k": {
                "type": "integer",
                "default": 8,
                "description": "Source chunks for answer generation",
            },
            "include_fulltext": {
                "type": "boolean",
                "default": False,
                "description": "Fetch PMC full text",
            },
            "rerank": {
                "type": "boolean",
                "default": True,
                "description": "Apply LLM reranking",
            },
            "search_recent": {
                "type": "boolean",
                "default": True,
                "description": "Include recent literature (last 2 years)",
            },
            "search_foundational": {
                "type": "boolean",
                "default": True,
                "description": "Include foundational papers",
            },
        },
        "required": ["query"],
    }

    _DEFAULT_URL = "https://pubmed-deep-research-agent-508047128875.europe-west1.run.app"

    def __init__(self) -> None:
        self._base_url = (
            os.getenv("PUBMED_DEEP_RESEARCH_AGENT") or self._DEFAULT_URL
        ).rstrip("/")
        self._timeout = 300
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "MedicalResearchAgent/2.0",
        })

    def call(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        query = input_dict["query"]
        max_papers = input_dict.get("max_papers", 50)
        top_k = input_dict.get("top_k", 8)
        include_fulltext = input_dict.get("include_fulltext", False)
        rerank = input_dict.get("rerank", True)
        search_recent = input_dict.get("search_recent", True)
        search_foundational = input_dict.get("search_foundational", True)

        try:
            logger.info(
                "search_pubmed_deep: max_papers=%d top_k=%d rerank=%s",
                max_papers, top_k, rerank,
            )

            response = self._session.post(
                f"{self._base_url}/ask",
                json={
                    "question": query,
                    "max_papers": max_papers,
                    "top_k": top_k,
                    "include_fulltext": include_fulltext,
                    "rerank": rerank,
                    "search_recent": search_recent,
                    "search_foundational": search_foundational,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            result = response.json()

            papers_analyzed = result.get("papers_analyzed", 0)
            citations = result.get("citations", [])
            confidence = self._calculate_confidence(papers_analyzed, len(citations))
            elapsed = time.time() - start

            logger.info(
                "search_pubmed_deep: %d papers, %d citations, confidence=%.2f",
                papers_analyzed, len(citations), confidence,
            )

            return self._success(
                results=citations,
                retrieval_time_sec=elapsed,
                answer=result.get("answer", ""),
                citations=citations,
                confidence=confidence,
                papers_analyzed=papers_analyzed,
                search_queries=result.get("search_queries", []),
            )

        except requests.exceptions.Timeout:
            elapsed = time.time() - start
            msg = f"Request timed out after {self._timeout}s"
            logger.error("search_pubmed_deep: %s", msg)
            return self._error(msg, elapsed)
        except requests.exceptions.ConnectionError as exc:
            elapsed = time.time() - start
            msg = f"Connection error: {exc}"
            logger.error("search_pubmed_deep: %s", msg)
            return self._error(msg, elapsed)
        except requests.exceptions.HTTPError as exc:
            elapsed = time.time() - start
            msg = f"HTTP {exc.response.status_code}: {exc.response.text}"
            logger.error("search_pubmed_deep: %s", msg)
            return self._error(msg, elapsed)
        except Exception as exc:
            elapsed = time.time() - start
            logger.error("search_pubmed_deep failed: %s", exc, exc_info=True)
            return self._error(str(exc), elapsed)

    @staticmethod
    def _calculate_confidence(papers_analyzed: int, citations_count: int) -> float:
        """Confidence from paper/citation counts (ported from legacy)."""
        if papers_analyzed == 0:
            return 0.0
        base = min(0.5, papers_analyzed / 20)
        citation_boost = min(0.4, citations_count / 10)
        return min(0.95, base + citation_boost + 0.1)

    def __del__(self) -> None:
        if hasattr(self, "_session"):
            self._session.close()
