"""
local_index_tool.py — MCP tool for local RAG index retrieval (HTTP proxy).

Ported from ``agentic-pipeline-clinical/local_agent_wrapper.py``.
This tool delegates to a deployed RAG service via HTTP — no direct
LLM/embedding calls in the wrapper itself.
"""

import logging
import os
import time
from typing import Any, Dict

import requests
from dotenv import load_dotenv

from evaluation_core import RuntimeDeadlineExceeded, stable_query_fingerprint
from evaluation_core.deadline import ensure_deadline, remaining_seconds
from tools.mcp_base import MCPToolBase

load_dotenv()

logger = logging.getLogger(__name__)


class LocalIndexTool(MCPToolBase):
    """Local RAG index retrieval tool (HTTP proxy).

    Forwards queries to a deployed RAG service and normalises the
    response into the standard MCP tool output schema.
    """

    name = "search_local_index"
    description = (
        "Medical research RAG system that answers queries based on "
        "academic papers via a deployed local index service."
    )
    triggers = [
        "local", "paper", "academic", "research",
        "literature", "document", "index",
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Research query"},
            "model_id": {
                "type": "string",
                "default": "medical_papers",
                "description": "GCP index identifier",
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "description": "Number of documents to retrieve",
            },
        },
        "required": ["query"],
    }

    def __init__(self) -> None:
        self._base_url = os.getenv("LOCAL_AGENT", "")
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def call(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        query = input_dict["query"]
        model_id = input_dict.get("model_id", "medical_papers")
        top_k = input_dict.get("top_k", 5)
        deadline_at = input_dict.get("_runtime_deadline_at_monotonic")

        if not self._base_url:
            return self._error(
                "LOCAL_AGENT env var not set — cannot reach deployed service",
                time.time() - start,
            )

        try:
            ensure_deadline(deadline_at)
            logger.info(
                "[%s] search_local_index query_sha256=%s query_length=%d "
                "stage=retrieval top_k=%d",
                input_dict.get("trace_id", "unknown"),
                stable_query_fingerprint(query),
                len(query),
                top_k,
            )

            response = self._session.post(
                f"{self._base_url}/query",
                json={"query": query, "model_id": model_id, "top_k": top_k},
                timeout=remaining_seconds(deadline_at, default=60),
            )
            response.raise_for_status()
            result = response.json()
            if result.get("answer") and not result.get("synthesis_context"):
                return self._error(
                    "local_index_exact_context_unavailable",
                    time.time() - start,
                )

            elapsed = time.time() - start
            return self._success(
                results=result.get("citations", []),
                retrieval_time_sec=elapsed,
                answer=result.get("answer", ""),
                citations=result.get("citations", []),
                confidence=0.80,
                synthesis_context=list(
                    result.get("synthesis_context") or []
                ),
            )

        except RuntimeDeadlineExceeded:
            return self._error(
                "runtime_deadline_exhausted", time.time() - start
            )
        except requests.exceptions.Timeout:
            elapsed = time.time() - start
            if deadline_at is not None:
                try:
                    remaining_seconds(deadline_at)
                except RuntimeDeadlineExceeded:
                    return self._error(
                        "runtime_deadline_exhausted", elapsed
                    )
            logger.error(
                "search_local_index HTTP failure error_type=Timeout"
            )
            return self._error("local_index_http_failed:Timeout", elapsed)
        except requests.exceptions.RequestException as exc:
            elapsed = time.time() - start
            error_type = type(exc).__name__
            logger.error(
                "search_local_index HTTP failure error_type=%s",
                error_type,
            )
            return self._error(f"local_index_http_failed:{error_type}", elapsed)
        except Exception as exc:
            elapsed = time.time() - start
            error_type = type(exc).__name__
            logger.error(
                "search_local_index failed error_type=%s", error_type
            )
            return self._error(f"local_index_failed:{error_type}", elapsed)

    def __del__(self) -> None:
        if hasattr(self, "_session"):
            self._session.close()
