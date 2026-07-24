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

        if not self._base_url:
            return self._error(
                "LOCAL_AGENT env var not set — cannot reach deployed service",
                time.time() - start,
            )

        try:
            logger.info(
                "search_local_index: querying %s model_id=%s top_k=%d",
                self._base_url, model_id, top_k,
            )

            response = self._session.post(
                f"{self._base_url}/query",
                json={"query": query, "model_id": model_id, "top_k": top_k},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            elapsed = time.time() - start
            return self._success(
                results=result.get("citations", []),
                retrieval_time_sec=elapsed,
                answer=result.get("answer", ""),
                citations=result.get("citations", []),
                confidence=0.80,
            )

        except requests.exceptions.RequestException as exc:
            elapsed = time.time() - start
            logger.error("search_local_index HTTP error: %s", exc)
            return self._error(f"HTTP error: {exc}", elapsed)
        except Exception as exc:
            elapsed = time.time() - start
            logger.error("search_local_index failed: %s", exc, exc_info=True)
            return self._error(str(exc), elapsed)

    def __del__(self) -> None:
        if hasattr(self, "_session"):
            self._session.close()
