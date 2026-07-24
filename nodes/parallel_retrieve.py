"""
Node 3: parallel_retrieve — Concurrent tool invocation.

Phase 9 integration: when an agent subgraph exists for a discovered skill,
the subgraph is invoked directly (expand_query → retrieve → rerank →
synthesise) instead of the legacy MCP wrapper.  Tools without a subgraph
(e.g. ``search_pubmed_deep``) fall back to the MCP registry.

Uses asyncio for concurrent execution with timeout protection per tool.
"""

import asyncio
from typing import Any, Dict, Optional
from agent_state import AgentState
from mcp_registry import mcp_registry
import logging
import time

logger = logging.getLogger(__name__)

# ======================================================================
# Phase 9 — agent subgraph mapping
# ======================================================================

# Maps MCP tool names (as returned by skill_router / YAML manifests)
# to ``(module_path, class_name)`` pairs.  Lazy-imported on first use
# to avoid heavy startup cost when the subgraphs are not needed.

_AGENT_GRAPH_MAP: Dict[str, tuple] = {
    "search_pubmed":          ("agents.pubmed_agent.graph",           "PubMedAgentGraph"),
    "search_fda":             ("agents.fda_agent.graph",              "FDAAgentGraph"),
    "search_clinical_trials": ("agents.clinical_trials_agent.graph",  "ClinicalTrialsAgentGraph"),
    "search_local_index":     ("agents.local_agent.graph",            "LocalAgentGraph"),
}

# Singleton cache so each subgraph class is only instantiated once.
_agent_instances: Dict[str, Any] = {}


def _get_agent_graph(tool_name: str) -> Optional[Any]:
    """Return a cached agent subgraph instance, or *None* if no mapping exists."""
    if tool_name not in _AGENT_GRAPH_MAP:
        return None

    if tool_name not in _agent_instances:
        module_path, class_name = _AGENT_GRAPH_MAP[tool_name]
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            _agent_instances[tool_name] = cls()
            logger.info(
                "parallel_retrieve: instantiated agent subgraph %s for %s",
                class_name, tool_name,
            )
        except Exception as exc:
            logger.warning(
                "parallel_retrieve: failed to load agent subgraph for %s, "
                "will fall back to MCP tool: %s",
                tool_name, exc,
            )
            # Mark as failed so we don't retry every call.
            _agent_instances[tool_name] = None

    return _agent_instances.get(tool_name)


def _agent_output_to_mcp_result(agent_output: Any) -> Dict[str, Any]:
    """Convert an ``AgentOutput`` dataclass to the MCP result dict
    consumed by downstream nodes (synthesise, score_confidence, etc.).

    The ``results`` list is built from ``AgentOutput.sources`` (the
    reranked retrieval documents).  Metadata keys are flattened to the
    top level so that ``synthesise.py`` can access ``title``, ``authors``,
    ``year``, ``abstract``, and ``doi`` directly.
    """
    results = []
    for src in (agent_output.sources or []):
        meta = src.get("metadata", {}) if isinstance(src, dict) else {}
        results.append({
            "title":    meta.get("title", ""),
            "authors":  meta.get("authors", []),
            "year":     meta.get("year", ""),
            "abstract": (src.get("text", "") if isinstance(src, dict) else ""),
            "doi":      meta.get("doi", ""),
            # Preserve original fields for traceability
            "score":    src.get("score", 0.0) if isinstance(src, dict) else 0.0,
        })

    return {
        "results":            results,
        "tokens_used":        0,
        "retrieval_time_sec": agent_output.execution_time_sec,
        "error":              agent_output.error,
        # Phase 9 extras — the subgraph's own synthesis output
        "agent_answer":       agent_output.answer,
        "agent_citations":    agent_output.citations,
        "agent_confidence":   agent_output.confidence,
        "agent_model_used":   agent_output.model_used,
        "agent_domain":       agent_output.domain,
    }


def parallel_retrieve(state: AgentState) -> AgentState:
    """
    Invoke all discovered tools in parallel.

    Phase 9: for each discovered skill, the node first checks whether an
    agent subgraph exists (``_AGENT_GRAPH_MAP``).  If so, the subgraph is
    invoked directly (expand_query -> retrieve -> rerank -> synthesise)
    and its ``AgentOutput`` is converted to the standard MCP result dict.
    Tools without a subgraph fall back to the MCP registry.

    Args:
        state: Current agent state

    Returns:
        Updated state with retrieval_results, tokens_used, retrieval_time_sec
    """
    trace_id = state.get("trace_id", "unknown")
    tools_to_invoke = state["discovered_skills"]

    if not tools_to_invoke:
        logger.warning(f"[{trace_id}] parallel_retrieve: No tools to invoke")
        state["retrieval_results"] = {}
        state["tokens_used"] = {}
        state["retrieval_time_sec"] = {}
        state["total_retrieval_time_sec"] = 0.0
        state["is_partial_response"] = True
        return state

    try:
        # Invoke tools in parallel using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def invoke_all_tools() -> Dict[str, Dict[str, Any]]:
            """Concurrently invoke all tools."""
            tasks = []
            tool_names_list = []

            for tool_name in tools_to_invoke:
                try:
                    # Phase 9: prefer agent subgraph when available
                    agent_graph = _get_agent_graph(tool_name)

                    if agent_graph is not None:
                        task = asyncio.create_task(
                            invoke_agent_with_timeout(
                                tool_name=tool_name,
                                agent_graph=agent_graph,
                                query=state["input_query"],
                                context=state["context"],
                                timeout_sec=30.0,  # subgraphs do more work
                                trace_id=trace_id,
                            )
                        )
                    else:
                        # Fallback to legacy MCP tool
                        tool = mcp_registry.get_tool(tool_name)
                        task = asyncio.create_task(
                            invoke_tool_with_timeout(
                                tool_name=tool_name,
                                tool=tool,
                                query=state["input_query"],
                                context=state["context"],
                                timeout_sec=5.0,
                                trace_id=trace_id,
                            )
                        )

                    tasks.append(task)
                    tool_names_list.append(tool_name)
                except Exception as e:
                    logger.error(
                        f"[{trace_id}] Failed to create task for tool {tool_name}: {str(e)}",
                        exc_info=True,
                    )

            # Gather results
            results = {}
            if tasks:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
                for tool_name, result in zip(tool_names_list, task_results):
                    if isinstance(result, BaseException):
                        results[tool_name] = {
                            "results": [],
                            "tokens_used": 0,
                            "retrieval_time_sec": 0.0,
                            "error": str(result),
                        }
                    else:
                        results[tool_name] = result

            return results

        # Run async invocation
        retrieval_results = loop.run_until_complete(invoke_all_tools())
        loop.close()

        # Extract tokens and timing
        tokens_used = {}
        retrieval_time_sec = {}
        total_time = 0.0

        for tool_name, result in retrieval_results.items():
            tokens_used[tool_name] = result.get("tokens_used", 0)
            retrieval_time_sec[tool_name] = result.get("retrieval_time_sec", 0.0)
            total_time += retrieval_time_sec[tool_name]

        state["retrieval_results"] = retrieval_results
        state["tokens_used"] = tokens_used
        state["retrieval_time_sec"] = retrieval_time_sec
        state["total_retrieval_time_sec"] = total_time

        # Check if any tools failed
        failed_tools = [
            t
            for t, r in retrieval_results.items()
            if r.get("error") is not None
        ]
        if failed_tools:
            state["is_partial_response"] = True
            state["error_messages"].extend([
                f"Tool {t} failed: {retrieval_results[t].get('error')}"
                for t in failed_tools
            ])
            logger.warning(
                f"[{trace_id}] parallel_retrieve: {len(failed_tools)} tools failed: {failed_tools}"
            )
        else:
            state["is_partial_response"] = False

        logger.info(
            f"[{trace_id}] parallel_retrieve: Invoked {len(tools_to_invoke)} tools. "
            f"Total time: {total_time:.2f}s. Failed: {len(failed_tools)}"
        )

    except Exception as e:
        logger.error(
            f"[{trace_id}] parallel_retrieve: Failed to invoke tools: {str(e)}",
            exc_info=True,
        )
        state["retrieval_results"] = {}
        state["tokens_used"] = {}
        state["retrieval_time_sec"] = {}
        state["total_retrieval_time_sec"] = 0.0
        state["is_partial_response"] = True
        state["error_occurred"] = True
        state["error_messages"].append(f"Parallel retrieval error: {str(e)}")

    return state


async def invoke_agent_with_timeout(
    tool_name: str,
    agent_graph: Any,
    query: str,
    context: Dict[str, Any],
    timeout_sec: float,
    trace_id: str,
) -> Dict[str, Any]:
    """Invoke an agent subgraph with timeout protection.

    The subgraph's ``invoke(query, context)`` runs synchronously in a
    thread-pool executor so it doesn't block the event loop.  The
    returned ``AgentOutput`` is converted to the standard MCP result dict.

    Parameters
    ----------
    tool_name : str
        MCP tool name (e.g. ``"search_pubmed"``).
    agent_graph : SubAgentGraph
        Instantiated agent subgraph.
    query, context, timeout_sec, trace_id :
        Same as :func:`invoke_tool_with_timeout`.

    Returns
    -------
    dict
        Standard MCP result dict (``results``, ``tokens_used``, etc.)
        enriched with ``agent_answer``, ``agent_citations``, and
        ``agent_confidence`` from the subgraph's own synthesis.
    """
    try:
        start_time = time.time()
        loop = asyncio.get_event_loop()

        agent_output = await asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: agent_graph.invoke(query, context)
            ),
            timeout=timeout_sec,
        )

        elapsed = time.time() - start_time
        result = _agent_output_to_mcp_result(agent_output)
        result["retrieval_time_sec"] = elapsed

        logger.info(
            "[%s] Agent subgraph %s: %d sources, confidence=%.2f, %.2fs",
            trace_id, tool_name,
            len(result.get("results", [])),
            agent_output.confidence,
            elapsed,
        )
        return result

    except asyncio.TimeoutError:
        error_msg = f"Agent subgraph {tool_name} timed out after {timeout_sec}s"
        logger.warning("[%s] %s", trace_id, error_msg)
        return {
            "results": [],
            "tokens_used": 0,
            "retrieval_time_sec": timeout_sec,
            "error": error_msg,
        }

    except Exception as e:
        error_msg = f"Agent subgraph {tool_name} error: {e}"
        logger.error("[%s] %s", trace_id, error_msg, exc_info=True)
        return {
            "results": [],
            "tokens_used": 0,
            "retrieval_time_sec": 0.0,
            "error": error_msg,
        }


async def invoke_tool_with_timeout(
    tool_name: str,
    tool: Any,
    query: str,
    context: Dict[str, Any],
    timeout_sec: float,
    trace_id: str,
) -> Dict[str, Any]:
    """
    Invoke a single MCP tool with timeout protection.

    Args:
        tool_name: Name of the tool
        tool: Tool instance
        query: User query
        context: Request context
        timeout_sec: Timeout in seconds
        trace_id: Trace ID for logging

    Returns:
        Tool result dict with standard schema:
        {
            "results": [...],
            "tokens_used": int,
            "retrieval_time_sec": float,
            "error": None or str
        }
    """
    try:
        start_time = time.time()

        # Call tool's async invoke method (or sync if not async)
        if hasattr(tool, "invoke_async"):
            result = await asyncio.wait_for(
                tool.invoke_async(query=query, context=context),
                timeout=timeout_sec,
            )
        else:
            # Fall back to sync invocation in executor
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: tool.invoke(query=query, context=context)
                ),
                timeout=timeout_sec,
            )

        elapsed_time = time.time() - start_time

        # Ensure result has standard schema
        if "tokens_used" not in result:
            result["tokens_used"] = 0
        if "retrieval_time_sec" not in result:
            result["retrieval_time_sec"] = elapsed_time
        if "error" not in result:
            result["error"] = None

        logger.debug(
            f"[{trace_id}] Tool {tool_name}: {len(result.get('results', []))} results, "
            f"{elapsed_time:.2f}s"
        )
        return result

    except asyncio.TimeoutError:
        error_msg = f"Tool {tool_name} timed out after {timeout_sec}s"
        logger.warning(f"[{trace_id}] {error_msg}")
        return {
            "results": [],
            "tokens_used": 0,
            "retrieval_time_sec": timeout_sec,
            "error": error_msg,
        }

    except Exception as e:
        error_msg = f"Tool {tool_name} error: {str(e)}"
        logger.error(f"[{trace_id}] {error_msg}", exc_info=True)
        return {
            "results": [],
            "tokens_used": 0,
            "retrieval_time_sec": 0.0,
            "error": error_msg,
        }
