"""
Node 3: parallel_retrieve — Concurrent tool invocation.

Phase 9 integration: when an agent subgraph exists for a discovered skill,
the subgraph is invoked directly (expand_query → retrieve → rerank →
synthesise) instead of the legacy MCP wrapper.  Tools without a subgraph
(e.g. ``search_pubmed_deep``) fall back to the MCP registry.

Uses asyncio for concurrent execution with timeout protection per tool.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
from agent_state import AgentState
from evaluation_core import EvaluationTrace, build_agent_evaluation_trace
from llm_client import LLMClient, LLMCallResult
from mcp_registry import mcp_registry
from runtime_verification import (
    RuntimeVerifier,
    aggregate_attempt_telemetry,
    build_attempt_event,
    build_retry_request,
    build_runtime_verifier,
    evidence_limited_decision,
    repair_agent_synthesis,
)
from runtime_verification.executor import (
    BoundedExecutor,
    ExecutorSaturatedError,
    get_runtime_executor,
)
import logging
import time

logger = logging.getLogger(__name__)


async def _run_blocking_with_timeout(
    call: Callable[[], Any],
    timeout: float,
    *,
    executor: Optional[BoundedExecutor] = None,
) -> Any:
    """Run blocking work in the process-bounded executor."""
    bounded_executor = executor or get_runtime_executor()
    future = bounded_executor.submit(call)
    wrapped = asyncio.wrap_future(future)
    try:
        return await asyncio.wait_for(
            asyncio.shield(wrapped), timeout=timeout
        )
    except asyncio.TimeoutError:
        bounded_executor.mark_timeout()
        raise

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
                "will fall back to MCP tool error_type=%s",
                tool_name,
                type(exc).__name__,
            )
            # Mark as failed so we don't retry every call.
            _agent_instances[tool_name] = None

    return _agent_instances.get(tool_name)


def _agent_output_to_mcp_result(
    agent_output: Any, trace: Optional[EvaluationTrace] = None
) -> Dict[str, Any]:
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
        results.append(
            {
                "document_id": (
                    trace.final_context_document_ids[len(results)]
                    if trace is not None
                    and len(results) < len(trace.final_context_document_ids)
                    else src.get("doc_id")
                ),
                "doc_id": src.get("doc_id"),
                "title": meta.get("title", ""),
                "authors": meta.get("authors", []),
                "year": meta.get("year", ""),
                "abstract": (
                    src.get("text", "") if isinstance(src, dict) else ""
                ),
                "doi": meta.get("doi", ""),
                "pmid": meta.get("pmid"),
                "nct_id": meta.get("nct_id"),
                "record_id": meta.get("record_id"),
                "journal": meta.get("journal", ""),
                "source": meta.get("source", agent_output.domain),
                "source_type": meta.get("source_type"),
                "authority": meta.get("authority"),
                "publication_type": meta.get("publication_type")
                or meta.get("study_type"),
                "publication_date": meta.get("publication_date")
                or meta.get("date")
                or meta.get("year"),
                "provenance": meta.get("provenance") or meta.get("url"),
                "metadata": dict(meta),
                "original_rank": src.get("original_rank"),
                "score": src.get("score", 0.0),
            }
        )

    tokens_used = (
        _normalized_token_total(trace.token_usage) if trace is not None else 0
    )
    cost = (
        sum(trace.cost_breakdown_usd.values()) if trace is not None else 0.0
    )

    return {
        "results":            results,
        "tokens_used":        tokens_used,
        "cost":               cost,
        "retrieval_time_sec": agent_output.execution_time_sec,
        "error":              (
            "agent_subgraph_error" if agent_output.error else None
        ),
        # Phase 9 extras — the subgraph's own synthesis output
        "agent_answer":       agent_output.answer,
        "agent_citations":    agent_output.citations,
        "agent_confidence":   agent_output.confidence,
        "agent_model_used":   agent_output.model_used,
        "agent_domain":       agent_output.domain,
        "evaluation_trace":   trace.to_dict() if trace is not None else None,
    }


def _attach_attempt_aggregate(
    result: Dict[str, Any], traces: List[Dict[str, Any]]
) -> Dict[str, Any]:
    aggregate = aggregate_attempt_telemetry(traces)
    result["tokens_used"] = aggregate["tokens_used"]
    result["cost"] = aggregate["cost_usd"]
    result["attempt_compute_time_sec"] = aggregate["attempt_compute_time_sec"]
    result["attempt_telemetry"] = aggregate["attempt_telemetry"]
    return result


def _invoke_agent_with_telemetry(
    agent_graph: Any, query: str, context: Dict[str, Any]
) -> tuple[Any, Dict[str, Any]]:
    client = LLMClient()
    before = client.thread_metrics()
    before_history = len(client.thread_call_history())
    output = agent_graph.invoke(query, context)
    after = client.thread_metrics()
    call_results = client.thread_call_history()[before_history:]
    return output, {
        "calls": max(0, int(after["calls"]) - int(before["calls"])),
        "tokens_in": max(
            0, int(after["tokens_in"]) - int(before["tokens_in"])
        ),
        "tokens_out": max(
            0, int(after["tokens_out"]) - int(before["tokens_out"])
        ),
        "cost_usd": max(
            0.0, float(after["cost_usd"]) - float(before["cost_usd"])
        ),
        "latency_sec": max(
            0.0, float(after["latency_sec"]) - float(before["latency_sec"])
        ),
        "model": after.get("last_model") or "",
        "model_revision": after.get("last_model_revision") or "",
        "call_results": call_results,
    }


def _apply_invocation_telemetry(
    trace: EvaluationTrace, telemetry: Dict[str, Any]
) -> None:
    if int(telemetry.get("calls") or 0) <= 0:
        return
    tokens_in = int(telemetry.get("tokens_in") or 0)
    tokens_out = int(telemetry.get("tokens_out") or 0)
    trace.token_usage = {
        "input": tokens_in,
        "output": tokens_out,
        "total": tokens_in + tokens_out,
        "calls": int(telemetry["calls"]),
    }
    trace.cost_breakdown_usd = {
        "llm": float(telemetry.get("cost_usd") or 0.0)
    }
    if telemetry.get("model"):
        trace.exact_model = str(telemetry["model"])
        trace.model_revision = str(telemetry.get("model_revision") or "")
    call_results = list(telemetry.get("call_results") or [])
    authoritative_stages = {
        (
            str(
                (call.provider_metadata or {}).get(
                    "telemetry_attempt_id"
                )
                or trace.attempt_id
            ),
            str(
                (call.provider_metadata or {}).get("telemetry_stage")
                or "agent_llm_call"
            ),
        )
        for call in call_results
    }
    if authoritative_stages:
        trace.attempt_events = [
            event
            for event in trace.attempt_events
            if (
                str(event.get("attempt_id") or ""),
                str(event.get("stage") or ""),
            )
            not in authoritative_stages
        ]
    existing_ids = {
        str(item.get("event_id"))
        for item in trace.attempt_events
        if item.get("event_id")
    }
    stage_occurrences: Dict[tuple[str, str], int] = {}
    for index, call_result in enumerate(call_results, 1):
        provider_metadata = dict(call_result.provider_metadata or {})
        stage = str(
            provider_metadata.get("telemetry_stage") or "agent_llm_call"
        )
        attempt_id = str(
            provider_metadata.get("telemetry_attempt_id")
            or trace.attempt_id
        )
        occurrence_key = (attempt_id, stage)
        occurrence = stage_occurrences.get(occurrence_key, 0) + 1
        stage_occurrences[occurrence_key] = occurrence
        base_event_id = f"{attempt_id}:{stage}"
        event_id = (
            base_event_id
            if occurrence == 1
            else f"{base_event_id}:call:{occurrence}"
        )
        if event_id in existing_ids:
            continue
        event = build_attempt_event(
            trace_id=trace.trace_id,
            attempt_id=attempt_id,
            parent_attempt_id=(
                provider_metadata.get("telemetry_parent_attempt_id")
                or trace.parent_attempt_id
            ),
            stage=stage,
            component=trace.agent_name,
            status=(
                "deadline_exhausted"
                if call_result.error_type
                == "RuntimeDeadlineExceeded"
                else str(call_result.status or "success")
            ),
            repair_status=str(
                provider_metadata.get("telemetry_repair_status")
                or "initial"
            ),
            model=call_result.model,
            model_revision=call_result.model_revision,
            tokens_in=call_result.tokens_in,
            tokens_out=call_result.tokens_out,
            cost_usd=call_result.cost_usd,
            latency_sec=call_result.latency_sec,
            finish_reason=call_result.finish_reason,
            deadline_exhausted=(
                call_result.error_type == "RuntimeDeadlineExceeded"
            ),
            error_type=call_result.error_type,
            provider_metadata=provider_metadata,
            event_id=event_id,
        )
        trace.attempt_events.append(event)
        existing_ids.add(event_id)


def _normalized_token_total(usage: Optional[Dict[str, Any]]) -> int:
    """Return canonical token total without double-counting aggregate keys."""
    if not usage:
        return 0
    total = usage.get("total")
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        return max(0, int(total))
    return max(0, int(usage.get("input") or 0)) + max(
        0, int(usage.get("output") or 0)
    )


def _record_failed_agent_repair(
    trace: EvaluationTrace,
    repair_context: Dict[str, Any],
    exc: BaseException,
) -> EvaluationTrace:
    """Persist a failed synthesis-repair provider call on the attempt trace."""
    from dataclasses import replace

    repair_trace = replace(
        trace,
        attempt_id=str(repair_context.get("attempt_id") or trace.attempt_id),
        parent_attempt_id=str(
            repair_context.get("parent_attempt_id") or trace.attempt_id
        ),
        errors=list(trace.errors)
        + [f"agent_synthesis_repair_failed:{type(exc).__name__}"],
    )
    history_reader = getattr(LLMClient(), "thread_call_history", None)
    history = (
        list(history_reader() or [])
        if callable(history_reader)
        else []
    )
    call_result = history[-1] if history else None
    if not isinstance(call_result, LLMCallResult):
        call_result = None
    latency_sec = (
        float(call_result.latency_sec)
        if call_result is not None
        else 0.0
    )
    tokens_in = call_result.tokens_in if call_result is not None else 0
    tokens_out = call_result.tokens_out if call_result is not None else 0
    cost_usd = float(call_result.cost_usd) if call_result is not None else 0.0
    model = (
        call_result.model
        if call_result is not None
        else str(repair_context.get("model_id") or repair_trace.exact_model or "")
    )
    model_revision = (
        call_result.model_revision if call_result is not None else ""
    )
    repair_trace.attempt_events.append(
        build_attempt_event(
            trace_id=repair_trace.trace_id,
            attempt_id=repair_trace.attempt_id,
            parent_attempt_id=repair_trace.parent_attempt_id,
            stage="agent_synthesis_repair",
            component=repair_trace.agent_name,
            status=(
                "deadline_exhausted"
                if isinstance(exc, TimeoutError)
                else "error"
            ),
            repair_status="synthesis_repair",
            model=model,
            model_revision=model_revision,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            latency_sec=latency_sec,
            finish_reason=(
                call_result.finish_reason
                if call_result is not None
                else "error"
            ),
            deadline_exhausted=isinstance(exc, TimeoutError),
            error_type=type(exc).__name__,
            provider_metadata=(
                call_result.provider_metadata if call_result is not None else None
            ),
            event_id=f"{repair_trace.attempt_id}:agent_synthesis_repair",
        )
    )
    repair_trace.token_usage = {
        "input": int(repair_trace.token_usage.get("input") or 0) + tokens_in,
        "output": int(repair_trace.token_usage.get("output") or 0) + tokens_out,
        "total": _normalized_token_total(repair_trace.token_usage)
        + tokens_in
        + tokens_out,
    }
    repair_trace.cost_breakdown_usd = dict(repair_trace.cost_breakdown_usd or {})
    repair_trace.cost_breakdown_usd["repair"] = float(
        repair_trace.cost_breakdown_usd.get("repair") or 0.0
    ) + cost_usd
    repair_trace.stage_latency_sec = dict(repair_trace.stage_latency_sec or {})
    repair_trace.stage_latency_sec["synthesis"] = float(
        repair_trace.stage_latency_sec.get("synthesis") or 0.0
    ) + latency_sec
    if model:
        repair_trace.exact_model = model
        repair_trace.model_revision = model_revision
    return repair_trace


def _append_retrieval_event(
    trace: EvaluationTrace,
    *,
    latency_sec: float,
    status: str,
    repair_status: str,
    error_type: str = "",
    deadline_exhausted: bool = False,
) -> None:
    """Attach one canonical, non-LLM retrieval-stage event."""
    existing_token_total = sum(
        int((item.get("token_usage") or {}).get("total") or 0)
        for item in trace.attempt_events
    )
    trace_usage = dict(trace.token_usage or {})
    trace_token_total = int(
        trace_usage.get("total")
        if isinstance(trace_usage.get("total"), (int, float))
        else int(trace_usage.get("input") or 0)
        + int(trace_usage.get("output") or 0)
    )
    trace_cost = sum(
        float(value)
        for value in dict(trace.cost_breakdown_usd or {}).values()
        if isinstance(value, (int, float))
    )
    legacy_tokens_in = int(trace_usage.get("input") or 0)
    legacy_tokens_out = int(trace_usage.get("output") or 0)
    if (
        legacy_tokens_in == 0
        and legacy_tokens_out == 0
        and trace_token_total > 0
    ):
        legacy_tokens_in = trace_token_total
    legacy_generation_latency = float(
        trace.stage_latency_sec.get("generation") or 0.0
    )
    if legacy_generation_latency <= 0.0:
        legacy_generation_latency = max(
            0.0,
            float(trace.stage_latency_sec.get("total") or 0.0)
            - float(trace.stage_latency_sec.get("retrieval") or 0.0)
            - float(trace.stage_latency_sec.get("verification") or 0.0),
        )
    if (
        not trace.attempt_events
        or (
            existing_token_total == 0
            and trace_token_total > 0
        )
    ) and (trace_token_total > 0 or trace_cost > 0.0):
        trace.attempt_events.append(
            build_attempt_event(
                trace_id=trace.trace_id,
                attempt_id=trace.attempt_id,
                parent_attempt_id=trace.parent_attempt_id,
                stage="agent_generation_legacy",
                component=trace.agent_name,
                status="error" if trace.errors else "success",
                repair_status=repair_status,
                model=trace.exact_model,
                model_revision=trace.model_revision,
                tokens_in=legacy_tokens_in,
                tokens_out=legacy_tokens_out,
                cost_usd=trace_cost,
                latency_sec=legacy_generation_latency,
                finish_reason="legacy_adapted",
                event_id=f"{trace.attempt_id}:agent_generation_legacy",
            )
        )
    event_id = f"{trace.attempt_id}:retrieval"
    if any(
        str(item.get("event_id") or "") == event_id
        for item in trace.attempt_events
    ):
        return
    trace.attempt_events.append(
        build_attempt_event(
            trace_id=trace.trace_id,
            attempt_id=trace.attempt_id,
            parent_attempt_id=trace.parent_attempt_id,
            stage="retrieval",
            component=trace.agent_name,
            status=status,
            repair_status=repair_status,
            latency_sec=latency_sec,
            deadline_exhausted=deadline_exhausted,
            error_type=error_type,
            event_id=event_id,
        )
    )


def _terminal_failure_result(
    *,
    tool_name: str,
    query: str,
    context: Dict[str, Any],
    trace_id: str,
    started_at: float,
    failed_check: str,
    error_message: str,
    attempt_id: str,
    parent_attempt_id: Optional[str] = None,
    traces: Optional[List[Dict[str, Any]]] = None,
    repair_history: Optional[List[Dict[str, Any]]] = None,
    error_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a schema-valid terminal attempt for timeout/error exits."""
    elapsed = time.time() - started_at
    trace_context = dict(context)
    trace_context.update(
        {
            "trace_id": trace_id,
            "attempt_id": attempt_id,
            "parent_attempt_id": parent_attempt_id,
            "retrieval_only": True,
        }
    )
    trace = build_agent_evaluation_trace(
        agent_name=tool_name,
        domain=tool_name,
        original_query=query,
        state={
            "expanded_query": query,
            "retrieval_results": [],
            "reranked_results": [],
            "answer": "",
            "citations": [],
            "model_used": str(context.get("model_id") or ""),
            "execution_time_sec": elapsed,
            "error": error_message,
        },
        context=trace_context,
    )
    decision = evidence_limited_decision(
        target_agent=tool_name,
        failed_check=failed_check,
        message=error_message,
        valid=False,
        error=error_message,
    )
    if error_metadata:
        decision.raw_decision.update(error_metadata)
    trace.verification_decisions.append(decision)
    _append_retrieval_event(
        trace,
        latency_sec=elapsed,
        status="deadline_exhausted"
        if "deadline" in failed_check
        else "error",
        repair_status=(
            "retry" if parent_attempt_id else "initial"
        ),
        error_type=failed_check,
        deadline_exhausted="deadline" in failed_check,
    )
    all_traces = list(traces or []) + [trace.to_dict()]
    result = {
        "results": [],
        "retrieval_time_sec": elapsed,
        "error": error_message,
        "evaluation_trace": trace.to_dict(),
        "evaluation_traces": all_traces,
        "repair_history": list(repair_history or []),
        "evidence_limited": True,
        "verification_decision": decision.to_dict(),
        "error_metadata": dict(error_metadata or {}),
    }
    return _attach_attempt_aggregate(result, all_traces)


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
        verifier = build_runtime_verifier(state.get("context", {}))
        max_retries = max(
            0, min(1, int(state.get("context", {}).get("max_agent_retries", 1)))
        )
        max_agent_synthesis_repairs = max(
            0,
            min(
                1,
                int(
                    state.get("context", {}).get(
                        "max_agent_synthesis_repairs", 1
                    )
                ),
            ),
        )
        overall_deadline_sec = max(
            1.0,
            float(
                state.get("context", {}).get(
                    "runtime_verification_deadline_sec", 60.0
                )
            ),
        )
        deadline_at = float(
            state.get("context", {}).get("_runtime_deadline_at_monotonic")
            or (time.monotonic() + overall_deadline_sec)
        )
        state["context"]["_runtime_deadline_at_monotonic"] = deadline_at
        agent_timeout_sec = max(
            0.1,
            float(state.get("context", {}).get("agent_timeout_sec", 30.0)),
        )

        # Invoke tools in parallel using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def invoke_all_tools() -> Dict[str, Dict[str, Any]]:
            """Concurrently invoke all tools."""
            tasks = []
            tool_names_list = []
            results: Dict[str, Dict[str, Any]] = {}

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
                                timeout_sec=agent_timeout_sec,
                                trace_id=trace_id,
                                verifier=verifier,
                                max_retries=max_retries,
                                max_synthesis_repairs=max_agent_synthesis_repairs,
                                deadline_at=deadline_at,
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
                                verifier=verifier,
                                max_retries=max_retries,
                                deadline_at=deadline_at,
                            )
                        )

                    tasks.append(task)
                    tool_names_list.append(tool_name)
                except Exception as e:
                    logger.error(
                        "[%s] Failed to create task for tool %s error_type=%s",
                        trace_id,
                        tool_name,
                        type(e).__name__,
                    )
                    error_message = (
                        f"Failed to create task for tool {tool_name} "
                        f"(error_type={type(e).__name__})"
                    )
                    results[tool_name] = _terminal_failure_result(
                        tool_name=tool_name,
                        query=state["input_query"],
                        context=state["context"],
                        trace_id=trace_id,
                        started_at=time.time(),
                        failed_check="tool_task_creation_failure",
                        error_message=error_message,
                        attempt_id=f"{trace_id}:{tool_name}:1",
                    )

            # Gather results
            if tasks:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
                for tool_name, result in zip(tool_names_list, task_results):
                    if isinstance(result, BaseException):
                        error_message = (
                            f"Unhandled tool task failure for {tool_name} "
                            f"(error_type={type(result).__name__})"
                        )
                        results[tool_name] = _terminal_failure_result(
                            tool_name=tool_name,
                            query=state["input_query"],
                            context=state["context"],
                            trace_id=trace_id,
                            started_at=time.time(),
                            failed_check="unhandled_tool_task_failure",
                            error_message=error_message,
                            attempt_id=f"{trace_id}:{tool_name}:1",
                        )
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
        evaluation_traces: List[Dict[str, Any]] = []
        verification_history: List[Dict[str, Any]] = []
        repair_history: List[Dict[str, Any]] = []
        attempt_telemetry: List[Dict[str, Any]] = []

        for tool_name, result in retrieval_results.items():
            tokens_used[tool_name] = result.get("tokens_used", 0)
            retrieval_time_sec[tool_name] = result.get("retrieval_time_sec", 0.0)
            total_time += retrieval_time_sec[tool_name]
            evaluation_traces.extend(result.get("evaluation_traces", []))
            decision = result.get("verification_decision")
            if decision:
                verification_history.append(decision)
            repair_history.extend(result.get("repair_history", []))
            attempt_telemetry.extend(result.get("attempt_telemetry", []))

        state["retrieval_results"] = retrieval_results
        state["tokens_used"] = tokens_used
        state["retrieval_time_sec"] = retrieval_time_sec
        state["total_retrieval_time_sec"] = total_time
        state["evaluation_traces"] = list(
            state.get("evaluation_traces", [])
        ) + evaluation_traces
        state["verification_history"] = list(
            state.get("verification_history", [])
        ) + verification_history
        state["repair_history"] = list(
            state.get("repair_history", [])
        ) + repair_history
        existing_attempts = list(
            state.get("attempt_telemetry") or []
        )
        seen_event_ids = {
            str(item.get("event_id"))
            for item in existing_attempts
            if isinstance(item, dict) and item.get("event_id")
        }
        new_attempts = []
        for item in attempt_telemetry:
            event_id = str(item.get("event_id") or "")
            if event_id and event_id in seen_event_ids:
                continue
            if event_id:
                seen_event_ids.add(event_id)
            new_attempts.append(item)
        state["attempt_telemetry"] = existing_attempts + new_attempts
        retrieval_tokens_in = sum(
            int((item.get("token_usage") or {}).get("input") or 0)
            for item in new_attempts
        )
        retrieval_tokens_out = sum(
            int((item.get("token_usage") or {}).get("output") or 0)
            for item in new_attempts
        )
        retrieval_token_total = sum(
            int(item.get("token_total") or 0) for item in new_attempts
        )
        existing_usage = dict(state.get("token_usage") or {})
        state["token_usage"] = {
            "input": int(existing_usage.get("input") or 0) + retrieval_tokens_in,
            "output": int(existing_usage.get("output") or 0)
            + retrieval_tokens_out,
            "total": int(existing_usage.get("total") or 0)
            + retrieval_token_total,
        }
        new_retrieval_cost = sum(
            float(item.get("cost_usd") or 0.0)
            for item in new_attempts
        )
        state["cost_estimate"] = float(
            state.get("cost_estimate", 0.0)
        ) + new_retrieval_cost
        state["runtime_executor_metrics"] = get_runtime_executor().metrics()

        # Check if any tools failed
        failed_tools = [
            t
            for t, r in retrieval_results.items()
            if r.get("error") is not None
        ]
        evidence_limited_tools = [
            tool_name
            for tool_name, result in retrieval_results.items()
            if result.get("evidence_limited")
        ]
        if failed_tools or evidence_limited_tools:
            state["is_partial_response"] = True
            if failed_tools:
                state["error_occurred"] = True
            state["error_messages"].extend([
                f"Tool {t} failed"
                for t in failed_tools
            ])
            logger.warning(
                f"[{trace_id}] parallel_retrieve: {len(failed_tools)} tools failed: {failed_tools}"
            )
        else:
            state["is_partial_response"] = False
        state["evidence_limited"] = bool(retrieval_results) and all(
            result.get("evidence_limited", False)
            or not result.get("results")
            for result in retrieval_results.values()
        )

        logger.info(
            f"[{trace_id}] parallel_retrieve: Invoked {len(tools_to_invoke)} tools. "
            f"Total time: {total_time:.2f}s. Failed: {len(failed_tools)}"
        )

    except Exception as e:
        logger.error(
            "[%s] parallel_retrieve failed error_type=%s",
            trace_id,
            type(e).__name__,
        )
        state["retrieval_results"] = {}
        state["tokens_used"] = {}
        state["retrieval_time_sec"] = {}
        state["total_retrieval_time_sec"] = 0.0
        state["is_partial_response"] = True
        state["error_occurred"] = True
        state["error_messages"].append(
            f"Parallel retrieval error_type={type(e).__name__}"
        )
        state["runtime_executor_metrics"] = get_runtime_executor().metrics()

    return state


async def invoke_agent_with_timeout(
    tool_name: str,
    agent_graph: Any,
    query: str,
    context: Dict[str, Any],
    timeout_sec: float,
    trace_id: str,
    verifier: Optional[RuntimeVerifier] = None,
    max_retries: int = 1,
    max_synthesis_repairs: int = 1,
    deadline_at: Optional[float] = None,
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
    verifier = verifier or build_runtime_verifier(context)
    max_retries = max(0, min(1, int(max_retries)))
    max_synthesis_repairs = max(0, min(1, int(max_synthesis_repairs)))
    deadline_at = deadline_at or (
        time.monotonic()
        + timeout_sec * (max_retries + max_synthesis_repairs + 1)
    )
    started_at = time.time()
    current_query = query
    current_context = dict(context)
    traces: List[Dict[str, Any]] = []
    repair_history: List[Dict[str, Any]] = []
    retrieval_retries_used = 0
    synthesis_repairs_used = 0
    attempt_sequence = 0

    while retrieval_retries_used <= max_retries:
        remaining = deadline_at - time.monotonic()
        if remaining <= 0:
            error_msg = f"Agent subgraph {tool_name} exceeded runtime deadline"
            logger.warning("[%s] %s", trace_id, error_msg)
            return _terminal_failure_result(
                tool_name=tool_name,
                query=query,
                context=current_context,
                trace_id=trace_id,
                started_at=started_at,
                failed_check="agent_deadline_exceeded",
                error_message=error_msg,
                attempt_id=str(
                    current_context.get("attempt_id")
                    or f"{trace_id}:{tool_name}:{attempt_sequence + 1}"
                ),
                parent_attempt_id=current_context.get("parent_attempt_id"),
                traces=traces,
                repair_history=repair_history,
            )

        invocation_context = dict(current_context)
        invocation_context.setdefault("trace_id", trace_id)
        invocation_context.setdefault("original_query", query)
        invocation_context.setdefault(
            "attempt_id", f"{trace_id}:{tool_name}:{attempt_sequence + 1}"
        )
        invocation_context["agent_name"] = tool_name
        invocation_context["_runtime_deadline_at_monotonic"] = min(
            deadline_at, time.monotonic() + min(timeout_sec, remaining)
        )

        try:
            agent_output, invocation_telemetry = await _run_blocking_with_timeout(
                lambda q=current_query, c=invocation_context: _invoke_agent_with_telemetry(
                    agent_graph, q, c
                ),
                timeout=min(timeout_sec, remaining),
            )
        except ExecutorSaturatedError as exc:
            metrics = get_runtime_executor().metrics()
            error_msg = (
                f"Agent subgraph {tool_name} could not start "
                f"(error_type={type(exc).__name__})"
            )
            logger.warning("[%s] %s metrics=%s", trace_id, error_msg, metrics)
            return _terminal_failure_result(
                tool_name=tool_name,
                query=query,
                context=invocation_context,
                trace_id=trace_id,
                started_at=started_at,
                failed_check="executor_saturated",
                error_message=error_msg,
                attempt_id=str(invocation_context["attempt_id"]),
                parent_attempt_id=invocation_context.get("parent_attempt_id"),
                traces=traces,
                repair_history=repair_history,
                error_metadata={"executor_metrics": metrics},
            )
        except asyncio.TimeoutError:
            error_msg = f"Agent subgraph {tool_name} timed out after {min(timeout_sec, remaining):.1f}s"
            logger.warning("[%s] %s", trace_id, error_msg)
            # run_in_executor cannot cancel a running Python thread. Retrying
            # here would overlap attempts and can outlive the request budget.
            return _terminal_failure_result(
                tool_name=tool_name,
                query=query,
                context=invocation_context,
                trace_id=trace_id,
                started_at=started_at,
                failed_check="agent_timeout",
                error_message=error_msg,
                attempt_id=str(invocation_context["attempt_id"]),
                parent_attempt_id=invocation_context.get("parent_attempt_id"),
                traces=traces,
                repair_history=repair_history,
                error_metadata={
                    "executor_metrics": get_runtime_executor().metrics()
                },
            )
        except Exception as exc:
            error_msg = (
                f"Agent subgraph {tool_name} failed "
                f"(error_type={type(exc).__name__})"
            )
            logger.error("[%s] %s", trace_id, error_msg)
            return _terminal_failure_result(
                tool_name=tool_name,
                query=query,
                context=invocation_context,
                trace_id=trace_id,
                started_at=started_at,
                failed_check="agent_error",
                error_message=error_msg,
                attempt_id=str(invocation_context["attempt_id"]),
                parent_attempt_id=invocation_context.get("parent_attempt_id"),
                traces=traces,
                repair_history=repair_history,
            )

        trace = getattr(agent_output, "evaluation_trace", None)
        if trace is None:
            synthetic_state = {
                "expanded_query": current_query,
                "retrieval_results": list(agent_output.sources or []),
                "reranked_results": list(agent_output.sources or []),
                "answer": agent_output.answer,
                "citations": agent_output.citations,
                "model_used": agent_output.model_used,
                "execution_time_sec": agent_output.execution_time_sec,
                "error": agent_output.error,
            }
            trace = build_agent_evaluation_trace(
                agent_name=tool_name,
                domain=agent_output.domain or tool_name,
                original_query=query,
                state=synthetic_state,
                context=invocation_context,
            )
        else:
            trace.trace_id = trace_id
            trace.agent_name = tool_name
            trace.original_query = query
            trace.attempt_id = str(invocation_context["attempt_id"])
            parent_attempt_id = invocation_context.get("parent_attempt_id")
            trace.parent_attempt_id = (
                str(parent_attempt_id) if parent_attempt_id else None
            )
            trace.retry_feedback = list(
                invocation_context.get("verification_feedback") or []
            )
            trace.repair_history = list(
                invocation_context.get("repair_history") or []
            )
        _apply_invocation_telemetry(trace, invocation_telemetry)
        _append_retrieval_event(
            trace,
            latency_sec=float(
                trace.stage_latency_sec.get("retrieval") or 0.0
            ),
            status="error" if trace.errors else "success",
            repair_status=(
                "retrieval_retry"
                if retrieval_retries_used
                else "initial"
            ),
            error_type="agent_retrieval_error" if trace.errors else "",
        )

        decision = verifier.verify(
            trace,
            retries_remaining=max(
                max_retries - retrieval_retries_used,
                max_synthesis_repairs - synthesis_repairs_used,
            ),
            retrieval_retries_remaining=(
                max_retries - retrieval_retries_used
            ),
            synthesis_repairs_remaining=(
                max_synthesis_repairs - synthesis_repairs_used
            ),
        )
        traces.append(trace.to_dict())
        result = _agent_output_to_mcp_result(agent_output, trace=trace)
        result.update(
            {
                "retrieval_time_sec": time.time() - started_at,
                "verification_decision": decision.to_dict(),
                "evaluation_traces": list(traces),
                "repair_history": list(repair_history),
                "evidence_limited": decision.status == "evidence_limited",
            }
        )
        _attach_attempt_aggregate(result, traces)

        logger.info(
            "[%s] Agent subgraph %s attempt=%d: %d sources, "
            "confidence=%.2f, verification=%s",
            trace_id,
            tool_name,
            attempt_sequence + 1,
            len(result.get("results", [])),
            agent_output.confidence,
            decision.status,
        )

        if (
            decision.status == "retry_synthesis"
            and synthesis_repairs_used < max_synthesis_repairs
        ):
            synthesis_repairs_used += 1
            attempt_sequence += 1
            _, repair_context, repair_event = build_retry_request(
                original_query=query,
                context=invocation_context,
                decision=decision,
                trace=trace,
                attempt_number=attempt_sequence,
            )
            repair_history.append(repair_event)
            remaining = deadline_at - time.monotonic()
            if remaining <= 0:
                error_msg = (
                    f"Agent synthesis repair {tool_name} exceeded runtime deadline"
                )
                return _terminal_failure_result(
                    tool_name=tool_name,
                    query=query,
                    context=repair_context,
                    trace_id=trace_id,
                    started_at=started_at,
                    failed_check="agent_repair_deadline_exceeded",
                    error_message=error_msg,
                    attempt_id=str(repair_context["attempt_id"]),
                    parent_attempt_id=repair_context.get("parent_attempt_id"),
                    traces=traces,
                    repair_history=repair_history,
                )
            try:
                repaired_output = await _run_blocking_with_timeout(
                    lambda: repair_agent_synthesis(
                        agent_output=agent_output,
                        trace=trace,
                        decision=decision,
                        original_query=query,
                        context=repair_context,
                    ),
                    timeout=min(timeout_sec, remaining),
                )
            except ExecutorSaturatedError as exc:
                metrics = get_runtime_executor().metrics()
                error_msg = (
                    f"Agent synthesis repair {tool_name} could not start "
                    f"(error_type={type(exc).__name__})"
                )
                return _terminal_failure_result(
                    tool_name=tool_name,
                    query=query,
                    context=repair_context,
                    trace_id=trace_id,
                    started_at=started_at,
                    failed_check="executor_saturated",
                    error_message=error_msg,
                    attempt_id=str(repair_context["attempt_id"]),
                    parent_attempt_id=repair_context.get("parent_attempt_id"),
                    traces=traces,
                    repair_history=repair_history,
                    error_metadata={"executor_metrics": metrics},
                )
            except asyncio.TimeoutError:
                error_msg = f"Agent synthesis repair {tool_name} timed out"
                failed_repair_trace = _record_failed_agent_repair(
                    trace, repair_context, asyncio.TimeoutError(error_msg)
                )
                traces.append(failed_repair_trace.to_dict())
                return _terminal_failure_result(
                    tool_name=tool_name,
                    query=query,
                    context=repair_context,
                    trace_id=trace_id,
                    started_at=started_at,
                    failed_check="agent_repair_timeout",
                    error_message=error_msg,
                    attempt_id=str(repair_context["attempt_id"]),
                    parent_attempt_id=repair_context.get("parent_attempt_id"),
                    traces=traces,
                    repair_history=repair_history,
                    error_metadata={
                        "executor_metrics": get_runtime_executor().metrics()
                    },
                )
            except Exception as exc:
                error_msg = (
                    f"Agent synthesis repair {tool_name} failed "
                    f"(error_type={type(exc).__name__})"
                )
                failed_repair_trace = _record_failed_agent_repair(
                    trace, repair_context, exc
                )
                traces.append(failed_repair_trace.to_dict())
                return _terminal_failure_result(
                    tool_name=tool_name,
                    query=query,
                    context=repair_context,
                    trace_id=trace_id,
                    started_at=started_at,
                    failed_check="agent_repair_error",
                    error_message=error_msg,
                    attempt_id=str(repair_context["attempt_id"]),
                    parent_attempt_id=repair_context.get("parent_attempt_id"),
                    traces=traces,
                    repair_history=repair_history,
                )

            repaired_trace = repaired_output.evaluation_trace
            repaired_decision = verifier.verify(
                repaired_trace,
                retries_remaining=0,
                retrieval_retries_remaining=0,
                synthesis_repairs_remaining=0,
            )
            traces.append(repaired_trace.to_dict())
            repaired_result = _agent_output_to_mcp_result(
                repaired_output, trace=repaired_trace
            )
            repaired_result.update(
                {
                    "retrieval_time_sec": time.time() - started_at,
                    "verification_decision": repaired_decision.to_dict(),
                    "evaluation_traces": list(traces),
                    "repair_history": list(repair_history),
                    "evidence_limited": (
                        repaired_decision.status == "evidence_limited"
                    ),
                }
            )
            _attach_attempt_aggregate(repaired_result, traces)
            return repaired_result

        if (
            decision.status == "retry_retrieval"
            and retrieval_retries_used < max_retries
        ):
            retrieval_retries_used += 1
            attempt_sequence += 1
            current_query, current_context, repair_event = build_retry_request(
                original_query=query,
                context=invocation_context,
                decision=decision,
                trace=trace,
                attempt_number=attempt_sequence,
            )
            repair_history.append(repair_event)
            continue

        return result

    raise RuntimeError("bounded agent retry loop terminated unexpectedly")


async def invoke_tool_with_timeout(
    tool_name: str,
    tool: Any,
    query: str,
    context: Dict[str, Any],
    timeout_sec: float,
    trace_id: str,
    verifier: Optional[RuntimeVerifier] = None,
    max_retries: int = 1,
    deadline_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Invoke and verify an MCP retrieval tool with one bounded retry."""
    verifier = verifier or build_runtime_verifier(context)
    max_retries = max(0, min(1, int(max_retries)))
    deadline_at = deadline_at or (
        time.monotonic() + timeout_sec * (max_retries + 1)
    )
    started_at = time.time()
    current_query = query
    current_context = dict(context)
    traces: List[Dict[str, Any]] = []
    repair_history: List[Dict[str, Any]] = []
    attempt_number = 0

    while attempt_number <= max_retries:
        attempt_started_at = time.time()
        remaining = deadline_at - time.monotonic()
        attempt_id = str(
            current_context.get("attempt_id")
            or f"{trace_id}:{tool_name}:{attempt_number + 1}"
        )
        parent_attempt_id = current_context.get("parent_attempt_id")
        if remaining <= 0:
            error_msg = f"Tool {tool_name} exceeded runtime deadline"
            return _terminal_failure_result(
                tool_name=tool_name,
                query=query,
                context=current_context,
                trace_id=trace_id,
                started_at=started_at,
                failed_check="tool_deadline_exceeded",
                error_message=error_msg,
                attempt_id=attempt_id,
                parent_attempt_id=parent_attempt_id,
                traces=traces,
                repair_history=repair_history,
            )

        invocation_context = dict(current_context)
        invocation_context.update(
            {
                "trace_id": trace_id,
                "original_query": query,
                "attempt_id": attempt_id,
                "agent_name": tool_name,
                "_runtime_deadline_at_monotonic": min(
                    deadline_at, time.monotonic() + min(timeout_sec, remaining)
                ),
            }
        )
        call_timeout = min(timeout_sec, remaining)
        try:
            if hasattr(tool, "invoke_async"):
                raw_result = await asyncio.wait_for(
                    tool.invoke_async(
                        query=current_query, context=invocation_context
                    ),
                    timeout=call_timeout,
                )
            else:
                raw_result = await _run_blocking_with_timeout(
                    lambda q=current_query, c=invocation_context: tool.invoke(
                        query=q, context=c
                    ),
                    timeout=call_timeout,
                )
            if not isinstance(raw_result, dict):
                raise TypeError("tool result must be a dictionary")
            result = dict(raw_result)
        except ExecutorSaturatedError as exc:
            metrics = get_runtime_executor().metrics()
            error_msg = (
                f"Tool {tool_name} could not start "
                f"(error_type={type(exc).__name__})"
            )
            return _terminal_failure_result(
                tool_name=tool_name,
                query=query,
                context=invocation_context,
                trace_id=trace_id,
                started_at=started_at,
                failed_check="executor_saturated",
                error_message=error_msg,
                attempt_id=attempt_id,
                parent_attempt_id=parent_attempt_id,
                traces=traces,
                repair_history=repair_history,
                error_metadata={"executor_metrics": metrics},
            )
        except asyncio.TimeoutError:
            error_msg = f"Tool {tool_name} timed out after {call_timeout:.1f}s"
            logger.warning("[%s] %s", trace_id, error_msg)
            return _terminal_failure_result(
                tool_name=tool_name,
                query=query,
                context=invocation_context,
                trace_id=trace_id,
                started_at=started_at,
                failed_check="tool_timeout",
                error_message=error_msg,
                attempt_id=attempt_id,
                parent_attempt_id=parent_attempt_id,
                traces=traces,
                repair_history=repair_history,
                error_metadata={
                    "executor_metrics": get_runtime_executor().metrics()
                },
            )
        except Exception as exc:
            error_msg = (
                f"Tool {tool_name} failed "
                f"(error_type={type(exc).__name__})"
            )
            logger.error("[%s] %s", trace_id, error_msg)
            result = {"results": [], "error": error_msg}

        if result.get("error"):
            # MCP tools are extensible and may return provider payloads or
            # echo medical query text. Keep only a stable boundary error.
            result["error"] = "tool_retrieval_error"

        documents = list(result.get("results") or [])
        answer = str(result.get("answer") or result.get("agent_answer") or "")
        trace_context = dict(invocation_context)
        trace_context["retrieval_only"] = not bool(answer.strip())
        trace = build_agent_evaluation_trace(
            agent_name=tool_name,
            domain=tool_name,
            original_query=query,
            state={
                "expanded_query": current_query,
                "retrieval_results": documents,
                "reranked_results": documents,
                "answer": answer,
                "citations": list(result.get("citations") or []),
                "synthesis_context": list(
                    result.get("synthesis_context") or []
                ),
                "model_used": str(result.get("model_used") or ""),
                "execution_time_sec": time.time() - attempt_started_at,
                "retrieval_time_sec": time.time() - attempt_started_at,
                "token_usage": {"total": int(result.get("tokens_used") or 0)},
                "cost_breakdown_usd": {
                    "tool": float(result.get("cost") or 0.0)
                },
                "error": result.get("error"),
                "attempt_events": list(result.get("attempt_events") or []),
            },
            context=trace_context,
        )
        _append_retrieval_event(
            trace,
            latency_sec=time.time() - attempt_started_at,
            status="error" if result.get("error") else "success",
            repair_status=(
                "retrieval_retry" if attempt_number else "initial"
            ),
            error_type=(
                "tool_retrieval_error" if result.get("error") else ""
            ),
        )
        decision = verifier.verify(
            trace, retries_remaining=max_retries - attempt_number
        )
        traces.append(trace.to_dict())
        result.setdefault("tokens_used", 0)
        result.setdefault("cost", 0.0)
        result["retrieval_time_sec"] = time.time() - started_at
        result.setdefault("error", None)
        result.update(
            {
                "evaluation_trace": trace.to_dict(),
                "evaluation_traces": list(traces),
                "verification_decision": decision.to_dict(),
                "repair_history": list(repair_history),
                "evidence_limited": decision.status == "evidence_limited",
            }
        )
        _attach_attempt_aggregate(result, traces)

        if (
            decision.status == "retry_retrieval"
            and attempt_number < max_retries
        ):
            attempt_number += 1
            current_query, current_context, repair_event = build_retry_request(
                original_query=query,
                context=invocation_context,
                decision=decision,
                trace=trace,
                attempt_number=attempt_number,
            )
            repair_history.append(repair_event)
            continue

        result["repair_history"] = list(repair_history)
        logger.debug(
            "[%s] Tool %s: %d results, verification=%s",
            trace_id,
            tool_name,
            len(documents),
            decision.status,
        )
        return result

    raise RuntimeError("bounded MCP tool retry loop terminated unexpectedly")
