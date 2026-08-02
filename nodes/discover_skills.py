"""
Node 2: discover_skills — Tool selection via semantic matching.

Uses skill router to rank available MCP tools by relevance to the query.
Selects top-K tools based on semantic similarity and keyword matching.
"""

from agent_state import AgentState
from evaluation_core import RuntimeDeadlineExceeded
from llm_client import LLMCallResult, LLMClient
from runtime_verification.telemetry import build_attempt_event
from skill_router import SkillRouter
import logging

logger = logging.getLogger(__name__)

_AGENT_ALIASES = {
    "local": "search_local_index",
    "pubmed": "search_pubmed",
    "pubmed_deep_research": "search_pubmed_deep",
    "clinical_trials": "search_clinical_trials",
    "fda": "search_fda",
}
_CANONICAL_AGENTS = set(_AGENT_ALIASES.values())


def _record_skill_telemetry(
    state: AgentState, call_results: list[LLMCallResult]
) -> None:
    """Add skill-routing provider calls to canonical request telemetry."""
    attempts = list(state.get("attempt_telemetry") or [])
    seen = {
        str(event.get("event_id"))
        for event in attempts
        if isinstance(event, dict) and event.get("event_id")
    }
    usage = dict(state.get("token_usage") or {})
    tokens_in = int(usage.get("input") or 0)
    tokens_out = int(usage.get("output") or 0)
    total_cost = float(state.get("cost_estimate") or 0.0)
    compute_time = float(
        state.get("attempt_compute_time_sec") or 0.0
    )
    trace_id = str(state.get("trace_id") or "")

    for index, call in enumerate(call_results, 1):
        stage = (
            "skill_discovery_embedding"
            if str(
                call.provider_metadata.get("telemetry_stage") or ""
            )
            == "embedding"
            else "skill_discovery_llm"
        )
        attempt_id = f"{trace_id}:skill-discovery:{index}"
        event = build_attempt_event(
            trace_id=trace_id,
            attempt_id=attempt_id,
            parent_attempt_id=None,
            stage=stage,
            component="skill_router",
            status=str(call.status or "success"),
            repair_status="initial",
            model=call.model,
            model_revision=call.model_revision,
            tokens_in=call.tokens_in,
            tokens_out=call.tokens_out,
            cost_usd=call.cost_usd,
            latency_sec=call.latency_sec,
            finish_reason=call.finish_reason,
            deadline_exhausted=(
                call.error_type == "RuntimeDeadlineExceeded"
            ),
            error_type=call.error_type,
            provider_metadata=call.provider_metadata,
            event_id=f"{attempt_id}:{stage}",
        )
        if event["event_id"] in seen:
            continue
        seen.add(event["event_id"])
        attempts.append(event)
        tokens_in += int(call.tokens_in or 0)
        tokens_out += int(call.tokens_out or 0)
        total_cost += float(call.cost_usd or 0.0)
        compute_time += float(call.latency_sec or 0.0)

    state["attempt_telemetry"] = attempts
    state["token_usage"] = {
        "input": tokens_in,
        "output": tokens_out,
        "total": tokens_in + tokens_out,
    }
    state["tokens_used"] = tokens_in + tokens_out
    state["cost_estimate"] = total_cost
    state["attempt_compute_time_sec"] = compute_time


def discover_skills(state: AgentState) -> AgentState:
    """
    Discover and rank relevant MCP tools for the query.

    Uses SkillRouter to perform semantic similarity + keyword matching.
    Respects explicit agent_to_use from context if provided (overrides discovery).

    Args:
        state: Current agent state

    Returns:
        Updated state with discovered_skills and skill_scores
    """
    trace_id = state.get("trace_id", "unknown")

    try:
        # Check if user explicitly specified agents (overrides discovery)
        if "agents_to_use" in state.get("context", {}) and state["context"]["agents_to_use"]:
            requested_agents = list(state["context"]["agents_to_use"])
            explicit_agents = [
                _AGENT_ALIASES.get(str(agent), str(agent))
                for agent in requested_agents
            ]
            unsupported = [
                agent
                for agent in explicit_agents
                if agent not in _CANONICAL_AGENTS
            ]
            if unsupported:
                state["discovered_skills"] = []
                state["skill_scores"] = {}
                state["error_occurred"] = True
                state["is_partial_response"] = True
                state["error_messages"].append(
                    "Unsupported explicit retrieval agent"
                )
                logger.warning(
                    "[%s] discover_skills rejected unsupported explicit "
                    "agents count=%d",
                    trace_id,
                    len(unsupported),
                )
                return state
            logger.info(
                "[%s] discover_skills using explicit agents count=%d",
                trace_id,
                len(explicit_agents),
            )
            state["discovered_skills"] = explicit_agents
            # Assign equal scores to explicit agents
            state["skill_scores"] = {agent: 1.0 for agent in explicit_agents}
            return state

        # Use SkillRouter for semantic discovery
        llm_client = LLMClient()
        history_reader = getattr(
            llm_client, "thread_call_history", None
        )
        history_start = (
            len(history_reader()) if callable(history_reader) else 0
        )
        router = SkillRouter()
        try:
            discovered_tools, scores = router.rank_tools(
                query=state["input_query"],
                top_k=3,  # Select top 3 tools by default
                deadline_at=state.get("context", {}).get(
                    "_runtime_deadline_at_monotonic"
                ),
            )
        finally:
            if callable(history_reader):
                new_calls = list(history_reader())[history_start:]
                _record_skill_telemetry(
                    state,
                    [
                        call
                        for call in new_calls
                        if isinstance(call, LLMCallResult)
                    ],
                )

        state["discovered_skills"] = discovered_tools
        state["skill_scores"] = dict(zip(discovered_tools, scores))

        logger.info(
            f"[{trace_id}] discover_skills: Discovered {len(discovered_tools)} tools: "
            f"{[(t, f'{s:.2f}') for t, s in zip(discovered_tools, scores)]}"
        )

    except RuntimeDeadlineExceeded:
        state["discovered_skills"] = []
        state["skill_scores"] = {}
        state["error_occurred"] = True
        state["is_partial_response"] = True
        state["error_messages"].append(
            "Skill discovery error_type=RuntimeDeadlineExceeded"
        )
        logger.warning(
            "[%s] discover_skills deadline exhausted", trace_id
        )
    except Exception as e:
        logger.error(
            "[%s] discover_skills failed error_type=%s",
            trace_id,
            type(e).__name__,
        )
        # Fallback: use all available tools
        all_tools = [
            "search_local_index",
            "search_pubmed",
            "search_pubmed_deep",
            "search_clinical_trials",
            "search_fda",
        ]
        state["discovered_skills"] = all_tools
        state["skill_scores"] = {tool: 0.5 for tool in all_tools}
        state["error_occurred"] = True
        state["error_messages"].append(
            f"Skill discovery error_type={type(e).__name__}; using all tools"
        )
        logger.warning(
            f"[{trace_id}] discover_skills: Fallback to all tools due to error"
        )

    return state
