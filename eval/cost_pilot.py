"""
Cost pilot aggregation for full LangGraph orchestrator runs.

Separates offline cost measurement from runtime verification logic.
Uses EvaluationTrace / attempt telemetry already captured in AgentState.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from evaluation_core.privacy import stable_query_fingerprint
from evaluation_core.schemas import EVALUATION_TRACE_SCHEMA_VERSION
from runtime_verification.telemetry import aggregate_attempt_telemetry

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX_PATH = _REPO_ROOT / "eval" / "configs" / "model_matrix.yaml"

# Mid-token full-orchestrator planning scenario (MedAgentsBench cost tables).
MID_TOKEN_PLANNING_INPUT = 55_000
MID_TOKEN_PLANNING_OUTPUT = 4_000
COST_PILOT_VERSION = "1.1.0"

# Short CLI aliases → MCP tool names used by skill discovery override.
AGENT_ALIASES: Dict[str, str] = {
    "pubmed": "search_pubmed",
    "fda": "search_fda",
    "clinical_trials": "search_clinical_trials",
    "local": "search_local_index",
    "pubmed_deep": "search_pubmed_deep",
}

# Default full-split sizes for projection (official benchmark test splits).
DEFAULT_FULL_SPLIT_SIZES: Dict[str, int] = {
    "medagentsbench_test_hard": 862,
    "medqa": 1273,
    "medmcqa": 4183,
    "pubmedqa_pqal": 500,
    "bioasq_task_b": 500,
    "custom": 100,
}

VERIFIER_STAGES = frozenset(
    {
        "conditional_claim_verification",
        "verification",
        "claim_verification",
    }
)

REPAIR_STATUSES = frozenset(
    {
        "synthesis_repair",
        "agent_synthesis_repair",
        "retrieval_retry",
        "fallback_regeneration",
    }
)


@dataclass
class QuestionCostRecord:
    """Per-question cost telemetry (no raw query text)."""

    question_id: str
    trace_id: str
    query_fingerprint: str
    query_length: int
    tokens_in: int
    tokens_out: int
    total_tokens: int
    cost_usd: float
    latency_sec: float
    retry_count: int
    repair_count: int
    verifier_calls: int
    evidence_limited: bool
    fallback_triggered: bool
    error_occurred: bool
    models_used: List[str] = field(default_factory=list)


@dataclass
class PilotAggregate:
    """Aggregate statistics across pilot questions."""

    n_samples: int
    n_errors: int
    tokens_in_total: int
    tokens_out_total: int
    tokens_total: int
    cost_usd_total: float
    cost_usd_mean: float
    cost_usd_median: float
    tokens_mean: float
    latency_sec_mean: float
    latency_sec_p50: float
    latency_sec_p95: float
    retry_count_total: int
    repair_count_total: int
    verifier_calls_total: int
    retry_count_mean: float
    repair_count_mean: float
    verifier_calls_mean: float
    evidence_limited_rate: float
    fallback_triggered_rate: float


@dataclass
class CostProjection:
    """Linear cost projection from pilot mean."""

    target_n: int
    projected_cost_usd: float
    projected_tokens: int
    projected_latency_sec_p50: float
    projected_latency_sec_p95: float


@dataclass
class MatrixPlanningRow:
    """Offline per-model cost row from matrix pricing × mid-token scenario."""

    model_id: str
    tokens_in: int
    tokens_out: int
    cost_per_question_usd: float
    target_n: int
    projected_cost_usd: float
    recommended_purchase_usd: float
    pricing_verified: bool
    pricing_label: str = ""
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


def load_model_matrix(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load eval/configs/model_matrix.yaml."""
    matrix_path = path or DEFAULT_MATRIX_PATH
    with matrix_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_cost_pilot_defaults(matrix: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return cost_pilot_defaults block from model matrix."""
    data = dict(matrix or load_model_matrix())
    defaults = dict(data.get("cost_pilot_defaults") or {})
    mid = dict(defaults.get("mid_tokens_per_question") or {})
    defaults.setdefault("projection_targets", [100, 500, 862])
    defaults.setdefault("mid_tokens_per_question", {
        "input": MID_TOKEN_PLANNING_INPUT,
        "output": MID_TOKEN_PLANNING_OUTPUT,
    })
    if mid.get("input"):
        defaults["mid_tokens_per_question"]["input"] = int(mid["input"])
    if mid.get("output"):
        defaults["mid_tokens_per_question"]["output"] = int(mid["output"])
    return defaults


def compute_mid_token_cost_per_question(
    cost_per_1k_input: float,
    cost_per_1k_output: float,
    *,
    tokens_in: int = MID_TOKEN_PLANNING_INPUT,
    tokens_out: int = MID_TOKEN_PLANNING_OUTPUT,
) -> float:
    """Cost for one full-orchestrator question at fixed token volumes."""
    return (tokens_in * cost_per_1k_input + tokens_out * cost_per_1k_output) / 1000.0


def recommend_credit_purchase_for_cost(
    cost_per_question_usd: float,
    *,
    target_n: int = 100,
    variance_buffer: float = 0.25,
    fixed_buffer_usd: float = 5.0,
) -> Dict[str, float]:
    """Credit recommendation from a scalar per-question cost (matrix planning)."""
    base = cost_per_question_usd * target_n
    recommended = base * (1.0 + variance_buffer) + fixed_buffer_usd
    return {
        "target_n": float(target_n),
        "base_projected_usd": round(base, 4),
        "variance_buffer_fraction": variance_buffer,
        "fixed_buffer_usd": fixed_buffer_usd,
        "recommended_purchase_usd": round(recommended, 2),
    }


def _expand_pricing_variants(candidate: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand date-gated pricing schedules into separate planning rows."""
    schedule = candidate.get("pricing_schedule")
    if not schedule:
        return [dict(candidate)]
    variants: List[Dict[str, Any]] = []
    for entry in schedule:
        variant = dict(candidate)
        variant["cost_per_1k_input"] = float(entry["cost_per_1k_input"])
        variant["cost_per_1k_output"] = float(entry["cost_per_1k_output"])
        variant["pricing_label"] = str(entry.get("label") or "")
        if entry.get("effective_through"):
            variant["pricing_effective"] = f"through {entry['effective_through']}"
        elif entry.get("effective_from"):
            variant["pricing_effective"] = f"from {entry['effective_from']}"
        else:
            variant["pricing_effective"] = variant["pricing_label"]
        variants.append(variant)
    return variants


def _display_model_id(candidate: Mapping[str, Any]) -> str:
    model_id = str(candidate.get("id") or "")
    label = str(candidate.get("pricing_label") or "").strip()
    effective = str(candidate.get("pricing_effective") or "").strip()
    if label and effective:
        return f"{model_id} [{label}; {effective}]"
    if label:
        return f"{model_id} [{label}]"
    return model_id


def build_matrix_planning_projections(
    *,
    matrix_path: Optional[Path] = None,
    target_n: int = DEFAULT_FULL_SPLIT_SIZES["medagentsbench_test_hard"],
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    variance_buffer: Optional[float] = None,
    fixed_buffer_usd: Optional[float] = None,
    include_deprecated: bool = False,
) -> Dict[str, Any]:
    """
    Offline model×N planning table from model_matrix.yaml prices × mid tokens.

    Does not call LLM APIs. Used in dry-run / credit planning for MedAgentsBench.
    """
    matrix = load_model_matrix(matrix_path)
    defaults = load_cost_pilot_defaults(matrix)
    mid = dict(defaults.get("mid_tokens_per_question") or {})
    tokens_in = int(tokens_in if tokens_in is not None else mid.get("input", MID_TOKEN_PLANNING_INPUT))
    tokens_out = int(tokens_out if tokens_out is not None else mid.get("output", MID_TOKEN_PLANNING_OUTPUT))
    variance_buffer = (
        float(variance_buffer)
        if variance_buffer is not None
        else float(defaults.get("variance_buffer", 0.25))
    )
    fixed_buffer_usd = (
        float(fixed_buffer_usd)
        if fixed_buffer_usd is not None
        else float(defaults.get("fixed_buffer_usd", 5.0))
    )

    rows: List[MatrixPlanningRow] = []
    generation = list((matrix.get("roles") or {}).get("generation", {}).get("candidates") or [])
    for candidate in generation:
        tier = str(candidate.get("tier") or "")
        if tier == "deprecated" and not include_deprecated:
            continue
        for variant in _expand_pricing_variants(candidate):
            in_rate = variant.get("cost_per_1k_input")
            out_rate = variant.get("cost_per_1k_output")
            if in_rate is None or out_rate is None:
                continue
            cost_per_q = compute_mid_token_cost_per_question(
                float(in_rate),
                float(out_rate),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
            rec = recommend_credit_purchase_for_cost(
                cost_per_q,
                target_n=target_n,
                variance_buffer=variance_buffer,
                fixed_buffer_usd=fixed_buffer_usd,
            )
            rows.append(
                MatrixPlanningRow(
                    model_id=_display_model_id(variant),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_per_question_usd=round(cost_per_q, 6),
                    target_n=target_n,
                    projected_cost_usd=round(rec["base_projected_usd"], 2),
                    recommended_purchase_usd=rec["recommended_purchase_usd"],
                    pricing_verified=bool(variant.get("pricing_verified", True)),
                    pricing_label=str(variant.get("pricing_label") or ""),
                    cost_per_1k_input=float(in_rate),
                    cost_per_1k_output=float(out_rate),
                )
            )

    rows.sort(key=lambda row: row.projected_cost_usd)
    return {
        "scenario": "mid_token_matrix_planning",
        "dataset": "medagentsbench_test_hard",
        "target_n": target_n,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "variance_buffer": variance_buffer,
        "fixed_buffer_usd": fixed_buffer_usd,
        "pricing_as_of": matrix.get("pricing_as_of"),
        "rows": [asdict(row) for row in rows],
    }


def render_matrix_planning_markdown(planning: Mapping[str, Any]) -> str:
    """Markdown table for offline matrix×N planning."""
    lines = [
        "## Matrix planning (mid-token scenario — offline)",
        "",
        f"- Dataset: `{planning.get('dataset', 'medagentsbench_test_hard')}`",
        f"- Target N: **{planning.get('target_n', 862)}**",
        f"- Tokens per question: **{planning.get('tokens_in'):,}** in / "
        f"**{planning.get('tokens_out'):,}** out",
        f"- Pricing as of: {planning.get('pricing_as_of', 'n/a')}",
        "",
        "| Model | $/question | Projected N | Recommended purchase | Verified |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in planning.get("rows") or []:
        verified = "yes" if row.get("pricing_verified") else "**no**"
        lines.append(
            f"| `{row['model_id']}` | ${row['cost_per_question_usd']:.4f} | "
            f"${row['projected_cost_usd']:,.2f} | "
            f"${row['recommended_purchase_usd']:,.2f} | {verified} |"
        )
    lines.extend(
        [
            "",
            "Formula: `projected = cost_per_question × N`; "
            "`recommended = projected × (1 + variance_buffer) + fixed_buffer`.",
        ]
    )
    return "\n".join(lines)

def normalize_agent_names(agents: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not agents:
        return None
    normalized: List[str] = []
    for name in agents:
        key = str(name).strip().lower()
        normalized.append(AGENT_ALIASES.get(key, name))
    return normalized


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile; returns 0.0 for empty input."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = max(1, math.ceil((p / 100.0) * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def _count_verifier_calls(events: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    for event in events:
        stage = str(event.get("stage") or "").lower()
        component = str(event.get("component") or "").lower()
        if stage in VERIFIER_STAGES or "verifier" in component or "verification" in stage:
            count += 1
    return count


def _count_repairs_and_retries(
    attempt_events: Iterable[Mapping[str, Any]],
    repair_history: Iterable[Mapping[str, Any]],
) -> tuple[int, int]:
    repair_count = 0
    retry_count = 0
    for event in attempt_events:
        repair_status = str(event.get("repair_status") or "initial").lower()
        if repair_status in REPAIR_STATUSES or repair_status.endswith("_repair"):
            repair_count += 1
        if repair_status in {"retrieval_retry", "retry"} or "retry" in repair_status:
            retry_count += 1
    for entry in repair_history:
        action = str(entry.get("action") or entry.get("repair_type") or "").lower()
        if "retry" in action:
            retry_count += 1
        elif action:
            repair_count += 1
    return retry_count, repair_count


def _models_from_events(events: Iterable[Mapping[str, Any]]) -> List[str]:
    seen: set[str] = set()
    models: List[str] = []
    for event in events:
        model = str(event.get("model") or "").strip()
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return models


def extract_question_metrics(
    *,
    question_id: str,
    final_state: Mapping[str, Any],
    query_text: str,
) -> QuestionCostRecord:
    """Extract cost record from a completed graph state."""
    traces = list(final_state.get("evaluation_traces") or [])
    attempt_telemetry = list(final_state.get("attempt_telemetry") or [])
    repair_history = list(final_state.get("repair_history") or [])

    aggregate = aggregate_attempt_telemetry(traces)
    events = list(aggregate.get("attempt_telemetry") or [])
    if not events and attempt_telemetry:
        events = attempt_telemetry

    token_usage = dict(final_state.get("token_usage") or {})
    tokens_in = int(token_usage.get("input") or 0)
    tokens_out = int(token_usage.get("output") or 0)
    for event in events:
        usage = dict(event.get("token_usage") or {})
        if not tokens_in and usage.get("input"):
            tokens_in = int(usage.get("input") or 0)
        if not tokens_out and usage.get("output"):
            tokens_out = int(usage.get("output") or 0)

    total_tokens = int(
        aggregate.get("tokens_used")
        if aggregate.get("tokens_used") is not None
        else token_usage.get("total")
        if isinstance(token_usage.get("total"), (int, float))
        else tokens_in + tokens_out
    )
    if not tokens_in and not tokens_out and total_tokens:
        tokens_in = total_tokens

    cost_usd = float(
        aggregate.get("cost_usd")
        if aggregate.get("cost_usd") is not None
        else final_state.get("cost_estimate") or 0.0
    )
    latency_sec = float(final_state.get("execution_time_sec") or 0.0)

    retry_count, repair_count = _count_repairs_and_retries(events, repair_history)
    verifier_calls = _count_verifier_calls(events)
    verification_history = final_state.get("verification_history") or []
    if isinstance(verification_history, list):
        verifier_calls = max(verifier_calls, len(verification_history))

    return QuestionCostRecord(
        question_id=question_id,
        trace_id=str(final_state.get("trace_id") or ""),
        query_fingerprint=stable_query_fingerprint(query_text),
        query_length=len(query_text),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        latency_sec=latency_sec,
        retry_count=retry_count,
        repair_count=repair_count,
        verifier_calls=verifier_calls,
        evidence_limited=bool(final_state.get("evidence_limited")),
        fallback_triggered=bool(final_state.get("fallback_triggered")),
        error_occurred=bool(final_state.get("error_occurred")),
        models_used=_models_from_events(events),
    )


def aggregate_records(records: Sequence[QuestionCostRecord]) -> PilotAggregate:
    """Compute aggregate pilot statistics."""
    n_samples = len(records)
    if n_samples == 0:
        return PilotAggregate(
            n_samples=0,
            n_errors=0,
            tokens_in_total=0,
            tokens_out_total=0,
            tokens_total=0,
            cost_usd_total=0.0,
            cost_usd_mean=0.0,
            cost_usd_median=0.0,
            tokens_mean=0.0,
            latency_sec_mean=0.0,
            latency_sec_p50=0.0,
            latency_sec_p95=0.0,
            retry_count_total=0,
            repair_count_total=0,
            verifier_calls_total=0,
            retry_count_mean=0.0,
            repair_count_mean=0.0,
            verifier_calls_mean=0.0,
            evidence_limited_rate=0.0,
            fallback_triggered_rate=0.0,
        )

    costs = [r.cost_usd for r in records]
    latencies = [r.latency_sec for r in records]
    tokens = [r.total_tokens for r in records]

    return PilotAggregate(
        n_samples=n_samples,
        n_errors=sum(1 for r in records if r.error_occurred),
        tokens_in_total=sum(r.tokens_in for r in records),
        tokens_out_total=sum(r.tokens_out for r in records),
        tokens_total=sum(r.total_tokens for r in records),
        cost_usd_total=sum(costs),
        cost_usd_mean=statistics.mean(costs),
        cost_usd_median=statistics.median(costs),
        tokens_mean=statistics.mean(tokens),
        latency_sec_mean=statistics.mean(latencies),
        latency_sec_p50=percentile(latencies, 50),
        latency_sec_p95=percentile(latencies, 95),
        retry_count_total=sum(r.retry_count for r in records),
        repair_count_total=sum(r.repair_count for r in records),
        verifier_calls_total=sum(r.verifier_calls for r in records),
        retry_count_mean=statistics.mean([r.retry_count for r in records]),
        repair_count_mean=statistics.mean([r.repair_count for r in records]),
        verifier_calls_mean=statistics.mean([r.verifier_calls for r in records]),
        evidence_limited_rate=sum(1 for r in records if r.evidence_limited) / n_samples,
        fallback_triggered_rate=sum(1 for r in records if r.fallback_triggered) / n_samples,
    )


def project_costs(
    aggregate: PilotAggregate,
    target_ns: Sequence[int],
    *,
    latency_p50_sec: Optional[float] = None,
    latency_p95_sec: Optional[float] = None,
) -> List[CostProjection]:
    """Linear projection from pilot per-question means."""
    p50 = latency_p50_sec if latency_p50_sec is not None else aggregate.latency_sec_p50
    p95 = latency_p95_sec if latency_p95_sec is not None else aggregate.latency_sec_p95
    projections: List[CostProjection] = []
    for n in target_ns:
        projections.append(
            CostProjection(
                target_n=int(n),
                projected_cost_usd=aggregate.cost_usd_mean * int(n),
                projected_tokens=int(round(aggregate.tokens_mean * int(n))),
                projected_latency_sec_p50=p50,
                projected_latency_sec_p95=p95,
            )
        )
    return projections


def recommend_credit_purchase_usd(
    aggregate: PilotAggregate,
    *,
    target_n: int = 100,
    variance_buffer: float = 0.25,
    fixed_buffer_usd: float = 5.0,
) -> Dict[str, float]:
    """
    Credit purchase recommendation from pilot mean cost.

    Formula:
        base = mean_cost_usd * target_n
        recommended = base * (1 + variance_buffer) + fixed_buffer_usd
    """
    base = aggregate.cost_usd_mean * target_n
    recommended = base * (1.0 + variance_buffer) + fixed_buffer_usd
    return {
        "target_n": float(target_n),
        "base_projected_usd": round(base, 4),
        "variance_buffer_fraction": variance_buffer,
        "fixed_buffer_usd": fixed_buffer_usd,
        "recommended_purchase_usd": round(recommended, 2),
    }


def build_initial_state(
    *,
    question: str,
    model_id: str,
    agents_to_use: Optional[List[str]] = None,
    top_k: int = 5,
    runtime_verification_deadline_sec: float = 60.0,
    trace_id: Optional[str] = None,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Mirror research_agent_api_v2 AgentState initialization."""
    trace_id = trace_id or str(uuid.uuid4())
    context: Dict[str, Any] = {
        "model_id": model_id,
        "db_name": "index",
        "top_k": top_k,
        "clinical_trials_top_k": 10,
        "fda_top_k": 10,
        "max_trials": 25,
        "max_agent_retries": 1,
        "max_agent_synthesis_repairs": 1,
        "max_synthesis_repairs": 1,
        "runtime_verification_deadline_sec": runtime_verification_deadline_sec,
        "_runtime_deadline_at_monotonic": (
            time.monotonic() + runtime_verification_deadline_sec
        ),
    }
    if agents_to_use:
        context["agents_to_use"] = agents_to_use
    if extra_context:
        context.update(extra_context)

    return {
        "input_query": question,
        "context": context,
        "trace_id": trace_id,
        "timestamp_start": datetime.now(timezone.utc),
        "is_medical_query": True,
        "classification_confidence": 0.0,
        "classification_reason": "",
        "discovered_skills": [],
        "skill_scores": {},
        "retrieval_results": {},
        "tokens_used": {},
        "retrieval_time_sec": {},
        "total_retrieval_time_sec": 0.0,
        "intermediate_answer": "",
        "intermediate_sources": [],
        "intermediate_model_used": "",
        "synthesis_tokens_in": 0,
        "synthesis_tokens_out": 0,
        "synthesis_time_sec": 0.0,
        "last_synthesis_cost_usd": 0.0,
        "synthesis_context": [],
        "confidence_score": 0.0,
        "coverage_explanation": "",
        "confidence_components": {},
        "runtime_quality_score": 0.0,
        "runtime_quality_explanation": "",
        "coherence_score": 0.0,
        "coherence_explanation": "",
        "should_fallback": False,
        "coherence_eval_model_used": "",
        "fallback_count": 0,
        "fallback_answer": None,
        "fallback_triggered": False,
        "fallback_reason": "",
        "evaluation_traces": [],
        "verification_history": [],
        "verification_decision": None,
        "repair_history": [],
        "evidence_limited": False,
        "attempt_telemetry": [],
        "token_usage": {"input": 0, "output": 0, "total": 0},
        "runtime_executor_metrics": {},
        "output_answer": "",
        "output_sources": [],
        "output_citations": [],
        "output_disclaimer": "",
        "timestamp_end": datetime.now(timezone.utc),
        "execution_time_sec": 0.0,
        "cost_estimate": 0.0,
        "error_occurred": False,
        "error_messages": [],
        "is_partial_response": False,
    }


def synthesize_mock_final_state(
    *,
    question_id: str,
    question: str,
    model_id: str,
    index: int,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministic mock graph output for dry-run / offline tests."""
    trace_id = trace_id or str(uuid.uuid4())
    tokens_in = 800 + (index * 37) % 400
    tokens_out = 200 + (index * 13) % 150
    cost_usd = round((tokens_in * 0.005 + tokens_out * 0.015) / 1000.0, 6)
    latency = 3.5 + (index % 5) * 0.8
    repair_count = 1 if index % 4 == 0 else 0
    retry_count = 1 if index % 7 == 0 else 0
    verifier_calls = 1 + (index % 3)

    attempt_events = [
        {
            "event_id": f"{trace_id}:intent",
            "stage": "intent_classification",
            "model": model_id,
            "token_usage": {"input": 120, "output": 10, "total": 130},
            "cost_usd": 0.0008,
            "latency_sec": 0.4,
            "repair_status": "initial",
        },
        {
            "event_id": f"{trace_id}:synthesis",
            "stage": "agent_synthesis",
            "model": model_id,
            "token_usage": {"input": tokens_in - 130, "output": tokens_out, "total": tokens_in - 130 + tokens_out},
            "cost_usd": cost_usd,
            "latency_sec": latency - 0.4,
            "repair_status": "synthesis_repair" if repair_count else "initial",
        },
    ]
    for v in range(verifier_calls):
        attempt_events.append(
            {
                "event_id": f"{trace_id}:verify:{v}",
                "stage": "conditional_claim_verification",
                "model": model_id,
                "token_usage": {"input": 50, "output": 20, "total": 70},
                "cost_usd": 0.0004,
                "latency_sec": 0.2,
                "repair_status": "initial",
            }
        )

    repair_history = [{"action": "synthesis_repair"}] if repair_count else []
    if retry_count:
        repair_history.append({"action": "retrieval_retry"})

    return {
        "trace_id": trace_id,
        "input_query": question,
        "execution_time_sec": latency,
        "cost_estimate": cost_usd + verifier_calls * 0.0004,
        "token_usage": {
            "input": tokens_in + verifier_calls * 50,
            "output": tokens_out + verifier_calls * 20,
            "total": tokens_in + tokens_out + verifier_calls * 70,
        },
        "evaluation_traces": [
            {
                "trace_id": trace_id,
                "schema_version": EVALUATION_TRACE_SCHEMA_VERSION,
                "attempt_events": attempt_events,
            }
        ],
        "attempt_telemetry": attempt_events,
        "repair_history": repair_history,
        "verification_history": [{}] * verifier_calls,
        "evidence_limited": index % 11 == 0,
        "fallback_triggered": index % 9 == 0,
        "error_occurred": False,
    }


def load_pilot_questions(
    dataset_path: Path,
    *,
    n_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load questions from medical_benchmark.json-style files."""
    with dataset_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if n_samples is not None:
        rows = rows[:n_samples]
    return rows


def render_markdown_summary(
    *,
    run_meta: Mapping[str, Any],
    aggregate: PilotAggregate,
    projections: Sequence[CostProjection],
    recommendations: Mapping[str, Mapping[str, float]],
    per_question: Sequence[QuestionCostRecord],
    matrix_planning: Optional[Mapping[str, Any]] = None,
) -> str:
    """Human-readable summary for credit purchase decisions."""
    lines = [
        "# Cost Pilot Summary",
        "",
        f"- Run ID: `{run_meta.get('run_id', 'n/a')}`",
        f"- Timestamp: {run_meta.get('timestamp', 'n/a')}",
        f"- Mode: {run_meta.get('mode', 'live')}",
        f"- Model: `{run_meta.get('model_id', 'n/a')}`",
        f"- Agents: {', '.join(run_meta.get('agents') or ['auto-discovery'])}",
        f"- Samples: {aggregate.n_samples} ({aggregate.n_errors} errors)",
        f"- EvaluationTrace schema: `{EVALUATION_TRACE_SCHEMA_VERSION}`",
        "",
        "## Aggregate telemetry",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Mean cost (USD) | ${aggregate.cost_usd_mean:.4f} |",
        f"| Median cost (USD) | ${aggregate.cost_usd_median:.4f} |",
        f"| Total cost (USD) | ${aggregate.cost_usd_total:.4f} |",
        f"| Mean tokens | {aggregate.tokens_mean:.0f} |",
        f"| Latency p50 (s) | {aggregate.latency_sec_p50:.2f} |",
        f"| Latency p95 (s) | {aggregate.latency_sec_p95:.2f} |",
        f"| Mean retries | {aggregate.retry_count_mean:.2f} |",
        f"| Mean repairs | {aggregate.repair_count_mean:.2f} |",
        f"| Mean verifier calls | {aggregate.verifier_calls_mean:.2f} |",
        f"| Evidence-limited rate | {aggregate.evidence_limited_rate:.1%} |",
        f"| Fallback rate | {aggregate.fallback_triggered_rate:.1%} |",
        "",
        "## Cost projections",
        "",
        "| Target N | Projected USD | Projected tokens |",
        "| --- | --- | --- |",
    ]
    for proj in projections:
        lines.append(
            f"| {proj.target_n} | ${proj.projected_cost_usd:.2f} | {proj.projected_tokens:,} |"
        )

    if matrix_planning:
        lines.extend(["", render_matrix_planning_markdown(matrix_planning)])

    lines.extend(["", "## Credit purchase recommendation", ""])
    for label, rec in recommendations.items():
        lines.append(
            f"- **{label}**: base ${rec['base_projected_usd']:.2f} → "
            f"recommended **${rec['recommended_purchase_usd']:.2f}** "
            f"(+{int(rec['variance_buffer_fraction'] * 100)}% variance, "
            f"+${rec['fixed_buffer_usd']:.2f} fixed buffer)"
        )

    lines.extend(
        [
            "",
            "## Per-question fingerprints (no raw text)",
            "",
            "| ID | fingerprint | cost USD | tokens | latency s | retries | repairs | verifiers |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in per_question:
        lines.append(
            f"| {row.question_id} | `{row.query_fingerprint}` | "
            f"${row.cost_usd:.4f} | {row.total_tokens} | {row.latency_sec:.2f} | "
            f"{row.retry_count} | {row.repair_count} | {row.verifier_calls} |"
        )

    lines.extend(
        [
            "",
            "## Formula",
            "",
            "```",
            "projected_usd(N) = mean_cost_usd_per_question * N",
            "recommended purchase (USD) = projected_usd(N) * (1 + variance_buffer) + fixed_buffer_usd",
            "default: variance_buffer=0.25, fixed_buffer_usd=5.00",
            "```",
        ]
    )
    return "\n".join(lines)


def serialize_pilot_result(
    *,
    run_meta: Mapping[str, Any],
    records: Sequence[QuestionCostRecord],
    aggregate: PilotAggregate,
    projections: Sequence[CostProjection],
    recommendations: Mapping[str, Mapping[str, float]],
    matrix_planning: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """JSON-serializable pilot artifact."""
    payload: Dict[str, Any] = {
        "run_meta": dict(run_meta),
        "schema": {
            "evaluation_trace_version": EVALUATION_TRACE_SCHEMA_VERSION,
            "cost_pilot_version": COST_PILOT_VERSION,
        },
        "aggregate": asdict(aggregate),
        "projections": [asdict(p) for p in projections],
        "recommendations": recommendations,
        "per_question": [asdict(r) for r in records],
    }
    if matrix_planning is not None:
        payload["matrix_planning"] = dict(matrix_planning)
    return payload


GraphInvoker = Callable[[Dict[str, Any]], Dict[str, Any]]


def run_pilot(
    questions: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    agents: Optional[List[str]] = None,
    graph_invoke: GraphInvoker,
    run_id: Optional[str] = None,
    projection_sizes: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Execute cost pilot over question list."""
    run_id = run_id or uuid.uuid4().hex[:8]
    normalized_agents = normalize_agent_names(agents)
    records: List[QuestionCostRecord] = []

    for index, item in enumerate(questions):
        question_id = str(item.get("id") or f"q{index + 1}")
        question = str(item.get("question") or "")
        extra_context: Dict[str, Any] = {}
        if item.get("top_k") is not None:
            extra_context["top_k"] = item["top_k"]
        if item.get("context"):
            extra_context.update(dict(item["context"]))

        initial = build_initial_state(
            question=question,
            model_id=model_id,
            agents_to_use=normalized_agents,
            extra_context=extra_context or None,
        )
        final_state = graph_invoke(initial)
        records.append(
            extract_question_metrics(
                question_id=question_id,
                final_state=final_state,
                query_text=question,
            )
        )

    aggregate = aggregate_records(records)
    defaults = load_cost_pilot_defaults()
    sizes = list(
        projection_sizes
        or defaults.get("projection_targets")
        or [100, 500, DEFAULT_FULL_SPLIT_SIZES["medagentsbench_test_hard"]]
    )

    projections = project_costs(aggregate, sizes)
    recommendations = {
        "100_question_pilot": recommend_credit_purchase_usd(aggregate, target_n=100),
        "500_question_run": recommend_credit_purchase_usd(aggregate, target_n=500),
    }
    medagents_n = DEFAULT_FULL_SPLIT_SIZES["medagentsbench_test_hard"]
    recommendations[f"medagentsbench_test_hard_{medagents_n}"] = (
        recommend_credit_purchase_usd(aggregate, target_n=medagents_n)
    )
    if sizes:
        full_n = max(sizes)
        if f"full_split_{full_n}" not in recommendations:
            recommendations[f"full_split_{full_n}"] = recommend_credit_purchase_usd(
                aggregate, target_n=full_n
            )

    matrix_planning = build_matrix_planning_projections(target_n=medagents_n)

    run_meta = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "agents": normalized_agents,
        "n_samples": len(questions),
        "mode": "live",
    }
    return serialize_pilot_result(
        run_meta=run_meta,
        records=records,
        aggregate=aggregate,
        projections=projections,
        recommendations=recommendations,
        matrix_planning=matrix_planning,
    )
