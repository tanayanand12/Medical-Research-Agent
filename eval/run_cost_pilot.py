#!/usr/bin/env python3
"""
Cost pilot runner — full LangGraph orchestrator telemetry for credit planning.

Usage:
    python eval/run_cost_pilot.py --n_samples 20 --models gpt-4o --agents pubmed
    python eval/run_cost_pilot.py --dry-run --n_samples 5 --output results/cost_pilot.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

# Ensure repo root is importable when invoked as script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from eval.cost_pilot import (
    DEFAULT_FULL_SPLIT_SIZES,
    build_matrix_planning_projections,
    load_cost_pilot_defaults,
    render_markdown_summary,
    run_pilot,
    synthesize_mock_final_state,
    load_pilot_questions,
    aggregate_records,
    project_costs,
    recommend_credit_purchase_usd,
    extract_question_metrics,
    serialize_pilot_result,
)
from evaluation_core.privacy import stable_query_fingerprint
from evaluation_core.schemas import EVALUATION_TRACE_SCHEMA_VERSION
from unicode_safe_logging import configure_all_loggers

configure_all_loggers()
logger = logging.getLogger(__name__)

DEFAULT_DATASET = _REPO_ROOT / "eval" / "data" / "medical_benchmark.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cost pilot against full LangGraph orchestrator.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of questions to run (default: 20 smoke size).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o"],
        help="Generation model ID(s); first entry is used for this pilot run.",
    )
    parser.add_argument(
        "--agents",
        nargs="*",
        default=None,
        help="Optional agent/tool override (pubmed, fda, clinical_trials, local).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Question JSON dataset (medical_benchmark.json format).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "results" / "cost_pilot.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional markdown summary path (default: alongside JSON).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Offline mock mode — no LLM/API calls.",
    )
    parser.add_argument(
        "--project-full-split",
        choices=sorted(DEFAULT_FULL_SPLIT_SIZES.keys()),
        default="custom",
        help="Include full-split N in projections.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier.",
    )
    return parser.parse_args()


def _safe_log_progress(question_id: str, query_text: str, index: int, total: int) -> None:
    logger.info(
        "Pilot progress %d/%d question_id=%s fingerprint=%s length=%d",
        index + 1,
        total,
        question_id,
        stable_query_fingerprint(query_text),
        len(query_text),
    )


def main() -> int:
    load_dotenv()
    args = _parse_args()
    model_id = args.models[0]
    run_id = args.run_id or uuid.uuid4().hex[:8]

    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        return 1

    questions = load_pilot_questions(args.dataset, n_samples=args.n_samples)
    if not questions:
        logger.error("No questions loaded from %s", args.dataset)
        return 1

    full_n = DEFAULT_FULL_SPLIT_SIZES.get(args.project_full_split, 100)
    defaults = load_cost_pilot_defaults()
    projection_sizes = list(defaults.get("projection_targets") or [100, 500, full_n])
    if full_n not in projection_sizes:
        projection_sizes.append(full_n)
    medagents_n = DEFAULT_FULL_SPLIT_SIZES["medagentsbench_test_hard"]
    matrix_planning = build_matrix_planning_projections(target_n=medagents_n)

    if args.dry_run:
        logger.info("Dry-run mode — synthesizing telemetry for %d questions", len(questions))

        def mock_invoke(initial_state: dict) -> dict:
            idx = hash(initial_state.get("trace_id", "")) % 997
            qid = stable_query_fingerprint(initial_state.get("input_query", ""))
            return synthesize_mock_final_state(
                question_id=qid,
                question=str(initial_state.get("input_query") or ""),
                model_id=model_id,
                index=idx,
                trace_id=str(initial_state.get("trace_id") or ""),
            )

        from eval.cost_pilot import (
            QuestionCostRecord,
            build_initial_state,
            normalize_agent_names,
        )

        normalized_agents = normalize_agent_names(args.agents)
        records: list[QuestionCostRecord] = []
        for index, item in enumerate(questions):
            question_id = str(item.get("id") or f"q{index + 1}")
            question = str(item.get("question") or "")
            _safe_log_progress(question_id, question, index, len(questions))
            initial = build_initial_state(
                question=question,
                model_id=model_id,
                agents_to_use=normalized_agents,
            )
            final_state = mock_invoke(initial)
            records.append(
                extract_question_metrics(
                    question_id=question_id,
                    final_state=final_state,
                    query_text=question,
                )
            )

        aggregate = aggregate_records(records)
        projections = project_costs(aggregate, projection_sizes)
        recommendations = {
            "100_question_pilot": recommend_credit_purchase_usd(aggregate, target_n=100),
            "500_question_run": recommend_credit_purchase_usd(aggregate, target_n=500),
            f"medagentsbench_test_hard_{medagents_n}": recommend_credit_purchase_usd(
                aggregate, target_n=medagents_n
            ),
            f"full_split_{full_n}": recommend_credit_purchase_usd(aggregate, target_n=full_n),
        }
        result = serialize_pilot_result(
            run_meta={
                "run_id": run_id,
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "model_id": model_id,
                "agents": normalized_agents,
                "n_samples": len(questions),
                "mode": "dry_run",
            },
            records=records,
            aggregate=aggregate,
            projections=projections,
            recommendations=recommendations,
            matrix_planning=matrix_planning,
        )
    else:
        from graph import get_graph

        graph = get_graph()

        def live_invoke(initial_state: dict) -> dict:
            return graph.invoke(initial_state)

        logger.info(
            "Live pilot model=%s agents=%s n=%d trace_schema=%s",
            model_id,
            args.agents or "auto",
            len(questions),
            EVALUATION_TRACE_SCHEMA_VERSION,
        )
        for index, item in enumerate(questions):
            _safe_log_progress(
                str(item.get("id") or f"q{index + 1}"),
                str(item.get("question") or ""),
                index,
                len(questions),
            )

        result = run_pilot(
            questions,
            model_id=model_id,
            agents=args.agents,
            graph_invoke=live_invoke,
            run_id=run_id,
            projection_sizes=projection_sizes,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    md_path = args.markdown or args.output.with_suffix(".md")
    from eval.cost_pilot import PilotAggregate, CostProjection, QuestionCostRecord

    aggregate = PilotAggregate(**result["aggregate"])
    projections = [CostProjection(**p) for p in result["projections"]]
    records = [QuestionCostRecord(**r) for r in result["per_question"]]
    md_text = render_markdown_summary(
        run_meta=result["run_meta"],
        aggregate=aggregate,
        projections=projections,
        recommendations=result["recommendations"],
        per_question=records,
        matrix_planning=result.get("matrix_planning"),
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")

    logger.info("Wrote JSON: %s", args.output)
    logger.info("Wrote markdown: %s", md_path)
    logger.info(
        "Mean cost: $%.4f | p95 latency: %.2fs | recommended (100q): $%.2f",
        aggregate.cost_usd_mean,
        aggregate.latency_sec_p95,
        result["recommendations"]["100_question_pilot"]["recommended_purchase_usd"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
