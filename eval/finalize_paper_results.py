#!/usr/bin/env python
"""Recompute final statistics and render manuscript-ready result tables."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.run_eval import _aggregate_results  # noqa: E402
from eval.metrics import EvalSample, citation_fidelity  # noqa: E402

METRIC_LABELS = {
    "faithfulness": "Faithfulness (higher is better)",
    "answer_relevancy": "Answer relevancy (higher is better)",
    "answer_correctness": "Answer correctness (higher is better)",
    "citation_fidelity": "Citation fidelity (higher is better)",
    "hallucination_rate": "Hallucination rate (lower is better)",
}
RETRIEVAL_METRIC_LABELS = {
    "context_relevance": "Context relevance",
    "context_sufficiency": "Context sufficiency",
    "citation_claim_support": "Citation-bearing claim support",
    "sentence_citation_coverage": "Sentence citation coverage",
}


def _metric_cell(group: Dict[str, Any], metric: str) -> str:
    score = group.get("scores", {}).get(metric, 0.0)
    lower, upper = group.get("ci95", {}).get(metric, [0.0, 0.0])
    valid = group.get("valid_counts", {}).get(metric, 0)
    total = group.get("sample_count", 0)
    return (
        f"{_format_decimal(score)} "
        f"({_format_decimal(lower)}–{_format_decimal(upper)}); {valid}/{total}"
    )


def _format_decimal(value: Any) -> str:
    return str(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


def _recompute_deterministic_metrics(
    results: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Recompute deterministic metrics from preserved question-level fields."""
    corrected = copy.deepcopy(results)
    for row in corrected:
        sample = EvalSample(
            question=row.get("question", ""),
            answer=row.get("agent_answer", ""),
            expected_answer=row.get("expected_answer", ""),
            retrieved_contexts=row.get("retrieved_contexts", []),
            citations=row.get("citations", []),
        )
        metric = citation_fidelity(sample)
        row.setdefault("metrics", {})["citation_fidelity"] = {
            "score": metric.score,
            "reason": metric.reason,
            "valid": metric.valid,
        }
    return corrected


def build_results_markdown(
    summary: Dict[str, Any],
    routing_summary: Dict[str, Any] | None = None,
    retrieval_summary: Dict[str, Any] | None = None,
) -> str:
    """Render aggregate results as journal-ready Markdown."""
    overall = summary["overall"]
    lines = [
        "# Evaluation Results",
        "",
        "Values are mean (bootstrap 95% CI); valid metric denominator / evaluated samples.",
        "",
        "## Overall",
        "",
        "| Metric | Result |",
        "|---|---:|",
    ]
    for metric, label in METRIC_LABELS.items():
        lines.append(f"| {label} | {_metric_cell(overall, metric)} |")

    latency = overall.get("latency_sec", {})
    if routing_summary:
        lines.extend(
            [
                "",
                "## Held-out routing",
                "",
                "| Routing metric | Result |",
                "|---|---:|",
                f"| Top-1 accuracy | {routing_summary.get('top1_accuracy', 0.0):.3f} |",
                f"| Top-3 accuracy | {routing_summary.get('top3_accuracy', 0.0):.3f} |",
                (
                    "| Mean reciprocal rank | "
                    f"{routing_summary.get('mean_reciprocal_rank', 0.0):.3f} |"
                ),
                f"| Questions | {routing_summary.get('n', 0)} |",
            ]
        )
    if retrieval_summary:
        lines.extend(
            [
                "",
                "## Retrieval-stage grounding",
                "",
                "| Retrieval metric | Result |",
                "|---|---:|",
            ]
        )
        for metric, label in RETRIEVAL_METRIC_LABELS.items():
            lines.append(
                f"| {label} | {_metric_cell(retrieval_summary, metric)} |"
            )
    lines.extend(
        [
            "",
            (
                f"Across {overall.get('sample_count', 0)} evaluations, "
                f"{overall.get('error_count', 0)} execution errors were recorded. "
                f"Latency was mean {latency.get('mean', 0.0):.3f} s, "
                f"median {latency.get('median', 0.0):.3f} s, and "
                f"p95 {latency.get('p95', 0.0):.3f} s."
            ),
            "",
            "## By agent",
            "",
            (
                "| Agent | Faithfulness | Relevancy | Correctness | Citation fidelity | "
                "Hallucination rate | Errors | Latency, mean s |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for agent, group in summary.get("by_agent", {}).items():
        lines.append(
            f"| {agent} | {_metric_cell(group, 'faithfulness')} | "
            f"{_metric_cell(group, 'answer_relevancy')} | "
            f"{_metric_cell(group, 'answer_correctness')} | "
            f"{_metric_cell(group, 'citation_fidelity')} | "
            f"{_metric_cell(group, 'hallucination_rate')} | "
            f"{group.get('error_count', 0)} | "
            f"{group.get('latency_sec', {}).get('mean', 0.0):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _latest_merged_result(results_dir: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in results_dir.glob("benchmark_all_agents_*.json")
            if not path.stem.endswith("_audited")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No benchmark_all_agents_*.json result was found")
    return candidates[0]


def finalize(input_path: Path | None = None) -> Dict[str, Any]:
    results_dir = _REPO_ROOT / "results"
    source_path = input_path or _latest_merged_result(results_dir)
    with open(source_path, "r", encoding="utf-8") as handle:
        source = json.load(handle)

    config = source.get("config", {})
    models = config.get("models", [])
    agents = config.get("agents", [])
    original_results = source.get("results", [])
    results = _recompute_deterministic_metrics(original_results)
    summary = _aggregate_results(results, models, agents)
    routing_path = results_dir / "routing_holdout_eval_final.json"
    routing_summary: Dict[str, Any] = {}
    if routing_path.exists():
        with open(routing_path, "r", encoding="utf-8") as handle:
            routing_summary = json.load(handle).get("summary", {})
    retrieval_path = results_dir / "retrieval_grounding_eval.json"
    retrieval_summary: Dict[str, Any] = {}
    if retrieval_path.exists():
        with open(retrieval_path, "r", encoding="utf-8") as handle:
            retrieval_summary = (
                json.load(handle).get("summary", {}).get("overall", {})
            )

    audited_source_path = source_path
    if results != original_results:
        audited_source_path = source_path.with_name(
            f"{source_path.stem}_audited{source_path.suffix}"
        )
        audited_source = copy.deepcopy(source)
        audited_source["results"] = results
        audited_source["summary"] = summary
        audited_source["audit"] = {
            "deterministic_metrics_recomputed": ["citation_fidelity"],
            "source_run": str(source_path),
        }
        with open(audited_source_path, "w", encoding="utf-8") as handle:
            json.dump(audited_source, handle, indent=2, ensure_ascii=False)

    finalized = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(audited_source_path),
        "raw_source_run": str(source_path),
        "config": config,
        "summary": summary,
        "routing_summary": routing_summary,
        "retrieval_summary": retrieval_summary,
    }
    summary_path = results_dir / "paper_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(finalized, handle, indent=2, ensure_ascii=False)

    paper_path = _REPO_ROOT / "paper" / "RESULTS_TABLES.md"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(
        build_results_markdown(summary, routing_summary, retrieval_summary),
        encoding="utf-8",
    )
    return finalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()
    finalized = finalize(args.input)
    print(json.dumps(finalized["summary"], indent=2))


if __name__ == "__main__":
    main()
