#!/usr/bin/env python
"""Post-hoc retrieval and grounding evaluation over preserved run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.metrics import EvalSample, MetricResult  # noqa: E402
from eval.retrieval_metrics import (  # noqa: E402
    compute_retrieval_grounding_metrics,
    sentence_citation_coverage,
)
from eval.run_eval import _bootstrap_mean_ci  # noqa: E402

METRIC_NAMES = (
    "context_relevance",
    "context_sufficiency",
    "citation_claim_support",
    "sentence_citation_coverage",
)


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores: Dict[str, float] = {}
    ci95: Dict[str, List[float]] = {}
    valid_counts: Dict[str, int] = {}
    for name in METRIC_NAMES:
        values = [
            float(row["metrics"][name]["score"])
            for row in rows
            if row.get("metrics", {}).get(name, {}).get("valid", False)
        ]
        scores[name] = round(sum(values) / len(values), 4) if values else 0.0
        ci95[name] = list(_bootstrap_mean_ci(values))
        valid_counts[name] = len(values)
    return {
        "scores": scores,
        "ci95": ci95,
        "valid_counts": valid_counts,
        "sample_count": len(rows),
        "mean_retrieved_contexts": (
            round(
                sum(row.get("retrieved_context_count", 0) for row in rows)
                / len(rows),
                3,
            )
            if rows
            else 0.0
        ),
    }


def aggregate_grounding_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate retrieval-stage metrics overall and by source agent."""
    agents = sorted({str(row.get("agent", "")) for row in rows if row.get("agent")})
    return {
        "overall": _summarize_rows(rows),
        "by_agent": {
            agent: _summarize_rows(
                [row for row in rows if row.get("agent") == agent]
            )
            for agent in agents
        },
    }


def run(
    input_path: Path,
    output_path: Path,
    judge_model: str,
) -> Dict[str, Any]:
    source = json.loads(input_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for source_row in source.get("results", []):
        sample = EvalSample(
            question=source_row.get("question", ""),
            answer=source_row.get("agent_answer", ""),
            expected_answer=source_row.get("expected_answer", ""),
            retrieved_contexts=source_row.get("retrieved_contexts", []),
            citations=source_row.get("citations", []),
        )
        try:
            metrics = compute_retrieval_grounding_metrics(
                sample, model=judge_model
            )
        except Exception as exc:
            metrics = [
                MetricResult(
                    name=name,
                    score=0.0,
                    reason=f"Retrieval judge failed: {exc}",
                    valid=False,
                )
                for name in METRIC_NAMES[:-1]
            ]
            metrics.append(sentence_citation_coverage(sample))

        rows.append(
            {
                "question": sample.question,
                "agent": source_row.get("agent", ""),
                "retrieved_context_count": len(sample.retrieved_contexts),
                "metrics": {
                    metric.name: {
                        "score": metric.score,
                        "reason": metric.reason,
                        "valid": metric.valid,
                        "raw": metric.raw,
                    }
                    for metric in metrics
                },
            }
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(input_path),
        "judge_model": judge_model,
        "logprob_assessment": {
            "available": False,
            "reason": (
                "The evaluated Ollama Qwen endpoints did not expose calibrated "
                "token log-probabilities. No token-confidence proxy was imputed."
            ),
        },
        "metric_scope": {
            "context_relevance": "Retrieved-set relevance to the question.",
            "context_sufficiency": (
                "Coverage of reference-answer facts by retrieved contexts."
            ),
            "citation_claim_support": (
                "Support for citation-bearing claims by the retrieved set; "
                "not strict source-specific entailment."
            ),
            "sentence_citation_coverage": (
                "Heuristic fraction of factual sentences with valid markers."
            ),
        },
        "summary": aggregate_grounding_rows(rows),
        "results": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "results" / "retrieval_grounding_eval.json",
    )
    parser.add_argument(
        "--judge_model",
        default="ollama/qwen2.5:3b",
    )
    args = parser.parse_args()
    result = run(args.input, args.output, args.judge_model)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
