#!/usr/bin/env python
"""Recompute evaluation metrics from preserved answers and retrieval contexts."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.metrics import EvalSample, compute_all_metrics  # noqa: E402
from eval.run_eval import _aggregate_results  # noqa: E402


def rejudge_payload(source: Dict[str, Any], judge_model: str) -> Dict[str, Any]:
    """Replace metric judgments while preserving generated outputs."""
    output = copy.deepcopy(source)
    rows = output.get("results", [])
    for row in rows:
        sample = EvalSample(
            question=row.get("question", ""),
            answer=row.get("agent_answer", ""),
            expected_answer=row.get("expected_answer", ""),
            retrieved_contexts=row.get("retrieved_contexts", []),
            citations=row.get("citations", []),
        )
        metrics = compute_all_metrics(sample, model=judge_model)
        row["metrics"] = {
            metric.name: {
                "score": metric.score,
                "reason": metric.reason,
                "valid": metric.valid,
                "raw": metric.raw,
            }
            for metric in metrics
        }

    config = dict(output.get("config", {}))
    config.update(
        {
            "judge_model": judge_model,
            "rejudged_from_preserved_outputs": True,
            "metric_judging_protocol": "separated_grounding_reference_v2",
            "rejudged_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    output["config"] = config
    models = list(config.get("models", []))
    agents = list(config.get("agents", []))
    output["summary"] = _aggregate_results(rows, models, agents)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge_model", required=True)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    output = rejudge_payload(source, args.judge_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
