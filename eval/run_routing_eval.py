#!/usr/bin/env python
"""Evaluate skill-router top-k accuracy on the source-stratified benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from skill_router import SkillRouter  # noqa: E402

AGENT_TO_TOOL = {
    "pubmed": "search_pubmed",
    "fda": "search_fda",
    "clinical_trials": "search_clinical_trials",
    "local": "search_local_index",
}


def summarize_routing(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    top1 = sum(bool(row.get("ranked")) and row["ranked"][0] == row["expected"] for row in rows)
    top3 = sum(row["expected"] in row.get("ranked", [])[:3] for row in rows)
    reciprocal_ranks = []
    for row in rows:
        try:
            rank = row.get("ranked", []).index(row["expected"]) + 1
        except ValueError:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / rank)
    return {
        "n": n,
        "top1_accuracy": round(top1 / n, 4) if n else 0.0,
        "top3_accuracy": round(top3 / n, 4) if n else 0.0,
        "mean_reciprocal_rank": (
            round(sum(reciprocal_ranks) / n, 4) if n else 0.0
        ),
    }


def run(benchmark_path: Path, output_path: Path) -> Dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    router = SkillRouter()
    rows = []
    for item in benchmark:
        target_agents = item.get("target_agents", [])
        if len(target_agents) != 1:
            continue
        expected = AGENT_TO_TOOL[target_agents[0]]
        ranked, scores = router.rank_tools(
            item["question"], top_k=len(router.available_skills), min_threshold=0.0
        )
        rows.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "expected": expected,
                "ranked": ranked,
                "scores": scores,
                "expected_rank": (
                    ranked.index(expected) + 1 if expected in ranked else None
                ),
            }
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(benchmark_path),
        "embedding_model": __import__("os").getenv("DEFAULT_EMBEDDING_MODEL"),
        "summary": summarize_routing(rows),
        "results": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=_REPO_ROOT / "eval" / "data" / "medical_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "results" / "routing_eval.json",
    )
    args = parser.parse_args()
    result = run(args.benchmark, args.output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
