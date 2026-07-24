#!/usr/bin/env python
"""
run_all_agents_eval.py — Run the Phase 8 eval harness for every sub-agent.

Executes agent-targeted questions from eval/data/medical_benchmark.json,
writes per-agent JSON results, a merged summary, and updates the
experiment registry for paper reproducibility.

Usage:
    python eval/run_all_agents_eval.py
    python eval/run_all_agents_eval.py --n_samples 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.run_eval import ALL_AGENT_NAMES, run_evaluation  # noqa: E402

logger = logging.getLogger(__name__)

BENCHMARK_PATH = _REPO_ROOT / "eval" / "data" / "medical_benchmark.json"
RESULTS_DIR = _REPO_ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all Medical Research Agent sub-agents")
    parser.add_argument("--n_samples", type=int, default=None, help="Limit questions per run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        help="LLM model (repeat for multiple). Default: DEFAULT_LLM_MODEL env var.",
    )
    parser.add_argument(
        "--judge_model",
        default=None,
        help=(
            "Independent evaluation model. "
            "Default: EVAL_JUDGE_MODEL or gemini/gemini-2.5-flash-lite."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    os.environ.setdefault(
        "DEFAULT_EMBEDDING_MODEL",
        os.getenv("DEFAULT_EMBEDDING_MODEL", "ollama/nomic-embed-text"),
    )

    models = args.models or [os.getenv("DEFAULT_LLM_MODEL", "gemini/gemma-4-26b-a4b-it")]
    judge_model = args.judge_model or os.getenv(
        "EVAL_JUDGE_MODEL", "gemini/gemini-2.5-flash-lite"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    per_agent_outputs = {}
    merged_results = []

    for agent_name in ALL_AGENT_NAMES:
        out_path = RESULTS_DIR / f"benchmark_{agent_name}_{ts}.json"
        logger.info("=== Evaluating agent: %s ===", agent_name)
        output = run_evaluation(
            dataset_name="custom",
            models=models,
            agent_names=[agent_name],
            judge_model=judge_model,
            n_samples=args.n_samples,
            seed=args.seed,
            output_path=str(out_path),
            dataset_path=str(BENCHMARK_PATH),
        )
        per_agent_outputs[agent_name] = str(out_path)
        merged_results.extend(output.get("results", []))

    # Combined summary
    from eval.run_eval import _aggregate_results  # noqa: E402

    summary = _aggregate_results(merged_results, models, ALL_AGENT_NAMES)
    merged = {
        "run_id": f"all_agents_{ts}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset": "medical_benchmark",
            "dataset_path": str(BENCHMARK_PATH),
            "models": models,
            "judge_model": judge_model,
            "agents": ALL_AGENT_NAMES,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "embedding_model": os.getenv("DEFAULT_EMBEDDING_MODEL"),
        },
        "per_agent_outputs": per_agent_outputs,
        "summary": summary,
        "results": merged_results,
    }

    merged_path = RESULTS_DIR / f"benchmark_all_agents_{ts}.json"
    with open(merged_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)

    paper_summary_path = RESULTS_DIR / "paper_eval_summary.json"
    with open(paper_summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_at": merged["timestamp"],
                "config": merged["config"],
                "summary": summary,
                "source_run": str(merged_path),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("All-agent evaluation complete")
    print("=" * 60)
    print(f"Merged results: {merged_path}")
    print(f"Paper summary:  {paper_summary_path}")
    print("\nOverall scores:")
    for metric, score in summary.get("overall", {}).get("scores", {}).items():
        print(f"  {metric:25s} {score:.4f}")
    print("\nBy agent:")
    for agent, agent_summary in summary.get("by_agent", {}).items():
        scores = agent_summary.get("scores", {})
        avg = sum(scores.values()) / len(scores) if scores else 0.0
        print(f"  {agent:20s} avg={avg:.4f}  {scores}")


if __name__ == "__main__":
    main()
