#!/usr/bin/env python
"""
run_eval.py — Phase 8: CLI evaluation runner for Medical Research Agent.

Orchestrates benchmark evaluation across datasets, one generation model, and
agents (``agents/`` subgraphs).  All agents are invoked via the migrated
LangGraph sub-agent interface — legacy wrappers are never used.

Usage examples::

    # Show help
    python eval/run_eval.py --help

    # Quick sanity check (10 MedQA samples, default model, PubMed agent)
    python eval/run_eval.py --dataset medqa --n_samples 10 \\
        --agents pubmed --output results/quick.json

    # Full benchmark: 200 samples, one generation model per process
    python eval/run_eval.py --dataset medqa --n_samples 200 \\
        --model gpt-4o \\
        --output results/full.json

    # BioASQ with specific agents
    python eval/run_eval.py --dataset bioasq --n_samples 50 \\
        --agents pubmed,fda,clinical_trials \\
        --output results/bioasq_run.json
"""

import argparse
import json
import logging
import os
import random
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path so we can import project modules
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.datasets import get_dataset
from eval.metrics import EvalSample, MetricResult, compute_all_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# Agent registry — maps CLI names to SubAgentGraph classes
# ---------------------------------------------------------------------- #

AGENT_REGISTRY: Dict[str, str] = {
    "pubmed": "agents.pubmed_agent.graph.PubMedAgentGraph",
    "fda": "agents.fda_agent.graph.FDAAgentGraph",
    "clinical_trials": "agents.clinical_trials_agent.graph.ClinicalTrialsAgentGraph",
    "local": "agents.local_agent.graph.LocalAgentGraph",
}

ALL_AGENT_NAMES = list(AGENT_REGISTRY.keys())


def _import_agent(dotted_path: str) -> Any:
    """Dynamically import a class from a dotted module path."""
    module_path, cls_name = dotted_path.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _instantiate_agents(names: List[str]) -> Dict[str, Any]:
    """Instantiate requested SubAgentGraph instances by CLI name."""
    agents: Dict[str, Any] = {}
    for name in names:
        dotted = AGENT_REGISTRY.get(name)
        if dotted is None:
            logger.warning(
                "Unknown agent '%s' — skipping. Available: %s",
                name,
                ALL_AGENT_NAMES,
            )
            continue
        try:
            cls = _import_agent(dotted)
            agents[name] = cls()
            logger.info("Loaded agent: %s (%s)", name, dotted)
        except Exception as exc:
            logger.error("Failed to load agent '%s': %s", name, exc)
    return agents


# ---------------------------------------------------------------------- #
# Single-sample evaluation
# ---------------------------------------------------------------------- #


def _retrieval_diagnostics(sources: List[Any]) -> Dict[str, Any]:
    """Summarize agent-native retrieval scores without treating them as calibrated."""
    dictionaries = [source for source in sources if isinstance(source, dict)]
    scores = [
        float(source["score"])
        for source in dictionaries
        if isinstance(source.get("score"), (int, float))
    ]
    overlap = sum(
        source.get("dense_rank") is not None
        and source.get("sparse_rank") is not None
        for source in dictionaries
    )
    return {
        "retrieved_count": len(dictionaries),
        "scores": scores,
        "score_top1": scores[0] if scores else None,
        "score_mean": round(sum(scores) / len(scores), 6) if scores else None,
        "score_margin": (
            round(scores[0] - scores[1], 6) if len(scores) >= 2 else None
        ),
        "hybrid_overlap_fraction": (
            round(overlap / len(dictionaries), 6) if dictionaries else None
        ),
        "comparability_note": (
            "Agent-native scores are descriptive and are not calibrated or "
            "comparable across retrievers."
        ),
    }


def _evaluate_sample(
    question: str,
    expected_answer: str,
    agent: Any,
    model: str,
    judge_model: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one question through an agent and compute all metrics.

    Returns a dict with the agent output, metric scores, and metadata.
    """
    ctx = context or {}
    ctx.setdefault("model_id", model)

    start = time.time()
    retrieved_contexts: List[str] = []
    retrieved_sources: List[Any] = []
    try:
        output = agent.invoke(query=question, context=ctx)
        elapsed = time.time() - start

        # Build retrieved context strings for metric evaluation
        retrieved_sources = list(getattr(output, "sources", []))
        for src in retrieved_sources:
            text = src.get("text", "") if isinstance(src, dict) else str(src)
            if text:
                retrieved_contexts.append(text)

        sample = EvalSample(
            question=question,
            answer=output.answer,
            expected_answer=expected_answer,
            retrieved_contexts=retrieved_contexts,
            citations=output.citations,
        )

        metrics = compute_all_metrics(sample, model=judge_model or model)

    except Exception as exc:
        elapsed = time.time() - start
        logger.warning("Agent invocation failed: %s", exc)
        err_msg = str(exc)
        metrics = [
            MetricResult(
                name="faithfulness", score=0.0, reason=f"error: {err_msg}", valid=False
            ),
            MetricResult(
                name="answer_relevancy",
                score=0.0,
                reason=f"error: {err_msg}",
                valid=False,
            ),
            MetricResult(
                name="answer_correctness",
                score=0.0,
                reason=f"error: {err_msg}",
                valid=False,
            ),
            MetricResult(
                name="citation_fidelity",
                score=0.0,
                reason=f"error: {err_msg}",
                valid=False,
            ),
            MetricResult(
                name="hallucination_rate",
                score=0.0,
                reason=f"error: {err_msg}",
                valid=False,
            ),
        ]
        output = None
    else:
        err_msg = output.error if output else None

    return {
        "question": question,
        "expected_answer": expected_answer,
        "agent_answer": output.answer if output else "",
        "agent_confidence": output.confidence if output else 0.0,
        "agent_model_used": output.model_used if output else "",
        "citations": output.citations if output else [],
        "retrieved_contexts": retrieved_contexts if output else [],
        "retrieved_sources": retrieved_sources if output else [],
        "retrieval_diagnostics": _retrieval_diagnostics(
            retrieved_sources if output else []
        ),
        "metrics": {
            m.name: {"score": m.score, "reason": m.reason, "valid": m.valid}
            for m in metrics
        },
        "execution_time_sec": round(elapsed, 3),
        "error": err_msg,
    }


# ---------------------------------------------------------------------- #
# Full evaluation run
# ---------------------------------------------------------------------- #


def _select_questions(
    questions: List[str],
    dataset: Any,
    agent_names: List[str],
    n_samples: Optional[int],
    seed: int,
) -> List[str]:
    """Filter target-compatible questions before deterministic sampling."""
    eligible = []
    for question in questions:
        targets = dataset.get_target_agents(question)
        if not targets or any(agent in targets for agent in agent_names):
            eligible.append(question)
    if n_samples is None or n_samples >= len(eligible):
        return eligible
    return random.Random(seed).sample(eligible, n_samples)


def _validate_generation_models(
    agents: Dict[str, Any], models: List[str]
) -> None:
    """Fail fast when requested labels differ from process-wide agent models."""
    actual_models = {
        getattr(getattr(agent, "llm", None), "default_model", None)
        for agent in agents.values()
    }
    actual_models.discard(None)
    if len(models) != 1 or actual_models != {models[0]}:
        raise ValueError(
            "Agent subgraphs use the process-wide DEFAULT_LLM_MODEL. "
            "Evaluate one generation model per process and set "
            "DEFAULT_LLM_MODEL to the same value passed via --model. "
            f"Requested={models}; actual={sorted(actual_models)}"
        )


def run_evaluation(
    dataset_name: str,
    models: List[str],
    agent_names: List[str],
    judge_model: Optional[str] = None,
    n_samples: Optional[int] = None,
    seed: int = 42,
    output_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a full evaluation run.

    Iterates over: questions x models x agents.
    Computes all five metrics for each combination.
    """
    run_id = str(uuid.uuid4())[:8]
    ts_start = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Starting evaluation run %s: dataset=%s models=%s agents=%s n_samples=%s",
        run_id,
        dataset_name,
        models,
        agent_names,
        n_samples,
    )

    # Load dataset
    dataset = get_dataset(dataset_name, n_samples=None, seed=seed, path=dataset_path)
    questions = _select_questions(
        dataset.get_questions(), dataset, agent_names, n_samples, seed
    )
    logger.info("Dataset loaded: %d questions", len(questions))

    # Instantiate agents
    agents = _instantiate_agents(agent_names)
    if not agents:
        logger.error("No agents loaded — aborting.")
        return {"error": "No agents could be loaded"}
    _validate_generation_models(agents, models)

    # Run evaluations
    results: List[Dict[str, Any]] = []
    total = 0
    for question in questions:
        target_agents = dataset.get_target_agents(question)
        for _model in models:
            for agent_name in agents:
                if target_agents and agent_name not in target_agents:
                    continue
                total += 1
    done = 0

    for q_idx, question in enumerate(questions):
        expected = dataset.get_expected_answer(question)
        item_context = dataset.get_context(question)
        target_agents = dataset.get_target_agents(question)
        for model in models:
            for agent_name, agent in agents.items():
                if target_agents and agent_name not in target_agents:
                    continue
                done += 1
                logger.info(
                    "[%d/%d] q=%d model=%s agent=%s",
                    done,
                    total,
                    q_idx,
                    model,
                    agent_name,
                )
                merged_context = {**item_context, "model_id": model}
                result = _evaluate_sample(
                    question=question,
                    expected_answer=expected,
                    agent=agent,
                    model=model,
                    judge_model=judge_model,
                    context=merged_context,
                )
                result["run_id"] = run_id
                result["dataset"] = dataset_name
                result["model"] = model
                result["agent"] = agent_name
                result["question_idx"] = q_idx
                results.append(result)

    # Aggregate summary
    summary = _aggregate_results(results, models, list(agents.keys()))

    output = {
        "run_id": run_id,
        "timestamp_start": ts_start,
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset": dataset_name,
            "models": models,
            "judge_model": judge_model or models[0],
            "agents": list(agents.keys()),
            "dataset_size": len(questions),
            "evaluated_samples": len(results),
            "seed": seed,
        },
        "summary": summary,
        "results": results,
    }

    # Write output
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        logger.info("Results written to %s", out)

    # Append to experiment registry
    _append_to_registry(output)

    return output


# ---------------------------------------------------------------------- #
# Aggregation
# ---------------------------------------------------------------------- #


def _aggregate_results(
    results: List[Dict[str, Any]],
    models: List[str],
    agents: List[str],
) -> Dict[str, Any]:
    """Compute mean scores per model, per agent, and overall."""
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "citation_fidelity",
        "hallucination_rate",
    ]

    def _percentile(values: List[float], probability: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = (len(ordered) - 1) * probability
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def _mean_scores(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores: Dict[str, float] = {}
        ci95: Dict[str, List[float]] = {}
        valid_counts: Dict[str, int] = {}
        for metric in metric_names:
            valid_values = [
                float(item["metrics"][metric]["score"])
                for item in items
                if item.get("metrics", {}).get(metric, {}).get("valid", False)
            ]
            valid_counts[metric] = len(valid_values)
            scores[metric] = (
                round(sum(valid_values) / len(valid_values), 4)
                if valid_values
                else 0.0
            )
            lower, upper = _bootstrap_mean_ci(valid_values)
            ci95[metric] = [lower, upper]
        latencies = [
            float(item.get("execution_time_sec", 0.0))
            for item in items
            if item.get("execution_time_sec") is not None
        ]
        return {
            "scores": scores,
            "ci95": ci95,
            "valid_counts": valid_counts,
            "sample_count": len(items),
            "error_count": sum(bool(item.get("error")) for item in items),
            "latency_sec": {
                "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
                "median": round(statistics.median(latencies), 3) if latencies else 0.0,
                "p95": round(_percentile(latencies, 0.95), 3) if latencies else 0.0,
            },
        }

    summary: Dict[str, Any] = {
        "overall": _mean_scores(results),
        "by_model": {},
        "by_agent": {},
    }

    for model in models:
        subset = [r for r in results if r.get("model") == model]
        summary["by_model"][model] = _mean_scores(subset)

    for agent in agents:
        subset = [r for r in results if r.get("agent") == agent]
        summary["by_agent"][agent] = _mean_scores(subset)

    return summary


def _bootstrap_mean_ci(
    values: List[float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a deterministic percentile-bootstrap CI for the sample mean."""
    if not values:
        return (0.0, 0.0)
    if len(set(values)) == 1:
        value = round(float(values[0]), 4)
        return (value, value)

    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in range(n))
        for _ in range(n_resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    lower_idx = max(0, int(alpha * n_resamples))
    upper_idx = min(n_resamples - 1, int((1.0 - alpha) * n_resamples) - 1)
    return (round(means[lower_idx], 4), round(means[upper_idx], 4))


# ---------------------------------------------------------------------- #
# Experiment registry
# ---------------------------------------------------------------------- #

_REGISTRY_PATH = Path(__file__).resolve().parent / "experiments" / "registry.yaml"


def _append_to_registry(output: Dict[str, Any]) -> None:
    """Append a run summary to the YAML experiment registry."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — skipping registry update.")
        return

    entry = {
        "run_id": output["run_id"],
        "timestamp": output["timestamp_start"],
        "dataset": output["config"]["dataset"],
        "models": output["config"]["models"],
        "agents": output["config"]["agents"],
        "n_samples": output["config"]["evaluated_samples"],
        "summary": output.get("summary", {}).get("overall", {}),
    }

    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: List[Dict[str, Any]] = []
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            if isinstance(data, dict):
                existing = data.get("experiments", [])

    existing.append(entry)

    with open(_REGISTRY_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(
            {"experiments": existing},
            fh,
            default_flow_style=False,
            sort_keys=False,
        )
    logger.info("Registry updated: %s", _REGISTRY_PATH)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description=(
            "Phase 8 evaluation harness for Medical Research Agent.  "
            "Benchmarks migrated agents (agents/ subgraphs) against "
            "MedQA, BioASQ, or custom datasets using RAGAS-style metrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python eval/run_eval.py --dataset medqa --n_samples 10 --agents pubmed\n"
            "  python eval/run_eval.py --dataset bioasq --model gpt-4o "
            "--output results/gpt4o.json\n"
            "  python eval/run_eval.py --dataset custom --dataset_path data/my_qs.csv\n"
        ),
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["medqa", "bioasq", "custom"],
        help="Benchmark dataset to evaluate against.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to custom dataset file (CSV or JSON). Required when --dataset=custom.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        help=(
            "Single generation model to evaluate (as registered in models.yaml). "
            "Use a separate process and output file for each model. "
            "Default: DEFAULT_LLM_MODEL env var or gpt-4o."
        ),
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help=(
            "Comma-separated list of agents to evaluate. "
            f"Available: {','.join(ALL_AGENT_NAMES)}. "
            "Default: all agents."
        ),
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help=(
            "Independent LLM-as-judge model. "
            "Default: EVAL_JUDGE_MODEL env var or the generation model."
        ),
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of questions to sample from the dataset. Default: all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sub-sampling (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for results. Default: results/<dataset>_<timestamp>.json.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Resolve models
    models = args.models or [os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")]

    # Resolve agents
    if args.agents:
        agent_names = [a.strip() for a in args.agents.split(",")]
    else:
        agent_names = ALL_AGENT_NAMES

    # Resolve output path
    output_path = args.output
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(_REPO_ROOT / "results" / f"{args.dataset}_{ts}.json")

    result = run_evaluation(
        dataset_name=args.dataset,
        models=models,
        agent_names=agent_names,
        judge_model=args.judge_model or os.getenv("EVAL_JUDGE_MODEL"),
        n_samples=args.n_samples,
        seed=args.seed,
        output_path=output_path,
        dataset_path=args.dataset_path,
    )

    # Print summary to stdout
    summary = result.get("summary", {})
    print(f"\n{'='*60}")
    print(f"Evaluation run: {result.get('run_id', '?')}")
    print(
        f"Dataset: {args.dataset} | "
        f"Evaluated samples: {result['config']['evaluated_samples']}"
    )
    print(f"Models: {models} | Agents: {agent_names}")
    print(f"{'='*60}")

    overall = summary.get("overall", {}).get("scores", {})
    if overall:
        print("\nOverall scores:")
        for metric, score in overall.items():
            print(f"  {metric:25s} {score:.4f}")

    by_model = summary.get("by_model", {})
    if len(by_model) > 1:
        print("\nBy model:")
        for model, model_summary in by_model.items():
            scores = model_summary.get("scores", {})
            avg = sum(scores.values()) / len(scores) if scores else 0
            print(f"  {model:25s} avg={avg:.4f}")

    by_agent = summary.get("by_agent", {})
    if len(by_agent) > 1:
        print("\nBy agent:")
        for agent, agent_summary in by_agent.items():
            scores = agent_summary.get("scores", {})
            avg = sum(scores.values()) / len(scores) if scores else 0
            print(f"  {agent:25s} avg={avg:.4f}")

    if output_path:
        print(f"\nResults: {output_path}")

    print()


if __name__ == "__main__":
    main()
