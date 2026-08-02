# Evaluation Protocol (Step 2A)

This document defines how **offline benchmarking** is conducted for the Medical Research Agent paper, separately from **runtime verification** (Phase 1). It is the authoritative protocol for smoke tests, cost pilots, and official stratified runs.

## 1. Scope and separation of concerns

| Layer | Purpose | When it runs | Primary artifacts |
| --- | --- | --- | --- |
| **Runtime verification** | Online claim checking, repair, deadline enforcement during live `/query` | Production / pilot orchestrator runs | `EvaluationTrace` v1.0.0, `AttemptEvent` v1.0.0, verification decisions |
| **Offline evaluation (`eval/`)** | Retrospective scoring against labeled benchmarks | Post-hoc harness runs | JSON result files, RAGAS-style metrics, experiment registry |

Runtime verification improves answer quality and traceability; it is **not** a substitute for benchmark accuracy. Offline eval scores answers against gold labels or judge rubrics without modifying the orchestrator loop.

## 2. Primary benchmarks

### 2.1 MedAgentsBench — primary hard agent-evaluation set

**MedAgentsBench** (Tang et al., 2025, [arXiv:2503.07459](https://arxiv.org/abs/2503.07459)) is the **primary hard agent-evaluation benchmark** for this project. It aggregates challenging medical MCQ items selected where baseline models achieve &lt;50% accuracy, and is the target split for official orchestrator accuracy reporting and cost projections at **N = 862**.

| Field | Value |
| --- | --- |
| **Paper** | Tang et al., 2025 — *MedAgentsBench* |
| **arXiv** | [2503.07459](https://arxiv.org/abs/2503.07459) |
| **Split** | `test_hard` only |
| **Official N** | **862** (authoritative; all registry and cost tables use this N) |
| **GitHub** | [gersteinlab/MedicalAgentsBench](https://github.com/gersteinlab/MedicalAgentsBench) |
| **Hugging Face (canonical)** | `super-dainiu/MedicalAgentsBench` |
| **HF alias (same repo)** | `super-dainiu/medagents-benchmark` → resolves to the same dataset id |
| **Avg question length** | ~147 tokens |
| **Options per item** | 3–10 MCQ options depending on source benchmark |

**HF load note:** Both HF URLs resolve to dataset id `super-dainiu/MedicalAgentsBench` (verified via Hugging Face Hub API, Aug 2026). Adapters must record the resolved id in run metadata. Per-source `test_hard` parquet shards are present on HF (e.g. `MedQA/test_hard-*.parquet`, `MedMCQA/test_hard-*.parquet`, …). Full split reconciliation happens at adapter preflight (Step 4+); **N = 862** remains authoritative if per-source counts differ slightly after load.

**Source-benchmark breakdown** (paper / GitHub hard table):

| Source benchmark | Items in `test_hard` |
| --- | ---: |
| MedQA | 100 |
| MedMCQA | 100 |
| PubMedQA | 100 |
| MedExQA | 100 |
| MMLU-Pro | 100 |
| MedXpertQA-Reasoning | 100 |
| MedXpertQA-Understanding | 100 |
| MedBullets | 89 |
| MMLU | 73 |
| AfriMedQA (optional shard) | 32 |
| **Core total (official N)** | **862** |
| **All ten sources** | **894** |

**AfriMedQA reconciliation:** Official **N = 862** is the core `test_hard` split used for evaluation and cost projections. The nine core sources sum to **862**. **AfriMedQA (32)** is an optional tenth shard present on GitHub; when included, loaded rows sum to **894**. Adapters must record `loaded_n` and `n_discrepancy` in run metadata; **862** remains authoritative for official runs and credit planning regardless of whether AfriMedQA is present in a given HF load.

If AfriMedQA is absent in a loaded HF shard but present on GitHub, record the discrepancy and keep **N = 862** as the official evaluation size (do not substitute the unfiltered pool).

#### Non-negotiable MedAgentsBench rules

1. **Separate rows:** `medagentsbench_test_hard` (N = 862) and standalone **MedQA** (full test ~1,273) are **always separate rows** in every paper table, experiment registry entry, and cost projection — never merged or aliased.
2. **No double-counting:** Do **not** merge “MedQA-hard” (100 items) into a standalone MedQA row — those 100 items are **already inside** the 862. Reporting both as independent totals double-counts.
3. **No unfiltered pool:** Do **not** use the unfiltered ~11k-item HF pool for evaluation or cost projections. Only the **862-item `test_hard`** split is valid for official runs.
4. **MedQA remains standalone:** MedAgentsBench does **not** replace MedQA as an independent benchmark; MedQA, MedMCQA, PubMedQA PQA-L, BioASQ, and MIRAGE retain their prior definitions and separate adapter paths.

### 2.2 Additional benchmarks (standalone; not substitutes for MedAgentsBench)

| Benchmark | Task type | Default split | Notes |
| --- | --- | --- | --- |
| **MedQA** (USMLE-style) | Multiple-choice clinical reasoning | Test (~1,273) | **Separate from** MedAgentsBench MedQA-hard (100); see §2.1 rule 2 |
| **MedMCQA** | Multiple-choice (Indian medical entrance) | Validation/test (~4,183+) | Standalone; MedAgentsBench includes 100 MedMCQA-hard items |
| **PubMedQA** (PQA-L) | Yes/No/Maybe literature QA | Labeled subset (500) | Requires retrieval grounding; distinct from MedAgentsBench PubMedQA-hard (100) |
| **BioASQ Task B** | Batch biomedical QA (factoid/list) | Official test batches | Batch-friendly; exact year split resolved at adapter time |
| **MIRAGE** (optional) | RAG-focused medical QA extension | Subset TBD | Optional RAG extension; does not replace MedQA or MedAgentsBench |

**MIRAGE** extends RAG evaluation but is not required for the initial cost pilot or 100-question stratified run.

## 3. Experimental conditions

Each run must declare one **condition** (stored in run metadata):

| Condition | Description |
| --- | --- |
| `full_orchestrator` | Default: classify → discover → parallel retrieve → synthesise → score → coherence → optional fallback → format + runtime verification |
| `closed_book` | No retrieval tools; generation only |
| `retrieval_only` | Retrieve and log context; no synthesis scoring |
| `single_source` | Exactly one MCP tool forced via `agents_to_use` |
| `no_routing` | Skip skill discovery; explicit tool list |
| `no_rerank` | Disable rerank flags in tool context |
| `no_fallback` | Set coherence threshold unreachable or bypass fallback edge |

Conditions are orthogonal to model choice and must be recorded in `eval/experiments/registry.yaml` (or successor registry).

## 4. Metrics by task type

| Task type | Primary metrics | Secondary metrics |
| --- | --- | --- |
| **MedAgentsBench `test_hard`** (862) | Accuracy (MCQ), macro-F1 | Per-source-benchmark breakdown (10 sources), latency, cost |
| **Multiple-choice** (MedQA, MedMCQA — standalone splits) | Accuracy, macro-F1 | Calibration (optional), per-subject breakdown |
| **PubMedQA (PQA-L)** | Accuracy (YNM), F1 on binary collapse | Faithfulness, citation fidelity |
| **BioASQ Task B** | Mean reciprocal rank, strict accuracy (factoid), F1 (list) | Retrieval recall@k |
| **RAG / orchestrator** | Faithfulness, answer relevancy, citation fidelity, hallucination rate | Latency p50/p95, cost USD, repair/verifier rates |

Offline metrics are implemented in `eval/metrics.py` and `eval/retrieval_metrics.py`. Runtime telemetry (tokens, repairs, verifier calls) comes from `evaluation_core` / `EvaluationTrace`.

## 5. Model roles

| Role | Typical models | Config source |
| --- | --- | --- |
| **Generation** | gpt-4o, gpt-4.1, gemini/gemini-2.5-flash, xai/grok-* | `models.yaml`, `eval/configs/model_matrix.yaml` |
| **Judge** (coherence, RAGAS, MCQ extraction) | gemini/gemini-2.5-flash, gpt-4o | Same; may differ from generation model |
| **Embeddings** | text-embedding-3-large, gemini/text-embedding-004, ollama/nomic-embed-text | `models.yaml` embeddings section |
| **Local laptop** | ollama/mistral, ollama/llama3, vLLM-served OSS | `OLLAMA_BASE_URL` or custom `api_base` |

**Preflight requirement:** Live model IDs and API revisions change. Before any paid run, resolve current IDs via `models.yaml` + provider dashboards and record resolved IDs in run metadata.

## 6. Sample-size plan

| Stage | N | Purpose | Gate |
| --- | --- | --- | --- |
| **Smoke** | 20 | Wiring, schema validation, dry-run | All offline tests green; traces validate against v1.0.0 |
| **Cost pilot** | 100 | Measured USD/tokens/latency; credit purchase | `eval/run_cost_pilot.py` markdown reviewed |
| **Official / full — MedAgentsBench** | **862** (`medagentsbench_test_hard`) | Paper-ready hard-agent numbers | Adapter preflight, stratified by source benchmark, frozen config hash |
| **Official / full — other benchmarks** | Full stratified split per benchmark | Standalone MedQA, MedMCQA, PubMedQA PQA-L, BioASQ | Adapters + registry entry; **separate row from MedAgentsBench** |

Stratification (by source benchmark within MedAgentsBench, and by subject/difficulty on standalone sets) is applied at the **official** stage, not the smoke pilot. Cost projections for the hard-agent paper column use **N = 862** only (not the ~11k unfiltered pool).

## 7. Artifact schema and reproducibility

Each evaluation run must produce:

```json
{
  "run_meta": {
    "run_id": "8-char hex",
    "timestamp": "ISO-8601 UTC",
    "protocol_version": "2A",
    "condition": "full_orchestrator",
    "dataset": "medagentsbench_test_hard",
    "model_id": "gpt-4o",
    "judge_model_id": "gemini/gemini-2.5-flash",
    "agents": ["search_pubmed"],
    "n_samples": 100,
    "git_commit": "<sha>",
    "models_yaml_hash": "<sha256 prefix>"
  },
  "schema": {
    "evaluation_trace_version": "1.0.0",
    "cost_pilot_version": "1.0.0"
  },
  "aggregate": { },
  "per_question": [ ]
}
```

**Reproducibility requirements:**

1. Pin `git commit`, `models.yaml` hash, and dataset version/path.
2. Store raw per-question outputs (redacted traces only in public artifacts).
3. Never commit API keys or raw PHI/query text to results buckets.
4. Use `evaluation_core.privacy` redaction for any trace exported from runtime.
5. Register completed runs in `eval/experiments/registry.yaml`.

Cost pilot artifacts follow `eval/cost_pilot.py` → `serialize_pilot_result()`.

## 8. Cost pilot and credit purchase

Run:

```bash
python eval/run_cost_pilot.py --n_samples 20 --models gpt-4o --agents pubmed --output results/cost_pilot.json
python eval/run_cost_pilot.py --dry-run --n_samples 5  # offline tests / CI
```

**Projection formula:**

```
projected_usd(N) = mean_cost_usd_per_question × N
recommended_purchase_usd = projected_usd(N) × (1 + variance_buffer) + fixed_buffer_usd
```

Defaults: `variance_buffer = 0.25` (25%), `fixed_buffer_usd = 5.00`.

Example: if mean = $0.04/question and N = 100 → base $4.00 → recommended **$10.00**.

Multi-model matrices multiply by the number of generation×agent cells and add the fixed buffer once per provider account if shared.

## 9. Claim boundaries

Retrospective benchmark results measure **information retrieval and reasoning under dataset conditions**. They do **not** establish:

- Clinical benefit, safety, or superiority for patient care
- Regulatory clearance or guideline compliance
- Real-world deployment readiness

All paper language must treat scores as **offline proxy metrics** on public benchmarks, not prospective clinical evidence.

## 10. Phase 1 runtime (frozen) vs deferred observability

**Frozen for offline use (EvaluationTrace v1.0.0):**

- Full LangGraph orchestrator with runtime verifier-and-repair loop
- Canonical attempt telemetry and terminal verification binding
- Privacy-safe logging (fingerprints, redaction)

**Deferred (Phase 5+ / observability — not blocking Step 2A):**

- Pre-retrieval deadline propagation (classify, discover, embeddings)
- `pubmed_deep` remote-service telemetry completeness
- Non-cancellable timeout overlap on rerank/fetch
- LangSmith export PHI policy and broad fetcher log redaction
- FDA / CT / local explicit `synthesis_context` manifest parity (PubMed path complete)

## 11. Next implementation slice (after cost pilot)

1. **MedAgentsBench adapter** in `eval/datasets.py`: `test_hard` only (N = 862), provenance metadata, stratified sampling by source benchmark; **never merged into MedQA loader output**.
2. **Dataset adapters** for standalone splits: MedMCQA, PubMedQA PQA-L, BioASQ Task B (no full corpus download in smoke).
3. **Orchestrator eval entrypoint** mirroring `run_cost_pilot.py` but emitting accuracy/RAGAS metrics.
4. **Registry automation** — append run metadata from CLI to `eval/experiments/registry.yaml` (MedAgentsBench and MedQA as separate rows).
5. **Preflight script** — resolve live model IDs from `eval/configs/model_matrix.yaml`.
