# Supplementary Material

## Precision Evidence Orchestration for Medical Research

### Supplementary Methods S1. Software components

The evaluated system is implemented in Python and uses:

- LangGraph for typed state-machine orchestration.
- A provider-agnostic `LLMClient`: LiteLLM for cloud providers and native HTTP for local Ollama chat and embeddings.
- BM25 sparse retrieval and FAISS HNSW dense retrieval.
- Reciprocal-rank fusion for sparse–dense result fusion.
- `ncbi/MedCPT-Cross-Encoder` for biomedical reranking.
- FastAPI for the query interface.
- LangSmith-compatible tracing, Prometheus metrics, and structured JSON logging.
- MCP-style tool definitions with JSON-schema validation.

The repository's canonical execution path is `research_agent_api_v2.py` → `graph.py` → `nodes/`. Domain sub-agents are located under `agents/`; the evaluation harness is under `eval/`.

### Supplementary Methods S2. Top-level graph

1. `classify_intent`: identifies medical-domain requests and supports nonmedical early exit.
2. `discover_skills`: combines trigger, high-specificity source-intent, domain, and optional embedding signals.
3. `parallel_retrieve`: invokes selected evidence tools while retaining per-tool errors and latency.
4. `synthesise`: produces an evidence-constrained answer through `LLMClient`.
5. `score_confidence`: computes evidence-source coverage.
6. `evaluate_coherence`: evaluates internal response quality.
7. `fallback_regen`: performs at most one conservative regeneration when coherence is low.
8. `format_response`: formats citations, disclaimer, fallback marker, trace ID, and timing.

### Supplementary Methods S3. Agent interface

All evaluated agents return:

```text
answer: string
citations: list[string]
confidence: float
sources: list[object]
model_used: string
domain: string
execution_time_sec: float
error: string | null
```

This interface permits agent-level evaluation without using the legacy wrappers.

### Supplementary Table S1. Pilot benchmark composition

| ID | Agent | Domain | Question focus |
|---|---|---|---|
| pubmed_01 | PubMed | Cardiology | SGLT2 inhibitors in HFpEF |
| pubmed_02 | PubMed | Endocrinology | First-line type 2 diabetes therapy |
| pubmed_03 | PubMed | Cardiology | Anticoagulation and stroke stratification in AF |
| pubmed_04 | PubMed | Oncology | Immune-checkpoint-inhibitor adverse effects |
| pubmed_05 | PubMed | Prevention | Statins for primary prevention |
| fda_01 | FDA | Regulatory | Semaglutide indications |
| fda_02 | FDA | Drug safety | Fluoroquinolone boxed warnings |
| fda_03 | FDA | Oncology safety | Pembrolizumab labeled adverse events |
| fda_04 | FDA | Emergency medicine | Naloxone nasal-spray indication |
| fda_05 | FDA | Contraindications | Metformin contraindications |
| ct_01 | ClinicalTrials.gov | Obesity | Phase 3 GLP-1 trials |
| ct_02 | ClinicalTrials.gov | Hematology | CAR-T trials in multiple myeloma |
| ct_03 | ClinicalTrials.gov | Neurology | Recruiting Alzheimer disease trials |
| ct_04 | ClinicalTrials.gov | Vaccines | mRNA influenza-vaccine trials |
| ct_05 | ClinicalTrials.gov | Nephrology | SGLT2-inhibitor trials in CKD |
| local_01 | Local index | Hypertension | Institutional blood-pressure target |
| local_02 | Local index | Endocrinology | Institutional diabetes threshold |
| local_03 | Local index | Orthopedics | Dental prophylaxis and prosthetic joints |
| local_04 | Local index | Pharmacy | Enoxaparin renal-dose threshold |
| local_05 | Local index | Infection control | C. difficile contact precautions |

The complete questions, reference answers, target-agent labels, retrieval limits, and local synthetic documents are versioned in `eval/data/medical_benchmark.json`.

An independent 20-question routing set, `eval/data/routing_holdout.json`, contains five previously unused source-intent questions per source. Router weights and priority cues were fixed before this set was executed. The routing evaluation reports top-1 accuracy, top-3 accuracy, and mean reciprocal rank.

### Supplementary Methods S4. Metric validity rules

**Faithfulness.** The judge receives retrieved passages and the generated answer. A score of 1 indicates all answer claims are supported by the context. The metric is invalid when no retrieved context is available or when the judge response cannot be parsed.

**Answer relevancy.** The judge receives the question and generated answer. It evaluates whether the answer is direct, specific, and complete.

**Answer correctness.** The judge compares the generated answer with the author-written reference answer for factual agreement and clinically important coverage while allowing wording differences.

**Citation fidelity.** Citation markers of the form `[N]` are checked against the numbered citation inventory returned by the agent, not the number of retrieved chunks. The metric is `valid markers / total markers`. An answer without citation markers receives 0.

**Hallucination rate.** The judge decomposes the answer into atomic claims and returns `unsupported claims / total claims`. Lower scores are better. The metric is invalid when no retrieved context is available.

**Judge-output validation and leakage control.** Faithfulness and hallucination rate are requested together using only the question, retrieved passages, and candidate answer. Answer relevancy and correctness are requested separately using the question, reference answer, and candidate answer. This prevents the reference answer from influencing context-grounding judgments. The parser accepts plain or fenced JSON, requires a numeric `score` in `[0,1]`, and invalidates only the malformed two-metric group. Citation fidelity is computed deterministically.

**Post-hoc retrieval-stage analysis.** A second structured judgment over each preserved question, reference answer, retrieved set, and candidate answer measured: (1) context relevance, the proportion of passages materially relevant to the question; (2) context sufficiency, coverage of clinically important reference-answer facts by the retrieved set; and (3) support for citation-bearing answer claims by any retrieved passage. The third measure is not source-specific citation entailment. Sentence citation coverage was computed deterministically as the fraction of factual sentences containing at least one marker that resolves to the agent citation inventory.

**Scores and uncertainty.** Reciprocal-rank-fusion and MedCPT scores influence ranking internally, but their raw scales are not calibrated across evidence agents and therefore were not pooled as quality outcomes. The revised harness preserves source objects and reports top-rank score, mean score, score margin, and sparse–dense overlap as descriptive diagnostics for future prospective runs. The definitive Ollama Qwen endpoints did not expose calibrated token log-probabilities. No missing token confidence was imputed, because raw generation likelihood is not a validated proxy for medical correctness.

### Supplementary Methods S5. Statistical reporting

For each outcome, the harness reports:

- mean score;
- valid metric denominator;
- deterministic percentile-bootstrap 95% confidence interval using 2,000 resamples and seed 42;
- number of evaluated samples;
- number of execution errors;
- mean, median, and 95th-percentile end-to-end latency.

No hypothesis tests or cross-agent rankings are reported because agent question sets differ in domain and source.

### Supplementary Methods S6. Reproduction

The definitive run used Python 3.12.9 and Ollama server 0.32.1 on Windows. The generator was Qwen2.5-Coder-7B-Instruct Q4_K_M, the judge was Qwen2.5-3B-Instruct Q4_K_M, and both were served locally. `requirements.txt` records the minimum tested direct dependencies; each result artifact records the model identifiers and seed.

From the `medical-research-agent` directory:

```powershell
$env:DEFAULT_LLM_MODEL='ollama/qwen2.5-coder:7b'
$env:EVAL_JUDGE_MODEL='ollama/qwen2.5:3b'
$env:DEFAULT_EMBEDDING_MODEL='ollama/nomic-embed-text'
.\venv\Scripts\python.exe -m pytest test_phase4_integration.py test_reliability.py eval\test_metrics.py eval\test_finalize_results.py eval\test_retrieval_metrics.py eval\test_routing_eval.py test_skill_router_precision.py -q
.\venv\Scripts\python.exe eval\run_routing_eval.py --benchmark eval\data\routing_holdout.json --output results\routing_holdout_eval_final.json
.\venv\Scripts\python.exe eval\run_all_agents_eval.py --model ollama/qwen2.5-coder:7b --judge_model ollama/qwen2.5:3b
.\venv\Scripts\python.exe eval\rejudge_results.py --input results\benchmark_all_agents_<timestamp>_audited.json --output results\benchmark_all_agents_<timestamp>_rejudged.json --judge_model ollama/qwen2.5:3b
.\venv\Scripts\python.exe eval\evaluate_retrieval_grounding.py --input results\benchmark_all_agents_<timestamp>_audited.json --judge_model ollama/qwen2.5:3b
.\venv\Scripts\python.exe eval\finalize_paper_results.py --input results\benchmark_all_agents_<timestamp>_rejudged.json
```

Expected generated artifacts:

```text
results/benchmark_<agent>_<timestamp>.json
results/benchmark_all_agents_<timestamp>.json
results/benchmark_all_agents_<timestamp>_rejudged.json
results/routing_holdout_eval_final.json
results/retrieval_grounding_eval.json
results/paper_eval_summary.json
eval/experiments/registry.yaml
```

Provider credentials are loaded from `.env` and are never written intentionally to manuscript artifacts. The exact generator/judge and embedding model identifiers are recorded in each result file.

### Supplementary Methods S7. Reporting-guideline applicability

This study is an in-silico technical validation:

- CONSORT-AI: not applicable because no randomized clinical trial was conducted.
- SPIRIT-AI: not applicable because this is not a clinical trial protocol.
- DECIDE-AI: not directly applicable because no live clinical decision or patient care was affected.
- WHO AI-for-health principles: used as design and disclosure guidance for human oversight, safety boundaries, transparency, accountability, and equity.

### Supplementary Methods S8. Predefined failure interpretation

An execution failure is not converted into a valid zero-quality answer. It is counted in the error rate and its context-dependent metrics are marked invalid. A judge-format failure is distinct from an agent failure and reduces only the valid denominator for the affected metric. Cloud completions and native Ollama chat or batch-embedding calls use bounded retry for recognized transient failures. Dense-index construction and dense query failures degrade to BM25 retrieval; the acquired source records are not discarded. Per-agent invocations are serialized because their ephemeral indexes are mutable. Authentication, invalid configuration, and exhausted retry budgets remain explicit failures. This separation prevents infrastructure and parser failures from being misreported as model-quality observations.

### Supplementary Limitations

The benchmark is a technical smoke test, not a clinical validation dataset. Public-source retrieval is time-dependent. Local passages are synthetic. Reference answers have not undergone independent clinical adjudication. Distinct 7-billion- and 3-billion-parameter generation and judge models belong to the same Qwen2.5 family; the smaller judge may miss subtle clinical errors. The retrieval-stage analysis was post hoc and uses the same judge family, and sentence citation coverage is a heuristic. No clinician passage-relevance labels were available, so standard gold-label retrieval measures such as precision@k, recall@k, and normalized discounted cumulative gain could not be estimated. Citation-bearing claim support against the retrieved set does not prove source-specific citation entailment. Raw retriever scores are uncalibrated across agents, and token log-probabilities were unavailable. Answer generation invoked designated domain agents directly, so the complete top-level graph was not outcome-evaluated. The system has not undergone prospective workflow, fairness, privacy, security, or patient-outcome evaluation.
