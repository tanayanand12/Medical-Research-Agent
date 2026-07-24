# Submission Package

## Recommended positioning

Position the work as a **methods and technical-validation study of precision evidence orchestration**, not as a clinically validated diagnostic or treatment system.

The strongest defensible claims are:

- source-specialized, LLM-agnostic medical evidence orchestration;
- explicit graph states and failure handling;
- live PubMed, FDA, and ClinicalTrials.gov retrieval plus a local-data pathway;
- hybrid retrieval and biomedical reranking;
- sparse fallback and bounded transient-failure recovery;
- held-out routing validation;
- reproducible, agent-stratified evaluation with complete metric denominators;
- retrieval-stage relevance, sufficiency, and citation-support analysis.

Do not claim:

- clinical effectiveness or patient benefit;
- diagnostic accuracy;
- individualized treatment selection;
- calibrated clinical confidence;
- regulatory compliance or readiness;
- superiority over clinicians or other systems;
- generalizability from the 20-question pilot.

## Journal strategy

The current package is best treated as a **pilot methods manuscript**. A top-quartile medical-AI journal may consider the architecture useful, but an A-tier empirical claim will normally require a larger external benchmark, baselines and ablations, and independent clinician review.

Potential journal families, subject to current scope and author preference:

1. Digital-medicine or medical-informatics methods journals.
2. Biomedical informatics system-evaluation journals.
3. Open clinical-AI or health-data-science journals accepting reproducible technical validations.

Before targeting a highly selective journal such as *npj Digital Medicine*, add:

- preregistered clinician-authored evaluation;
- MedQA and BioASQ benchmark subsets;
- non-RAG and single-source baselines;
- reranker, skill-router, and fallback ablations;
- two or more blinded clinician raters with inter-rater agreement;
- claim-level citation-entailment assessment;
- subgroup, fairness, security, and privacy analyses;
- prospective human-factors evaluation if used in a live workflow.

## Files

- `MANUSCRIPT.md` — main blinded manuscript.
- `SUPPLEMENTARY_MATERIAL.md` — architecture, benchmark, metrics, reproducibility, and limitations.
- `COVER_LETTER.md` — adaptable editor cover letter.
- `RESULTS_TABLES.md` — reproducibly generated routing and performance tables.
- `../results/paper_eval_summary.json` — machine-readable final aggregate results.
- `../results/benchmark_all_agents_20260724_214522_rejudged_audited.json` — definitive question-level results, with reference-independent grounding judgments, separate reference-based quality judgments, and audited deterministic citation metrics.
- `../results/benchmark_all_agents_20260724_214522_rejudged.json` — preserved output of the separated rejudging pass before deterministic citation postprocessing.
- `../results/benchmark_all_agents_20260724_214522_audited.json` — preserved predecessor used as input to rejudging; deterministic citation fidelity was recomputed against each agent's actual citation inventory.
- `../results/benchmark_all_agents_20260724_214522.json` — immutable raw run before deterministic postprocessing.
- `../results/routing_holdout_eval_final.json` — held-out source-routing results.
- `../results/retrieval_grounding_eval.json` — post-hoc retrieval relevance, sufficiency, citation-support, and citation-coverage results.
- `../eval/data/medical_benchmark.json` — versioned pilot benchmark.
- `../eval/data/routing_holdout.json` — versioned held-out routing set.
- `../eval/experiments/registry.yaml` — experiment registry.

## Mandatory author actions before submission

These items cannot be inferred from source code:

- insert author names, affiliations, ORCID identifiers, and corresponding-author details;
- confirm CRediT roles;
- confirm funding and conflicts of interest;
- select a journal and apply its word, reference, figure, and structured-abstract limits;
- confirm whether institutional ethics review or exemption is required locally;
- archive a release and add its DOI or permanent repository URL;
- have qualified clinicians review all benchmark reference answers and the manuscript's clinical statements;
- run plagiarism, reference, and authorship-policy checks;
- disclose generative-AI assistance according to the target journal's current policy.

## Final integrity check

Do not submit if the manuscript still contains `{{...}}`, `TBD`, or unverified performance claims. Every reported value must be traceable to `paper_eval_summary.json` and the question-level merged result.
