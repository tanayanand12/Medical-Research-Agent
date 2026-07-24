# Evaluation Results

Values are mean (bootstrap 95% CI); valid metric denominator / evaluated samples.

## Overall

| Metric | Result |
|---|---:|
| Faithfulness (higher is better) | 0.905 (0.790–0.990); 20/20 |
| Answer relevancy (higher is better) | 0.290 (0.160–0.450); 20/20 |
| Answer correctness (higher is better) | 0.218 (0.080–0.375); 20/20 |
| Citation fidelity (higher is better) | 0.738 (0.538–0.900); 20/20 |
| Hallucination rate (lower is better) | 0.090 (0.005–0.210); 20/20 |

## Held-out routing

| Routing metric | Result |
|---|---:|
| Top-1 accuracy | 0.850 |
| Top-3 accuracy | 1.000 |
| Mean reciprocal rank | 0.908 |
| Questions | 20 |

## Retrieval-stage grounding

| Retrieval metric | Result |
|---|---:|
| Context relevance | 0.387 (0.215–0.572); 20/20 |
| Context sufficiency | 0.334 (0.158–0.532); 20/20 |
| Citation-bearing claim support | 0.324 (0.147–0.525); 20/20 |
| Sentence citation coverage | 0.337 (0.193–0.482); 20/20 |

Across 20 evaluations, 0 execution errors were recorded. Latency was mean 137.888 s, median 137.394 s, and p95 222.620 s.

## By agent

| Agent | Faithfulness | Relevancy | Correctness | Citation fidelity | Hallucination rate | Errors | Latency, mean s |
|---|---:|---:|---:|---:|---:|---:|---:|
| pubmed | 0.860 (0.660–1.000); 5/5 | 0.240 (0.040–0.520); 5/5 | 0.180 (0.000–0.460); 5/5 | 1.000 (1.000–1.000); 5/5 | 0.120 (0.000–0.320); 5/5 | 0 | 143.330 |
| fda | 0.960 (0.880–1.000); 5/5 | 0.240 (0.040–0.520); 5/5 | 0.350 (0.000–0.730); 5/5 | 0.550 (0.150–0.950); 5/5 | 0.040 (0.000–0.120); 5/5 | 0 | 168.365 |
| clinical_trials | 0.800 (0.400–1.000); 5/5 | 0.220 (0.040–0.460); 5/5 | 0.100 (0.020–0.200); 5/5 | 0.400 (0.000–0.800); 5/5 | 0.200 (0.000–0.600); 5/5 | 0 | 172.816 |
| local | 1.000 (1.000–1.000); 5/5 | 0.460 (0.100–0.800); 5/5 | 0.240 (0.000–0.600); 5/5 | 1.000 (1.000–1.000); 5/5 | 0.000 (0.000–0.000); 5/5 | 0 | 67.043 |
