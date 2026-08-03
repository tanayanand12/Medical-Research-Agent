# Architecture Documentation Index

**System:** Medical Research Agent (`medical-research-agent`)  
**Repo:** [tanayanand12/Medical-Research-Agent](https://github.com/tanayanand12/Medical-Research-Agent)  
**Document set version:** 1.0 (aligned with remote `main` @ Step 2B / Phase 1 freeze)  
**Audience:** engineers, reviewers, and paper authors

This folder is the authoritative **system architecture** documentation for the production LangGraph agent and its offline evaluation harness. It complements (does not replace) the scientific evaluation protocol.

| Document | Purpose |
|----------|---------|
| [HLD.md](./HLD.md) | High-level design: context, containers, major flows, boundaries |
| [LLD.md](./LLD.md) | Low-level design: modules, state contracts, algorithms, interfaces |
| [ADR.md](./ADR.md) | Architecture Decision Records (why we chose X over Y) |
| [../evaluation_protocol.md](../evaluation_protocol.md) | Offline benchmarking protocol (MedAgentsBench, cost pilots, metrics) |

## Quick orientation

```
Client → FastAPI (/query) → LangGraph StateGraph (8 nodes)
                              ├─ MCP tools / domain sub-agent graphs
                              ├─ Runtime verifier-and-repair (online)
                              └─ EvaluationTrace / AttemptEvent telemetry
Offline: eval/ cost pilot + datasets (never mutates live answers)
```

## Critical invariant

**Runtime verification ≠ offline benchmarking.**

- Online (`runtime_verification/`, graph nodes): qrel-free checks, bounded retries/repairs, evidence-limited answers.
- Offline (`eval/`): labeled benchmarks, accuracy/RAGAS/cost tables for the paper. Gold answers never enter production.

## Status snapshot

| Layer | Status |
|-------|--------|
| LangGraph orchestration (Phase 4) | Complete |
| Runtime verifier-and-repair (Phase 1) | Frozen on `main` |
| Evaluation protocol + cost pilot (Step 2A) | Frozen on `main` |
| MedAgentsBench adapter stub (Step 2B) | Frozen on `main` |
| LangSmith / Prometheus (Phase 5) | Deferred |
| Full official benchmark runs | Not started (await live 20q cost pilots + credits) |

## How to keep docs honest

When changing graph topology, state fields, verification budgets, or eval separation rules:

1. Update HLD/LLD diagrams and contracts in the same PR when possible.
2. Add or amend an ADR if the decision is non-obvious or reversible-costly.
3. Never claim clinical benefit from retrospective benchmarks in architecture docs.
