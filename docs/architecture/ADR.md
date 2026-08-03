# Architecture Decision Records (ADR)

**System:** Medical Research Agent  
**Format:** Context → Decision → Consequences → Status  
**Companion:** [HLD.md](./HLD.md) · [LLD.md](./LLD.md)

---

## ADR-001: Separate runtime verification from offline benchmarking

| | |
|--|--|
| **Status** | Accepted (Phase 1 / Step 2A) |
| **Date** | 2026-07 / 2026-08 |

### Context

The project needs (a) online quality control during `/query` and (b) scientific benchmark scores for a paper. Collapsing both into one path risks gold-label leakage and invalidates claims.

### Decision

Maintain two planes:

1. **Runtime verifier-and-repair** (`runtime_verification/`, graph nodes) — qrel-free.
2. **Offline eval** (`eval/`, `docs/evaluation_protocol.md`) — labeled, post-hoc.

Shared schemas (`EvaluationTrace`, `AttemptEvent`) may be reused; scoring against gold never runs inside the production graph.

### Consequences

- Clearer manuscript claims; no “self-grading” production path.
- Two harnesses to maintain; orchestrator eval must not call into runtime with references.

---

## ADR-002: LangGraph StateGraph over ThreadPool orchestration

| | |
|--|--|
| **Status** | Accepted (Phase 4) |
| **Supersedes** | Legacy `agentic-pipeline-clinical` ThreadPool orchestrator |

### Context

Legacy system used imperative `ThreadPoolExecutor` with feature flags. Hard to checkpoint, reason about branches, or attach typed state.

### Decision

Top-level orchestration is an 8-node `langgraph.StateGraph` with typed `AgentState` and explicit conditional edges.

### Consequences

- Clear control flow and testability.
- Learning curve; large nodes (especially `parallel_retrieve`) need discipline.
- Checkpointing/LangSmith integration deferred but structurally ready.

---

## ADR-003: LiteLLM single LLM gateway

| | |
|--|--|
| **Status** | Accepted (Phase 1 LLM abstraction) |

### Context

Hardcoded OpenAI calls blocked OSS/enterprise comparison required for the paper.

### Decision

All LLM calls go through `LLMClient` + `models.yaml` (LiteLLM). Model matrix for eval is separate but must sync prices before paid runs.

### Consequences

- Swap backends via config.
- Provider-specific quirks still surface; preflight required for new IDs (e.g. retired Grok Fast SKUs).

---

## ADR-004: Skill-discovery YAML manifests (not static feature flags)

| | |
|--|--|
| **Status** | Accepted (Phase 3) |

### Context

Legacy booleans (`pubmed=true`, `fda=false`) do not scale and hide routing quality.

### Decision

Each tool has `skills/*.yaml`; `skill_router` selects via semantic + keyword signals. API may still override with `agents_to_use`.

### Consequences

- Extensible sources without graph edits.
- Routing itself needs offline evaluation (`run_routing_eval.py`).

---

## ADR-005: In-process MCP tools + optional sub-agent graphs

| | |
|--|--|
| **Status** | Accepted |

### Context

True multi-process MCP servers add ops cost; research codebase needs tight iteration.

### Decision

Tools run in-process behind an MCP-style registry contract. Preferred path for core sources is a 4-node `SubAgentGraph` invoked from `parallel_retrieve`; unmatched tools use MCP wrappers.

### Consequences

- Lower latency, shared process resources (reranker singleton).
- Weaker isolation; one bad tool can affect the process.
- `pubmed_deep` remote path remains a telemetry/ops special case.

---

## ADR-006: Bounded, selective repair (not unbounded agent debate)

| | |
|--|--|
| **Status** | Accepted (Phase 1) |

### Context

Unconstrained multi-agent debate is expensive, hard to reproduce, and can amplify hallucinations.

### Decision

- Separate budgets for retrieval retry vs frozen-evidence synthesis repair.
- Caps typically 0–1 at agent and top-level.
- Exhaustion → `evidence_limited` answer, not unrepaired free text.
- High-risk / unknown-entity paths trigger semantic verification; others may accept faster.

### Consequences

- Predictable cost/latency envelopes for pilots.
- Some failures surface as evidence-limited rather than “best effort” prose.

---

## ADR-007: EvaluationTrace as sidecar (do not replace AgentOutput)

| | |
|--|--|
| **Status** | Accepted | Schema | `1.0.0` frozen for pilots |

### Context

Need rich attempt history without breaking existing agent return shapes.

### Decision

Keep agent/MCP outputs; attach versioned `EvaluationTrace` / `AttemptEvent` sidecars on state. API returns redacted traces only on opt-in.

### Consequences

- Backward-compatible agent interfaces.
- Schema evolution must bump versions; v1 frozen for Step 2 pilots.

---

## ADR-008: Evidence-limited degradation over silent pass

| | |
|--|--|
| **Status** | Accepted |

### Context

Earlier behaviors could deliver unrepaired answers after budget exhaustion, or pass when coherence eval failed.

### Decision

Prefer explicit `evidence_limited` terminal states and conservative legacy coherence fallback only when verifier decision is invalid/missing.

### Consequences

- Safer clinician-facing posture.
- Lower “fluent but unsupported” rate; more abstentions.

---

## ADR-009: MedAgentsBench test_hard as primary hard eval row

| | |
|--|--|
| **Status** | Accepted (Step 2B) |
| **N** | 862 official |

### Context

Need a hard, agent-relevant MCQ suite comparable to Tang et al. 2025, without conflating with full MedQA.

### Decision

- Primary hard row: MedAgentsBench `test_hard` (N=862).
- Standalone MedQA remains a separate row.
- Never merge MedQA-hard (100) into MedAgentsBench (already included).
- Never project costs on the unfiltered ~11k pool.
- AfriMedQA (+32) may appear in loads (894); official reporting N stays 862.

### Consequences

- Clean comparison to published MedAgentsBench.
- Adapter must record `loaded_n` / `n_discrepancy`.

---

## ADR-010: Cost pilot before full paid matrices

| | |
|--|--|
| **Status** | Accepted (Step 2A) |

### Context

Full orchestrator RAG multiplies tokens; list prices ≠ measured spend.

### Decision

1. Offline matrix planning at mid tokens (55k in / 4k out) for N=862.
2. Live 20q pilots on cheap verified models first.
3. Credit purchase from measured means: `projected × 1.25 + $5`.

### Consequences

- Avoids premature large credit buys.
- Matrix table is an upper-bound sanity check, not a substitute for telemetry.

---

## ADR-011: No persona tracking in target API

| | |
|--|--|
| **Status** | Accepted |

### Context

Legacy persona HTTP service customised tone; conflicts with evidence-grounded clinician-facing product and complicates evaluation.

### Decision

Remove persona from target API/response path.

### Consequences

- Simpler state and ethics posture.
- No personalized answer adaptation.

---

## ADR-012: Defer Phase 5 observability and CI/CD

| | |
|--|--|
| **Status** | Accepted (explicit product choice) |

### Context

LangSmith/Prometheus and container CI were planned; user cancelled CI/CD; privacy/PHI export policy unfinished.

### Decision

Ship architecture with `trace_id` and local structured logs; defer LangSmith export policy, Prometheus, and CI/CD until requested. Document privacy as a known gap.

### Consequences

- Faster path to scientific eval.
- Production ops maturity incomplete; do not claim full observability.

---

## ADR-013: Date-gated Claude Sonnet pricing in model matrix

| | |
|--|--|
| **Status** | Accepted (Step 2A.1) |

### Context

Claude Sonnet 5 introductory $2/$10 ends 2026-08-31; then $3/$15.

### Decision

Store both rates + date gate in `model_matrix.yaml`; cost tables show intro vs standard rows; pin run dates in registry metadata.

### Consequences

- Avoids underestimating post-cliff spend.
- Pilots before Sep 1 should record which schedule applied.

---

## ADR index

| ID | Title | Status |
|----|-------|--------|
| 001 | Separate runtime vs offline eval | Accepted |
| 002 | LangGraph over ThreadPool | Accepted |
| 003 | LiteLLM gateway | Accepted |
| 004 | Skill YAML discovery | Accepted |
| 005 | In-process MCP + sub-agent graphs | Accepted |
| 006 | Bounded selective repair | Accepted |
| 007 | EvaluationTrace sidecar | Accepted |
| 008 | Evidence-limited degradation | Accepted |
| 009 | MedAgentsBench hard primary row | Accepted |
| 010 | Cost pilot before full matrices | Accepted |
| 011 | No persona tracking | Accepted |
| 012 | Defer Phase 5 / CI-CD | Accepted |
| 013 | Sonnet date-gated pricing | Accepted |

To add a new ADR: append `ADR-0xx` with Status `Proposed`, discuss, then set `Accepted` / `Superseded` / `Rejected`.
