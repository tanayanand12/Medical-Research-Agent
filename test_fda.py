"""
test_fda_agent.py — Integration smoke-test for the FDA regulatory sub-agent.

Mirrors the PubMed agent test pattern exactly.  Validates:
  - Graph topology can be compiled and rendered.
  - FDAFetcher concept extraction + deterministic URL builder work end-to-end.
  - Hybrid retrieval (BM25 + HNSW + RRF) + cross-encoder reranking run without error.
  - Synthesis produces a non-empty answer with FDA record ID citations.

Run from the project root:
    python test_fda_agent.py

Requirements:
    - OPENAI_API_KEY (or configured LiteLLM provider) in .env or environment.
    - rag_engine package installed (SemanticChunker, BM25Index, DenseIndex,
      HybridRetriever, Embedder).
    - agents.base with SubAgentGraph, AgentOutput, load_prompt, _RetrievalDoc.
"""

from dotenv import load_dotenv  # type: ignore
load_dotenv()

from agents.fda_agent.graph import FDAAgentGraph
from graph import get_graph, get_graph_diagram  # type: ignore

# ====================================================================== #
# Orchestrator graph
# ====================================================================== #

print("=== Main Orchestrator Graph ===")
print(get_graph_diagram())

# ====================================================================== #
# FDA sub-agent graph topology
# ====================================================================== #

print("=== FDA Sub-Agent Graph ===")
agent = FDAAgentGraph()
compiled = agent.graph

try:
    compiled.get_graph().print_ascii()
except Exception:
    print("Install grandalf for ASCII: pip install grandalf")

try:
    mermaid = compiled.get_graph().draw_mermaid()
    print(mermaid)
except Exception as e:
    print("Mermaid unavailable:", e)

print()
print("Nodes:", list(compiled.get_graph().nodes.keys()))

# ====================================================================== #
# Query 1 — Adverse events (drug/event endpoint)
# ====================================================================== #

print()
print("=" * 60)
print("=== Running FDA Agent: Adverse Events Query ===")
print("=" * 60)

result = agent.invoke(
    "What are the serious adverse events and safety signals for metformin "
    "in patients with type 2 diabetes, including lactic acidosis reports?"
)

print("=== ANSWER ===")
print(result.answer)
print()
print("=== CITATIONS (FDA Record IDs) ===")
for c in result.citations:
    print(" -", c)
print()
print("=== CONFIDENCE ===", result.confidence)
print("=== MODEL ===", result.model_used)
print("=== TIME ===", f"{result.execution_time_sec:.1f}s")

# ====================================================================== #
# Query 2 — Drug label / regulatory (drug/label endpoint)
# ====================================================================== #

print()
print("=" * 60)
print("=== Running FDA Agent: Drug Label Query ===")
print("=" * 60)

result2 = agent.invoke(
    "What are the contraindications, warnings, and boxed warnings for "
    "warfarin sodium according to FDA drug labeling?"
)

print("=== ANSWER ===")
print(result2.answer)
print()
print("=== CITATIONS (FDA Record IDs) ===")
for c in result2.citations:
    print(" -", c)
print()
print("=== CONFIDENCE ===", result2.confidence)
print("=== MODEL ===", result2.model_used)
print("=== TIME ===", f"{result2.execution_time_sec:.1f}s")

# ====================================================================== #
# Query 3 — Recall (device/recall or food/enforcement endpoint)
# ====================================================================== #

print()
print("=" * 60)
print("=== Running FDA Agent: Recall Query ===")
print("=" * 60)

result3 = agent.invoke(
    "What FDA recalls exist for heparin products due to contamination, "
    "and what were the recall classifications and corrective actions?"
)

print("=== ANSWER ===")
print(result3.answer)
print()
print("=== CITATIONS (FDA Record IDs) ===")
for c in result3.citations:
    print(" -", c)
print()
print("=== CONFIDENCE ===", result3.confidence)
print("=== MODEL ===", result3.model_used)
print("=== TIME ===", f"{result3.execution_time_sec:.1f}s")

# ====================================================================== #
# AgentOutput contract validation
# ====================================================================== #

print()
print("=" * 60)
print("=== AgentOutput Contract Validation ===")
print("=" * 60)

for label, r in [("Q1 adverse_events", result), ("Q2 drug_label", result2), ("Q3 recall", result3)]:
    has_answer    = bool(r.answer and len(r.answer) > 50)
    has_citations = len(r.citations) > 0
    has_conf      = 0.0 <= r.confidence <= 1.0
    has_domain    = r.domain == "fda"
    has_sources   = isinstance(r.sources, list)

    status = "PASS" if all([has_answer, has_citations, has_conf, has_domain, has_sources]) else "FAIL"
    print(
        f"[{status}] {label:30s} | "
        f"answer={has_answer} citations={has_citations} "
        f"conf={has_conf} domain={has_domain} sources={has_sources}"
    )