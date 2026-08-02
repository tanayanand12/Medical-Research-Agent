if __name__ != "__main__":
    import pytest

    pytest.skip(
        "manual online PubMed provider smoke script", allow_module_level=True
    )

from dotenv import load_dotenv  # type: ignore
load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)

logging.getLogger("perf").setLevel(logging.INFO)

from agents.pubmed_agent.graph import PubMedAgentGraph
from graph import get_graph, get_graph_diagram

print("=== Main Orchestrator Graph ===")
print(get_graph_diagram())

print("=== PubMed Sub-Agent Graph ===")
agent = PubMedAgentGraph()
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

print()
print("=== Running PubMed Agent ===")
result = agent.invoke("What is the evidence for SGLT2 inhibitors in heart failure with reduced ejection fraction?")

print("=== ANSWER ===")
print(result.answer)
print()
print("=== CITATIONS (AMA) ===")
for c in result.citations:
    print(" -", c)
print()
print("=== CONFIDENCE ===", result.confidence)
print("=== MODEL ===", result.model_used)
print("=== TIME ===", f"{result.execution_time_sec:.1f}s")