from dotenv import load_dotenv # type: ignore 
load_dotenv()

from agents.clinical_trials_agent.graph import ClinicalTrialsAgentGraph
from graph import get_graph, get_graph_diagram

print("=== Main Orchestrator Graph ===")
print(get_graph_diagram())

print("=== Clinical Trials Sub-Agent Graph ===")
agent = ClinicalTrialsAgentGraph()
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
print("=== Running Clinical Trials Agent ===")
result = agent.invoke("What SGLT2 inhibitor trials exist for type 2 diabetes?")

print("=== ANSWER ===")
print(result.answer)
print()
print("=== CITATIONS ===")
for c in result.citations:
    print(" -", c)
print()
print("=== CONFIDENCE ===", result.confidence)