#!/usr/bin/env python3
"""
Script to visualize the Medical Research Agent graph in different formats.
"""

from graph import get_graph_diagram, print_graph_ascii

def main():
    print("=== Medical Research Agent Graph Visualization ===\n")

    print("1. Mermaid Diagram (for documentation/rendering):")
    print("-" * 50)
    diagram = get_graph_diagram()
    print(diagram)
    print()

    print("2. ASCII Diagram (terminal display):")
    print("-" * 50)
    ascii_art = print_graph_ascii()
    print(ascii_art)
    print()

    print("3. Graph Structure Summary:")
    print("-" * 50)
    print("Nodes (8 total):")
    nodes = [
        "classify_intent - Medical domain filtering",
        "discover_skills - Tool selection via semantic matching",
        "parallel_retrieve - Concurrent MCP tool invocation",
        "synthesise - LLM answer generation",
        "score_confidence - Coverage-based confidence calculation",
        "evaluate_coherence - Coherence scoring for fallback decision",
        "fallback_regen - Fallback answer regeneration",
        "format_response - AMA citation formatting + disclaimers"
    ]
    for node in nodes:
        print(f"  • {node}")

    print("\nFlow:")
    print("  classify_intent → [conditional] → discover_skills OR format_response")
    print("  discover_skills → parallel_retrieve → synthesise → score_confidence → evaluate_coherence")
    print("  evaluate_coherence → [conditional] → fallback_regen OR format_response")
    print("  fallback_regen → format_response → END")

if __name__ == "__main__":
    main()