"""
eval — Phase 8: Evaluation harness for Medical Research Agent.

Provides benchmark datasets, RAGAS-style metrics, and a CLI runner
for evaluating migrated agents (``agents/`` subgraphs) across models,
datasets, and agent combinations.

Public API::

    from eval.metrics import (
        faithfulness, answer_relevancy,
        citation_fidelity, hallucination_rate,
    )
    from eval.datasets import MedQADataset, BioASQDataset
"""
