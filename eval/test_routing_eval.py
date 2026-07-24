from eval.run_routing_eval import summarize_routing


def test_summarize_routing_reports_top1_and_top3_accuracy():
    rows = [
        {"expected": "search_pubmed", "ranked": ["search_pubmed", "search_fda"]},
        {
            "expected": "search_fda",
            "ranked": ["search_pubmed", "search_fda", "search_local_index"],
        },
    ]

    summary = summarize_routing(rows)

    assert summary["n"] == 2
    assert summary["top1_accuracy"] == 0.5
    assert summary["top3_accuracy"] == 1.0
